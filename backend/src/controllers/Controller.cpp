#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pwd.h>
#include <string>
#include <thread>
#include <unordered_map>

#include "controllers/Controller.h"
#include "services/Service.h"
#include "utils/MinIOHTTPClient.h"
#include "utils/httplib.h"
#include "utils/utils.h"

class ThreadPool {
private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;

public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
      workers.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F> void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
      worker.join();
  }
};

// Global thread pool
ThreadPool extracting_thread_pool(std::thread::hardware_concurrency());

namespace controller {

void registerExtractRoute(httplib::Server &server, TritonClient *tritonClient) {
  server.Post("/extract", [tritonClient](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    try {
      auto pdf_file = req.get_file_value("pdfFile");

      if (pdf_file.filename.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing required PDF file\"}", "application/json");
        return;
      }

      std::string jobId = utils::generateRandomId();

      // Note: We'll use jobId for now, but later we can update it with QR info
      // The actual folder name on MinIO will be determined after QR code reading

      // Initialize job progress
      {
        std::lock_guard<std::mutex> lock(progressMutex);
        jobProgressMap[jobId] = JobProgress{};
        auto &progress = jobProgressMap[jobId];
        progress.processCompleted = false;
        progress.pdfFilename = pdf_file.filename;
        progress.currentStage = "initializing";
        progress.currentStep = "Starting extracting process...";
        progress.progressPercent = 0.0;
      }

      // Queue the extracting task
      extracting_thread_pool.enqueue([=]() {
        bool success =
            examark::services::extract_all_exams_answers(pdf_file.filename, pdf_file.content, tritonClient, jobId);
        if (!success) {
          utils::updateJobProgress(jobId, "error", "extracting process failed", 0, 0, 0.0, true,
                                   "Failed to complete extracting");
          Logger::error("EXTRACTING", "Failed to complete extracting");
        }
      });

      nlohmann::json response;
      response["jobId"] = jobId;
      response["message"] = "Extracting started successfully";
      res.set_content(response.dump(), "application/json");

    } catch (const std::exception &e) {
      res.status = 500;
      nlohmann::json error_response;
      error_response["error"] = "Server error: " + std::string(e.what());
      res.set_content(error_response.dump(), "application/json");
    }
  });

  // Status endpoint
  server.Get("/status/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string jobId = req.path_params.at("jobId");

    std::lock_guard<std::mutex> lock(progressMutex);
    auto it = jobProgressMap.find(jobId);

    nlohmann::json response;
    if (it != jobProgressMap.end()) {
      const auto &progress = it->second;

      if (progress.hasError) {
        response["status"] = "error";
        response["error"] = progress.errorMessage;
      } else if (progress.processCompleted) {
        response["status"] = "completed";
      } else {
        response["status"] = "processing";
      }

      response["currentStage"] = progress.currentStage;
      response["currentStep"] = progress.currentStep;
      response["currentPage"] = progress.currentPage;
      response["totalPages"] = progress.totalPages;
      response["progress"] = progress.progressPercent;
    } else {
      response["status"] = "not_found";
      response["message"] = "Job not found";
    }

    res.set_content(response.dump(), "application/json");
  });

  // Get images list endpoint - returns MinIO URLs
  server.Get("/results/:jobId/images", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string jobId = req.path_params.at("jobId");

    // Check job status
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end() || !it->second.processCompleted) {
        res.status = 404;
        res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
        return;
      }
    }

    try {
      // Initialize MinIO client
      MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);

      // Get QR info and determine folder name
      std::string folderName = jobId;
      {
        std::lock_guard<std::mutex> lock(progressMutex);
        auto it = jobProgressMap.find(jobId);
        if (it != jobProgressMap.end() && !it->second.qrInfo.empty()) {
          folderName = it->second.qrInfo;
        }
      }

      std::vector<std::string> allFiles = minioClient.listFiles(folderName + "/");

      if (allFiles.empty() && folderName != jobId) {
        allFiles = minioClient.listFiles(jobId + "/");
        folderName = jobId;
      }

      // Filter and process image files
      nlohmann::json response;
      nlohmann::json imageUrls = nlohmann::json::array();

      for (const std::string &filename : allFiles) {
        // Only include image files
        if (filename.find(".jpg") != std::string::npos || filename.find(".png") != std::string::npos ||
            filename.find(".jpeg") != std::string::npos) {

          nlohmann::json imageInfo;
          imageInfo["name"] = filename;

          std::string fullObjectName = folderName + "/" + filename;
          imageInfo["url"] = minioClient.getFileUrl(fullObjectName);
          imageUrls.emplace_back(imageInfo);
        }
      }

      // Sort images by page number (page_1.jpg, page_2.jpg, etc.)
      std::sort(imageUrls.begin(), imageUrls.end(), [](const nlohmann::json &a, const nlohmann::json &b) {
        std::string nameA = a["name"];
        std::string nameB = b["name"];

        // Extract page numbers for proper sorting
        auto extractPageNum = [](const std::string &name) -> int {
          size_t start = name.find("page_");
          if (start != std::string::npos) {
            start += 5;
            size_t end = name.find(".", start);
            if (end != std::string::npos) {
              try {
                return std::stoi(name.substr(start, end - start));
              } catch (...) {
                return 0;
              }
            }
          }
          return 0;
        };

        return extractPageNum(nameA) < extractPageNum(nameB);
      });

      response["images"] = imageUrls;

      res.set_header("Content-Type", "application/json");
      res.set_content(response.dump(), "application/json");

    } catch (const std::exception &e) {
      res.status = 500;
      nlohmann::json errorResponse;
      errorResponse["error"] = "Failed to fetch images: " + std::string(e.what());
      res.set_content(errorResponse.dump(), "application/json");
    }
  });

  // Get CSV endpoint - fetches from MinIO or local
  server.Get("/results/:jobId/csv", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string jobId = req.path_params.at("jobId");

    // Check job status
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end() || !it->second.processCompleted) {
        res.status = 404;
        res.set_content("Results not ready or job not found", "text/plain");
        return;
      }
    }

    // Fetch CSV from MinIO
    try {
      MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);

      // Get QR info and determine folder name
      std::string folderName = jobId;
      {
        std::lock_guard<std::mutex> lock(progressMutex);
        auto it = jobProgressMap.find(jobId);
        if (it != jobProgressMap.end() && !it->second.qrInfo.empty()) {
          folderName = it->second.qrInfo;
        }
      }

      // Try to get CSV from QR-based folder first, then fallback to jobId
      std::vector<std::string> objects = minioClient.listFiles(folderName + "/");
      std::string csvObjectName;

      for (const std::string &objectName : objects) {
        if (objectName.find(".csv") != std::string::npos) {
          csvObjectName = objectName;
          break;
        }
      }

      // If not found in QR folder and folderName is not jobId, try jobId folder
      if (csvObjectName.empty() && folderName != jobId) {
        objects = minioClient.listFiles(jobId + "/");
        folderName = jobId; // Update folderName for download
        for (const std::string &objectName : objects) {
          if (objectName.find(".csv") != std::string::npos) {
            csvObjectName = objectName;
            break;
          }
        }
      }

      if (csvObjectName.empty()) {
        res.status = 404;
        res.set_content("CSV file not found", "text/plain");
        return;
      }

      // Download CSV content from MinIO
      std::string fullObjectName = folderName + "/" + csvObjectName;
      std::string csvContent = minioClient.downloadCSV(fullObjectName);
      if (csvContent.empty()) {
        res.status = 500;
        res.set_content("Failed to download CSV from storage", "text/plain");
        return;
      }

      // Return CSV content
      res.set_header("Content-Type", "text/csv");
      res.set_header("Content-Disposition", "attachment; filename=\"results.csv\"");
      res.set_content(csvContent, "text/csv");

    } catch (const std::exception &e) {
      res.status = 500;
      res.set_content("Failed to fetch CSV: " + std::string(e.what()), "text/plain");
    }
  });

  // Upload CSV endpoint - uploads updated CSV to MinIO
  server.Post("/upload-csv/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string jobId = req.path_params.at("jobId");

    // Check if job exists
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end()) {
        res.status = 404;
        res.set_content("{\"error\":\"Job not found\"}", "application/json");
        return;
      }
    }

    try {
      // Check if CSV file is present in request
      if (!req.has_file("csvFile")) {
        res.status = 400;
        res.set_content("{\"error\":\"No CSV file provided\"}", "application/json");
        return;
      }

      const auto &file = req.get_file_value("csvFile");
      std::string csvContent = file.content;

      if (csvContent.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Empty CSV content\"}", "application/json");
        return;
      }

      // Upload to MinIO
      MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);

      // Get QR info and determine folder name
      std::string folderName = jobId;
      {
        std::lock_guard<std::mutex> lock(progressMutex);
        auto it = jobProgressMap.find(jobId);
        if (it != jobProgressMap.end() && !it->second.qrInfo.empty()) {
          folderName = it->second.qrInfo;
        }
      }

      // Try to find existing CSV file in QR-based folder first, then fallback to jobId
      std::vector<std::string> objects = minioClient.listFiles(folderName + "/");
      std::string csvObjectName;

      for (const std::string &objectName : objects) {
        if (objectName.find(".csv") != std::string::npos) {
          csvObjectName = folderName + "/" + objectName;
          break;
        }
      }

      // If not found in QR folder and folderName is not jobId, try jobId folder
      if (csvObjectName.empty() && folderName != jobId) {
        objects = minioClient.listFiles(jobId + "/");
        for (const std::string &objectName : objects) {
          if (objectName.find(".csv") != std::string::npos) {
            csvObjectName = jobId + "/" + objectName;
            folderName = jobId; // Update folderName for consistency
            break;
          }
        }
      }

      if (csvObjectName.empty()) {
        // Create new CSV file name if none exists - use QR folder if available
        csvObjectName = folderName + "/results.csv";
      }

      // Upload the updated CSV content
      if (!minioClient.uploadCSV(csvObjectName, csvContent)) {
        res.status = 500;
        res.set_content("{\"error\":\"Failed to upload CSV to MinIO\"}", "application/json");
        return;
      }

      nlohmann::json response;
      response["message"] = "CSV uploaded successfully";
      response["objectName"] = csvObjectName;
      res.set_content(response.dump(), "application/json");

    } catch (const std::exception &e) {
      res.status = 500;
      nlohmann::json errorResponse;
      errorResponse["error"] = "Failed to upload CSV: " + std::string(e.what());
      res.set_content(errorResponse.dump(), "application/json");
    }
  });
}

} // namespace controller