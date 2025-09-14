#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pwd.h>
#include <string>
#include <unordered_map>

#include "controllers/Controller.h"
#include "services/Service.h"
#include "utils/Logger.h"
#include "utils/MinIOHTTPClient.h"
#include "utils/httplib.h"
#include "utils/utils.h"

using json = nlohmann::json;

// Global map to store completed results
std::unordered_map<std::string, std::vector<nlohmann::json>> client_results;
std::mutex client_results_mutex;

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
ThreadPool grading_thread_pool(std::thread::hardware_concurrency());

namespace controller {
void registerSSEEndpoint(httplib::Server &server) {
  server.Get("/events/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    std::string jobId = req.path_params.at("jobId");

    // Add CORS and SSE headers
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // Add the content provider to stream events
    res.set_content_provider("text/event-stream", [jobId](size_t offset, httplib::DataSink &sink) {
      while (true) {
        {
          std::lock_guard<std::mutex> lock(progressMutex);
          auto it = jobProgressMap.find(jobId);
          if (it != jobProgressMap.end()) {
            json event;
            event["currentStage"] = it->second.currentStage;
            event["currentStep"] = it->second.currentStep;
            event["progress"] = it->second.progressPercent;
            event["status"] =
                it->second.processCompleted ? "completed" : (it->second.hasError ? "error" : "processing");
            event["currentPage"] = it->second.currentPage;
            event["totalPages"] = it->second.totalPages;

            std::string data = "data: " + event.dump() + "\n\n";
            sink.write(data.c_str(), data.length());
            sink.write(":\n\n", 3);

            if (it->second.processCompleted || it->second.hasError) {
              return false;
            }
          } else {
            // Job not found
            std::string data = "data: {\"status\":\"not_found\",\"message\":\"Job not found\"}\n\n";
            sink.write(data.c_str(), data.length());
            return false;
          }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      return false;
    });
  });
}

void registerExtractRoute(httplib::Server &server, TritonClient *tritonClient) {
  server.Post("/extract", [tritonClient](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    try {
      // Parse file and parameters
      auto range = req.files.equal_range("pdfFiles");
      if (range.first == req.files.end()) {
        res.status = 400;
        res.set_content("{\"error\":\"No PDF files provided\"}", "application/json");
        return;
      }

      // Parse qr-info from request
      std::unordered_set<std::string> allowedQRCodes;
      auto qrInfoIt = req.files.find("qr-info");
      if (qrInfoIt != req.files.end()) {
        try {
          nlohmann::json qrInfo = nlohmann::json::parse(qrInfoIt->second.content);

          for (auto it = qrInfo.begin(); it != qrInfo.end(); ++it) {
            if (it.value().is_string()) {
              allowedQRCodes.insert(it.value().get<std::string>());
            }
          }

        } catch (const std::exception &e) {
          Logger::error("EXTRACTING", "Failed to parse qr-info: " + std::string(e.what()));
        }
      }

      // Generate a unique session ID for this client's batch
      std::string sessionId = utils::generateUUIDv4();

      // Create session folder
      std::string sessionFolder = "sessions/" + sessionId;
      try {
        if (!std::filesystem::create_directories(sessionFolder)) {
          Logger::error("EXTRACTING", "Failed to create session folder: " + sessionFolder);
          res.status = 500;
          res.set_content("{\"error\":\"Failed to create session folder\"}", "application/json");
          return;
        }
      } catch (const std::exception &e) {
        Logger::error("EXTRACTING", "Exception creating session folder: " + std::string(e.what()));
        res.status = 500;
        res.set_content("{\"error\":\"Failed to create session folder: " + std::string(e.what()) + "\"}",
                        "application/json");
        return;
      }

      // Initialize session in client_results
      {
        std::lock_guard<std::mutex> lock(client_results_mutex);
        client_results[sessionId] = std::vector<nlohmann::json>();
      }

      // First, save all PDF files to the session folder
      std::vector<std::pair<std::string, std::string>> fileData;
      std::vector<std::pair<std::string, std::string>> invalidFiles;
      int count = 0;

      for (auto it = range.first; it != range.second; ++it) {
        const auto &file = it->second;
        if (file.filename.find(".pdf") == std::string::npos) {
          continue;
        }

        // Save file to session folder
        std::string filePath = sessionFolder + "/" + file.filename;
        std::ofstream outFile(filePath, std::ios::binary);
        if (outFile.is_open()) {
          outFile.write(file.content.c_str(), file.content.size());
          outFile.close();

          if (!allowedQRCodes.empty()) {
            std::string extractedQR = examark::services::get_pdf_qr_code(file.content);
            if (extractedQR.empty()) {
              // Return immediately if a file has no QR code
              nlohmann::json errorResponse;
              errorResponse["status"] = "error";
              errorResponse["message"] = "Invalid QR code detected in uploaded file";

              nlohmann::json fileInfo;
              fileInfo["filename"] = file.filename;
              fileInfo["qr_code"] = "";
              fileInfo["error"] = "QR code not found";

              errorResponse["invalid_files"] = nlohmann::json::array();
              errorResponse["invalid_files"].push_back(fileInfo);

              std::filesystem::remove_all(sessionFolder);

              res.status = 400;
              res.set_content(errorResponse.dump(), "application/json");
              return;
            } else if (allowedQRCodes.find(extractedQR) == allowedQRCodes.end()) {
              // Return immediately if a file has an invalid QR code
              nlohmann::json errorResponse;
              errorResponse["status"] = "error";
              errorResponse["message"] = "Invalid QR code detected in uploaded file";

              nlohmann::json fileInfo;
              fileInfo["filename"] = file.filename;
              fileInfo["qr_code"] = extractedQR;
              fileInfo["error"] = "QR code does not match allowed classes";

              errorResponse["invalid_files"] = nlohmann::json::array();
              errorResponse["invalid_files"].push_back(fileInfo);

              std::filesystem::remove_all(sessionFolder);

              res.status = 400;
              res.set_content(errorResponse.dump(), "application/json");
              return;
            }
          }

          fileData.push_back({file.filename, file.content});
          ++count;
        }
      }

      // Now process files sequentially
      nlohmann::json response;
      response["metadata"]["sessionId"] = sessionId;
      response["metadata"]["totalPDFs"] = count;
      response["metadata"]["timestamp"] = utils::getCurrentTimestamp();
      response["data"] = nlohmann::json::array();

      // Create jobs and enqueue them for asynchronous processing
      for (const auto &[filename, content] : fileData) {
        std::string jobId = utils::generateUUIDv4();

        Logger::info("EXTRACTING", "Processing file: " + filename + " with jobId: " + jobId);

        // Initialize progress tracking
        {
          std::lock_guard<std::mutex> lock(progressMutex);
          jobProgressMap[jobId] = JobProgress{};
          auto &progress = jobProgressMap[jobId];
          progress.processCompleted = false;
          progress.pdfFilename = filename;
          progress.currentStage = "queued";
          progress.currentStep = "Job queued for processing";
          progress.progressPercent = 0.0;
        }

        // Add job to response
        nlohmann::json jobResult;
        jobResult["jobId"] = jobId;
        jobResult["pdf"] = filename;
        jobResult["status"] = "queued";
        response["data"].push_back(jobResult);

        Logger::info("EXTRACTING", "Queuing file: " + filename + " with jobId: " + jobId);

        // Enqueue the processing task to thread pool
        grading_thread_pool.enqueue(
            [=]() { processFileAsync(filename, content, tritonClient, jobId, sessionId, sessionFolder); });
      }

      // Return response immediately with all job IDs
      res.set_header("Content-Type", "application/json");
      res.set_content(response.dump(), "application/json");

      Logger::info("EXTRACTING",
                   "Started processing " + std::to_string(count) + " files asynchronously for session: " + sessionId);

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

  // Get completed results for a specific client session
  server.Get("/results/session/:sessionId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string sessionId = req.path_params.at("sessionId");

    nlohmann::json response;
    response["metadata"]["timestamp"] = utils::getCurrentTimestamp();
    response["metadata"]["sessionId"] = sessionId;
    response["data"] = nlohmann::json::array();

    {
      std::lock_guard<std::mutex> lock(client_results_mutex);
      auto it = client_results.find(sessionId);
      if (it != client_results.end()) {
        // Return all results (both completed and error)
        for (const auto &result : it->second) {
          response["data"].push_back(result);
        }
        response["metadata"]["totalPDFs"] = response["data"].size();
      } else {
        response["metadata"]["totalPDFs"] = 0;
      }
    }

    res.set_content(response.dump(), "application/json");
  });

  // Get specific job result (for individual job queries)
  server.Get("/results/:jobId/complete", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string jobId = req.path_params.at("jobId");

    {
      std::lock_guard<std::mutex> lock(client_results_mutex);
      // Search through all sessions for this jobId
      for (const auto &[sessionId, results] : client_results) {
        for (const auto &result : results) {
          if (result["jobId"] == jobId) {
            res.set_content(result.dump(), "application/json");
            return;
          }
        }
      }
    }

    res.status = 404;
    res.set_content("{\"error\":\"Job result not found\"}", "application/json");
  });

  server.Delete("/results/session/:sessionId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    std::string sessionId = req.path_params.at("sessionId");

    {
      std::lock_guard<std::mutex> lock(client_results_mutex);
      auto it = client_results.find(sessionId);
      if (it != client_results.end()) {
        client_results.erase(it);
        res.set_content("{\"message\":\"Session data cleared\"}", "application/json");
      } else {
        res.status = 404;
        res.set_content("{\"error\":\"Session not found\"}", "application/json");
      }
    }
  });
}

// Add this new function to handle individual file processing
void processFileAsync(const std::string &filename, const std::string &fileContent, TritonClient *tritonClient,
                      const std::string &jobId, const std::string &sessionId, const std::string &sessionFolder) {
  try {
    Logger::info("EXTRACTING", "Starting async processing for file: " + filename + " with jobId: " + jobId);

    // Update progress to processing - fix the function call
    utils::updateJobProgress(jobId, "initializing", "Starting extraction process...", 0, 0, 0.0, false, "");

    // Process the file
    std::string extraction_result =
        examark::services::extract_all_exams_answers(filename, fileContent, tritonClient, jobId);

    // Parse and handle results
    try {
      nlohmann::json result = nlohmann::json::parse(extraction_result);

      if (result["status"] == "error") {
        utils::updateJobProgress(jobId, "error", "Extraction process failed", 0, 0, 0.0, true, result["message"]);
        Logger::error("EXTRACTING", "Failed to complete extracting for jobId: " + jobId);

        // Store error result
        {
          std::lock_guard<std::mutex> lock(client_results_mutex);
          auto it = client_results.find(sessionId);
          if (it != client_results.end()) {
            nlohmann::json jobResult;
            jobResult["jobId"] = jobId;
            jobResult["pdf"] = filename;
            jobResult["status"] = "error";
            jobResult["error"] = result["message"];
            it->second.push_back(jobResult);
          }
        }
      } else {
        // Mark as completed - fix the function call
        utils::updateJobProgress(jobId, "completed", "All processing completed successfully", 0, 0, 100.0, false, "");

        // Store successful result
        {
          std::lock_guard<std::mutex> lock(client_results_mutex);
          auto it = client_results.find(sessionId);
          if (it != client_results.end()) {
            nlohmann::json jobResult;
            jobResult["jobId"] = jobId;
            jobResult["pdf"] = result["pdf"];
            jobResult["class"] = result["class"];
            jobResult["csv"] = result["csv"];
            jobResult["images"] = result["images"];
            jobResult["status"] = result["status"];
            it->second.push_back(jobResult);
          }
        }

        Logger::info("EXTRACTING", "Successfully completed extracting for jobId: " + jobId);
      }
    } catch (const std::exception &e) {
      utils::updateJobProgress(jobId, "error", "Failed to parse extraction result", 0, 0, 0.0, true, e.what());
      Logger::error("EXTRACTING", "Failed to parse extraction result for jobId: " + jobId + " - " + e.what());

      // Store error result
      {
        std::lock_guard<std::mutex> lock(client_results_mutex);
        auto it = client_results.find(sessionId);
        if (it != client_results.end()) {
          nlohmann::json jobResult;
          jobResult["jobId"] = jobId;
          jobResult["pdf"] = filename;
          jobResult["status"] = "error";
          jobResult["error"] = e.what();
          it->second.push_back(jobResult);
        }
      }
    }

  } catch (const std::exception &e) {
    Logger::error("EXTRACTING", "Exception during async processing for jobId " + jobId + ": " + e.what());
    utils::updateJobProgress(jobId, "error", "Processing failed", 0, 0, 0.0, true, e.what());

    // Store error result
    {
      std::lock_guard<std::mutex> lock(client_results_mutex);
      auto it = client_results.find(sessionId);
      if (it != client_results.end()) {
        nlohmann::json jobResult;
        jobResult["jobId"] = jobId;
        jobResult["pdf"] = filename;
        jobResult["status"] = "error";
        jobResult["error"] = "Processing failed: " + std::string(e.what());
        it->second.push_back(jobResult);
      }
    }
  }
}

} // namespace controller