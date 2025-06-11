#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "utils/httplib.h"
#include "controllers/Controller.h"

const std::string USER_NAME = []() -> std::string {
  struct passwd *pw = getpwuid(getuid());
  return pw ? std::string(pw->pw_name) : "root";
}();

class ThreadPool {
private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;
    
public:
  /**
   * @brief Constructs a ThreadPool with a specified number of worker threads.
   *
   * Initializes the thread pool and starts the given number of worker threads.
   * Each worker thread continuously waits for tasks to be available in the queue,
   * and executes them as they arrive. The threads will exit when the pool is stopped
   * and there are no remaining tasks.
   *
   * @param threads The number of worker threads to create in the pool.
   */
  ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
      workers.emplace_back([this] {
        while(true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] { 
              return this->stop || !this->tasks.empty(); 
            });
            
            if(this->stop && this->tasks.empty())
              return;
                
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }
  
  /**
   * @brief Enqueues a new task into the thread pool.
   *
   * Adds a callable object (such as a lambda, function, or functor) to the task queue.
   * The task will be executed by one of the worker threads when available.
   *
   * @tparam F The type of the callable object to enqueue.
   * @param f The callable object to be executed by the thread pool.
   *
   * @note This function is thread-safe and can be called concurrently from multiple threads.
   */
  template<class F>
  void enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
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
      for(std::thread &worker: workers)
        worker.join();
    }
};

// Global thread pool
ThreadPool grading_thread_pool(std::thread::hardware_concurrency());

// Tracking grading progress
struct JobProgress {
  bool processCompleted;
  std::string outputDir;
  std::string pdfFilename;
  std::string csvContent;
  std::string currentStage;
  std::string currentStep;
  int currentPage;
  int totalPages;
  double progressPercent;
  bool hasError;
  std::string errorMessage;
  
  JobProgress() : processCompleted(false), currentPage(0), totalPages(0), 
                  progressPercent(0.0), hasError(false) {}
};

// Global map to track grading results
std::unordered_map<std::string, JobProgress> jobProgressMap;
std::mutex progressMutex;

void updateJobProgress(
  const std::string& jobId, const std::string& stage, 
  const std::string& step, int currentPage = 0, int totalPages = 0, 
  double progressPercent = 0.0, bool isError = false, const std::string& errorMsg = "") 
{
  std::lock_guard<std::mutex> lock(progressMutex);
  auto& progress = jobProgressMap[jobId];
  progress.currentStage = stage;
  progress.currentStep = step;
  progress.currentPage = currentPage;
  progress.totalPages = totalPages;
  progress.progressPercent = progressPercent;
  progress.hasError = isError;
  progress.errorMessage = errorMsg;
  
  std::string logMsg = "Job " + jobId + " [" + stage + "]: " + step;
  if (totalPages > 0) {
    logMsg += " (" + std::to_string(currentPage) + "/" + std::to_string(totalPages) + ")";
  }
  if (progressPercent > 0) {
    logMsg += " - " + std::to_string((int)progressPercent) + "%";
  }
}

void registerGradingRoute(httplib::Server& server, TritonClient* tritonClient) {
  // Main grading endpoint
  server.Post("/grade", [tritonClient](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
        
    auto pdfFile = req.get_file_value("pdfFile");
    if (pdfFile.filename.empty() || pdfFile.content.empty()) {
      res.status = 400;
      res.set_content("PDF file not provided or is empty.", "text/plain");
      Logger::error("CONTROLLER", "PDF file not provided or is empty.");
      return;
    }

    auto csvFile = req.get_file_value("csvFile");
    if (csvFile.filename.empty() || csvFile.content.empty()) {
      res.status = 400;
      res.set_content("CSV file not provided or is empty.", "text/plain");
      Logger::error("CONTROLLER", "CSV file not provided or is empty.");
      return;
    }

    // Generate a job ID based on filename and timestamp
    std::string timestamp = std::to_string(std::time(nullptr));
    std::string baseName = pdfFile.filename.substr(0, pdfFile.filename.find_last_of('.'));
    std::string jobId = baseName + "_" + timestamp;
    
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto& progress = jobProgressMap[jobId];
      progress.processCompleted = false;
      progress.pdfFilename = pdfFile.filename;
      progress.csvContent = csvFile.content;
      progress.currentStage = "initializing";
      progress.currentStep = "Starting processing...";
      progress.progressPercent = 0.0;
    }
        
    // Respond with job ID (use JSON format)
    res.set_header("Content-Type", "application/json");
    res.set_content("{\"jobId\":\"" + jobId + "\",\"message\":\"Grading request received.\"}", "application/json");
    
    grading_thread_pool.enqueue([pdfFile, csvFile, jobId, timestamp, tritonClient]() {
      std::string baseName = pdfFile.filename.substr(0, pdfFile.filename.find_last_of('.'));
      std::string outputDir = "/home/" + USER_NAME + "/examark-data/" + baseName + "_" + timestamp;
        
      bool success = grading(pdfFile.filename, pdfFile.content, csvFile.content, outputDir, tritonClient, jobId);
      
      std::lock_guard<std::mutex> lock(progressMutex);
      auto& progress = jobProgressMap[jobId];
      if (success) {
        progress.processCompleted = true;
        progress.outputDir = outputDir;
        progress.currentStage = "completed";
        progress.currentStep = "Grading completed successfully";
        progress.progressPercent = 100.0;
      } else {
        progress.hasError = true;
        if (progress.errorMessage.empty()) {
          progress.errorMessage = "Grading process failed";
        }
        progress.currentStage = "error";
        progress.currentStep = "Grading failed";
      }
    });
  });
    
  // Status endpoint
  server.Get("/status/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
        
    std::string jobId = req.path_params.at("jobId");
        
    std::lock_guard<std::mutex> lock(progressMutex);
    auto it = jobProgressMap.find(jobId);
    if (it == jobProgressMap.end()) {
      res.status = 404;
      res.set_content("{\"error\":\"Job not found\"}", "application/json");
      return;
    }
        
    const auto& progress = it->second;
    nlohmann::json response;
    
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
    
    res.set_content(response.dump(), "application/json");
  });
    
  // Get CSV results endpoint
  server.Get("/results/:jobId/csv", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
        
    std::string jobId = req.path_params.at("jobId");
        
    // Get the job result
    std::string outputDir;
    std::string filename;
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end() || !it->second.processCompleted) {
        res.status = 404;
        res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
        return;
      }
      outputDir = it->second.outputDir;
      filename = it->second.pdfFilename;
    }
    
    // Find the CSV file
    std::string baseName = filename.substr(0, filename.find_last_of('.'));
    std::string csvPath = outputDir + "/" + baseName + ".csv";
    
    if (!std::filesystem::exists(csvPath)) {
      res.status = 404;
      res.set_content("{\"error\":\"CSV file not found\"}", "application/json");
      return;
    }
    
    // Read the CSV file
    std::ifstream csvFile(csvPath);
    if (!csvFile.is_open()) {
      res.status = 500;
      res.set_content("{\"error\":\"Failed to read CSV file\"}", "application/json");
      return;
    }
    
    std::string csvContent((std::istreambuf_iterator<char>(csvFile)), std::istreambuf_iterator<char>());
    
    res.set_header("Content-Type", "text/csv");
    res.set_header("Content-Disposition", "attachment; filename=\"results.csv\"");
    res.set_content(csvContent, "text/csv");
  });
    
  // Get images list endpoint
  server.Get("/results/:jobId/images", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
      
    std::string jobId = req.path_params.at("jobId");
      
    // Get the job result
    std::string outputDir;
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end() || !it->second.processCompleted) {
        res.status = 404;
        res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
        return;
      }
      outputDir = it->second.outputDir;
    }
      
    // Get list of images
    std::vector<std::string> imageList;
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
      if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
        imageList.push_back(entry.path().filename().string());
      }
    }
      
    // Create JSON response with image URLs
    nlohmann::json response;
    response["images"] = imageList;
      
    res.set_header("Content-Type", "application/json");
    res.set_content(response.dump(), "application/json");
  });
  
  // Get specific image endpoint
  server.Get("/results/:jobId/images/:imageName", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
      
    std::string jobId = req.path_params.at("jobId");
    std::string imageName = req.path_params.at("imageName");
      
    // Get the job result
    std::string outputDir;
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it == jobProgressMap.end() || !it->second.processCompleted) {
        res.status = 404;
        res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
        return;
      }
      outputDir = it->second.outputDir;
    }
      
    // Find the image
    std::string imagePath = outputDir + "/" + imageName;
    if (!std::filesystem::exists(imagePath)) {
      res.status = 404;
      res.set_content("{\"error\":\"Image not found\"}", "application/json");
      return;
    }
      
    // Read the image file
    std::ifstream imageFile(imagePath, std::ios::binary);
    if (!imageFile.is_open()) {
      res.status = 500;
      res.set_content("{\"error\":\"Failed to read image file\"}", "application/json");
      return;
    }
      
    std::string imageData((std::istreambuf_iterator<char>(imageFile)), std::istreambuf_iterator<char>());
      
    // Determine content type based on extension
    std::string contentType = "image/jpeg";  // Default
    if (imageName.substr(imageName.find_last_of(".") + 1) == "png") {
      contentType = "image/png";
    }
      
    res.set_header("Content-Type", contentType);
    res.set_content(imageData, contentType);
  });
  
  // Handle CORS preflight requests
  server.Options("/grade", [](const httplib::Request&, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 204;
  });
  
  server.Options("/results/(.*)", [](const httplib::Request&, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 204;
  });

  server.Post("/cancel/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    
    std::string jobId = req.path_params.at("jobId");
    
    {
      std::lock_guard<std::mutex> lock(progressMutex);
      auto it = jobProgressMap.find(jobId);
      if (it != jobProgressMap.end()) {
        auto& progress = it->second;
        progress.hasError = true;
        progress.errorMessage = "Job cancelled by user";
        progress.currentStage = "cancelled";
        progress.currentStep = "Process terminated";
        
        // Note: In a more sophisticated implementation, you would need to
        // actually kill the running thread/process. For now, we just mark it as cancelled.
        // The grading function should check for cancellation status periodically.
      }
    }
    
    res.set_header("Content-Type", "application/json");
    res.set_content("{\"message\":\"Job cancellation requested\"}", "application/json");
  });

  // Handle CORS preflight for cancel endpoint
  server.Options("/cancel/(.*)", [](const httplib::Request&, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 204;
  });

  // Re-grade endpoint
  server.Post("/regrade", [tritonClient](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    
    // Check if request is multipart form data
    if (req.is_multipart_form_data()) {
      auto jobIdPart = req.get_file_value("jobId");
      auto csvFilePart = req.get_file_value("csvFile");
      auto answerKeyPart = req.get_file_value("answerKey");

      std::string jobId = jobIdPart.content;
      
      // Check if any required field is missing
      if (jobId.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing jobId parameter\"}", "application/json");
        Logger::error("CONTROLLER", "Missing jobId in regrade request");
        return;
      }
      
      if (csvFilePart.content.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing or empty csvFile\"}", "application/json");
        Logger::error("CONTROLLER", "Missing or empty csvFile in regrade request");
        return;
      }
      
      if (answerKeyPart.content.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing or empty answerKey file\"}", "application/json");
        Logger::error("CONTROLLER", "Missing or empty answerKey in regrade request");
        return;
      }
      
      std::string csvData = csvFilePart.content;
      std::string answerKeyData = answerKeyPart.content;
      
      // Check if the original job exists and get output directory
      std::string outputDir;
      {
        std::lock_guard<std::mutex> lock(progressMutex);
        auto it = jobProgressMap.find(jobId);
        if (it == jobProgressMap.end() || !it->second.processCompleted) {
          res.status = 404;
          res.set_content("{\"error\":\"Original job not found or not completed\"}", "application/json");
          Logger::error("CONTROLLER", "Original job not found: " + jobId);
          return;
        }
        outputDir = it->second.outputDir;
      }
      
      try {
        // Call the re-grading service directly (synchronous)
        bool success = regradeExam(outputDir, csvData, answerKeyData, jobId, jobId);
        
        if (success) {
          {
            std::lock_guard<std::mutex> lock(progressMutex);
            auto originalIt = jobProgressMap.find(jobId);
            if (originalIt != jobProgressMap.end()) {
              originalIt->second.processCompleted = true;
              originalIt->second.currentStage = "completed";
              originalIt->second.currentStep = "Re-grading completed";
              originalIt->second.progressPercent = 100.0;
            }
          }
          
          // Respond with success
          res.set_header("Content-Type", "application/json");
          res.set_content("{\"status\":\"success\",\"jobId\":\"" + jobId + "_regrade\",\"message\":\"Re-grading completed successfully\"}", "application/json");
          
        } else {
          res.status = 500;
          res.set_content("{\"error\":\"Re-grading process failed\"}", "application/json");
        }
        
      } catch (const std::exception& e) {
        Logger::error("CONTROLLER", "Exception in re-grading: " + std::string(e.what()));
        res.status = 500;
        res.set_content("{\"error\":\"Re-grading exception: " + std::string(e.what()) + "\"}", "application/json");
      }
      
    } else {
      res.status = 400;
      res.set_content("{\"error\":\"Request must be multipart form data\"}", "application/json");
      Logger::error("CONTROLLER", "Regrade request is not multipart form data");
    }
  });

  // Handle CORS preflight for regrade endpoint
  server.Options("/regrade", [](const httplib::Request&, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 204;
  });
}

