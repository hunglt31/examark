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

// Structure to track grading job results
struct GradingResult {
  bool completed;
  std::string outputDir;
  std::string pdfFilename;
};

// Global map to track grading results
std::unordered_map<std::string, GradingResult> gradingResults;
std::mutex resultsMutex;


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
      std::lock_guard<std::mutex> lock(resultsMutex);
      gradingResults[jobId] = {false, "", pdfFile.filename};
    }
        
    // Respond with job ID (use JSON format)
    res.set_header("Content-Type", "application/json");
    res.set_content("{\"jobId\":\"" + jobId + "\",\"message\":\"Grading request received.\"}", "application/json");
    
    grading_thread_pool.enqueue([pdfFile, csvFile, jobId, timestamp, tritonClient]() {
      std::string baseName = pdfFile.filename.substr(0, pdfFile.filename.find_last_of('.'));
      std::string outputDir = "/home/" + USER_NAME + "/examark-data/" + baseName + "_" + timestamp;
        
      bool success = grading(pdfFile.filename, pdfFile.content, csvFile.content, outputDir, tritonClient);
      std::lock_guard<std::mutex> lock(resultsMutex);
      if (success) {
        gradingResults[jobId] = {true, outputDir, pdfFile.filename};
      }
    });
  });
    
  // Add endpoint to check status
  server.Get("/status/:jobId", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
        
    std::string jobId = req.path_params.at("jobId");
        
    std::lock_guard<std::mutex> lock(resultsMutex);
    auto it = gradingResults.find(jobId);
    if (it == gradingResults.end()) {
      res.status = 404;
      res.set_content("{\"error\":\"Job not found\"}", "application/json");
      return;
    }
        
      bool completed = it->second.completed;
      std::string status = completed ? "completed" : "processing";
        
      res.set_header("Content-Type", "application/json");
      res.set_content("{\"status\":\"" + status + "\"}", "application/json");
    });
    
  // Add endpoint to get CSV results
  server.Get("/results/:jobId/csv", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
        
    std::string jobId = req.path_params.at("jobId");
        
    // Get the job result
    std::string outputDir;
    std::string filename;
    {
      std::lock_guard<std::mutex> lock(resultsMutex);
      auto it = gradingResults.find(jobId);
      if (it == gradingResults.end() || !it->second.completed) {
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
    
  // Add endpoint to get the list of images
  server.Get("/results/:jobId/images", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
      
    std::string jobId = req.path_params.at("jobId");
      
    // Get the job result
    std::string outputDir;
    {
      std::lock_guard<std::mutex> lock(resultsMutex);
      auto it = gradingResults.find(jobId);
      if (it == gradingResults.end() || !it->second.completed) {
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
    std::string jsonResponse = "{\"images\":[";
    for (size_t i = 0; i < imageList.size(); ++i) {
      jsonResponse += "\"" + imageList[i] + "\"";
      if (i < imageList.size() - 1) {
        jsonResponse += ",";
      }
    }
    jsonResponse += "]}";
      
    res.set_header("Content-Type", "application/json");
    res.set_content(jsonResponse, "application/json");
  });
  
  // Add endpoint to get a specific image
  server.Get("/results/:jobId/images/:imageName", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
      
    std::string jobId = req.path_params.at("jobId");
    std::string imageName = req.path_params.at("imageName");
      
    // Get the job result
    std::string outputDir;
    {
      std::lock_guard<std::mutex> lock(resultsMutex);
      auto it = gradingResults.find(jobId);
      if (it == gradingResults.end() || !it->second.completed) {
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

  server.Post("/grade-exam", [](const httplib::Request &req, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Content-Type", "application/json");
    
    try {
      // Parse JSON request body
      nlohmann::json requestData = nlohmann::json::parse(req.body);
      std::string jobId = requestData["jobId"];
      
      // Get stored results path
      std::string outputDir;
      std::string filename;
      {
        std::lock_guard<std::mutex> lock(resultsMutex);
        auto it = gradingResults.find(jobId);
        if (it == gradingResults.end() || !it->second.completed) {
          res.status = 404;
          res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
          return;
        }
        outputDir = it->second.outputDir;
        filename = it->second.pdfFilename;
      }
      
      // Read the stored answer key CSV
      std::string answerKeyPath = outputDir + "/answer_key.csv";
      std::ifstream answerKeyFile(answerKeyPath);
      if (!answerKeyFile.is_open()) {
        res.status = 500;
        res.set_content("{\"error\":\"Answer key file not found\"}", "application/json");
        return;
      }
      
      // Parse answer key CSV
      std::vector<std::vector<std::string>> answerKeyData;
      std::string line;
      while (std::getline(answerKeyFile, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
          // Remove any quotes and trim whitespace
          if (!cell.empty() && cell.front() == '"' && cell.back() == '"') {
            cell = cell.substr(1, cell.length() - 2);
          }
          cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
          cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
          }).base(), cell.end());
          
          row.push_back(cell);
        }
        answerKeyData.push_back(row);
      }
      answerKeyFile.close();
      
      // Parse answer key CSV format based on the provided sample
      // Format: First row has headers like ",Exam ID,101,102"
      // Subsequent rows: "Part,Question,Key,Key" then actual answers
      std::map<std::string, std::vector<std::string>> examAnswerKeys;
      std::vector<int> pointValues(24, 1); // Default 1 point per question

      if (answerKeyData.size() < 3) {
        res.status = 400;
        res.set_content("{\"error\":\"Answer key CSV must have at least 3 rows\"}", "application/json");
        return;
      }

      // Get all ExamIDs from first row (skip first two columns: empty and "Exam ID")
      std::vector<std::string> examIds;
      for (int col = 2; col < answerKeyData[0].size(); col++) {
        std::string examId = answerKeyData[0][col];
        if (!examId.empty()) {
          examIds.push_back(examId);
        }
      }

      // Extract answers for each ExamID (skip header row, start from row 2)
      for (int i = 0; i < examIds.size(); i++) {
        std::vector<std::string> answers;
        int columnIndex = 2 + i; // Skip first two columns (Part and Question)
        
        // Extract answers from rows 2-25 (24 questions total)
        for (int row = 2; row < answerKeyData.size() && answers.size() < 24; row++) {
          if (answerKeyData[row].size() > columnIndex) {
            answers.push_back(answerKeyData[row][columnIndex]);
          }
        }
        
        if (answers.size() == 24) {
          examAnswerKeys[examIds[i]] = answers;
        } else {
          Logger::error("CONTROLLER", "Incomplete answers for ExamID: " + examIds[i] + 
                        " (found " + std::to_string(answers.size()) + ")");
        }
      }

      if (examAnswerKeys.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"No valid answer keys found in CSV\"}", "application/json");
        return;
      }
      
      // Read student answers CSV
      std::string baseName = filename.substr(0, filename.find_last_of('.'));
      std::string csvPath = outputDir + "/" + baseName + ".csv";
      
      std::ifstream csvFile(csvPath);
      if (!csvFile.is_open()) {
        res.status = 500;
        res.set_content("{\"error\":\"Student answers CSV file not found\"}", "application/json");
        return;
      }
      
      // Parse student answers CSV (transposed format)
      std::vector<std::vector<std::string>> csvData;
      while (std::getline(csvFile, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
          row.push_back(cell);
        }
        csvData.push_back(row);
      }
      csvFile.close();

      // Remove existing grading result rows if they exist
      // Look for grading result rows by checking for specific headers
      std::vector<std::vector<std::string>> cleanedCsvData;
      for (int i = 0; i < csvData.size(); i++) {
        // Check if this row is a grading result row
        bool isGradingRow = false;
        if (!csvData[i].empty()) {
          std::string firstCell = csvData[i][0];
          // Check for grading headers
          if (firstCell == "Part 1" || firstCell == "Part 2" || firstCell == "Total") {
            // Also check second column to be more specific
            if (csvData[i].size() > 1) {
              std::string secondCell = csvData[i][1];
              if (secondCell == "Correct" || secondCell == "Points") {
                isGradingRow = true;
              }
            }
          }
        }
        
        // Only keep non-grading rows
        if (!isGradingRow) {
          cleanedCsvData.push_back(csvData[i]);
        }
      }

      // Update csvData to use the cleaned version
      csvData = cleanedCsvData;

      // Process each student and add grading results
      ExamGrader grader;
      std::vector<std::vector<std::string>> gradingResultsData = csvData;

      // Add 6 new rows for grading results
      std::vector<std::string> part1CorrectRow, part1PointsRow, part2CorrectRow;
      std::vector<std::string> part2PointsRow, totalCorrectRow, totalPointsRow;

      // Add headers for new columns
      part1CorrectRow.push_back("Part 1");
      part1PointsRow.push_back("Part 1");
      part2CorrectRow.push_back("Part 2");
      part2PointsRow.push_back("Part 2");
      totalCorrectRow.push_back("Total");
      totalPointsRow.push_back("Total");

      // Add subheaders
      part1CorrectRow.push_back("Correct");
      part1PointsRow.push_back("Points");
      part2CorrectRow.push_back("Correct");
      part2PointsRow.push_back("Points");
      totalCorrectRow.push_back("Correct");
      totalPointsRow.push_back("Points");

      int studentsGraded = 0;
      int studentsWithAnswerKey = 0;

      // Process each student (skip first 2 header columns)
      for (int studentCol = 2; studentCol < csvData[0].size(); studentCol++) {
        // Extract student data from this column
        std::vector<std::string> studentAnswers;
        std::string studentExamId = "";
        
        for (int row = 0; row < csvData.size(); row++) {
          if (studentCol < csvData[row].size()) {
            studentAnswers.push_back(csvData[row][studentCol]);
          } else {
            studentAnswers.push_back("");
          }
        }
        
        // Get student's ExamID (should be at index 2 in their answers)
        if (studentAnswers.size() > 2) {
          studentExamId = studentAnswers[2];
        }
        
        studentsGraded++;
        
        // Find matching answer key
        auto keyIt = examAnswerKeys.find(studentExamId);
        if (keyIt != examAnswerKeys.end()) {
          studentsWithAnswerKey++;
          
          // Grade this student
          auto gradingResult = grader.gradeStudentExam(studentAnswers, keyIt->second, pointValues);
          
          // Add results to new rows
          part1CorrectRow.push_back(std::to_string(gradingResult.part1CorrectCount));
          part1PointsRow.push_back(std::to_string(gradingResult.part1TotalPoints));
          part2CorrectRow.push_back(std::to_string(gradingResult.part2CorrectCount));
          part2PointsRow.push_back(std::to_string(gradingResult.part2TotalPoints));
          totalCorrectRow.push_back(std::to_string(gradingResult.totalCorrectCount));
          totalPointsRow.push_back(std::to_string(gradingResult.totalPoints));
          
        } else {
          // No answer key found for this ExamID
          part1CorrectRow.push_back("N/A");
          part1PointsRow.push_back("N/A");
          part2CorrectRow.push_back("N/A");
          part2PointsRow.push_back("N/A");
          totalCorrectRow.push_back("N/A");
          totalPointsRow.push_back("N/A");
          
          Logger::warning("CONTROLLER", "No answer key found for ExamID: " + studentExamId);
        }
      }

      // Add the new grading result rows to CSV
      gradingResultsData.push_back(part1CorrectRow);
      gradingResultsData.push_back(part1PointsRow);
      gradingResultsData.push_back(part2CorrectRow);
      gradingResultsData.push_back(part2PointsRow);
      gradingResultsData.push_back(totalCorrectRow);
      gradingResultsData.push_back(totalPointsRow);
      
      // Write updated CSV
      std::ofstream outCsvFile(csvPath);
      if (!outCsvFile.is_open()) {
        res.status = 500;
        res.set_content("{\"error\":\"Failed to write updated CSV file\"}", "application/json");
        return;
      }
      
      for (int i = 0; i < gradingResultsData.size(); i++) {
        for (int j = 0; j < gradingResultsData[i].size(); j++) {
          outCsvFile << gradingResultsData[i][j];
          if (j < gradingResultsData[i].size() - 1) {
            outCsvFile << ",";
          }
        }
        outCsvFile << "\n";
      }
      outCsvFile.close();
      
      // Success response
      nlohmann::json response;
      response["success"] = true;
      response["message"] = "Grading completed successfully";
      response["studentsGraded"] = studentsGraded;
      response["studentsWithAnswerKey"] = studentsWithAnswerKey;
      response["answerKeysUsed"] = examAnswerKeys.size();
      response["examIds"] = examIds;
      
      res.set_content(response.dump(), "application/json");
      
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content("{\"error\":\"Failed to process grading request: " + std::string(e.what()) + "\"}", "application/json");
    }
  });

  // Add CORS handler for the grading endpoint
  server.Options("/grade-exam", [](const httplib::Request&, httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.status = 204;
  });
}

