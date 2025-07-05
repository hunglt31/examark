#include <iostream>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <functional>
#include "utils/Logger.h"
#include "services/Service.h"
#include "utils/ImageProcessor.h"
#include "utils/ExamGrader.h"
#include "utils/ExamConfig.h"
#include "utils/MinIOHTTPClient.h"

using json = nlohmann::json;

// Add MinIO configuration
const std::string MINIO_ENDPOINT = "127.0.0.1:9000";
const std::string MINIO_ACCESS_KEY = "minioadmin";
const std::string MINIO_SECRET_KEY = "minioadmin123";
const std::string MINIO_BUCKET = "grading-jobs";

// Function to update job progress
extern void updateJobProgress(
  const std::string& jobId, 
  const std::string& stage, 
  const std::string& step, 
  int currentPage = 0, 
  int totalPages = 0, 
  double progressPercent = 0.0, 
  bool isError = false, 
  const std::string& errorMsg = ""
);

// Constants for CSV header
const std::vector<std::string> HEADER_1 = {
  "", "", "", "Part", 
  "1", "1", "1", "1", "1", "1", "1", "1", 
  "1", "1", "1", "1", "1", "1", "1", "1", 
  "2", "2", "2", "2", "2", "2", "2", "2",
  "Part 1", "Part 2", "Total"
};

const std::vector<std::string> HEADER_2 = {
  "Image name", "Student ID", "Exam ID", "Question",
  "1", "2", "3", "4", "5", "6", "7", "8",
  "9", "10", "11", "12", "13", "14", "15", "16",
  "1", "2", "3", "4", "5", "6", "7", "8",
  "Correct", "Correct", "Points"
};

// Helper function to generate CSV string
std::string generateCSVString(const std::vector<std::vector<std::string>>& results) {
  std::ostringstream csvStream;
  size_t numOriginalRows = results.size();    
  size_t numOriginalCols = results[0].size(); 
  
  for (size_t i = 0; i < numOriginalCols; ++i) {
    for (size_t j = 0; j < numOriginalRows; ++j) {
      csvStream << results[j][i];
      if (j < numOriginalRows - 1) {
        csvStream << ",";
      }
    }
    csvStream << "\n";
  }
  
  return csvStream.str();
}

bool examark::services::grade(
  const std::string& pdfFileName, 
  const std::string& pdfData, 
  const std::string& answerKeyCSV, 
  const std::string& outputDir, 
  TritonClient* tritonClient, 
  const std::string& jobId
) {
  try {
    /* ============================================== */
    /* ===== Stage 1: Reading Answer Key (0-5%) ===== */
    /* ============================================== */

    updateJobProgress(jobId, "reading_key", "Initializing grading process...", 0, 0, 0.0);
    if (!std::filesystem::exists(outputDir)) {
      std::filesystem::create_directory(outputDir);
    }

    std::map<std::string, std::vector<std::string>> examAnswerKeys;
    std::vector<std::vector<std::string>> answerKeyData;
    std::stringstream answerKeyStream(answerKeyCSV);
    std::string line;
    
    while (std::getline(answerKeyStream, line)) {
      std::vector<std::string> row;
      std::stringstream ss(line);
      std::string cell;
      
      while (std::getline(ss, cell, ',')) {
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
    
    // Check answer key 
    if (answerKeyData.size() <= ANSWER_KEY_START_COLUMN) {
      updateJobProgress(jobId, "reading_key", "Error: Answer key is empty", 0, 0, 0.0, true, "Answer key is empty");
      return false;
    }  

    std::vector<std::string> examIds;
    for (int col = ANSWER_KEY_START_COLUMN; col < answerKeyData[0].size(); col++) {
      std::string examId = answerKeyData[0][col];
      if (!examId.empty()) {
        examIds.push_back(examId);
      }
    }

    // Extract answers for each ExamID
    for (int i = 0; i < examIds.size(); i++) {
      std::vector<std::string> answers;
      int columnIndex = ANSWER_KEY_START_COLUMN + i;
      
      for (int row = ANSWER_KEY_START_INDEX; row < answerKeyData.size() && answers.size() < TOTAL_QUESTIONS; row++) {
        if (answerKeyData[row].size() > columnIndex) {
          std::string answer = answerKeyData[row][columnIndex];
          if (answers.size() >= PART_1_NUM_QUESTIONS) { 
            // Part 2
            std::string convertedAnswer = "SSSSSS"; 
            for (char c : answer) {
              if (c >= 'A' && c <= 'F') {
                int position = c - 'A'; 
                if (position < PART_2_STRING_SIZE) {
                  convertedAnswer[position] = 'D';
                }
              }
            }
            answers.push_back(convertedAnswer);
          } else {
            // Part 1
            answers.push_back(answer);
          }
        }
      }
      if (answers.size() == TOTAL_QUESTIONS) {
        examAnswerKeys[examIds[i]] = answers;
      }
    }

    // Create JSON content
    nlohmann::json answerKeyJson;
    answerKeyJson["exam_answer_keys"] = json::object();
    for (const auto& [examId, answers] : examAnswerKeys) {
      answerKeyJson["exam_answer_keys"][examId] = answers;
    }
    std::string answerKeyJsonStr = answerKeyJson.dump(2);

    // Save answer key locally (for regrade compatibility)
    std::string answerKeyPath = outputDir + "/answer_key.json";
    std::ofstream answerKeyFile(answerKeyPath);
    if (!answerKeyFile.is_open()) {
      updateJobProgress(jobId, "reading_key", "Error: Failed to save local answer key", 0, 0, 0.0, true, "Failed to save answer key JSON locally");
      return false;
    }
    answerKeyFile << answerKeyJsonStr;
    answerKeyFile.close();

    // Upload answer key JSON to MinIO
    MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
    std::string jsonObjectName = jobId + "/answer_key.json";
    if (!minioClient.uploadJSON(jsonObjectName, answerKeyJsonStr)) {
      updateJobProgress(jobId, "reading_key", "Error: Failed to upload answer key to storage", 0, 0, 0.0, true, "Failed to save answer key JSON to MinIO");
      return false;
    }
    
    updateJobProgress(jobId, "reading_key", "Answer key saved successfully", 0, 0, 5.0);
    
    /* ============================================= */
    /* ===== Stage 2: Rendering Images (5-75%) ===== */
    /* ============================================= */

    updateJobProgress(jobId, "rendering_images", "Starting PDF conversion...", 0, 0, 5.0);
    ImageProcessor imgProc;
    std::vector<cv::Mat> images;

    auto progressCallback = [&jobId](int currentPage, int totalPages, double percent) {
      std::string message;
      if (totalPages > 0 && currentPage > 0) {
        message = "Rendered " + std::to_string(currentPage) + " of " + std::to_string(totalPages) + " pages";
      } else if (totalPages > 0) {
        message = "Found " + std::to_string(totalPages) + " pages, starting conversion...";
      } else {
        message = "Loading PDF document...";
      }
      updateJobProgress(jobId, "rendering_images", message, currentPage, totalPages, percent);
    };
    
    if (!imgProc.getRequestImagesWithProgress(pdfData.c_str(), pdfData.size(), images, progressCallback, 300.0)) {
        updateJobProgress(jobId, "rendering_images", "Error: Failed to convert PDF", 0, 0, 0.0, true, "Failed to convert PDF to images");
        return false;
    }
    
    updateJobProgress(jobId, "rendering_images", "PDF converted successfully - " + std::to_string(images.size()) + " pages rendered", images.size(), images.size(), 75.0);
    
    /* ================================================ */
    /* ===== Stage 3: Upload Images to MinIO (75-80%) ===== */
    /* ================================================ */

    updateJobProgress(jobId, "uploading_images", "Uploading images to storage...", 0, images.size(), 75.0);
    
    std::vector<std::string> uploadedImageNames;
    for (size_t i = 0; i < images.size(); ++i) {
      std::string imageBasename = "page_" + std::to_string(i + 1);
      std::string minioObjectName = jobId + "/" + imageBasename + ".jpg";
      
      std::string localImagePath = outputDir + "/" + imageBasename + ".jpg";
      cv::imwrite(localImagePath, images[i]);
      
      if (!minioClient.uploadImage(minioObjectName, images[i])) {
        updateJobProgress(jobId, "uploading_images", "Failed to upload image " + std::to_string(i + 1), 
                         i + 1, images.size(), 0.0, true, "Failed to upload image to storage");
        return false;
      }
      
      uploadedImageNames.push_back(minioObjectName);
      
      // Update upload progress (75% to 80%)
      double uploadProgress = 75.0 + (double(i + 1) / images.size()) * 5.0;
      updateJobProgress(jobId, "uploading_images", "Uploaded " + std::to_string(i + 1) + " of " + std::to_string(images.size()) + " images", 
                       i + 1, images.size(), uploadProgress);
    }
    
    /* =========================================== */
    /* ===== Stage 4: Grading Exams (80-95%) ===== */
    /* =========================================== */

    updateJobProgress(jobId, "grading_exams", "Starting YOLO detection and grading...", 0, images.size(), 80.0);
    
    // Process images for grading
    std::vector<std::vector<std::string>> results;
    results.push_back(HEADER_1);
    results.push_back(HEADER_2);
    
    ExamGrader grader;
    int numImages = images.size();
    for (int i = 0; i < numImages; ++i) {
      std::string imageBasename = "page_" + std::to_string(i + 1);

      std::vector<cv::Mat> metadataImages, contentImages;
      if (!imgProc.splitImage(images[i], metadataImages, contentImages)) {
        double currentProgress = 80.0 + (double(i + 1) / numImages) * 15.0; 
        updateJobProgress(jobId, "grading_exams", "Error: Failed to split image page " + std::to_string(i + 1), i + 1, images.size(), currentProgress, true, "Failed to split image page " + std::to_string(i + 1));
        continue;
      }
    
      // Grading the exam with YOLO detection
      std::vector<std::vector<Detection>> metadataDetections = tritonClient->inference(metadataImages, "metadata_model");
      std::vector<std::vector<Detection>> contentDetections = tritonClient->inference(contentImages, "content_model");
      std::vector<std::string> result = grader.extractAnswersAndGradeExam(imageBasename, metadataDetections, contentDetections, examAnswerKeys);
      results.push_back(result);

      // Update grading progress (80% to 95%)
      if ((i + 1) % 5 == 0 || i == numImages - 1) {
        double currentProgress = 80.0 + (double(i + 1) / numImages) * 15.0; 
        std::string message = "Graded " + std::to_string(i + 1) + " of " + std::to_string(numImages) + " pages";
        updateJobProgress(jobId, "grading_exams", message, i + 1, numImages, currentProgress);
      }
    }
    updateJobProgress(jobId, "grading_exams", "All exams graded successfully", numImages, numImages, 95.0);
    
    /* ================================================= */
    /* ===== Stage 5: Save Results to Storage (95-100%) ===== */
    /* ================================================= */
    updateJobProgress(jobId, "saving_results", "Saving results to storage...", 0, images.size(), 95.0);
    
    // Generate CSV content
    std::string csvContent = generateCSVString(results);
    
    // Save CSV locally (for regrade function)
    std::string csvBasename = pdfFileName.substr(0, pdfFileName.find_last_of('.'));
    std::string csvFilePath = outputDir + "/" + csvBasename + ".csv";
    std::ofstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
      updateJobProgress(jobId, "saving_results", "Error: Failed to save local CSV", 0, 0, 0.0, true, "Failed to save results locally");
      return false;
    }
    csvFile << csvContent;
    csvFile.close();
    
    // Upload CSV to MinIO
    std::string csvObjectName = jobId + "/" + csvBasename + ".csv";
    if (!minioClient.uploadCSV(csvObjectName, csvContent)) {
      updateJobProgress(jobId, "saving_results", "Error: Failed to upload results to storage", 0, 0, 0.0, true, "Failed to upload CSV to storage");
      return false;
    }
    
    updateJobProgress(jobId, "completed", "All processing completed successfully", numImages, numImages, 100.0);
    return true;
    
  } catch (const std::exception& e) {
    updateJobProgress(jobId, "error", "Grading failed: " + std::string(e.what()), 0, 0, 0.0, true, e.what());
    return false;
  }
}

bool examark::services::regrade(
  const std::string& outputDir, 
  const std::string& csvData, 
  const std::string& answerKeyData, 
  const std::string& regradeJobId, 
  const std::string& originalJobId
) {
  try {
    // Initialize MinIO client
    MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
    
    // Parse the NEW answer key from the uploaded CSV file
    std::map<std::string, std::vector<std::string>> examAnswerKeys;
    
    if (!answerKeyData.empty()) {
      // Parse the new answer key CSV
      std::vector<std::vector<std::string>> answerKeyRows;
      std::stringstream answerKeyStream(answerKeyData);
      std::string line;
      
      while (std::getline(answerKeyStream, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
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
        answerKeyRows.push_back(row);
      }
      
      // Process the answer key (same logic as in grading function)
      if (answerKeyRows.size() > ANSWER_KEY_START_COLUMN) {
        std::vector<std::string> examIds;
        for (int col = ANSWER_KEY_START_COLUMN; col < answerKeyRows[0].size(); col++) {
          std::string examId = answerKeyRows[0][col];
          if (!examId.empty()) {
            examIds.push_back(examId);
          }
        }
        
        // Extract answers for each ExamID
        for (int i = 0; i < examIds.size(); i++) {
          std::vector<std::string> answers;
          int columnIndex = ANSWER_KEY_START_COLUMN + i;
          
          for (int row = ANSWER_KEY_START_INDEX; row < answerKeyRows.size() && answers.size() < TOTAL_QUESTIONS; row++) {
            if (answerKeyRows[row].size() > columnIndex) {
              std::string answer = answerKeyRows[row][columnIndex];
              if (answers.size() >= PART_1_NUM_QUESTIONS) { 
                // Part 2
                std::string convertedAnswer = "SSSSSS"; 
                for (char c : answer) {
                  if (c >= 'A' && c <= 'F') {
                    int position = c - 'A'; 
                    if (position < PART_2_STRING_SIZE) {
                      convertedAnswer[position] = 'D';
                    }
                  }
                }
                answers.push_back(convertedAnswer);
              } else {
                // Part 1
                answers.push_back(answer);
              }
            }
          }
          if (answers.size() == TOTAL_QUESTIONS) {
            examAnswerKeys[examIds[i]] = answers;
          }
        }
        
        // Save the NEW answer key to JSON and upload to MinIO
        nlohmann::json newAnswerKeyJson;
        newAnswerKeyJson["exam_answer_keys"] = json::object();
        for (const auto& [examId, answers] : examAnswerKeys) {
          newAnswerKeyJson["exam_answer_keys"][examId] = answers;
        }
        std::string newAnswerKeyJsonStr = newAnswerKeyJson.dump(2);
        
        // Update local JSON file
        std::string answerKeyPath = outputDir + "/answer_key.json";
        std::ofstream answerKeyFile(answerKeyPath);
        if (answerKeyFile.is_open()) {
          answerKeyFile << newAnswerKeyJsonStr;
          answerKeyFile.close();
        }
        
        // Upload updated JSON to MinIO
        std::string jsonObjectName = originalJobId + "/answer_key.json";
        if (!minioClient.uploadJSON(jsonObjectName, newAnswerKeyJsonStr)) {
          Logger::error("REGRADE", "Failed to upload new answer key JSON to MinIO for job: " + originalJobId);
        }
      }
    }
    
    // Fallback to existing answer key if new one failed to parse
    if (examAnswerKeys.empty()) {
      Logger::info("REGRADE", "Failed to parse new answer key, falling back to existing for job: " + originalJobId);
      
      // Try to load from MinIO first
      std::string jsonObjectName = originalJobId + "/answer_key.json";
      std::string answerKeyJsonStr = minioClient.downloadJSON(jsonObjectName);
      
      if (!answerKeyJsonStr.empty()) {
        try {
          nlohmann::json answerKeyJson = nlohmann::json::parse(answerKeyJsonStr);
          if (answerKeyJson.contains("exam_answer_keys")) {
            for (const auto& [examId, answers] : answerKeyJson["exam_answer_keys"].items()) {
              examAnswerKeys[examId] = answers.get<std::vector<std::string>>();
            }
          }
        } catch (const std::exception& e) {
          Logger::error("REGRADE", "Failed to parse JSON from MinIO: " + std::string(e.what()) + " for job: " + originalJobId);
        }
      }
      
      // Fallback to local file if MinIO failed
      if (examAnswerKeys.empty()) {
        std::string answerKeyPath = outputDir + "/answer_key.json";
        if (std::filesystem::exists(answerKeyPath)) {
          std::ifstream answerKeyFile(answerKeyPath);
          if (answerKeyFile.is_open()) {
            try {
              nlohmann::json answerKeyJson;
              answerKeyFile >> answerKeyJson;
              answerKeyFile.close();
              
              if (answerKeyJson.contains("exam_answer_keys")) {
                for (const auto& [examId, answers] : answerKeyJson["exam_answer_keys"].items()) {
                  examAnswerKeys[examId] = answers.get<std::vector<std::string>>();
                }
              }
            } catch (const std::exception& e) {
              Logger::error("REGRADE", "Failed to parse local answer key JSON: " + std::string(e.what()) + " for job: " + originalJobId);
            }
          } else {
            Logger::error("REGRADE", "Failed to open local answer key file: " + answerKeyPath);
          }
        } else {
          Logger::error("REGRADE", "Local answer key file does not exist: " + answerKeyPath);
        }
      }
    }
    
    if (examAnswerKeys.empty()) {
      Logger::error("REGRADE", "No answer keys available for regrading job: " + originalJobId);
      return false;
    }
    
    // Parse the CSV data
    std::vector<std::vector<std::string>> csvRows;
    std::stringstream csvStream(csvData);
    std::string csvLine;
    
    while (std::getline(csvStream, csvLine)) {
      std::vector<std::string> row;
      std::stringstream ss(csvLine);
      std::string cell;
      
      while (std::getline(ss, cell, ',')) {
        // Clean up cell data
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
      csvRows.push_back(row);
    }
    
    if (csvRows.size() < 4) {
      Logger::error("REGRADE", "CSV data has insufficient rows (" + std::to_string(csvRows.size()) + ") for job: " + originalJobId);
      return false;
    }
    
    // Re-grade each exam
    ExamGrader grader;
    
    for (size_t col = 2; col < csvRows[0].size(); col++) {
      std::vector<std::string> studentData;
      
      // Get image name, student ID, exam ID
      if (col < csvRows[0].size()) {
        studentData.push_back(csvRows[0][col]);
      } else {
        studentData.push_back("page_" + std::to_string(col - 1));
      }
      
      if (csvRows.size() > 1 && col < csvRows[1].size()) {
        studentData.push_back(csvRows[1][col]);
      } else {
        studentData.push_back("");
      }
      
      if (csvRows.size() > 2 && col < csvRows[2].size()) {
        studentData.push_back(csvRows[2][col]);
      } else {
        studentData.push_back("");
      }
      
      studentData.push_back("Answers");
      
      // Extract answers
      std::vector<std::string> part1Answers;
      std::vector<std::string> part2Answers;
      
      for (size_t row = 4; row < csvRows.size(); row++) {
        if (csvRows[row].size() > 1) {
          std::string partNumber = csvRows[row][0];
          
          if (partNumber == "1" && part1Answers.size() < 16) {
            if (col < csvRows[row].size()) {
              part1Answers.push_back(csvRows[row][col]);
            } else {
              part1Answers.push_back("_");
            }
          } else if (partNumber == "2" && part2Answers.size() < 8) {
            if (col < csvRows[row].size()) {
              part2Answers.push_back(csvRows[row][col]);
            } else {
              part2Answers.push_back("_");
            }
          }
        }
      }
      
      while (part1Answers.size() < 16) {
        part1Answers.push_back("_");
      }
      while (part2Answers.size() < 8) {
        part2Answers.push_back("_");
      }
      
      studentData.insert(studentData.end(), part1Answers.begin(), part1Answers.end());
      studentData.insert(studentData.end(), part2Answers.begin(), part2Answers.end());
      
      // Re-grade using the NEW answer key
      std::vector<std::string> regradedResult = grader.extractAnswersAndRegradeExam(studentData, examAnswerKeys);
      
      // Update scores
      if (!regradedResult.empty() && regradedResult.size() >= 3) {
        for (int row = csvRows.size() - 3; row < csvRows.size(); row++) {
          if (row >= 0 && col < csvRows[row].size()) {
            int scoreIndex = row - (csvRows.size() - 3);
            if (scoreIndex < 3) {
              size_t resultIndex = regradedResult.size() - 3 + scoreIndex;
              if (resultIndex < regradedResult.size()) {
                csvRows[row][col] = regradedResult[resultIndex];
              }
            }
          }
        }
      }
    }
    
    // Save updated CSV locally
    std::string csvFilePath;
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
      if (entry.path().extension() == ".csv" && entry.path().filename() != "answer_key.json") {
        csvFilePath = entry.path().string();
        break;
      }
    }
    
    if (csvFilePath.empty()) {
      csvFilePath = outputDir + "/results.csv";
    }
    
    std::ofstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
      Logger::error("REGRADE", "Failed to open CSV file for writing: " + csvFilePath);
      return false;
    }
    
    // Write CSV data back
    for (size_t row = 0; row < csvRows.size(); ++row) {
      for (size_t col = 0; col < csvRows[row].size(); ++col) {
        csvFile << csvRows[row][col];
        if (col < csvRows[row].size() - 1) {
          csvFile << ",";
        }
      }
      csvFile << "\n";
    }
    csvFile.close();
    
    // Upload updated CSV back to MinIO
    std::ostringstream updatedCsvStream;
    for (size_t row = 0; row < csvRows.size(); ++row) {
      for (size_t col = 0; col < csvRows[row].size(); ++col) {
        updatedCsvStream << csvRows[row][col];
        if (col < csvRows[row].size() - 1) {
          updatedCsvStream << ",";
        }
      }
      updatedCsvStream << "\n";
    }
    
    // Update both the original CSV and create a timestamped version
    std::string originalCsvBasename;
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
      if (entry.path().extension() == ".csv") {
        originalCsvBasename = entry.path().stem().string();
        break;
      }
    }
    
    if (originalCsvBasename.empty()) {
      originalCsvBasename = "results";
    }
    
    std::string updatedCsvObjectName = originalJobId + "/" + originalCsvBasename + ".csv";
    if (!minioClient.uploadCSV(updatedCsvObjectName, updatedCsvStream.str())) {
      Logger::error("REGRADE", "Failed to upload updated CSV to MinIO: " + updatedCsvObjectName);
    }
    
    return true;
    
  } catch (const std::exception& e) {
    Logger::error("REGRADE", "Exception in regrade process: " + std::string(e.what()) + " for job: " + originalJobId);
    return false;
  }
}
