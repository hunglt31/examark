#include <iostream>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <functional>
#include "services/Service.h"
#include "utils/ImageProcessor.h"
#include "utils/ExamGrader.h"
#include "utils/ExamConfig.h"

using json = nlohmann::json;

// Function to update job progress
extern void updateJobProgress(
  const std::string& jobId, const std::string& stage, 
  const std::string& step, int currentPage = 0, int totalPages = 0, 
  double progressPercent = 0.0, bool isError = false, const std::string& errorMsg = "");

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

bool grading(const std::string& pdfFileName, const std::string& pdfData, 
             const std::string& answerKeyCSV, const std::string& outputDir, 
             TritonClient* tritonClient, const std::string& jobId) {
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

    // Save answer key for reference in JSON format
    std::string answerKeyPath = outputDir + "/answer_key.json";
    std::ofstream answerKeyFile(answerKeyPath);
    if (!answerKeyFile.is_open()) {
      updateJobProgress(jobId, "reading_key", "Error: Failed to save answer key", 0, 0, 0.0, true, "Failed to save answer key JSON");
      return false;
    }
    
    json answerKeyJson;
    answerKeyJson["exam_answer_keys"] = json::object();
    for (const auto& [examId, answers] : examAnswerKeys) {
      answerKeyJson["exam_answer_keys"][examId] = answers;
    }

    answerKeyFile << answerKeyJson.dump(2); 
    answerKeyFile.close();
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
    
    /* =========================================== */
    /* ===== Stage 3: Grading Exams (75-95%) ===== */
    /* =========================================== */
    updateJobProgress(jobId, "grading_exams", "Grading exams...", 0, images.size(), 75.0);
    
    // Process images
    std::vector<std::vector<std::string>> results;
    results.push_back(HEADER_1);
    results.push_back(HEADER_2);
    
    ExamGrader grader;
    int numImages = images.size();
    for (int i = 0; i < numImages; ++i) {
      std::string imageBasename = "page_" + std::to_string(i + 1);
      std::string outputFilepath = outputDir + "/" + imageBasename + ".jpg";
      cv::imwrite(outputFilepath, images[i]);

      std::vector<cv::Mat> metadataImages, contentImages;
      if (!imgProc.splitImage(images[i], metadataImages, contentImages)) {
        double currentProgress = 75.0 + (double(i + 1) / numImages) * 20.0; 
        updateJobProgress(jobId, "grading_exams", "Error: Failed to split image page " + std::to_string(i + 1), i + 1, images.size(), currentProgress, true, "Failed to split image page " + std::to_string(i + 1));
        continue;
      }
    
      // Grading the exam
      std::vector<std::vector<Detection>> metadataDetections = tritonClient->inference(metadataImages, "metadata_model");
      std::vector<std::vector<Detection>> contentDetections = tritonClient->inference(contentImages, "content_model");
      std::vector<std::string> result = grader.extractAnswersAndGradeExam(imageBasename, metadataDetections, contentDetections, examAnswerKeys);
      results.push_back(result);

      if ((i + 1) % 5 == 0 || i == numImages - 1) {
        double currentProgress = 75.0 + (double(i + 1) / numImages) * 20.0; 
        std::string message = "Graded " + std::to_string(i + 1) + " of " + std::to_string(numImages) + " pages";
        updateJobProgress(jobId, "grading_exams", message, i + 1, numImages, currentProgress);
      }
    }
    updateJobProgress(jobId, "grading_exams", "All exams graded successfully", numImages, numImages, 95.0);
    
    /* ================================================= */
    /* ===== Step 4: Save results to CSV (95-100%) ===== */
    /* ================================================= */
    updateJobProgress(jobId, "saving_results", "Saving results to system...", 0, images.size(), 97.0);
    size_t numOriginalRows = results.size();    
    size_t numOriginalCols = results[0].size(); 
    
    std::string csvFilePath = outputDir + "/" + pdfFileName.substr(0, pdfFileName.find_last_of('.')) + ".csv";
    std::ofstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
      updateJobProgress(jobId, "saving_results", "Error: Failed to save results", 0, 0, 0.0, true, "Failed to save results to system");
      return false;
    }
    
    for (size_t i = 0; i < numOriginalCols; ++i) {
      for (size_t j = 0; j < numOriginalRows; ++j) {
        csvFile << results[j][i];
        if (j < numOriginalRows - 1) {
          csvFile << ",";
        }
      }
      csvFile << "\n";
    }
    csvFile.close();
    
    updateJobProgress(jobId, "saving_results", "Saved all exams to system", numImages, numImages, 100.0);
    return true;
    
  } catch (const std::exception& e) {
    updateJobProgress(jobId, "error", "Grading failed: " + std::string(e.what()), 0, 0, 0.0, true, e.what());
    return false;
  }
}

bool regradeExam(const std::string& outputDir, const std::string& csvData, 
                 const std::string& answerKeyData, const std::string& regradeJobId, 
                 const std::string& originalJobId) {
  try {
    // Load answer key from JSON file
    std::string answerKeyPath = outputDir + "/answer_key.json";
    if (!std::filesystem::exists(answerKeyPath)) {
      return false;
    }
    
    std::ifstream answerKeyFile(answerKeyPath);
    if (!answerKeyFile.is_open()) {
      return false;
    }
    
    nlohmann::json answerKeyJson;
    answerKeyFile >> answerKeyJson;
    answerKeyFile.close();
    
    // Convert JSON back to map format
    std::map<std::string, std::vector<std::string>> examAnswerKeys;
    if (answerKeyJson.contains("exam_answer_keys")) {
      for (const auto& [examId, answers] : answerKeyJson["exam_answer_keys"].items()) {
        examAnswerKeys[examId] = answers.get<std::vector<std::string>>();
      }
    }
    
    if (examAnswerKeys.empty()) {
      return false;
    }
    
    // Parse the CSV data
    std::vector<std::vector<std::string>> csvRows;
    std::stringstream csvStream(csvData);
    std::string line;
    
    while (std::getline(csvStream, line)) {
      std::vector<std::string> row;
      std::stringstream ss(line);
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
      return false;
    }
    
    // Re-grade each exam (each column starting from column 2)
    ExamGrader grader;
    
    // Process each student (each column from index 2 onwards)
    for (size_t col = 2; col < csvRows[0].size(); col++) {
      // Extract student data for this column
      std::vector<std::string> studentData;
      
      // Get image name (row 0), student ID (row 1), exam ID (row 2)
      if (col < csvRows[0].size()) {
        studentData.push_back(csvRows[0][col]);  // Image name
      } else {
        studentData.push_back("page_" + std::to_string(col - 1));
      }
      
      if (csvRows.size() > 1 && col < csvRows[1].size()) {
        studentData.push_back(csvRows[1][col]);  // Student ID
      } else {
        studentData.push_back("");
      }
      
      if (csvRows.size() > 2 && col < csvRows[2].size()) {
        studentData.push_back(csvRows[2][col]);  // Exam ID
      } else {
        studentData.push_back("");
      }
      
      // Add "Answers" label
      studentData.push_back("Answers");
      
      // Extract answers from rows 4 onwards
      // Part 1 answers (16 questions) - rows where column 0 is "1"
      std::vector<std::string> part1Answers;
      std::vector<std::string> part2Answers;
      
      for (size_t row = 4; row < csvRows.size(); row++) {
        if (csvRows[row].size() > 1) {
          std::string partNumber = csvRows[row][0];  // Part number (1 or 2)
          
          if (partNumber == "1" && part1Answers.size() < 16) {
            // Part 1 question
            if (col < csvRows[row].size()) {
              part1Answers.push_back(csvRows[row][col]);
            } else {
              part1Answers.push_back("_");
            }
          } else if (partNumber == "2" && part2Answers.size() < 8) {
            // Part 2 question
            if (col < csvRows[row].size()) {
              part2Answers.push_back(csvRows[row][col]);
            } else {
              part2Answers.push_back("_");
            }
          }
        }
      }
      
      // Ensure we have the right number of answers
      while (part1Answers.size() < 16) {
        part1Answers.push_back("_");
      }
      while (part2Answers.size() < 8) {
        part2Answers.push_back("_");
      }
      
      // Add answers to student data (Part 1 first, then Part 2)
      studentData.insert(studentData.end(), part1Answers.begin(), part1Answers.end());
      studentData.insert(studentData.end(), part2Answers.begin(), part2Answers.end());
      
      // Re-grade using the CSV data (which includes user edits)
      std::string imageBasename = studentData[0];
      std::vector<std::string> regradedResult = grader.regradeExamFromCsv(imageBasename, studentData, examAnswerKeys);
      
      // Update ONLY the score rows with new grades (keep existing answers)
      if (!regradedResult.empty() && regradedResult.size() >= 3) {
        // Find the score rows (last 3 rows that contain "Part 1", "Part 2", "Total")
        for (int row = csvRows.size() - 3; row < csvRows.size(); row++) {
          if (row >= 0 && col < csvRows[row].size()) {
            int scoreIndex = row - (csvRows.size() - 3); // 0, 1, or 2
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
    
    // Find the original CSV file in the output directory
    std::string csvFilePath;
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
      if (entry.path().extension() == ".csv" && entry.path().filename() != "answer_key.json") {
        csvFilePath = entry.path().string();
        break;
      }
    }
    
    // If no CSV found, create one with a default name
    if (csvFilePath.empty()) {
      csvFilePath = outputDir + "/results.csv";
    }
    
    std::ofstream csvFile(csvFilePath);
    if (!csvFile.is_open()) {
      return false;
    }
    
    // Write CSV data back in original format
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
    
    return true;
    
  } catch (const std::exception& e) {
    return false;
  }
}