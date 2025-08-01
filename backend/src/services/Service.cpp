#include "services/Service.h"
#include "utils/ExamConfig.h"
#include "utils/ExamGrader.h"
#include "utils/ImageProcessor.h"
#include "utils/Logger.h"
#include "utils/MinIOHTTPClient.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <unistd.h>

using json = nlohmann::json;

// Add MinIO configuration
const std::string MINIO_ENDPOINT = "127.0.0.1:9000";
const std::string MINIO_ACCESS_KEY = "minioadmin";
const std::string MINIO_SECRET_KEY = "minioadmin123";
const std::string MINIO_BUCKET = "grading-jobs";

// Function to update job progress
extern void updateJobProgress(const std::string &jobId, const std::string &stage, const std::string &step,
                              int currentPage = 0, int totalPages = 0, double progressPercent = 0.0,
                              bool isError = false, const std::string &errorMsg = "");

// Constants for CSV header
const std::vector<std::string> HEADER_1 = {"",  "",  "",  "Part", "1", "1", "1", "1", "1",      "1",      "1",
                                           "1", "1", "1", "2",    "2", "2", "2", "2", "Part 1", "Part 2", "Total"};

const std::vector<std::string> HEADER_2 = {
    "Image name", "Student ID", "Exam ID", "Question", "1",  "2",  "3",  "4",  "5",       "6",       "7",
    "8",          "9",          "10",      "11",       "12", "13", "14", "15", "Correct", "Correct", "Points"};

// Helper function to generate CSV string
std::string generateCSVString(const std::vector<std::vector<std::string>> &results) {
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

// Helper function to convert JSON answer key to internal format
std::map<std::string, std::vector<std::string>> parseJsonAnswerKey(const std::string &answerKeyJson) {
  std::map<std::string, std::vector<std::string>> examAnswerKeys;

  try {
    nlohmann::json answerData = nlohmann::json::parse(answerKeyJson);

    // Handle array of objects: [{ exam_id: "101", "1": "...", ..., "15": "..." }, ...]
    if (answerData.is_array()) {
      for (const auto &examObj : answerData) {
        if (!examObj.contains("exam_id"))
          continue;
        std::string examId = examObj["exam_id"].get<std::string>();
        if (examId.empty())
          continue;

        std::vector<std::string> answers;
        for (int i = 1; i <= TOTAL_QUESTIONS; ++i) {
          std::string key = std::to_string(i);
          std::string answer = examObj.contains(key) ? examObj[key].get<std::string>() : "";
          answers.push_back(answer);
        }
        examAnswerKeys[examId] = answers;
      }
      return examAnswerKeys;
    }
  } catch (const std::exception &e) {
    Logger::error("SERVICE", "Failed to parse JSON answer key: " + std::string(e.what()));
    return examAnswerKeys;
  }

  return examAnswerKeys;
}

bool examark::services::extract_all_exams_answers(const std::string &pdfFileName, const std::string &pdfData,
                                                  const std::string &outputDir, TritonClient *tritonClient,
                                                  const std::string &jobId) {
  try {
    /* ============================================= */
    /* ===== Stage 1: Rendering images (0-75%) ===== */
    /* ============================================= */
    MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
    ImageProcessor imgProc;
    std::vector<cv::Mat> images;
    updateJobProgress(jobId, "rendering_images", "Starting PDF conversion...", 0, 0, 70.0);

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
      updateJobProgress(jobId, "rendering_images", "Error: Failed to convert PDF", 0, 0, 0.0, true,
                        "Failed to convert PDF to images");
      Logger::error("SERVICE", "Failed to convert PDF to images");
      return false;
    }

    updateJobProgress(jobId, "rendering_images",
                      "PDF converted successfully - " + std::to_string(images.size()) + " pages rendered",
                      images.size(), images.size(), 75.0);

    /* ====================================================== */
    /* ===== Stage 2: Read QR code information (70-75%) ===== */
    /* ====================================================== */
    updateJobProgress(jobId, "reading_qr_codes", "Reading QR codes from images...", 0, images.size(), 75.0);

    int first_exam_image_idx = 0;
    bool get_qr_info = false;
    std::string qr_info;

    for (first_exam_image_idx = 0; first_exam_image_idx < images.size(); ++first_exam_image_idx) {
      std::string temp_qr_info;
      if (!imgProc.get_qr_code_info(images[first_exam_image_idx], temp_qr_info)) {
        break;
      }

      if (!get_qr_info) {
        qr_info = temp_qr_info;
        std::string subject_id, class_id, random_id;
        {
          std::istringstream iss(qr_info);
          std::string token;
          std::vector<std::string> parts;
          while (std::getline(iss, token, '-')) {
            parts.push_back(token);
          }
          if (parts.size() == 3) {
            subject_id = parts[0];
            class_id = parts[1];
            random_id = parts[2];
          } else {
            subject_id = class_id = random_id = "";
            Logger::error("SERVICE", "QR info format invalid: " + qr_info);
          }
        }
        std::cout << "[DEBUG] QR INFO: " << qr_info << std::endl;
        get_qr_info = true;
      }
    }

    /* ==================================================== */
    /* ===== Stage 3: Upload images to MinIO (75-80%) ===== */
    /* ==================================================== */
    std::vector<std::string> uploadedImageNames;
    std::vector<cv::Mat> uploaded_images;
    updateJobProgress(jobId, "uploading_images", "Uploading images to storage...", 0, images.size(), 75.0);

    for (int i = first_exam_image_idx; i < images.size(); ++i) {
      cv::Mat corrected;
      cv::LUT(images[i], GAMMA_LUT, corrected);

      cv::Mat aligned_image = imgProc.alignImage(corrected);

      std::string imageBasename = "page_" + std::to_string(i - first_exam_image_idx + 1);
      std::string minioObjectName = jobId + "/" + imageBasename + ".jpg";

      // std::string localImagePath = outputDir + "/" + imageBasename + ".jpg";
      // cv::imwrite(localImagePath, aligned_image);

      if (!minioClient.uploadImage(minioObjectName, aligned_image)) {
        updateJobProgress(jobId, "uploading_images", "Failed to upload image " + std::to_string(i + 1), i + 1,
                          images.size(), 0.0, true, "Failed to upload image to storage");
        Logger::error("SERVICE", "Failed to upload image to MinIO.");
        return false;
      }

      uploaded_images.emplace_back(aligned_image);
      uploadedImageNames.emplace_back(minioObjectName);

      // Update upload progress (75% to 80%)
      double uploadProgress = 75.0 + (double(i + 1) / images.size()) * 5.0;
      updateJobProgress(jobId, "uploading_images",
                        "Uploaded " + std::to_string(i + 1) + " of " + std::to_string(images.size()) + " images", i + 1,
                        images.size(), uploadProgress);
    }

    /* =================================================== */
    /* ===== Stage 4: Extract exams answers (80-95%) ===== */
    /* =================================================== */
    Logger::info("SERVICE", "Starting to extract answers from " + std::to_string(uploaded_images.size()) + " pages");
    updateJobProgress(jobId, "extracting_answers", "Starting YOLO detection and extracting answers...", 0,
                      images.size(), 80.0);

    // Process images for extracting answers
    std::vector<std::vector<std::string>> results;
    results.emplace_back(HEADER_1);
    results.emplace_back(HEADER_2);

    ExamGrader grader;
    int numImages = uploaded_images.size();
    for (int i = 0; i < numImages; ++i) {
      std::string imageBasename = "page_" + std::to_string(i + 1);

      std::vector<cv::Mat> metadataImages, contentImages;
      if (!imgProc.splitImage(uploaded_images[i], metadataImages, contentImages)) {
        double currentProgress = 80.0 + (double(i + 1) / numImages) * 15.0;
        updateJobProgress(jobId, "extracting_answers", "Error: Failed to split image page " + std::to_string(i + 1),
                          i + 1, numImages, currentProgress, true,
                          "Failed to split image page " + std::to_string(i + 1));
        continue;
      }

      // Extract exam answers with YOLO detection
      std::vector<std::vector<Detection>> metadataDetections =
          tritonClient->inference(metadataImages, "metadata_model");
      std::vector<std::vector<Detection>> contentDetections = tritonClient->inference(contentImages, "content_model");
      std::vector<std::string> result =
          grader.extract_answers_from_detections(imageBasename, metadataDetections, contentDetections);
      results.emplace_back(result);

      // Update grading progress (80% to 95%)
      int gradedCount = i + 1;
      if (gradedCount % 5 == 0 || gradedCount == numImages - 1) {
        double currentProgress = 80.0 + (gradedCount / numImages) * 15.0;
        std::string message =
            "Extracted answers from " + std::to_string(gradedCount) + " of " + std::to_string(numImages) + " pages";
        updateJobProgress(jobId, "extracting_answers", message, gradedCount, numImages, currentProgress);
      }
    }
    updateJobProgress(jobId, "extracting_answers", "All answers extracted successfully", numImages, numImages, 95.0);
    Logger::info("SERVICE", "All answers extracted successfully from " + std::to_string(numImages) + " pages");

    /* ====================================================== */
    /* ===== Stage 5: Save Results to Storage (95-100%) ===== */
    /* ====================================================== */
    updateJobProgress(jobId, "saving_results", "Saving results to storage...", 0, numImages, 95.0);

    // Generate CSV content
    std::string csvContent = generateCSVString(results);
    // std::string csvBasename = pdfFileName.substr(0, pdfFileName.find_last_of('.'));

    std::string csvBasename;
    if (!qr_info.empty()) {
      // Clean QR info to make it filename-safe
      std::string cleanQrInfo = qr_info;
      // Replace spaces with underscores
      std::replace(cleanQrInfo.begin(), cleanQrInfo.end(), ' ', '_');
      // Remove any other invalid characters if needed
      csvBasename = cleanQrInfo;
      Logger::info("SERVICE", "Using QR info for filename: " + qr_info + " -> " + csvBasename);
    } else {
      // Fallback to PDF filename if QR info is not available
      csvBasename = pdfFileName.substr(0, pdfFileName.find_last_of('.'));
      Logger::info("SERVICE", "QR info empty, using PDF filename: " + csvBasename);
    }
    // // Save CSV locally (for regrade function)
    // std::string csvFilePath = outputDir + "/" + csvBasename + ".csv";
    // std::ofstream csvFile(csvFilePath);
    // if (!csvFile.is_open()) {
    //   updateJobProgress(jobId, "saving_results", "Error: Failed to save local CSV", 0, 0, 0.0, true,
    //                     "Failed to save results locally");
    //   Logger::error("SERVICE", "Failed to save results locally to " + csvFilePath);
    //   return false;
    // }
    // csvFile << csvContent;
    // csvFile.close();
    // Logger::info("SERVICE", "Results saved locally to " + csvFilePath);

    // Upload CSV to MinIO
    std::string csvObjectName = jobId + "/" + csvBasename + ".csv";
    if (!minioClient.uploadCSV(csvObjectName, csvContent)) {
      updateJobProgress(jobId, "saving_results", "Error: Failed to upload results to storage", 0, 0, 0.0, true,
                        "Failed to upload CSV to storage");
      Logger::error("SERVICE", "Failed to upload CSV to MinIO.");
      return false;
    }
    Logger::info("SERVICE", "Results uploaded to MinIO at " + csvObjectName + " (QR: " + qr_info + ")");

    updateJobProgress(jobId, "completed", "All processing completed successfully", numImages, numImages, 100.0);
    return true;
  } catch (const std::exception &e) {
    updateJobProgress(jobId, "error", "Extracting answers failed: " + std::string(e.what()), 0, 0, 0.0, true, e.what());
    Logger::error("SERVICE", "Extracting answers failed: " + std::string(e.what()));
    return false;
  }
}

bool examark::services::regradeWithJson(const std::string &outputDir, const std::string &csvData,
                                        const std::string &answerKeyJson, const std::string &regradeJobId,
                                        const std::string &originalJobId) {
  try {
    // Initialize MinIO client
    MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);

    // Parse the new JSON answer key
    std::map<std::string, std::vector<std::string>> examAnswerKeys;

    if (!answerKeyJson.empty()) {
      try {
        examAnswerKeys = parseJsonAnswerKey(answerKeyJson);

        // Save the new answer key JSON
        nlohmann::json newAnswerKeyJsonObj;
        newAnswerKeyJsonObj["exam_answer_keys"] = json::object();
        for (const auto &[examId, answers] : examAnswerKeys) {
          newAnswerKeyJsonObj["exam_answer_keys"][examId] = answers;
        }
        std::string newAnswerKeyJsonStr = newAnswerKeyJsonObj.dump(2);

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

      } catch (const std::exception &e) {
        Logger::error("REGRADE", "Failed to parse new JSON answer key: " + std::string(e.what()));
        // Fall back to existing answer key
      }
    }

    // Fallback to existing answer key if new one failed to parse
    if (examAnswerKeys.empty()) {
      Logger::info("REGRADE", "Using existing answer key for job: " + originalJobId);

      // Try to load from MinIO first
      std::string jsonObjectName = originalJobId + "/answer_key.json";
      std::string answerKeyJsonStr = minioClient.downloadJSON(jsonObjectName);

      if (!answerKeyJsonStr.empty()) {
        try {
          nlohmann::json answerKeyJsonObj = nlohmann::json::parse(answerKeyJsonStr);
          if (answerKeyJsonObj.contains("exam_answer_keys")) {
            for (const auto &[examId, answers] : answerKeyJsonObj["exam_answer_keys"].items()) {
              examAnswerKeys[examId] = answers.get<std::vector<std::string>>();
            }
          }
        } catch (const std::exception &e) {
          Logger::error("REGRADE", "Failed to parse JSON from MinIO: " + std::string(e.what()));
        }
      }

      // Fallback to local file if MinIO failed
      if (examAnswerKeys.empty()) {
        std::string answerKeyPath = outputDir + "/answer_key.json";
        if (std::filesystem::exists(answerKeyPath)) {
          std::ifstream answerKeyFile(answerKeyPath);
          if (answerKeyFile.is_open()) {
            try {
              nlohmann::json answerKeyJsonObj;
              answerKeyFile >> answerKeyJsonObj;
              answerKeyFile.close();

              if (answerKeyJsonObj.contains("exam_answer_keys")) {
                for (const auto &[examId, answers] : answerKeyJsonObj["exam_answer_keys"].items()) {
                  examAnswerKeys[examId] = answers.get<std::vector<std::string>>();
                }
              }
            } catch (const std::exception &e) {
              Logger::error("REGRADE", "Failed to parse local answer key JSON: " + std::string(e.what()));
            }
          }
        }
      }
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
        cell.erase(cell.begin(),
                   std::find_if(cell.begin(), cell.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                   cell.end());

        row.push_back(cell);
      }
      csvRows.push_back(row);
    }

    if (csvRows.size() < 4) {
      Logger::error("REGRADE", "CSV data has insufficient rows (" + std::to_string(csvRows.size()) +
                                   ") for job: " + originalJobId);
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

          if (partNumber == "1" && part1Answers.size() < PART_1_NUM_QUESTIONS) {
            if (col < csvRows[row].size()) {
              part1Answers.push_back(csvRows[row][col]);
            } else {
              part1Answers.push_back("_");
            }
          } else if (partNumber == "2" && part2Answers.size() < PART_2_NUM_QUESTIONS) {
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
    for (const auto &entry : std::filesystem::directory_iterator(outputDir)) {
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
    for (const auto &entry : std::filesystem::directory_iterator(outputDir)) {
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

  } catch (const std::exception &e) {
    Logger::error("REGRADE", "Exception in regrade process: " + std::string(e.what()) + " for job: " + originalJobId);
    return false;
  }
}
