#include <iostream>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "services/Service.h"
#include "utils/ImageProcessor.h"
#include "utils/ExamGrader.h"

// Constants for CSV header
const std::vector<std::string> HEADER_1 = {
  "", "", "", "Part", 
  "1", "1", "1", "1", "1", "1", "1", "1", 
  "1", "1", "1", "1", "1", "1", "1", "1", 
  "2", "2", "2", "2", "2", "2", "2", "2"
};
const std::vector<std::string> HEADER_2 = {
  "Image name", "Student ID", "Exam ID", "Question",
  "1", "2", "3", "4", "5", "6", "7", "8",
  "9", "10", "11", "12", "13", "14", "15", "16",
  "1", "2", "3", "4", "5", "6", "7", "8"
};

bool grading(const std::string& pdfFileName, const std::string& pdfData, 
             const std::string& answerKeyCSV, const std::string& outputDir, 
             TritonClient* tritonClient) {
  Logger::info("SERVICE", "Starting grading process for: " + pdfFileName);
  
  // Save answer key CSV
  if (!std::filesystem::exists(outputDir)) {
    std::filesystem::create_directory(outputDir);
  }

  std::string answerKeyPath = outputDir + "/answer_key.csv";
  std::ofstream answerKeyFile(answerKeyPath);
  if (!answerKeyFile.is_open()) {
    Logger::error("SERVICE", "Failed to save answer key CSV: " + answerKeyPath);
    return false;
  }
  answerKeyFile << answerKeyCSV;
  answerKeyFile.close();
  Logger::info("SERVICE", "Answer key saved to: " + answerKeyPath);
  
  // Process PDF data
  ImageProcessor imgProc;
  Logger::info("SERVICE", "Converting PDF to images...");
  std::vector<cv::Mat> images;
  if (!imgProc.getRequestImages(pdfData.c_str(), pdfData.size(), images)) {
      Logger::error("SERVICE", "Failed to convert PDF to images.");
      return false;
  }

  if (!std::filesystem::exists(outputDir)) {
    std::filesystem::create_directory(outputDir);
  }

  // Process images
  std::vector<std::vector<std::string>> results;
  results.push_back(HEADER_1);
  results.push_back(HEADER_2);
  
  ExamGrader grader;
  int numImages = images.size();
  std::string numImagesStr = std::to_string(numImages);
  for (int i = 0; i < numImages; ++i) {
    Logger::info("SERVICE", "Processing page " + std::to_string(i + 1) + "/" + numImagesStr);
    std::string imageBasename = "page_" + std::to_string(i + 1);
    std::string outputFilepath = outputDir + "/" + imageBasename + ".jpg";
    cv::imwrite(outputFilepath, images[i]);

    std::vector<cv::Mat> metadataImages, contentImages;
    if (!imgProc.splitImage(images[i], metadataImages, contentImages)) {
      Logger::error("SERVICE", "Failed to split image for page " + std::to_string(i + 1));
      continue;
    }
  
    // Extract metadata and content detections
    std::vector<std::vector<Detection>> metadataDetections = 
      tritonClient->inference(metadataImages, "metadata_model");
    
    std::vector<std::vector<Detection>> contentDetections = 
      tritonClient->inference(contentImages, "content_model");
    
    // tritonClient->drawAndSaveResults(metadataImages, metadataDetections, "../assets/results");
    // tritonClient->drawAndSaveResults(contentImages, contentDetections, "../assets/results");

    std::vector<std::string> result = grader.extractAnswers(imageBasename, metadataDetections, contentDetections);
    results.push_back(result);
  }

  // 4. Save results to CSV
  size_t numOriginalRows = results.size();    
  size_t numOriginalCols = results[0].size(); 
  
  std::string csvFilePath = outputDir + "/" + pdfFileName.substr(0, pdfFileName.find_last_of('.')) + ".csv";
  std::ofstream csvFile(csvFilePath);
  if (!csvFile.is_open()) {
    Logger::error("SERVICE", "Failed to open CSV file for writing: " + csvFilePath);
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
  Logger::success("SERVICE", "Answer extraction completed. Results saved to: " + csvFilePath);
  return true;
} 
