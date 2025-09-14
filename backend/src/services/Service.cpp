#include "services/Service.h"
#include "controllers/Controller.h"
#include "utils/ExamConfig.h"
#include "utils/ExamExtractor.h"
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

std::string examark::services::get_pdf_qr_code(const std::string &pdfData) {
  try {
    ImageProcessor imgProc;
    std::vector<cv::Mat> images;

    auto progressCallback = [](int currentPage, int totalPages, double percent) {};

    if (!imgProc.renderImages(pdfData.c_str(), pdfData.size(), images, progressCallback, 300.0, 1)) {
      Logger::error("SERVICE", "Failed to render first page of PDF");
      return "";
    }
    if (images.empty()) {
      Logger::error("SERVICE", "No images rendered from PDF");
      return "";
    }

    std::string qr_info;
    if (!imgProc.get_qr_code_info(images[0], qr_info)) {
      Logger::error("SERVICE", "No QR code found in the first page");
      return "";
    }

    std::replace(qr_info.begin(), qr_info.end(), ' ', '-');
    return qr_info;
  } catch (const std::exception &e) {
    Logger::error("SERVICE", "Error extracting QR code: " + std::string(e.what()));
    return "";
  }
}

std::string examark::services::extract_all_exams_answers(const std::string &pdfFileName, const std::string &pdfData,
                                                         TritonClient *tritonClient, const std::string &jobId) {
  try {
    /* ============================================= */
    /* ===== Stage 1: Rendering images (0-9%) ===== */
    /* ============================================= */
    MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
    ImageProcessor imgProc;
    std::vector<cv::Mat> images;
    utils::updateJobProgress(jobId, "rendering_images", "Starting PDF conversion...", 0, 0, 0.0, false, "");

    auto progressCallback = [&jobId](int currentPage, int totalPages, double percent) {
      std::string message;
      if (totalPages > 0 && currentPage > 0) {
        message = "Rendered " + std::to_string(currentPage) + " of " + std::to_string(totalPages) + " pages";
      } else if (totalPages > 0) {
        message = "Found " + std::to_string(totalPages) + " pages, starting conversion...";
      } else {
        message = "Loading PDF document...";
      }
      utils::updateJobProgress(jobId, "rendering_images", message, currentPage, totalPages, percent, false, "");
    };

    if (!imgProc.renderImages(pdfData.c_str(), pdfData.size(), images, progressCallback, 300.0)) {
      utils::updateJobProgress(jobId, "rendering_images", "Error: Failed to convert PDF", 0, 0, 0.0, true,
                               "Failed to convert PDF to images");
      Logger::error("SERVICE", "Failed to convert PDF to images");
      nlohmann::json error_response;
      error_response["status"] = "error";
      error_response["message"] = "Failed to convert PDF to images";
      return error_response.dump();
    }

    utils::updateJobProgress(jobId, "rendering_images",
                             "PDF converted successfully - " + std::to_string(images.size()) + " pages rendered",
                             images.size(), images.size(), 9.0, false, "");

    /* ====================================================== */
    /* ===== Stage 2: Read QR code information (9-10%) ===== */
    /* ====================================================== */
    utils::updateJobProgress(jobId, "reading_qr_codes", "Reading QR codes from images...", 0, images.size(), 9.0, false,
                             "");

    int first_exam_image_idx = 0;
    bool get_qr_info = false;
    bool finish_qr_info_pages = false;
    std::string qr_info = jobId;
    std::string tmp_qr_info;

    for (first_exam_image_idx = 0; first_exam_image_idx < images.size(); ++first_exam_image_idx) {
      if (imgProc.get_qr_code_info(images[first_exam_image_idx], tmp_qr_info)) {
        if (!get_qr_info) {
          qr_info = tmp_qr_info;
          get_qr_info = true;
        }
      } else {
        finish_qr_info_pages = true;
        break;
      }
    }

    if (!get_qr_info) {
      utils::updateJobProgress(jobId, "reading_qr_codes", "Warning: No QR code found in images", 0, images.size(), 0.0,
                               false, "No QR code found in images, using default filename");
      Logger::error("SERVICE", "No QR code found in images");
    } else {
      std::replace(qr_info.begin(), qr_info.end(), ' ', '-');
    }

    utils::updateJobQrInfo(jobId, qr_info);

    utils::updateJobProgress(jobId, "reading_qr_codes", "QR code information read successfully - " + qr_info + " found",
                             first_exam_image_idx, images.size(), 10.0, false, "");

    /* =================================================================== */
    /* ===== Stage 3: Preprocess and upload images to MinIO (10-90%) ===== */
    /* =================================================================== */
    std::vector<cv::Mat> exam_images;
    nlohmann::json images_map = nlohmann::json::object();

    utils::updateJobProgress(jobId, "processing_images", "Preprocessing and uploading images to storage...", 0,
                             images.size(), 10.0, false, "");

    const int total_images = images.size();
    const int total_exam_images = total_images - first_exam_image_idx;

    for (int i = first_exam_image_idx; i < total_images; ++i) {
      cv::Mat exam_image = imgProc.preprocessImage(images[i]);

      std::string page_key = "page_" + std::to_string(i + 1);
      std::string minioObjectName = qr_info + "/" + page_key + ".jpg";

      if (!minioClient.uploadImage(minioObjectName, exam_image)) {
        utils::updateJobProgress(
            jobId, "processing_images", "Failed to process image " + std::to_string(i - first_exam_image_idx + 1),
            i - first_exam_image_idx + 1, total_exam_images, 0.0, true, "Failed to upload image to storage");
      }

      exam_images.emplace_back(exam_image);
      images_map[page_key] = minioClient.getFileUrl(minioObjectName);

      double uploadProgress = 10.0 + (double(i - first_exam_image_idx + 1) / total_exam_images) * 80.0;
      utils::updateJobProgress(jobId, "processing_images",
                               "Processed " + std::to_string(i - first_exam_image_idx + 1) + " of " +
                                   std::to_string(total_exam_images) + " images",
                               i - first_exam_image_idx + 1, total_exam_images, uploadProgress, false, "");
    }

    /* =================================================== */
    /* ===== Stage 4: Extract exams answers (90-99%) ===== */
    /* =================================================== */
    utils::updateJobProgress(jobId, "extracting_answers", "Starting YOLO detection and extracting answers...", 0,
                             exam_images.size(), 90.0, false, "");

    // Process images for extracting answers
    std::vector<std::vector<std::string>> results;
    results.emplace_back(HEADER_1);
    results.emplace_back(HEADER_2);

    ExamExtractor extractor;
    for (int i = 0; i < total_exam_images; ++i) {
      std::string imageBasename = "page_" + std::to_string(i + first_exam_image_idx + 1);

      std::vector<cv::Mat> metadataImages, contentImages;
      if (!imgProc.splitImage(exam_images[i], metadataImages, contentImages)) {
        double currentProgress = 90.0 + (double(i + 1) / total_exam_images) * 9.0;
        utils::updateJobProgress(jobId, "extracting_answers",
                                 "Error: Failed to split image page " + std::to_string(i + 1), i + 1, total_exam_images,
                                 currentProgress, true, "Failed to split image page " + std::to_string(i + 1));
        continue;
      }

      // Extract exam answers with YOLO detection
      std::vector<std::vector<Detection>> metadataDetections =
          tritonClient->inference(metadataImages, "metadata_model");
      std::vector<std::vector<Detection>> contentDetections = tritonClient->inference(contentImages, "content_model");
      std::vector<std::string> result =
          extractor.extract_answers_from_detections(imageBasename, metadataDetections, contentDetections);
      results.emplace_back(result);

      // Update grading progress (85% to 95%)
      int gradedCount = i + 1;
      if (gradedCount % 5 == 0 || gradedCount == total_exam_images - 1) {
        double currentProgress = 90.0 + (gradedCount / total_exam_images) * 9.0;
        std::string message = "Extracted answers from " + std::to_string(gradedCount) + " of " +
                              std::to_string(total_exam_images) + " pages";
        utils::updateJobProgress(jobId, "extracting_answers", message, gradedCount, total_exam_images, currentProgress,
                                 false, "");
      }
    }
    utils::updateJobProgress(jobId, "extracting_answers", "All answers extracted successfully", total_exam_images,
                             total_exam_images, 99.0, false, "");
    Logger::info("SERVICE", "All answers extracted successfully from " + std::to_string(total_exam_images) + " pages");

    /* ====================================================== */
    /* ===== Stage 5: Save Results to Storage (99-100%) ===== */
    /* ====================================================== */
    utils::updateJobProgress(jobId, "saving_results", "Saving results to storage...", 0, total_exam_images, 99.0, false,
                             "");

    std::string csvContent = generateCSVString(results);

    // Upload CSV to MinIO
    std::string csvObjectName = qr_info + "/" + qr_info + ".csv";
    if (!minioClient.uploadCSV(csvObjectName, csvContent)) {
      utils::updateJobProgress(jobId, "saving_results", "Error: Failed to upload results to storage", 0, 0, 0.0, true,
                               "Failed to upload CSV to storage");
      Logger::error("SERVICE", "Failed to upload CSV to MinIO.");
      nlohmann::json error_response;
      error_response["status"] = "error";
      error_response["message"] = "Failed to upload CSV to storage";
      return error_response.dump();
    }
    Logger::info("SERVICE", "Results uploaded to MinIO at " + csvObjectName + " (QR: " + qr_info + ")");

    utils::updateJobProgress(jobId, "completed", "All processing completed successfully", total_exam_images,
                             total_exam_images, 100.0, false, "");

    nlohmann::json success_response;
    success_response["pdf"] = pdfFileName;
    success_response["class"] = qr_info;
    success_response["csv"] = minioClient.getFileUrl(csvObjectName);
    success_response["images"] = images_map;
    success_response["status"] = "completed";

    return success_response.dump();

  } catch (const std::exception &e) {
    utils::updateJobProgress(jobId, "error", "Extracting answers failed: " + std::string(e.what()), 0, 0, 0.0, true,
                             e.what());
    Logger::error("SERVICE", "Extracting answers failed: " + std::string(e.what()));
    nlohmann::json error_response;
    error_response["status"] = "error";
    error_response["message"] = "Extracting answers failed: " + std::string(e.what());
    return error_response.dump();
  }
}
