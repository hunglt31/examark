#include "utils/ImageProcessor.h"
#include "utils/Logger.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/wechat_qrcode.hpp>

// Constants for qr reader
static const std::string detect_model = "../assets/wechat_qr_decoder/detect.prototxt";
static const std::string detect_weights = "../assets/wechat_qr_decoder/detect.caffemodel";
static const std::string sr_model = "../assets/wechat_qr_decoder/sr.prototxt";
static const std::string sr_weights = "../assets/wechat_qr_decoder/sr.caffemodel";

// Constants for image processor
const std::string REF_IMG_PATH = "../assets/reference.jpg";
const cv::Mat REF_IMG_ORI = cv::imread(REF_IMG_PATH, cv::IMREAD_GRAYSCALE);

cv::Mat REF_IMG_GRAY = [] {
  cv::Mat tmp;
  cv::resize(REF_IMG_ORI, tmp, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
  return tmp;
}();

const float LOWE_RATIO_THRESHOLD = 0.7f;
const float RANSAC_THRESHOLD = 5.0f;
const int ITERATIONS = 2000;

// --- 3. Coordinates for metadata ---
const int STUDENT_ID_CONTOUR_1_COORD_X = 167;
const int STUDENT_ID_CONTOUR_1_COORD_Y = 372;
const int STUDENT_ID_CONTOUR_2_COORD_X = 676;
const int STUDENT_ID_CONTOUR_2_COORD_Y = 1109;

const int EXAM_ID_CONTOUR_1_COORD_X = 677;
const int EXAM_ID_CONTOUR_1_COORD_Y = 372;
const int EXAM_ID_CONTOUR_2_COORD_X = 888;
const int EXAM_ID_CONTOUR_2_COORD_Y = 1109;

// --- 4. Coordinates for content ---
const int CONTENT_11_CONTOUR_1_COORD_X = 230;
const int CONTENT_11_CONTOUR_1_COORD_Y = 1265;
const int CONTENT_11_CONTOUR_2_COORD_X = 710;
const int CONTENT_11_CONTOUR_2_COORD_Y = 1605;

const int CONTENT_12_CONTOUR_1_COORD_X = 750;
const int CONTENT_12_CONTOUR_1_COORD_Y = 1265;
const int CONTENT_12_CONTOUR_2_COORD_X = 1230;
const int CONTENT_12_CONTOUR_2_COORD_Y = 1605;

const int CONTENT_13_CONTOUR_1_COORD_X = 1270;
const int CONTENT_13_CONTOUR_1_COORD_Y = 1265;
const int CONTENT_13_CONTOUR_2_COORD_X = 1750;
const int CONTENT_13_CONTOUR_2_COORD_Y = 1605;

const int CONTENT_14_CONTOUR_1_COORD_X = 1790;
const int CONTENT_14_CONTOUR_1_COORD_Y = 1265;
const int CONTENT_14_CONTOUR_2_COORD_X = 2270;
const int CONTENT_14_CONTOUR_2_COORD_Y = 1605;

const int CONTENT_21_CONTOUR_1_COORD_X = 230;
const int CONTENT_21_CONTOUR_1_COORD_Y = 1685;
const int CONTENT_21_CONTOUR_2_COORD_X = 710;
const int CONTENT_21_CONTOUR_2_COORD_Y = 2180;

const int CONTENT_22_CONTOUR_1_COORD_X = 750;
const int CONTENT_22_CONTOUR_1_COORD_Y = 1685;
const int CONTENT_22_CONTOUR_2_COORD_X = 1230;
const int CONTENT_22_CONTOUR_2_COORD_Y = 2180;

const int CONTENT_23_CONTOUR_1_COORD_X = 1270;
const int CONTENT_23_CONTOUR_1_COORD_Y = 1685;
const int CONTENT_23_CONTOUR_2_COORD_X = 1750;
const int CONTENT_23_CONTOUR_2_COORD_Y = 2180;

const int CONTENT_24_CONTOUR_1_COORD_X = 1790;
const int CONTENT_24_CONTOUR_1_COORD_Y = 1685;
const int CONTENT_24_CONTOUR_2_COORD_X = 2270;
const int CONTENT_24_CONTOUR_2_COORD_Y = 2180;

ImageProcessor::ImageProcessor() {
  sift = cv::SIFT::create();
  flann_matcher =
      cv::makePtr<cv::FlannBasedMatcher>(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(50));
}

bool ImageProcessor::renderImages(const char *pdfData, int dataSize, std::vector<cv::Mat> &images,
                                  ProgressCallback progressCallback, double dpi) {
  images.clear();
  if (progressCallback)
    progressCallback(0, 0, 0.0);

  std::unique_ptr<poppler::document> doc(poppler::document::load_from_raw_data(pdfData, dataSize));
  if (!doc)
    return false;

  int numPages = doc->pages();
  if (numPages == 0)
    return false;

  if (progressCallback)
    progressCallback(0, numPages, 0.0);

  images.reserve(numPages);
  poppler::page_renderer renderer;

  for (int i = 0; i < numPages; ++i) {
    std::unique_ptr<poppler::page> page(doc->create_page(i));
    if (!page)
      continue;

    poppler::image popImg = renderer.render_page(page.get(), dpi, dpi);
    if (!popImg.is_valid() || popImg.width() <= 0 || popImg.height() <= 0)
      continue;

    cv::Mat image_bgra(popImg.height(), popImg.width(), CV_8UC4, (void *)popImg.data(), popImg.bytes_per_row());
    std::memcpy(image_bgra.data, popImg.data(), popImg.bytes_per_row() * popImg.height());

    cv::Mat image_bgr;
    cv::cvtColor(image_bgra, image_bgr, cv::COLOR_BGRA2BGR);
    images.emplace_back(image_bgr);

    if (progressCallback) {
      double pageProgress = ((double)(i + 1) / numPages) * 9.0;
      progressCallback(i + 1, numPages, pageProgress);
    }
  }

  if (progressCallback)
    progressCallback(numPages, numPages, 9.0);

  return !images.empty();
}

cv::Mat ImageProcessor::preprocessImage(const cv::Mat &imgScan, cv::Size imgSize) {
  cv::Mat img_scan_gray = cv::Mat(imgScan.size(), CV_8UC1);
  cv::cvtColor(imgScan, img_scan_gray, cv::COLOR_BGR2GRAY);

  // SIFT detector
  std::vector<cv::KeyPoint> kpsScan, kpsRef;
  cv::Mat descScan, descRef;
  sift->detectAndCompute(img_scan_gray, cv::noArray(), kpsScan, descScan);
  sift->detectAndCompute(REF_IMG_GRAY, cv::noArray(), kpsRef, descRef);

  // FLANN matcher parameters
  std::vector<std::vector<cv::DMatch>> knnMatches;
  flann_matcher->knnMatch(descScan, descRef, knnMatches, 2);

  // Filter matches using Lowe's ratio test
  std::vector<cv::DMatch> goodMatches;
  for (const auto &m_n : knnMatches) {
    if (m_n[0].distance < LOWE_RATIO_THRESHOLD * m_n[1].distance)
      goodMatches.push_back(m_n[0]);
  }

  // Align images using homography
  std::vector<cv::Point2f> kpsScanPt, kpsRefPt;
  for (const auto &m : goodMatches) {
    kpsScanPt.push_back(kpsScan[m.queryIdx].pt);
    kpsRefPt.push_back(kpsRef[m.trainIdx].pt);
  }

  cv::Mat H = cv::findHomography(kpsScanPt, kpsRefPt, cv::RANSAC, RANSAC_THRESHOLD, cv::noArray(), ITERATIONS);
  cv::Mat imgAligned;
  cv::warpPerspective(imgScan, imgAligned, H, imgSize, cv::INTER_LINEAR);

  cv::Mat contrast_corrected;
  applyGammaCorrection_CUDA(imgAligned, contrast_corrected, GAMMA_LUT_CUDA);

  return contrast_corrected;
}

cv::Mat ImageProcessor::paddingImage(const cv::Mat &image, cv::Size paddingSize) {
  int height = image.rows, width = image.cols;
  float scale = static_cast<float>(paddingSize.width) / std::max(height, width);
  int newW = static_cast<int>(width * scale);
  int newH = static_cast<int>(height * scale);

  cv::Mat resized = cv::Mat(newH, newW, CV_8UC3);
  cv::resize(image, resized, resized.size(), 0, 0, cv::INTER_LINEAR);

  cv::Mat padded(paddingSize, CV_8UC3, cv::Scalar(0, 0, 0));

  int x = (paddingSize.width - newW) / 2;
  int y = (paddingSize.height - newH) / 2;

  resized.copyTo(padded(cv::Rect(x, y, newW, newH)));
  return padded;
}

bool ImageProcessor::splitImage(const cv::Mat &image, std::vector<cv::Mat> &metadataImages,
                                std::vector<cv::Mat> &contentImages) {
  try {
    // Metadata
    cv::Mat studentId = image(cv::Range(STUDENT_ID_CONTOUR_1_COORD_Y, STUDENT_ID_CONTOUR_2_COORD_Y),
                              cv::Range(STUDENT_ID_CONTOUR_1_COORD_X, STUDENT_ID_CONTOUR_2_COORD_X));
    cv::Mat examId = image(cv::Range(EXAM_ID_CONTOUR_1_COORD_Y, EXAM_ID_CONTOUR_2_COORD_Y),
                           cv::Range(EXAM_ID_CONTOUR_1_COORD_X, EXAM_ID_CONTOUR_2_COORD_X));

    // Content part 1
    cv::Mat content11 = image(cv::Range(CONTENT_11_CONTOUR_1_COORD_Y, CONTENT_11_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_11_CONTOUR_1_COORD_X, CONTENT_11_CONTOUR_2_COORD_X));
    cv::Mat content12 = image(cv::Range(CONTENT_12_CONTOUR_1_COORD_Y, CONTENT_12_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_12_CONTOUR_1_COORD_X, CONTENT_12_CONTOUR_2_COORD_X));
    cv::Mat content13 = image(cv::Range(CONTENT_13_CONTOUR_1_COORD_Y, CONTENT_13_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_13_CONTOUR_1_COORD_X, CONTENT_13_CONTOUR_2_COORD_X));
    cv::Mat content14 = image(cv::Range(CONTENT_14_CONTOUR_1_COORD_Y, CONTENT_14_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_14_CONTOUR_1_COORD_X, CONTENT_14_CONTOUR_2_COORD_X));

    // Content part 2
    cv::Mat content21 = image(cv::Range(CONTENT_21_CONTOUR_1_COORD_Y, CONTENT_21_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_21_CONTOUR_1_COORD_X, CONTENT_21_CONTOUR_2_COORD_X));
    cv::Mat content22 = image(cv::Range(CONTENT_22_CONTOUR_1_COORD_Y, CONTENT_22_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_22_CONTOUR_1_COORD_X, CONTENT_22_CONTOUR_2_COORD_X));
    cv::Mat content23 = image(cv::Range(CONTENT_23_CONTOUR_1_COORD_Y, CONTENT_23_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_23_CONTOUR_1_COORD_X, CONTENT_23_CONTOUR_2_COORD_X));
    cv::Mat content24 = image(cv::Range(CONTENT_24_CONTOUR_1_COORD_Y, CONTENT_24_CONTOUR_2_COORD_Y),
                              cv::Range(CONTENT_24_CONTOUR_1_COORD_X, CONTENT_24_CONTOUR_2_COORD_X));

    metadataImages.clear();
    metadataImages.emplace_back(paddingImage(studentId));
    metadataImages.emplace_back(paddingImage(examId));

    contentImages.clear();
    contentImages.emplace_back(paddingImage(content11));
    contentImages.emplace_back(paddingImage(content12));
    contentImages.emplace_back(paddingImage(content13));
    contentImages.emplace_back(paddingImage(content14));
    contentImages.emplace_back(paddingImage(content21));
    contentImages.emplace_back(paddingImage(content22));
    contentImages.emplace_back(paddingImage(content23));
    contentImages.emplace_back(paddingImage(content24));

    return true;
  } catch (const cv::Exception &e) {
    Logger::error("IMAGE PROCESSOR", "Image splitting failed: " + std::string(e.what()));
    return false;
  }
}

bool ImageProcessor::get_qr_code_info(const cv::Mat &image, std::string &qr_info) {
  static cv::wechat_qrcode::WeChatQRCode detector(detect_model, detect_weights, sr_model, sr_weights);
  std::vector<cv::Mat> qr_imgs;
  std::vector<std::string> results = detector.detectAndDecode(image, qr_imgs);
  qr_info = results.empty() ? "" : results[0];
  return !qr_info.empty();
}
