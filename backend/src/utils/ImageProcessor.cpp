#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdlib> 
#include <cstdio>
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <poppler/cpp/poppler-image.h>
#include <poppler/cpp/poppler-page-renderer.h>
#include <memory>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils/ImageProcessor.h"

// Constants for image processor
// --- 1. Align image ---
const std::string REF_IMG_PATH = "../assets/references/reference.png";
const cv::Mat REF_IMG_ORI = cv::imread(REF_IMG_PATH, cv::IMREAD_GRAYSCALE);
cv::Mat REF_IMG_GRAY = [] {
    cv::Mat tmp;
    cv::resize(REF_IMG_ORI, tmp, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
    return tmp;
}();

const int MIN_GOOD_MATCHES = 15;
const float RANSAC_THRESHOLD = 3.0f;
const int ITERATIONS = 2000;

// --- 2. Apply constrast ---
const float DEFAULT_GAMMA = 2.2f;

cv::Mat createGammaLUT(float gamma) {
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();
    const double inv255 = 1.0 / 255.0;
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(std::pow(i * inv255, gamma) * 255.0);
    }
    return lut;
}

const cv::Mat GAMMA_LUT = createGammaLUT(DEFAULT_GAMMA);

// --- 3. Coordinates for metadata ---
const int STUDENT_ID_CONTOUR_1_COORD_X = 187;
const int STUDENT_ID_CONTOUR_1_COORD_Y = 372;
const int STUDENT_ID_CONTOUR_2_COORD_X = 656;
const int STUDENT_ID_CONTOUR_2_COORD_Y = 1109;

const int EXAM_ID_CONTOUR_1_COORD_X = 697;
const int EXAM_ID_CONTOUR_1_COORD_Y = 372;
const int EXAM_ID_CONTOUR_2_COORD_X = 868;
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

bool ImageProcessor::getRequestImagesWithProgress(
  const char* pdfData, int dataSize, std::vector<cv::Mat> &images, 
  ProgressCallback progressCallback, double dpi) 
{
  images.clear();
  
  // Load PDF document
  if (progressCallback) {
    progressCallback(0, 0, 5.0);
  }
  
  std::unique_ptr<poppler::document> doc(poppler::document::load_from_raw_data(pdfData, dataSize));
  if (!doc) {
    return false;
  }
  
  int numPages = doc->pages();
  if (numPages == 0) {
    return false;
  }

  if (progressCallback) {
    progressCallback(0, numPages, 10.0);
  }

  poppler::page_renderer renderer;
  images.reserve(numPages);
  
  for (int i = 0; i < numPages; ++i) {    
    std::unique_ptr<poppler::page> page(doc->create_page(i));
    if (!page) {
      continue;
    }

    poppler::image popImg = renderer.render_page(page.get(), dpi, dpi); 
    if (!popImg.is_valid()) {
      continue;
    }
    
    cv::Mat img(popImg.height(), popImg.width(), CV_8UC4, (void*)popImg.data(), popImg.bytes_per_row());
    cv::Mat imgBGR;
    cv::cvtColor(img, imgBGR, cv::COLOR_BGRA2BGR);
    cv::Mat imgAligned = alignImage(imgBGR);

    cv::Mat corrected;
    cv::LUT(imgAligned, GAMMA_LUT, corrected);
    images.emplace_back(corrected);
    
    if (progressCallback) {
      double pageProgress = 10.0 + ((double)(i + 1) / numPages) * 65.0;
      progressCallback(i + 1, numPages, pageProgress);
    }
  }
  
  if (progressCallback) {
    progressCallback(numPages, numPages, 75.0); 
  }
  
  return !images.empty();
}

bool ImageProcessor::getRequestImages(const char* pdfData, int dataSize, std::vector<cv::Mat> &images, double dpi) {
  images.clear();
  // Load PDF document
  std::unique_ptr<poppler::document> doc(poppler::document::load_from_raw_data(pdfData, dataSize));
  if (!doc) {
    Logger::error("IMAGE PROCESSOR", "Failed to load PDF document from raw data.");
    return false;
  }
  
  int numPages = doc->pages();
  if (numPages == 0) {
    Logger::error("IMAGE PROCESSOR", "No pages found in the PDF document.");
    return false;
  }

  // Render pages to images
  poppler::page_renderer renderer;
  images.reserve(numPages);
  for (int i = 0; i < numPages; ++i) {
    std::unique_ptr<poppler::page> page(doc->create_page(i));
    if (!page) {
      continue;
    }

    poppler::image popImg = renderer.render_page(page.get(), dpi, dpi); 
    if (!popImg.is_valid()) {
      continue;
    }
    
    cv::Mat img(popImg.height(), popImg.width(), CV_8UC4, (void*)popImg.data(), popImg.bytes_per_row());
    cv::Mat imgBGR;
    cv::cvtColor(img, imgBGR, cv::COLOR_BGRA2BGR);

    cv::Mat imgAligned = alignImage(imgBGR);
    
    cv::Mat corrected;
    cv::LUT(imgAligned, GAMMA_LUT, corrected);
    images.emplace_back(corrected);
  }
  return !images.empty();
}

cv::Mat ImageProcessor::alignImage(const cv::Mat &imgScan, cv::Size imgSize) {
  cv::Mat imgScanGray;
  cv::cvtColor(imgScan, imgScanGray, cv::COLOR_BGR2GRAY);

  // Create SIFT detector
  auto sift = cv::SIFT::create();
  std::vector<cv::KeyPoint> kpsScan, kpsRef;
  cv::Mat descScan, descRef;
  sift->detectAndCompute(imgScanGray, cv::noArray(), kpsScan, descScan);
  sift->detectAndCompute(REF_IMG_GRAY, cv::noArray(), kpsRef, descRef);

  // FLANN matcher parameters
  cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(50));
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher.knnMatch(descScan, descRef, knnMatches, 2);

  // Filter matches using Lowe's ratio test
  std::vector<cv::DMatch> goodMatches;
  double threshold = 0.2; 
  
  while (goodMatches.size() < MIN_GOOD_MATCHES && threshold <= 0.7) {
    goodMatches.clear();
    for (const auto &m_n : knnMatches) {
      if (m_n.size() < 2)
        continue;
      if (m_n[0].distance < threshold * m_n[1].distance)
        goodMatches.push_back(m_n[0]);
    }
    threshold += 0.1;  
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
  
  return imgAligned;
}

cv::Mat ImageProcessor::paddingImage(cv::Mat &image, cv::Size paddingSize) {
  int height = image.rows, width = image.cols;
  float scale = static_cast<float>(paddingSize.width) / std::max(height, width);
  int newW = static_cast<int>(width * scale), newH = static_cast<int>(height * scale);

  cv::Mat tmp;
  cv::resize(image, tmp, cv::Size(newW, newH));

  int padW = paddingSize.width - newW, padH = paddingSize.height - newH;
  int left = padW / 2, right = padW - left;
  int top = padH / 2, bottom = padH - top;

  cv::copyMakeBorder(tmp, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return image;
}

bool ImageProcessor::splitImage(
  const cv::Mat &image,
  std::vector<cv::Mat>& metadataImages,
  std::vector<cv::Mat>& contentImages) 
{   
  try {
    // Extract regions for metadata
    cv::Mat studentId = image(
      cv::Range(STUDENT_ID_CONTOUR_1_COORD_Y, STUDENT_ID_CONTOUR_2_COORD_Y), 
      cv::Range(STUDENT_ID_CONTOUR_1_COORD_X, STUDENT_ID_CONTOUR_2_COORD_X));
    cv::Mat examId = image(
      cv::Range(EXAM_ID_CONTOUR_1_COORD_Y, EXAM_ID_CONTOUR_2_COORD_Y), 
      cv::Range(EXAM_ID_CONTOUR_1_COORD_X, EXAM_ID_CONTOUR_2_COORD_X));
                                
    // Extract content region                             
    cv::Mat content11 = image(
      cv::Range(CONTENT_11_CONTOUR_1_COORD_Y, CONTENT_11_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_11_CONTOUR_1_COORD_X, CONTENT_11_CONTOUR_2_COORD_X));
    cv::Mat content12 = image(
      cv::Range(CONTENT_12_CONTOUR_1_COORD_Y, CONTENT_12_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_12_CONTOUR_1_COORD_X, CONTENT_12_CONTOUR_2_COORD_X));
    cv::Mat content13 = image(
      cv::Range(CONTENT_13_CONTOUR_1_COORD_Y, CONTENT_13_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_13_CONTOUR_1_COORD_X, CONTENT_13_CONTOUR_2_COORD_X));
    cv::Mat content14 = image(
      cv::Range(CONTENT_14_CONTOUR_1_COORD_Y, CONTENT_14_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_14_CONTOUR_1_COORD_X, CONTENT_14_CONTOUR_2_COORD_X));                     
    cv::Mat content21 = image(
      cv::Range(CONTENT_21_CONTOUR_1_COORD_Y, CONTENT_21_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_21_CONTOUR_1_COORD_X, CONTENT_21_CONTOUR_2_COORD_X));
    cv::Mat content22 = image(
      cv::Range(CONTENT_22_CONTOUR_1_COORD_Y, CONTENT_22_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_22_CONTOUR_1_COORD_X, CONTENT_22_CONTOUR_2_COORD_X));
    cv::Mat content23 = image(
      cv::Range(CONTENT_23_CONTOUR_1_COORD_Y, CONTENT_23_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_23_CONTOUR_1_COORD_X, CONTENT_23_CONTOUR_2_COORD_X));
    cv::Mat content24 = image(
      cv::Range(CONTENT_24_CONTOUR_1_COORD_Y, CONTENT_24_CONTOUR_2_COORD_Y), 
      cv::Range(CONTENT_24_CONTOUR_1_COORD_X, CONTENT_24_CONTOUR_2_COORD_X));

    // Padding
    studentId = paddingImage(studentId);
    examId = paddingImage(examId);

    content11 = paddingImage(content11);
    content12 = paddingImage(content12);
    content13 = paddingImage(content13);
    content14 = paddingImage(content14);
    content21 = paddingImage(content21);
    content22 = paddingImage(content22);
    content23 = paddingImage(content23);
    content24 = paddingImage(content24);

    // Batching 
    metadataImages.clear();
    metadataImages.push_back(studentId);
    metadataImages.push_back(examId);

    contentImages.clear();
    contentImages.push_back(content11);
    contentImages.push_back(content12);
    contentImages.push_back(content13);
    contentImages.push_back(content14);
    contentImages.push_back(content21);
    contentImages.push_back(content22);
    contentImages.push_back(content23);
    contentImages.push_back(content24);

    return true;
  } catch (const std::exception &e) {
    Logger::error("IMAGE PROCESSOR", "Image splitting failed: " + std::string(e.what()));
    return false;
  }
}
