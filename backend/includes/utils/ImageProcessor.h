#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <functional>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-image.h>
#include <poppler/cpp/poppler-page-renderer.h>
#include <poppler/cpp/poppler-page.h>

#include "kernels/gamma_correction.h"
#include "utils/Logger.h"

const int IMAGE_WIDTH = 2480;
const int IMAGE_HEIGHT = 3508;

class ImageProcessor {
private:
  // Variables for image alignment
  cv::Ptr<cv::SIFT> sift;
  cv::Ptr<cv::FlannBasedMatcher> flann_matcher;

  /**
   * @brief Resizes the image maintaining its aspect ratio and pads it to 640x640 pixels.
   * This function scales the input image so that its largest dimension becomes 640 pixels.
   * It then adds white padding (using cv::BORDER_CONSTANT) to achieve a final size of 640x640.    *
   *
   * @param image The image to be padded.
   * @param paddingSize The target size for the padded image (default is 640x640).
   * @return cv::Mat The padded image.
   */
  cv::Mat paddingImage(const cv::Mat &image, cv::Size paddingSize = cv::Size(640, 640));

public:
  ImageProcessor();
  ~ImageProcessor() = default;
  /**
   * @brief Callback type for progress updates during image processing.
   *
   * This callback is invoked to report the progress of image extraction.
   * It provides the current page number, total pages, and the percentage of completion.
   */
  using ProgressCallback = std::function<void(int currentPage, int totalPages, double percent)>;

  /**
   * @brief Aligns an input image to a reference grayscale image.
   *
   * This function converts the input color image to grayscale, detects SIFT keypoints
   * and computes descriptors, matches them using FLANN, and estimates a homography matrix.
   * It then warps the scan image to the specified target size.
   *
   * @param imgScan The image to be aligned.
   * @param imgSize The target size for the aligned image.
   * @return cv::Mat The aligned image.
   */
  cv::Mat preprocessImage(const cv::Mat &imgScan, cv::Size imgSize = cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

  /**
   * @brief Converts PDF data to images with callback.
   *
   * This function reads PDF raw data and converts each page into an image.
   * The images are resized to a standard size and padded to ensure uniformity.
   *
   * @param pdfData The data of the PDF file.
   * @param dataSize The size of the PDF data.
   * @param images Vector to store the converted images.
   * @param progressCallback A callback function to report progress during the conversion.
   * @param dpi The DPI for the conversion (default is 300).
   * @param max_pages Maximum number of pages to convert (-1 for all pages).
   * @return true if the conversion is successful, false otherwise.
   */
  bool renderImages(const char *pdfData, int dataSize, std::vector<cv::Mat> &images, ProgressCallback progressCallback,
                    double dpi = 300.0, int max_pages = -1);

  /**
   * @brief Splits the scanned image into metadata and content regions.
   *
   * This function aligns the scanned image to the reference grayscale image,
   * extracts regions corresponding to metadata (studentId and examId) and content
   * (content11, content12, content13, content14, content21, content22, content23, content24),
   * applies padding to each region to produce 640x640 images, and batches the results
   * into two provided vectors.
   *
   * @param imgScan The scanned image.
   * @param imgRefGray The reference grayscale image.
   * @param metadataImages Vector to store the padded metadata images.
   * @param contentImages Vector to store the padded content images.
   * @return true if the splitting process completes successfully, false otherwise.
   */
  bool splitImage(const cv::Mat &imgage, std::vector<cv::Mat> &metadataImages, std::vector<cv::Mat> &contentImages);

  /** * @brief Extracts QR code information from an image.
   *
   * This function uses OpenCV's QRCodeDetector to detect and decode QR codes
   * in the provided image. If a QR code is found, it returns the decoded string.
   *
   * @param image The image to extract QR code information from.
   * @param qr_info Reference to a string where the extracted QR code information will be stored.
   * @return true if a QR code is successfully detected and decoded, false otherwise.
   */
  bool get_qr_code_info(const cv::Mat &image, std::string &qr_info);
};

#endif // IMAGE_PROCESSOR_H
