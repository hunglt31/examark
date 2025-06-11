#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <string>
#include <vector>
#include "utils/Logger.h"

const int IMAGE_WIDTH = 2480;
const int IMAGE_HEIGHT = 3508;

using ProgressCallback = std::function<void(int currentPage, int totalPages, double progressPercent)>;

class ImageProcessor {
private: 
  /** 
   * @brief Resizes the image maintaining its aspect ratio and pads it to 640x640 pixels.
   * This function scales the input image so that its largest dimension becomes 640 pixels.
   * It then adds white padding (using cv::BORDER_CONSTANT) to achieve a final size of 640x640.    *
   *
   * @param image The image to be padded.
   * @param paddingSize The target size for the padded image (default is 640x640).
   * @return cv::Mat The padded image.
   */             
  cv::Mat paddingImage(cv::Mat &image, cv::Size paddingSize = cv::Size(640, 640));

  /**
   * @brief Aligns an input image to a reference grayscale image.
   *
   * This function converts the input color image to grayscale, detects SIFT keypoints
   * and computes descriptors, matches them using FLANN, and estimates a homography matrix.
   * It then warps the scan image to the specified target size.
   *
   * @param imgScan The image to be aligned.
   * @param imgRefGray The reference image in grayscale.
   * @param imgSize The target size for the aligned image.
   * @return cv::Mat The aligned image.
   */
  cv::Mat alignImage(const cv::Mat &imgScan, cv::Size imgSize = cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

public:
  /**
   * @brief Callback type for progress updates during image processing.
   *
   * This callback is invoked to report the progress of image extraction.
   * It provides the current page number, total pages, and the percentage of completion.
   */
  using ProgressCallback = std::function<void(int currentPage, int totalPages, double percent)>;
  
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
   * @return true if the conversion is successful, false otherwise.
   */
  bool getRequestImagesWithProgress(const char* pdfData, int dataSize, std::vector<cv::Mat> &images, 
                                    ProgressCallback progressCallback, double dpi = 300.0);

  /**
   * @brief Reads PDF data and extracts images from it.
   *
   * This function reads PDF raw data and extracts images from each page. 
   * The extracted images are aligned and saved in the provided vector.
   *
   * @param pdfData The data of the PDF file.
   * @param images Vector to store the extracted images.
   * @param dpi The DPI for the conversion (default is 300).
   * @return true if the extraction is successful, false otherwise.
   */
  bool getRequestImages(const char* pdfData, int dataSize, std::vector<cv::Mat> &images, double dpi=300);
  
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
  bool splitImage(const cv::Mat &imgage,
                  std::vector<cv::Mat>& metadataImages, 
                  std::vector<cv::Mat>& contentImages);
};
#endif // IMAGE_PROCESSOR_H
