#pragma once
#include <opencv2/opencv.hpp>

cv::Mat alignImages(const cv::Mat& scan_img, const cv::Mat& ref_img_gray);