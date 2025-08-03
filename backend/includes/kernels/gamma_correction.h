#ifndef GAMMA_CORRECTION_CUDA_H
#define GAMMA_CORRECTION_CUDA_H

#include <opencv2/opencv.hpp>

// Default gamma value
extern const float DEFAULT_GAMMA;

// Host function to create gamma LUT
cv::Mat createGammaLUT_CUDA(float gamma = 2.2f);

// Function to apply gamma correction to an image using CUDA
void applyGammaCorrection_CUDA(const cv::Mat &input, cv::Mat &output, const cv::Mat &lut);

// Pre-computed gamma LUT with default gamma value
extern const cv::Mat GAMMA_LUT_CUDA;

#endif // GAMMA_CORRECTION_CUDA_H