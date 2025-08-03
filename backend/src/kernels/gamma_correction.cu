#include "kernels/gamma_correction.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Define constants
const float DEFAULT_GAMMA = 2.2f;

// Error checking macro
#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// CUDA kernel for gamma LUT creation
__global__ void gammaLutKernel(uchar *lut, float gamma) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 256) {
    float normalized = idx / 255.0f;
    lut[idx] = (uchar)(powf(normalized, gamma) * 255.0f + 0.5f);
  }
}

// CUDA kernel for applying LUT to an image
__global__ void applyLutKernel(const uchar *input, uchar *output, const uchar *lut, int width, int height,
                               int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    for (int c = 0; c < channels; c++) {
      int idx = (y * width + x) * channels + c;
      output[idx] = lut[input[idx]];
    }
  }
}

// Host function to create gamma LUT
cv::Mat createGammaLUT_CUDA(float gamma) {
  cv::Mat lut(1, 256, CV_8UC1);

  // Allocate device memory
  uchar *d_lut;
  CHECK_CUDA(cudaMalloc(&d_lut, 256 * sizeof(uchar)));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (256 + threadsPerBlock - 1) / threadsPerBlock;
  gammaLutKernel<<<blocksPerGrid, threadsPerBlock>>>(d_lut, gamma);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(lut.ptr(), d_lut, 256 * sizeof(uchar), cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK_CUDA(cudaFree(d_lut));

  return lut;
}

// Create once and reuse
const cv::Mat GAMMA_LUT_CUDA = createGammaLUT_CUDA(DEFAULT_GAMMA);

// Function to apply gamma correction to an image using CUDA
void applyGammaCorrection_CUDA(const cv::Mat &input, cv::Mat &output, const cv::Mat &lut) {
  // Ensure output has the same size and type as input
  if (output.empty() || output.size() != input.size() || output.type() != input.type()) {
    output.create(input.size(), input.type());
  }

  // Get image dimensions
  int width = input.cols;
  int height = input.rows;
  int channels = input.channels();
  size_t imageSize = width * height * channels * sizeof(uchar);

  // Allocate device memory
  uchar *d_input, *d_output, *d_lut;
  CHECK_CUDA(cudaMalloc(&d_input, imageSize));
  CHECK_CUDA(cudaMalloc(&d_output, imageSize));
  CHECK_CUDA(cudaMalloc(&d_lut, 256 * sizeof(uchar)));

  // Copy input image and LUT to device
  CHECK_CUDA(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_lut, lut.data, 256 * sizeof(uchar), cudaMemcpyHostToDevice));

  // Define thread and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Launch kernel
  applyLutKernel<<<gridSize, blockSize>>>(d_input, d_output, d_lut, width, height, channels);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_lut));
}