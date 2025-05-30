#ifndef DL_MODEL_H
#define DL_MODEL_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <random>
#include <memory>
#include <NvInferPlugin.h>

#include "utils/Detection.h"
#include "utils/Logger.h"
#include "models/ModelConfig.h"

class ModelBuilder {
public:
  /**
   * @brief Constructs the engine with specified ONNX model path, engine path, and input dimensions.
   * 
   * @param modelPath Path to the ONNX model file.
   * @param enginePath Path to the serialized engine file.
   * @param inputWidth Width of the input tensor.
   * @param inputHeight Height of the input tensor.
   * @param batchSize Number of inputs in a batch.
   * @param maxOutputBoxes The maximum number of boxes to keep after applying NMS.
   */
  explicit ModelBuilder(const std::string& modelPath, const std::string& enginePath, 
                   int inputWidth, int inputHeight, int batchSize,
                   int topK, int maxOutputBoxes); 
  
  /**
   * @brief Destructor. Releases all allocated resources.
   */
  ~ModelBuilder();

  /**
   * @brief Loads the engine from a serialized engine file if it exists;
   *        otherwise, builds the engine from scratch.
   * 
   * @return true if the engine is successfully loaded or built; false otherwise.
   */
  bool loadModelBuilder();  

  /**
   * @brief Performs inference on a batch of images.
   * 
   * @param originImgs A vector of input images in cv::Mat format.
   * @return A vector of vectors containing Detection objects for each image.
   */
  std::vector<std::vector<Detection>> inference(const std::vector<cv::Mat>& originImgs);

private:
  bool buildEngine();
  bool loadEngine();
  void allocateBuffers();
  void cleanup();

  std::string modelPath_;
  std::string enginePath_;
  int inputWidth_;
  int inputHeight_;
  int batchSize_;
  int topK_;
  int maxOutputBoxes_;

  nvinfer1::ILogger* logger_;
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;

  std::vector<void*> hostBuffers_;
  std::vector<void*> deviceBuffers_;
  std::vector<size_t> bufferSizes_;
  std::vector<cudaStream_t> streams_;
};
#endif // DL_MODEL_H
