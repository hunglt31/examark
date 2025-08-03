#include <NvInferPlugin.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "models/ModelBuilder.h"

class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      Logger::warning("TENSORRT", msg);
    if (severity == Severity::kERROR)
      Logger::error("TENSORRT", msg);
  }
} trtLogger;

// Class ModelBuilder
ModelBuilder::ModelBuilder(const std::string &modelPath, const std::string &enginePath, int inputWidth, int inputHeight,
                           int batchSize, int topK, int maxOutputBoxes)
    : modelPath_(modelPath), enginePath_(enginePath), inputWidth_(inputWidth), inputHeight_(inputHeight),
      batchSize_(batchSize), topK_(topK), maxOutputBoxes_(maxOutputBoxes), logger_(&trtLogger), runtime_(nullptr),
      engine_(nullptr), context_(nullptr) {
  streams_.resize(batchSize_);
  for (int i = 0; i < batchSize_; i++) {
    cudaStreamCreate(&streams_[i]);
  }
}

ModelBuilder::~ModelBuilder() { cleanup(); }

bool ModelBuilder::loadModelBuilder() {
  if (access(enginePath_.c_str(), F_OK) == 0) {
    Logger::info("MODEL BUILDER", "Engine file found at " + enginePath_ + " . Loading engine...");
    return loadEngine();
  } else {
    Logger::info("MODEL BUILDER", "Engine file not found at " + enginePath_ + " . Building engine...");
    return buildEngine();
  }
}

/**
 * @brief Creates an EfficientNMS plugin for use in TensorRT.
 *
 * This function initializes and returns an EfficientNMS plugin with the
 * specified configuration (top-K, keepTop-K, score threshold, IOU threshold,
 * etc.). The plugin is used to apply non-maximum suppression (NMS) to bounding
 * box predictions in object detection models.
 *
 * @param topK The number of top-scoring boxes to keep before applying NMS.
 * @param maxOutputBoxes The maximum number of boxes to keep after applying NMS.
 * @param scoreThreshold The minimum score threshold for a box to be kept.
 * @param iouThreshold The Intersection-over-Union (IOU) threshold for
 * suppressing overlapping boxes.
 * @return IPluginV2* A pointer to the created EfficientNMS plugin, or nullptr
 * if an error occurs.
 */
nvinfer1::IPluginV2 *createEfficientNMS(int topK, int maxOutputBoxes, float scoreThreshold, float iouThreshold) {
  nvinfer1::IPluginCreator *creator = getPluginRegistry()->getPluginCreator("EfficientNMS_TRT", "1");
  if (!creator) {
    Logger::error("MODEL BUILDER", "Failed to find EfficientNMS_TRT plugin creator.");
    return nullptr;
  }

  std::vector<nvinfer1::PluginField> pluginFields;
  nvinfer1::PluginFieldCollection pluginFieldCollection;

  int backgroundClass = -1;    // No background class
  int boxCoding = 1;           // Box encoding: 0 = corner-based, 1 = center-based
  int32_t scoreActivation = 0; // No need activation

  pluginFields.emplace_back("background_class", &backgroundClass, nvinfer1::PluginFieldType::kINT32, 1);
  pluginFields.emplace_back("box_coding", &boxCoding, nvinfer1::PluginFieldType::kINT32, 1);
  pluginFields.emplace_back("top_k", &topK, nvinfer1::PluginFieldType::kINT32, 1);
  pluginFields.emplace_back("max_output_boxes", &maxOutputBoxes, nvinfer1::PluginFieldType::kINT32, 1);
  pluginFields.emplace_back("score_activation", &scoreActivation, nvinfer1::PluginFieldType::kINT32, 1);
  pluginFields.emplace_back("score_threshold", &scoreThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);
  pluginFields.emplace_back("iou_threshold", &iouThreshold, nvinfer1::PluginFieldType::kFLOAT32, 1);

  pluginFieldCollection.nbFields = pluginFields.size();
  pluginFieldCollection.fields = pluginFields.data();

  return creator->createPlugin("EfficientNMS_TRT", &pluginFieldCollection);
}

bool ModelBuilder::buildEngine() {
  initLibNvInferPlugins(&trtLogger, "");
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(*logger_);
  if (!builder) {
    Logger::error("MODEL BUILDER", "Failed to create TensorRT builder.");
  }

  nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
    config->setFlag(nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
  }

  Logger::info("MODEL BUILDER", "Building engine from ONNX model: " + modelPath_);
  auto network =
      builder->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto parser = nvonnxparser::createParser(*network, *logger_);
  if (!parser->parseFromFile(modelPath_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    Logger::error("MODEL BUILDER", "Failed to parse ONNX model: " + modelPath_);
    return false;
  }

  auto bboxSlice = network->getOutput(0);
  auto scoreSlice = network->getOutput(1);
  nvinfer1::ITensor *inputs[] = {bboxSlice, scoreSlice};
  nvinfer1::IPluginV2 *dualScoreNMS = createEfficientNMS(topK_, maxOutputBoxes_, SCORE_THRESHOLD, IOU_THRESHOLD);

  auto nmsLayer = network->addPluginV2(inputs, 2, *dualScoreNMS);
  nmsLayer->setName("NMSLayer");
  nmsLayer->getOutput(1)->setName("bboxes");
  nmsLayer->getOutput(2)->setName("scores");
  nmsLayer->getOutput(3)->setName("classes");

  // Set network output
  int numOutputs = network->getNbOutputs();
  for (int i = 0; i < numOutputs; ++i) {
    nvinfer1::ITensor *output = network->getOutput(0);
    network->unmarkOutput(*output);
  }
  network->markOutput(*nmsLayer->getOutput(1));
  network->markOutput(*nmsLayer->getOutput(2));
  network->markOutput(*nmsLayer->getOutput(3));

  Logger::info("MODEL BUILDER", "Building serialized engine...");
  auto serializedEngine = builder->buildSerializedNetwork(*network, *config);
  if (!serializedEngine) {
    Logger::error("MODEL BUILDER", "Failed to build serialized engine from ONNX model: " + modelPath_);
    return false;
  }

  std::ofstream outFile(enginePath_, std::ios::binary);
  outFile.write(static_cast<const char *>(serializedEngine->data()), serializedEngine->size());
  outFile.close();
  Logger::info("MODEL BUILDER", "Engine built and saved to: " + enginePath_);

  return loadEngine();
}

bool ModelBuilder::loadEngine() {
  if (!initLibNvInferPlugins(nullptr, "")) {
    Logger::error("MODEL BUILDER", "Failed to initialize TensorRT plugins.");
    return false;
  }
  std::ifstream file(enginePath_, std::ios::binary);
  if (!file.good()) {
    Logger::error("MODEL BUILDER", "Engine file not found at: " + enginePath_);
    return false;
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> engineData(size);
  file.read(engineData.data(), size);

  runtime_ = nvinfer1::createInferRuntime(*logger_);
  engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
  if (!engine_) {
    Logger::error("MODEL BUILDER", "Failed to deserialize engine from file: " + enginePath_);
    return false;
  }
  context_ = engine_->createExecutionContext();
  if (!context_) {
    Logger::error("MODEL BUILDER", "Failed to create execution context from engine.");
    return false;
  }

  allocateBuffers();
  return true;
}

void ModelBuilder::allocateBuffers() {
  int nIO = engine_->getNbIOTensors();
  hostBuffers_.resize(nIO, nullptr);
  deviceBuffers_.resize(nIO, nullptr);
  bufferSizes_.resize(nIO);

  size_t size = 3 * inputHeight_ * inputWidth_ * sizeof(u_int8_t);
  bufferSizes_[0] = size;
  hostBuffers_[0] = new char[batchSize_ * size];
  cudaMalloc(&deviceBuffers_[0], batchSize_ * bufferSizes_[0]);

  for (int i = 1; i < nIO; ++i) {
    auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));
    size = sizeof(float);
    for (int j = 1; j < dims.nbDims; ++j) {
      size *= dims.d[j];
    }
    bufferSizes_[i] = batchSize_ * size;
    hostBuffers_[i] = new char[batchSize_ * size];
    cudaMalloc(&deviceBuffers_[i], bufferSizes_[i]);
  }
}

std::vector<std::vector<Detection>> ModelBuilder::inference(const std::vector<cv::Mat> &images) {
  if (!context_) {
    Logger::error("MODEL BUILDER", "Inference context is not initialized.");
    return std::vector<std::vector<Detection>>();
  }
  streams_.resize(batchSize_);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Copy input data from host to device
  for (int i = 0; i < batchSize_; i++) {
    cudaMemcpyAsync((uint8_t *)deviceBuffers_[0] + i * bufferSizes_[0], (uint8_t *)images[i].data, bufferSizes_[0],
                    cudaMemcpyHostToDevice, streams_[i]);
  }
  for (int i = 0; i < batchSize_; i++) {
    cudaStreamSynchronize(streams_[i]);
  }
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    context_->setTensorAddress(engine_->getIOTensorName(i), deviceBuffers_[i]);
  }

  // Execute the engine
  context_->enqueueV3(stream);
  cudaStreamSynchronize(stream);

  // Copy output data from device to host
  std::vector<float> bboxesData;
  std::vector<float> scoresData;
  std::vector<int> classesData;

  size_t bboxSize = bufferSizes_[1] / sizeof(float);
  bboxesData.resize(bboxSize);
  cudaMemcpyAsync(bboxesData.data(), deviceBuffers_[1], bufferSizes_[1], cudaMemcpyDeviceToHost, stream);

  size_t scoresSize = bufferSizes_[2] / sizeof(float);
  scoresData.resize(scoresSize);
  cudaMemcpyAsync(scoresData.data(), deviceBuffers_[2], bufferSizes_[2], cudaMemcpyDeviceToHost, stream);

  size_t classesSize = bufferSizes_[3] / sizeof(int);
  classesData.resize(classesSize);
  cudaMemcpyAsync(classesData.data(), deviceBuffers_[3], bufferSizes_[3], cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  // Process the output
  std::vector<std::vector<Detection>> results(batchSize_);
  for (int b = 0; b < batchSize_; b++) {
    std::vector<Detection> dets;

    cv::Mat gray;
    cv::cvtColor(images[b], gray, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < maxOutputBoxes_; i++) {
      if (scoresData[b * maxOutputBoxes_ + i] < SCORE_THRESHOLD) {
        continue;
      }
      int bboxIdx = b * maxOutputBoxes_ * 4 + i * 4;
      int scoreIdx = b * maxOutputBoxes_ + i;
      float x1 = bboxesData[bboxIdx + 0];
      float y1 = bboxesData[bboxIdx + 1];
      float x2 = bboxesData[bboxIdx + 2];
      float y2 = bboxesData[bboxIdx + 3];
      float score = scoresData[scoreIdx];
      int classId = classesData[scoreIdx];

      cv::Rect box(cvRound(x1), cvRound(y1), cvRound(x2 - x1), cvRound(y2 - y1));
      cv::Rect safeBox = box & cv::Rect(0, 0, gray.cols, gray.rows);
      float avgGray = 0.0f;
      cv::Scalar meanVal = cv::mean(gray(safeBox));
      avgGray = meanVal[0];

      dets.emplace_back(Detection(box, score, classId, avgGray));
    }
    results[b] = dets;
  }

  cudaStreamDestroy(stream);
  return results;
}

void ModelBuilder::cleanup() {
  for (auto buffer : deviceBuffers_) {
    cudaFree(buffer);
  }
  for (auto buffer : hostBuffers_) {
    delete[] static_cast<char *>(buffer);
  }
  for (auto &stream : streams_) {
    cudaStreamDestroy(stream);
  }
  if (context_)
    delete context_;
  if (engine_)
    delete engine_;
  if (runtime_)
    delete runtime_;
}
