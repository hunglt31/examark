#ifndef TRITON_CLIENT_H
#define TRITON_CLIENT_H

#pragma once

#include <grpc_client.h>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include "models/ModelConfig.h"
#include "utils/Detection.h"
#include "utils/Logger.h"

class TritonClient {
public:
  TritonClient(const std::string &url);
  ~TritonClient();

  bool connect();

  std::vector<std::vector<Detection>> inference(const std::vector<cv::Mat> &images, const std::string &modelName);

  void drawAndSaveResults(const std::vector<cv::Mat> &images, const std::vector<std::vector<Detection>> &detections,
                          const std::string &outputDir);

private:
  std::string m_url;
  std::unique_ptr<triton::client::InferenceServerGrpcClient> m_client;

  std::vector<std::vector<Detection>> postprocess(const std::vector<cv::Mat> &images,
                                                  const triton::client::InferResult *result, int numImages,
                                                  const std::string &modelName);

  std::string getCurrentTimestamp();
};

#endif // TRITON_CLIENT_H