#include "controllers/Controller.h"
#include "models/TritonClient.h"
#include "services/Service.h"
#include "utils/Logger.h"
#include "utils/httplib.h"
#include <csignal>
#include <iostream>
#include <memory>

/* Version 1: Triton Inference Server */
std::unique_ptr<TritonClient> g_tritonClient;

void signalHandler(int signum) {
  Logger::warning("SERVER", "Interrupt signal (" + std::to_string(signum) + ") received. Cleaning up...");
  g_tritonClient.reset();
  exit(signum);
}

bool initializeTritonClient() {
  Logger::info("SERVER", "Initializing Triton Client...");

  g_tritonClient = std::make_unique<TritonClient>("localhost:8001");
  if (!g_tritonClient->connect()) {
    Logger::error("SERVER", "Failed to connect to Triton Inference Server.");
    return false;
  }

  Logger::success("SERVER", "Triton Client initialized successfully.");
  return true;
}

/* Version 2: TensorRT Engine */
// std::unique_ptr<ModelBuilder> g_metadataModel;
// std::unique_ptr<ModelBuilder> g_contentModel;

// void signalHandler(int signum) {
//   Logger::warning("SERVER", "Interrupt signal (" + std::to_string(signum) + ") received. Cleaning up...");
//   g_metadataModel.reset();
//   g_contentModel.reset();
//   exit(signum);
// }

// bool initializeModels() {
//   Logger::info("SERVER", "Initializing TensorRT models...");

//   g_metadataModel = std::make_unique<ModelBuilder>(METADATA_MODEL_PATH, METADATA_ENGINE_PATH, INPUT_WIDTH,
//   INPUT_HEIGHT,
//                                                    METADATA_BATCH_SIZE, METADATA_TOP_K, METADATA_MAX_OUTPUT_BOXES);

//   g_contentModel = std::make_unique<ModelBuilder>(CONTENT_MODEL_PATH, CONTENT_ENGINE_PATH, INPUT_WIDTH, INPUT_HEIGHT,
//                                                   CONTENT_BATCH_SIZE, CONTENT_TOP_K, CONTENT_MAX_OUTPUT_BOXES);

//   if (!g_metadataModel->loadModelBuilder()) {
//     Logger::error("SERVER", "Failed to load metadata detection model.");
//     return false;
//   }
//   if (!g_contentModel->loadModelBuilder()) {
//     Logger::error("SERVER", "Failed to load content detection model.");
//     return false;
//   }
//   Logger::success("SERVER", "TensorRT models initialized successfully.");
//   return true;
// }

int main() {
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  if (!initializeTritonClient()) {
    Logger::error("SERVER", "Triton Client initialization failed. Exiting.");
    return 1;
  }

  // if (!initializeModels()) {
  //   Logger::error("SERVER", "TensorRT model initialization failed. Exiting.");
  //   return 1;
  // }

  Logger::info("SERVER", "Starting Examark Server...");
  httplib::Server examarkServer;
  registerGradingRouteTriton(examarkServer, g_tritonClient.get());
  // registerGradingRouteTRT(examarkServer, g_metadataModel.get(), g_contentModel.get());

  Logger::success("SERVER", "Server listening on port 8080...");
  examarkServer.listen("0.0.0.0", 8080);

  return 0;
}