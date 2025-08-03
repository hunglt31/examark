#include "controllers/Controller.h"
#include "models/TritonClient.h"
#include "services/Service.h"
#include "utils/Logger.h"
#include "utils/httplib.h"
#include <csignal>
#include <iostream>
#include <memory>

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

int main() {
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  if (!initializeTritonClient()) {
    Logger::error("SERVER", "Triton Client initialization failed. Exiting.");
    return 1;
  }

  Logger::info("SERVER", "Starting Examark Server...");
  httplib::Server examarkServer;
  controller::registerExtractRoute(examarkServer, g_tritonClient.get());

  Logger::success("SERVER", "Server listening on port 8080...");
  examarkServer.listen("0.0.0.0", 8080);

  return 0;
}