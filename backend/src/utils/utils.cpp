#include <iomanip>
#include <random>
#include <sstream>
#include <string>

#include "utils/utils.h"

std::unordered_map<std::string, JobProgress> jobProgressMap;
std::mutex progressMutex;

namespace utils {
std::string generateUUIDv4() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

  uint32_t data[4];
  for (auto &d : data)
    d = dis(gen);

  data[1] = (data[1] & 0xFFFF0FFF) | 0x00004000;
  data[2] = (data[2] & 0x3FFFFFFF) | 0x80000000;

  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(8) << data[0] << "-" << std::setw(4) << ((data[1] >> 16) & 0xFFFF)
      << "-" << std::setw(4) << (data[1] & 0xFFFF) << "-" << std::setw(4) << (data[2] >> 16) << "-" << std::setw(4)
      << (data[2] & 0xFFFF) << std::setw(8) << data[3];
  return oss.str();
}

void updateJobProgress(const std::string &jobId, const std::string &stage, const std::string &step, int currentPage,
                       int totalPages, double progressPercent, bool isError, const std::string &errorMsg) {
  std::lock_guard<std::mutex> lock(progressMutex);

  auto &progress = jobProgressMap[jobId];
  progress.currentStage = stage;
  progress.currentStep = step;
  progress.currentPage = currentPage;
  progress.totalPages = totalPages;
  progress.progressPercent = progressPercent;
  progress.hasError = isError;
  progress.errorMessage = errorMsg;

  if (stage == "completed" || stage == "error") {
    progress.processCompleted = true;
  }
}

void updateJobQrInfo(const std::string &jobId, const std::string &qrInfo) {
  std::lock_guard<std::mutex> lock(progressMutex);

  auto it = jobProgressMap.find(jobId);
  if (it != jobProgressMap.end()) {
    it->second.qrInfo = qrInfo;
  }
}

std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

} // namespace utils