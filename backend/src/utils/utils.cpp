#include "utils/utils.h"

std::unordered_map<std::string, JobProgress> jobProgressMap;
std::mutex progressMutex;

namespace utils {
std::string generateRandomId(int length) {
  const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string result;
  std::srand(std::time(nullptr));
  for (int i = 0; i < length; ++i) {
    result += chars[std::rand() % chars.size()];
  }
  return result;
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

} // namespace utils