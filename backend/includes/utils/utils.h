#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

struct JobProgress {
  bool processCompleted;
  std::string pdfFilename;
  std::string currentStage;
  std::string currentStep;
  int currentPage;
  int totalPages;
  double progressPercent;
  bool hasError;
  std::string errorMessage;
  std::string qrInfo;

  JobProgress() : processCompleted(false), currentPage(0), totalPages(0), progressPercent(0.0), hasError(false) {}
};

extern std::mutex progressMutex;
extern std::unordered_map<std::string, JobProgress> jobProgressMap;

namespace utils {

/**
 * @brief Generate an UUID v4 string.
 * @return An UUID v4 string.
 */
std::string generateUUIDv4();

/**
 * @brief Update the progress of a grading job.
 * @param jobId ID of the job to update.
 * @param stage Current stage of the job.
 * @param step Current step of the job.
 * @param currentPage Current page being processed.
 * @param totalPages Total number of pages in the job.
 * @param progressPercent Percentage of the job completed.
 * @param isError Whether the job encountered an error.
 * @param errorMsg Error message, if any.
 */
void updateJobProgress(const std::string &jobId, const std::string &stage, const std::string &step, int currentPage,
                       int totalPages, double progressPercent, bool isError, const std::string &errorMsg);

/** * @brief Update the QR information for a job.
 * @param jobId ID of the job to update.
 * @param qrInfo QR information to store.
 */
void updateJobQrInfo(const std::string &jobId, const std::string &qrInfo);

/**
 * @brief Get the current timestamp in ISO 8601 format.
 * @return Current timestamp as a string.
 */
std::string getCurrentTimestamp();
} // namespace utils

#endif // UTILS_H