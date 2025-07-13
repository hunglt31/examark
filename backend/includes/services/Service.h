#ifndef EXAMARK_SERVICES_H
#define EXAMARK_SERVICES_H

#include "models/ModelBuilder.h"
// #include "models/TritonClient.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pwd.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace examark {
namespace services {

// Triton client grade
// bool grade1(const std::string &pdfFileName, const std::string &pdfData,
//             const std::string &answerKeyCSV, const std::string &outputDir,
//             TritonClient *tritonClient, const std::string &jobId);

// TensorRT engine grade
bool grade(const std::string &pdfFileName, const std::string &pdfData,
           const std::string &answerKeyCSV, const std::string &outputDir,
           ModelBuilder *metadataModel, ModelBuilder *contentModel,
           const std::string &jobId);

bool regrade(const std::string &outputDir, const std::string &csvData,
             const std::string &answerKeyData, const std::string &regradeJobId,
             const std::string &originalJobId);

// Grade with answer key in JSON format
bool gradeWithJson(const std::string &pdfFileName, const std::string &pdfData,
                   const std::string &answerKeyJson,
                   const std::string &outputDir, ModelBuilder *metadataModel,
                   ModelBuilder *contentModel, const std::string &jobId);

// Regrade with answer key in JSON format
bool regradeWithJson(const std::string &outputDir, const std::string &csvData,
                     const std::string &answerKeyJson,
                     const std::string &regradeJobId,
                     const std::string &originalJobId);

} // namespace services
} // namespace examark

#endif // EXAMARK_SERVICES_H