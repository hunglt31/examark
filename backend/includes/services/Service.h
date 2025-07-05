#ifndef EXAMARK_SERVICES_H
#define EXAMARK_SERVICES_H

#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <pwd.h>
#include <vector>

#include "models/TritonClient.h"

namespace examark {
namespace services {

bool grade(
  const std::string &pdfFileName, 
  const std::string &pdfData,
  const std::string &answerKeyCSV,
  const std::string &outputDir,
  TritonClient *tritonClient,
  const std::string &jobId
);

bool regrade(
  const std::string& outputDir, 
  const std::string& csvData, 
  const std::string& answerKeyData, 
  const std::string& regradeJobId, 
  const std::string& originalJobId
);

} // namespace services
} // namespace examark

#endif // EXAMARK_SERVICES_H