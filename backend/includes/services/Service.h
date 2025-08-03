#ifndef EXAMARK_SERVICES_H
#define EXAMARK_SERVICES_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <pwd.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "models/TritonClient.h"
#include "utils/minio_config.h"
#include "utils/utils.h"

namespace examark {
namespace services {

bool extract_all_exams_answers(const std::string &pdfFileName, const std::string &pdfData, TritonClient *tritonClient,
                               const std::string &jobId);

} // namespace services
} // namespace examark

#endif // EXAMARK_SERVICES_H