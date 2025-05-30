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
#include "utils/Logger.h"
                                          
bool grading(const std::string &pdfFileName, 
             const std::string &pdfData,
             const std::string &answerKeyCSV,
             const std::string &outputDir,
             TritonClient *tritonClient);

#endif // EXAMARK_SERVICES_H