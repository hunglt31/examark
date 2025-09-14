#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "models/TritonClient.h"
#include "services/Service.h"
#include "utils/ExamExtractor.h"
#include "utils/Logger.h"
#include "utils/httplib.h"
#include "utils/minio_config.h"
#include "utils/utils.h"
#include <pwd.h>

namespace controller {
/**
 * @brief Register the SSE endpoint with the server.
 * @param server The HTTP server instance.
 */
void registerSSEEndpoint(httplib::Server &server);

/**
 * @brief Register the grading route with the server.
 * @param server The HTTP server instance.
 * @param tritonClient The Triton client instance for model inference.
 */
void registerExtractRoute(httplib::Server &server, TritonClient *tritonClient);

void processFileAsync(const std::string &filename, const std::string &fileContent, TritonClient *tritonClient,
                      const std::string &jobId, const std::string &sessionId, const std::string &sessionFolder);

} // namespace controller

#endif // CONTROLLER_H