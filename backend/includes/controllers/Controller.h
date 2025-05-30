#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <pwd.h>
#include "utils/httplib.h"
#include "models/TritonClient.h"
#include "services/Service.h"
#include "utils/Logger.h"
#include "utils/ExamGrader.h"

/**
 * @brief Register the grading route with the server.
 * 
 * This function sets up the main grading endpoint, status check, and CSV results retrieval.
 * It handles incoming requests, processes the PDF files, and manages the grading results.
 * 
 * @param server The HTTP server instance.
 * @param tritonClient The Triton client instance for model inference.
 */
void registerGradingRoute(httplib::Server& server, TritonClient* tritonClient);

#endif // CONTROLLER_H