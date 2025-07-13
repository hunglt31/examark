#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "models/ModelBuilder.h"
// #include "models/TritonClient.h"
#include "services/Service.h"
#include "utils/ExamGrader.h"
#include "utils/Logger.h"
#include "utils/httplib.h"
#include <pwd.h>

// /**
//  * @brief Register the grading route with the server.
//  *
//  * This function sets up the main grading endpoint, status check, and CSV
//  * results retrieval. It handles incoming requests, processes the PDF files,
//  and
//  * manages the grading results.
//  *
//  * @param server The HTTP server instance.
//  * @param tritonClient The Triton client instance for model inference.
//  */
// void registerGradingRouteTriton(httplib::Server &server,
//                                 TritonClient *tritonClient);

/**
 * @brief Register the grading route with the server.
 *
 * This function sets up the main grading endpoint, status check, and CSV
 * results retrieval. It handles incoming requests, processes the PDF files, and
 * manages the grading results.
 *
 * @param server The HTTP server instance.
 * @param metadataModel The model builder for metadata detection.
 * @param contentModel The model builder for content detection.
 */
void registerGradingRouteTRT(httplib::Server &server,
                             ModelBuilder *metadataModel,
                             ModelBuilder *contentModel);

#endif // CONTROLLER_H