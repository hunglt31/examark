#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <pwd.h>
#include <algorithm>
#include "utils/httplib.h"
#include "controllers/Controller.h"
#include "utils/MinIOHTTPClient.h"
#include "services/Service.h"

const std::string USER_NAME = []() -> std::string {
    struct passwd *pw = getpwuid(getuid());
    return pw ? std::string(pw->pw_name) : "unknown";
}();

// Add MinIO configuration (same as Service.cpp)
const std::string MINIO_ENDPOINT = "127.0.0.1:9000";
const std::string MINIO_ACCESS_KEY = "minioadmin";
const std::string MINIO_SECRET_KEY = "minioadmin123";
const std::string MINIO_BUCKET = "examark-images";

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) worker.join();
    }
};

// Global thread pool
ThreadPool grading_thread_pool(std::thread::hardware_concurrency());

// Tracking grading progress
struct JobProgress {
    bool processCompleted;
    std::string outputDir;
    std::string pdfFilename;
    std::string csvContent;
    std::string currentStage;
    std::string currentStep;
    int currentPage;
    int totalPages;
    double progressPercent;
    bool hasError;
    std::string errorMessage;
    
    JobProgress() : processCompleted(false), currentPage(0), totalPages(0), 
                    progressPercent(0.0), hasError(false) {}
};

std::unordered_map<std::string, JobProgress> jobProgressMap;
std::mutex progressMutex;

void updateJobProgress(const std::string& jobId, const std::string& stage, 
                       const std::string& step, int currentPage, int totalPages, 
                       double progressPercent, bool isError, const std::string& errorMsg) {
  std::lock_guard<std::mutex> lock(progressMutex);
  
  auto& progress = jobProgressMap[jobId];
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

std::string generateRandomId(int length = 8) {
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string result;
    std::srand(std::time(nullptr));
    for (int i = 0; i < length; ++i) {
        result += chars[std::rand() % chars.size()];
    }
    return result;
}

void registerGradingRoute(httplib::Server& server, TritonClient* tritonClient) {

    // Main grading endpoint
    server.Post("/grade", [tritonClient](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        try {
            auto pdf_file = req.get_file_value("pdfFile");
            auto csv_file = req.get_file_value("csvFile");

            if (pdf_file.filename.empty() || csv_file.filename.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"Missing required files\"}", "application/json");
                return;
            }

            std::string jobId = generateRandomId();
            std::string outputDir = "/tmp/examark_" + USER_NAME + "_" + jobId;
            
            // Initialize job progress
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                jobProgressMap[jobId] = JobProgress{};
                auto& progress = jobProgressMap[jobId];
                progress.processCompleted = false;
                progress.outputDir = outputDir;
                progress.pdfFilename = pdf_file.filename;
                progress.csvContent = csv_file.content;
                progress.currentStage = "initializing";
                progress.currentStep = "Starting grading process...";
                progress.progressPercent = 0.0;
            }

            // Queue the grading task
            grading_thread_pool.enqueue([=]() {
                bool success = grading(pdf_file.filename, pdf_file.content, csv_file.content, outputDir, tritonClient, jobId);
                if (!success) {
                    updateJobProgress(jobId, "error", "Grading process failed", 0, 0, 0.0, true, "Failed to complete grading");
                }
            });

            nlohmann::json response;
            response["jobId"] = jobId;
            response["message"] = "Grading started successfully";
            res.set_content(response.dump(), "application/json");
            
        } catch (const std::exception& e) {
            res.status = 500;
            nlohmann::json error_response;
            error_response["error"] = "Server error: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
        }
    });

    // Status endpoint
    server.Get("/status/:jobId", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        std::string jobId = req.path_params.at("jobId");
        
        std::lock_guard<std::mutex> lock(progressMutex);
        auto it = jobProgressMap.find(jobId);
        
        nlohmann::json response;
        if (it != jobProgressMap.end()) {
            const auto& progress = it->second;
            
            if (progress.hasError) {
                response["status"] = "error";
                response["error"] = progress.errorMessage;
            } else if (progress.processCompleted) {
                response["status"] = "completed";
            } else {
                response["status"] = "processing";
            }
            
            response["currentStage"] = progress.currentStage;
            response["currentStep"] = progress.currentStep;
            response["currentPage"] = progress.currentPage;
            response["totalPages"] = progress.totalPages;
            response["progress"] = progress.progressPercent;
        } else {
            response["status"] = "not_found";
            response["message"] = "Job not found";
        }
        
        res.set_content(response.dump(), "application/json");
    });

    // Get images list endpoint - returns MinIO URLs
    server.Get("/results/:jobId/images", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        std::string jobId = req.path_params.at("jobId");
        
        // Check job status
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            auto it = jobProgressMap.find(jobId);
            if (it == jobProgressMap.end() || !it->second.processCompleted) {
                res.status = 404;
                res.set_content("{\"error\":\"Results not ready or job not found\"}", "application/json");
                return;
            }
        }
        
        try {
            // Initialize MinIO client
            MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
            
            // Get list of all files from MinIO for this job
            std::vector<std::string> allFiles = minioClient.listFiles(jobId + "/");
            
            // Filter and process image files
            nlohmann::json response;
            nlohmann::json imageUrls = nlohmann::json::array();
            
            for (const std::string& filename : allFiles) {
                // Only include image files
                if (filename.find(".jpg") != std::string::npos || 
                    filename.find(".png") != std::string::npos ||
                    filename.find(".jpeg") != std::string::npos) {
                    
                    nlohmann::json imageInfo;
                    imageInfo["name"] = filename;

                    std::string fullObjectName = jobId + "/" + filename;
                    imageInfo["url"] = minioClient.getFileUrl(fullObjectName);
                    imageUrls.push_back(imageInfo);
                }
            }
            
            // Sort images by page number (page_1.jpg, page_2.jpg, etc.)
            std::sort(imageUrls.begin(), imageUrls.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
                std::string nameA = a["name"];
                std::string nameB = b["name"];
                
                // Extract page numbers for proper sorting
                auto extractPageNum = [](const std::string& name) -> int {
                    size_t start = name.find("page_");
                    if (start != std::string::npos) {
                        start += 5; // length of "page_"
                        size_t end = name.find(".", start);
                        if (end != std::string::npos) {
                            try {
                                return std::stoi(name.substr(start, end - start));
                            } catch (...) {
                                return 0;
                            }
                        }
                    }
                    return 0;
                };
                
                return extractPageNum(nameA) < extractPageNum(nameB);
            });
            
            response["images"] = imageUrls;
            
            res.set_header("Content-Type", "application/json");
            res.set_content(response.dump(), "application/json");
            
        } catch (const std::exception& e) {
            res.status = 500;
            nlohmann::json errorResponse;
            errorResponse["error"] = "Failed to fetch images: " + std::string(e.what());
            res.set_content(errorResponse.dump(), "application/json");
        }
    });

    // Get CSV endpoint - fetches from MinIO or local
    server.Get("/results/:jobId/csv", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        std::string jobId = req.path_params.at("jobId");
        
        // Check job status
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            auto it = jobProgressMap.find(jobId);
            if (it == jobProgressMap.end() || !it->second.processCompleted) {
                res.status = 404;
                res.set_content("Results not ready or job not found", "text/plain");
                return;
            }
        }
        
        try {
            // First try to get CSV from local file (for backward compatibility and regrade)
            std::string outputDir;
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                auto it = jobProgressMap.find(jobId);
                if (it != jobProgressMap.end()) {
                    outputDir = it->second.outputDir;
                }
            }
            
            // Try to find local CSV file first
            std::string csvContent;
            bool foundLocal = false;
            
            if (!outputDir.empty() && std::filesystem::exists(outputDir)) {
                for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
                    if (entry.path().extension() == ".csv") {
                        std::ifstream csvFile(entry.path());
                        if (csvFile.is_open()) {
                            std::stringstream buffer;
                            buffer << csvFile.rdbuf();
                            csvContent = buffer.str();
                            foundLocal = true;
                            csvFile.close();
                            break;
                        }
                    }
                }
            }
            
            // If no local file found, try to download from MinIO
            if (!foundLocal) {
                MinIOHTTPClient minioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET);
                
                // List objects to find CSV file
                std::vector<std::string> objects = minioClient.listFiles(jobId + "/");
                std::string csvObjectName;
                
                for (const std::string& objectName : objects) {
                    if (objectName.find(".csv") != std::string::npos) {
                        csvObjectName = objectName;
                        break;
                    }
                }
                
                if (csvObjectName.empty()) {
                    res.status = 404;
                    res.set_content("CSV file not found", "text/plain");
                    return;
                }
                
                // Download CSV content from MinIO
                csvContent = minioClient.downloadCSV(csvObjectName);
                if (csvContent.empty()) {
                    res.status = 500;
                    res.set_content("Failed to download CSV from storage", "text/plain");
                    return;
                }
            }
            
            // Return CSV content
            res.set_header("Content-Type", "text/csv");
            res.set_header("Content-Disposition", "attachment; filename=\"results.csv\"");
            res.set_content(csvContent, "text/csv");
            
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("Failed to fetch CSV: " + std::string(e.what()), "text/plain");
        }
    });

    // Regrade endpoint
    server.Post("/regrade", [tritonClient](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        try {
            nlohmann::json requestData = nlohmann::json::parse(req.body);
            
            if (!requestData.contains("jobId") || !requestData.contains("csvData") || !requestData.contains("answerKey")) {
                res.status = 400;
                res.set_content("{\"error\":\"Missing required fields: jobId, csvData, answerKey\"}", "application/json");
                return;
            }
            
            std::string originalJobId = requestData["jobId"];
            std::string csvData = requestData["csvData"];
            std::string answerKeyData = requestData["answerKey"];
            
            std::string regradeJobId = generateRandomId();
            std::string outputDir;
            
            // Get output directory from original job
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                auto it = jobProgressMap.find(originalJobId);
                if (it == jobProgressMap.end()) {
                    res.status = 404;
                    res.set_content("{\"error\":\"Original job not found\"}", "application/json");
                    return;
                }
                outputDir = it->second.outputDir;
            }
            
            if (!std::filesystem::exists(outputDir)) {
                res.status = 404;
                res.set_content("{\"error\":\"Original job results not found\"}", "application/json");
                return;
            }
            
            // Initialize regrade job progress
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                jobProgressMap[regradeJobId] = JobProgress{};
                auto& progress = jobProgressMap[regradeJobId];
                progress.processCompleted = false;
                progress.outputDir = outputDir;
                progress.pdfFilename = "";
                progress.csvContent = csvData;
                progress.currentStage = "regrading";
                progress.currentStep = "Starting regrade process...";
                progress.progressPercent = 0.0;
            }
            
            // Queue the regrade task
            grading_thread_pool.enqueue([=]() {
                updateJobProgress(regradeJobId, "regrading", "Reprocessing grades with updated data...", 0, 0, 50.0, false, "");
                
                bool success = regradeExam(outputDir, csvData, answerKeyData, regradeJobId, originalJobId);
                
                if (success) {
                    updateJobProgress(regradeJobId, "completed", "Regrade completed successfully", 0, 0, 100.0, false, "");
                } else {
                    updateJobProgress(regradeJobId, "error", "Regrade process failed", 0, 0, 0.0, true, "Failed to complete regrade");
                }
            });
            
            nlohmann::json response;
            response["regrade_job_id"] = regradeJobId;
            response["message"] = "Regrade started successfully";
            res.set_content(response.dump(), "application/json");
            
        } catch (const nlohmann::json::exception& e) {
            res.status = 400;
            nlohmann::json error_response;
            error_response["error"] = "Invalid JSON: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            nlohmann::json error_response;
            error_response["error"] = "Server error: " + std::string(e.what());
            res.set_content(error_response.dump(), "application/json");
        }
    });

    // Cancel job endpoint
    server.Post("/cancel/:jobId", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        std::string jobId = req.path_params.at("jobId");
        
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            auto it = jobProgressMap.find(jobId);
            if (it != jobProgressMap.end()) {
                auto& progress = it->second;
                progress.hasError = true;
                progress.errorMessage = "Job cancelled by user";
                progress.currentStage = "cancelled";
                progress.currentStep = "Process terminated";
                progress.processCompleted = true;
            }
        }
        
        nlohmann::json response;
        response["message"] = "Job cancellation requested";
        res.set_content(response.dump(), "application/json");
    });

    // Handle CORS preflight requests
    server.Options(".*", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });
}