#pragma once
#include <string>
#include <vector>
#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <iomanip>
#include <sstream>

struct WriteCallbackData {
    std::string response;
};

class MinIOHTTPClient {
private:
    std::string endpoint;
    std::string accessKey;
    std::string secretKey;
    std::string bucket;
    std::string region = "us-east-1"; // Default region for MinIO
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, WriteCallbackData* data);
    static size_t ReadCallback(void* ptr, size_t size, size_t nmemb, void* userdata);
    
    // AWS4 signature helpers
    std::string getCurrentTimestamp();
    std::string getCurrentDate();
    std::string sha256(const std::string& data);
    std::string hmacSha256(const std::string& key, const std::string& data);
    std::string createSignature(const std::string& method, const std::string& uri, 
                              const std::string& queryString, const std::string& headers,
                              const std::string& payload, const std::string& timestamp);
    std::string createAuthorizationHeader(const std::string& method, const std::string& uri,
                                        const std::string& queryString, const std::string& headers,
                                        const std::string& payload, const std::string& timestamp);
    
public:
    MinIOHTTPClient(const std::string& endpoint, const std::string& accessKey, 
                   const std::string& secretKey, const std::string& bucket);
    ~MinIOHTTPClient();
    
    bool createBucketIfNotExists();
    
    // Upload methods
    bool uploadImage(const std::string& objectName, const cv::Mat& image);
    bool uploadCSV(const std::string& objectName, const std::string& csvContent);
    bool uploadJSON(const std::string& objectName, const std::string& jsonContent);
    bool uploadFile(const std::string& objectName, const std::string& content, const std::string& contentType);
    
    // Download methods
    std::string downloadFile(const std::string& objectName);
    std::string downloadCSV(const std::string& objectName);
    std::string downloadJSON(const std::string& objectName);
    
    // URL and listing methods
    std::string getFileUrl(const std::string& objectName);
    std::string getImageUrl(const std::string& objectName);
    std::vector<std::string> listFiles(const std::string& prefix = "");
    std::vector<std::string> listImages(const std::string& prefix);
    
    // Management methods
    bool deleteObject(const std::string& objectName);
    bool objectExists(const std::string& objectName);
    
    // Utility methods
    std::vector<std::string> listFilesByExtension(const std::string& prefix, const std::string& extension);
    bool testConnection();
};