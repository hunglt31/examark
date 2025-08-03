#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

#include "utils/Logger.h"
#include "utils/MinIOHTTPClient.h"

struct UploadData {
  const char *data;
  size_t size;
  size_t offset;
};

MinIOHTTPClient::MinIOHTTPClient(const std::string &ep, const std::string &ak, const std::string &sk,
                                 const std::string &b)
    : endpoint(ep), accessKey(ak), secretKey(sk), bucket(b) {

  curl_global_init(CURL_GLOBAL_DEFAULT);

  bool bucketResult = createBucketIfNotExists();
  if (!bucketResult) {
    Logger::error("MINIO CLIENT", "Failed to create or verify bucket: " + bucket);
  }
}

MinIOHTTPClient::~MinIOHTTPClient() { curl_global_cleanup(); }

size_t MinIOHTTPClient::WriteCallback(void *contents, size_t size, size_t nmemb, WriteCallbackData *data) {
  size_t totalSize = size * nmemb;
  data->response.append((char *)contents, totalSize);
  return totalSize;
}

size_t MinIOHTTPClient::ReadCallback(void *ptr, size_t size, size_t nmemb, void *userdata) {
  UploadData *upload = (UploadData *)userdata;
  size_t maxSize = size * nmemb;

  if (upload->offset >= upload->size) {
    return 0;
  }

  size_t copySize = std::min(maxSize, upload->size - upload->offset);
  memcpy(ptr, upload->data + upload->offset, copySize);
  upload->offset += copySize;

  return copySize;
}

bool MinIOHTTPClient::createBucketIfNotExists() {
  // Set up MinIO alias (silently)
  std::string mcAliasCmd =
      "docker exec minio mc alias set local http://localhost:9000 " + accessKey + " " + secretKey + " >/dev/null 2>&1";
  int aliasResult = system(mcAliasCmd.c_str());

  if (aliasResult != 0) {
    Logger::error("MINIO CLIENT", "Failed to set MinIO alias");
    return false;
  }

  // Create bucket (silently)
  std::string mbCmd = "docker exec minio mc mb local/" + bucket + " >/dev/null 2>&1 || true";
  system(mbCmd.c_str());

  // Set public policy (silently)
  std::string policyCmd = "docker exec minio mc anonymous set public local/" + bucket + " >/dev/null 2>&1 || true";
  system(policyCmd.c_str());

  return true;
}

bool MinIOHTTPClient::uploadFile(const std::string &objectName, const std::string &content,
                                 const std::string &contentType) {
  // Create temporary file
  std::string timestamp = std::to_string(std::time(nullptr));
  std::string sanitizedName = objectName;

  // Replace problematic characters
  std::replace(sanitizedName.begin(), sanitizedName.end(), '/', '_');
  std::replace(sanitizedName.begin(), sanitizedName.end(), ' ', '_');
  std::replace(sanitizedName.begin(), sanitizedName.end(), '\\', '_');

  std::string tempFilename = "minio_upload_" + timestamp + "_" + sanitizedName;
  std::string tempPath = "/tmp/" + tempFilename;

  // Write content to temporary file
  std::ofstream tempFile(tempPath, std::ios::binary);
  if (!tempFile.is_open()) {
    Logger::error("MINIO CLIENT", "Failed to create temp file: " + tempPath);
    return false;
  }

  tempFile.write(content.c_str(), content.length());
  tempFile.close();

  // Copy file to MinIO container
  std::string copyCmd = "docker cp \"" + tempPath + "\" minio:/tmp/" + tempFilename + " >/dev/null 2>&1";
  int copyResult = system(copyCmd.c_str());

  if (copyResult != 0) {
    Logger::error("MINIO CLIENT", "Failed to copy file to container for object: " + objectName);
    std::remove(tempPath.c_str());
    return false;
  }

  // Upload to MinIO using mc
  std::string uploadCmd =
      "docker exec minio mc cp /tmp/" + tempFilename + " local/" + bucket + "/" + objectName + " >/dev/null 2>&1";
  int uploadResult = system(uploadCmd.c_str());

  // Cleanup temporary files
  std::remove(tempPath.c_str());
  system(("docker exec minio rm -f /tmp/" + tempFilename + " >/dev/null 2>&1").c_str());

  if (uploadResult != 0) {
    Logger::error("MINIO CLIENT", "Failed to upload object: " + objectName);
    return false;
  }

  return true;
}

bool MinIOHTTPClient::testConnection() {
  std::string testCmd = "docker exec minio mc admin info local >/dev/null 2>&1";
  int result = system(testCmd.c_str());

  if (result != 0) {
    Logger::error("MINIO CLIENT", "Connection test failed");
    return false;
  }

  return true;
}

bool MinIOHTTPClient::uploadImage(const std::string &objectName, const cv::Mat &image) {
  std::vector<uchar> buffer;
  if (!cv::imencode(".jpg", image, buffer)) {
    Logger::error("MINIO CLIENT", "Failed to encode image to JPEG: " + objectName);
    return false;
  }

  std::string imageData(buffer.begin(), buffer.end());
  return uploadFile(objectName, imageData, "image/jpeg");
}

bool MinIOHTTPClient::uploadCSV(const std::string &objectName, const std::string &csvContent) {
  return uploadFile(objectName, csvContent, "text/csv");
}

bool MinIOHTTPClient::uploadJSON(const std::string &objectName, const std::string &jsonContent) {
  return uploadFile(objectName, jsonContent, "application/json");
}

std::string MinIOHTTPClient::getFileUrl(const std::string &objectName) {
  return "http://" + endpoint + "/" + bucket + "/" + objectName;
}

std::string MinIOHTTPClient::getImageUrl(const std::string &objectName) { return getFileUrl(objectName); }

std::vector<std::string> MinIOHTTPClient::listFiles(const std::string &prefix) {
  std::vector<std::string> files;

  // Use mc command to list files
  std::string listCmd = "docker exec minio mc ls local/" + bucket + "/" + prefix + " --json 2>/dev/null";

  FILE *pipe = popen(listCmd.c_str(), "r");
  if (!pipe) {
    Logger::error("MINIO CLIENT", "Failed to execute list command for prefix: " + prefix);
    return files;
  }

  char buffer[1024];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  // Parse JSON output line by line
  std::istringstream iss(result);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("\"key\"") != std::string::npos) {
      size_t start = line.find("\"key\":\"") + 7;
      size_t end = line.find("\"", start);
      if (start != std::string::npos && end != std::string::npos) {
        std::string fullKey = line.substr(start, end - start);

        // Extract just the filename from the full key
        std::string filename;
        size_t lastSlash = fullKey.find_last_of('/');
        if (lastSlash != std::string::npos) {
          filename = fullKey.substr(lastSlash + 1);
        } else {
          filename = fullKey;
        }

        if (!filename.empty()) {
          files.emplace_back(filename);
        }
      }
    }
  }

  return files;
}

std::string MinIOHTTPClient::downloadFile(const std::string &objectName) {
  // Create temporary file for download
  std::string timestamp = std::to_string(std::time(nullptr));
  std::string tempFilename = "minio_download_" + timestamp;
  std::string tempPath = "/tmp/" + tempFilename;

  // Download using mc command
  std::string downloadCmd =
      "docker exec minio mc cp local/" + bucket + "/" + objectName + " /tmp/" + tempFilename + " >/dev/null 2>&1";

  int downloadResult = system(downloadCmd.c_str());
  if (downloadResult != 0) {
    Logger::error("MINIO CLIENT", "Failed to download object from MinIO: " + objectName);
    return "";
  }

  // Copy from container to host
  std::string copyCmd = "docker cp minio:/tmp/" + tempFilename + " " + tempPath + " >/dev/null 2>&1";
  int copyResult = system(copyCmd.c_str());

  if (copyResult != 0) {
    Logger::error("MINIO CLIENT", "Failed to copy object from container: " + objectName);
    system(("docker exec minio rm -f /tmp/" + tempFilename + " >/dev/null 2>&1").c_str());
    return "";
  }

  // Check if file exists locally
  if (!std::filesystem::exists(tempPath)) {
    Logger::error("MINIO CLIENT", "Downloaded file doesn't exist: " + objectName);
    system(("docker exec minio rm -f /tmp/" + tempFilename + " >/dev/null 2>&1").c_str());
    return "";
  }

  // Read file content
  std::ifstream file(tempPath, std::ios::binary);
  if (!file.is_open()) {
    Logger::error("MINIO CLIENT", "Failed to open downloaded file: " + objectName);
    std::remove(tempPath.c_str());
    system(("docker exec minio rm -f /tmp/" + tempFilename + " >/dev/null 2>&1").c_str());
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();
  file.close();

  // Cleanup temporary files
  std::remove(tempPath.c_str());
  system(("docker exec minio rm -f /tmp/" + tempFilename + " >/dev/null 2>&1").c_str());

  return content;
}

std::string MinIOHTTPClient::downloadCSV(const std::string &objectName) { return downloadFile(objectName); }

std::string MinIOHTTPClient::downloadJSON(const std::string &objectName) { return downloadFile(objectName); }

bool MinIOHTTPClient::objectExists(const std::string &objectName) {
  std::string statCmd = "docker exec minio mc stat local/" + bucket + "/" + objectName + " >/dev/null 2>&1";
  return system(statCmd.c_str()) == 0;
}

bool MinIOHTTPClient::deleteObject(const std::string &objectName) {
  std::string delCmd = "docker exec minio mc rm local/" + bucket + "/" + objectName + " >/dev/null 2>&1";
  int result = system(delCmd.c_str());

  if (result != 0) {
    Logger::error("MINIO CLIENT", "Failed to delete object: " + objectName);
    return false;
  }

  return true;
}

std::vector<std::string> MinIOHTTPClient::listImages(const std::string &prefix) {
  std::vector<std::string> allFiles = listFiles(prefix);
  std::vector<std::string> images;

  for (const auto &file : allFiles) {
    if (file.find(".jpg") != std::string::npos || file.find(".jpeg") != std::string::npos ||
        file.find(".png") != std::string::npos) {
      images.emplace_back(file);
    }
  }

  return images;
}

std::vector<std::string> MinIOHTTPClient::listFilesByExtension(const std::string &prefix,
                                                               const std::string &extension) {
  std::vector<std::string> allFiles = listFiles(prefix);
  std::vector<std::string> filtered;

  for (const auto &file : allFiles) {
    if (file.find(extension) != std::string::npos) {
      filtered.emplace_back(file);
    }
  }

  return filtered;
}

// Remove AWS4 signature methods (keep empty implementations for compatibility)
std::string MinIOHTTPClient::getCurrentTimestamp() { return ""; }
std::string MinIOHTTPClient::getCurrentDate() { return ""; }
std::string MinIOHTTPClient::sha256(const std::string &data) { return ""; }
std::string MinIOHTTPClient::hmacSha256(const std::string &key, const std::string &data) { return ""; }
std::string MinIOHTTPClient::createSignature(const std::string &method, const std::string &uri,
                                             const std::string &queryString, const std::string &headers,
                                             const std::string &payload, const std::string &timestamp) {
  return "";
}
std::string MinIOHTTPClient::createAuthorizationHeader(const std::string &method, const std::string &uri,
                                                       const std::string &queryString, const std::string &headers,
                                                       const std::string &payload, const std::string &timestamp) {
  return "";
}