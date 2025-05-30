#include <opencv2/opencv.hpp>
#include <iostream>
#include <NvInferPlugin.h>

#include "models/ModelBuilder.h"
#include "models/ModelConfig.h"
#include "utils/Logger.h"

int main() {   
    ModelBuilder metadataModel(
        METADATA_MODEL_PATH,
        METADATA_ENGINE_PATH,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        METADATA_BATCH_SIZE,
        METADATA_TOP_K,
        METADATA_MAX_OUTPUT_BOXES
    );
    
    ModelBuilder contentModel(
        CONTENT_MODEL_PATH,
        CONTENT_ENGINE_PATH,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        CONTENT_BATCH_SIZE,
        CONTENT_TOP_K,
        CONTENT_MAX_OUTPUT_BOXES
    );
    
    // Load or build engines
    Logger::info("MAIN", "Loading or building models...");
    if (!metadataModel.loadModelBuilder()) {
        Logger::error("MAIN", "Failed to build/load Metadata model.");
        return 1;
    }
    Logger::success("MAIN", "Metadata model loaded successfully.");
    
    if (!contentModel.loadModelBuilder()) {
        Logger::error("MAIN", "Failed to build/load Content model.");
        return 1;
    }
    Logger::success("MAIN", "Content model loaded successfully.");
    
    // Create dummy input images
    Logger::info("MAIN", "Creating dummy input images for inference...");
    std::vector<cv::Mat> metadataImages(METADATA_BATCH_SIZE);
    std::vector<cv::Mat> contentImages(CONTENT_BATCH_SIZE);
    
    for (int i = 0; i < METADATA_BATCH_SIZE; i++) {
        metadataImages[i] = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
    }
    for (int i = 0; i < CONTENT_BATCH_SIZE; i++) {
        contentImages[i] = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
    }
    Logger::success("MAIN", "Dummy input images created successfully.");
    
    // Run inference
    try {
        Logger::info("MAIN", "Running inference on Metadata model...");
        auto metadataResults = metadataModel.inference(metadataImages);
        Logger::success("MAIN", "Metadata model inference complete.");
        
        Logger::info("MAIN", "Running inference on Content model...");
        auto contentResults = contentModel.inference(contentImages);
        Logger::success("MAIN", "Content model inference complete.");
    }
    catch (const std::exception& e) {
        Logger::error("MAIN", "Inference failed: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}