#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "models/TritonClient.h"

namespace tc = triton::client;

TritonClient::TritonClient(const std::string &url) : m_url(url) {}

TritonClient::~TritonClient() {}

bool TritonClient::connect() {
  try {
    tc::Error err = tc::InferenceServerGrpcClient::Create(&m_client, m_url);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to create Triton Client: " + std::string(err.Message()));
      return false;
    }

    bool is_live = false;
    err = m_client->IsServerLive(&is_live);
    if (!err.IsOk() || !is_live) {
      Logger::error("TRITON CLIENT", "Triton Inference Server is not live or reachable at " + m_url);
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    Logger::error("TRITON CLIENT", "Exception during connection: " + std::string(e.what()));
    return false;
  }
}

std::vector<std::vector<Detection>> TritonClient::inference(const std::vector<cv::Mat> &images,
                                                            const std::string &modelName) {
  static std::mutex inferenceMutex;
  if (!m_client || images.empty()) {
    return {};
  }

  try {
    std::vector<uint8_t> rawData;
    for (const auto &img : images) {
      cv::Mat continuous = img.isContinuous() ? img : img.clone();
      const uint8_t *imgData = continuous.data;
      rawData.insert(rawData.end(), imgData, imgData + continuous.total() * continuous.elemSize());
    }

    // Thread safety for gRPC calls
    std::lock_guard<std::mutex> lock(inferenceMutex);

    // Create inference options
    tc::InferOptions options(modelName);
    options.model_version_ = "";

    // Input shape based on the model
    int batchSize = images.size();
    std::vector<int64_t> shape;
    if (modelName == "metadata_model") {
      shape = {METADATA_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, 3};
    } else {
      shape = {CONTENT_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, 3};
    }

    // Prepare input data
    tc::InferInput *input;
    tc::Error err = tc::InferInput::Create(&input, "input0", shape, "UINT8");
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to create inference input: " + std::string(err.Message()));
      return {};
    }

    std::shared_ptr<tc::InferInput> input_ptr(input);
    err = input_ptr->AppendRaw(rawData);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to append raw data to input: " + std::string(err.Message()));
      return {};
    }

    std::vector<tc::InferInput *> inputs = {input_ptr.get()};

    // Request all outputs
    std::vector<const tc::InferRequestedOutput *> outputs;
    tc::InferRequestedOutput *output_bboxes;
    tc::InferRequestedOutput *output_scores;
    tc::InferRequestedOutput *output_classes;

    tc::Error err_out;
    err_out = tc::InferRequestedOutput::Create(&output_bboxes, "bboxes");
    if (!err_out.IsOk()) {
      std::cerr << "Failed to create bboxes output: " << err_out << std::endl;
      return {};
    }
    err_out = tc::InferRequestedOutput::Create(&output_scores, "scores");
    if (!err_out.IsOk()) {
      std::cerr << "Failed to create scores output: " << err_out << std::endl;
      return {};
    }
    err_out = tc::InferRequestedOutput::Create(&output_classes, "classes");
    if (!err_out.IsOk()) {
      std::cerr << "Failed to create classes output: " << err_out << std::endl;
      return {};
    }

    // Use shared_ptr for memory management
    std::shared_ptr<tc::InferRequestedOutput> output1_ptr(output_bboxes);
    std::shared_ptr<tc::InferRequestedOutput> output2_ptr(output_scores);
    std::shared_ptr<tc::InferRequestedOutput> output3_ptr(output_classes);

    outputs.emplace_back(output1_ptr.get());
    outputs.emplace_back(output2_ptr.get());
    outputs.emplace_back(output3_ptr.get());

    // Infer call
    tc::InferResult *result = nullptr;
    err = m_client->Infer(&result, options, inputs, outputs);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Inference failed: " + std::string(err.Message()));
      return {};
    }

    std::unique_ptr<tc::InferResult> result_ptr(result);

    // Process the response
    return postprocess(images, result_ptr.get(), batchSize, modelName);
  } catch (const std::exception &e) {
    Logger::error("TRITON CLIENT", "Exception during inference: " + std::string(e.what()));
    return {};
  }
}

std::vector<std::vector<Detection>> TritonClient::postprocess(const std::vector<cv::Mat> &images,
                                                              const tc::InferResult *result, int numImages,
                                                              const std::string &modelName) {

  std::vector<std::vector<Detection>> results(numImages);

  try {
    tc::Error err;

    // Get bounding boxes
    std::vector<float> bboxes;
    size_t bbox_size;
    const uint8_t *bbox_data;
    size_t bbox_byte_size;
    err = result->RawData("bboxes", &bbox_data, &bbox_byte_size);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to get bounding boxes: " + std::string(err.Message()));
      return results;
    }
    bbox_size = bbox_byte_size / sizeof(float);
    bboxes.resize(bbox_size);
    memcpy(bboxes.data(), bbox_data, bbox_byte_size);

    // Get scores
    std::vector<float> scores;
    size_t score_size;
    const uint8_t *score_data;
    size_t score_byte_size;
    err = result->RawData("scores", &score_data, &score_byte_size);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to get scores: " + std::string(err.Message()));
      return results;
    }
    score_size = score_byte_size / sizeof(float);
    scores.resize(score_size);
    memcpy(scores.data(), score_data, score_byte_size);

    // Get classes
    std::vector<int32_t> classes;
    size_t class_size;
    const uint8_t *class_data;
    size_t class_byte_size;
    err = result->RawData("classes", &class_data, &class_byte_size);
    if (!err.IsOk()) {
      Logger::error("TRITON CLIENT", "Failed to get classes: " + std::string(err.Message()));
      return results;
    }
    class_size = class_byte_size / sizeof(int32_t);
    classes.resize(class_size);
    memcpy(classes.data(), class_data, class_byte_size);

    int maxOutputBoxes;
    if (modelName == "metadata_model") {
      maxOutputBoxes = METADATA_MAX_OUTPUT_BOXES;
    } else {
      maxOutputBoxes = CONTENT_MAX_OUTPUT_BOXES;
    }

    // Populate the results
    cv::Mat gray_image, cropped, binary_image;
    cv::Rect roi;

    for (int i = 0; i < numImages; i++) {
      cv::cvtColor(images[i], gray_image, cv::COLOR_BGR2GRAY);

      for (int j = 0; j < maxOutputBoxes; j++) {
        int detIdx = i * maxOutputBoxes + j;
        if (scores[detIdx] < 0.5f)
          continue;

        Detection det;
        int bboxIdx = detIdx * 4;
        float x1 = bboxes[bboxIdx];
        float y1 = bboxes[bboxIdx + 1];
        float x2 = bboxes[bboxIdx + 2];
        float y2 = bboxes[bboxIdx + 3];
        float width = x2 - x1;
        float height = y2 - y1;

        roi.x = cvRound(x1 + width / 4);
        roi.y = cvRound(y1 + height / 4);
        roi.width = cvRound(width / 2);
        roi.height = cvRound(height / 2);

        roi &= cv::Rect(0, 0, gray_image.cols, gray_image.rows);
        cropped = gray_image(roi);

        const uchar color_threshold = 175;
        cv::threshold(cropped, binary_image, color_threshold, 255, cv::THRESH_BINARY_INV);

        int pixel_count = cv::countNonZero(binary_image);
        int total_pixels = cropped.total();

        results[i].emplace_back(Detection(cv::Rect(cvRound(x1), cvRound(y1), cvRound(width), cvRound(height)),
                                          scores[detIdx], classes[detIdx], cv::mean(cropped)[0],
                                          static_cast<float>(pixel_count) / total_pixels));
      }
    }
    return results;
  } catch (const std::exception &e) {
    Logger::error("TRITON CLIENT", "Exception during postprocessing: " + std::string(e.what()));
    return results;
  }
}

std::string TritonClient::getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S") << "_" << std::setw(3) << std::setfill('0')
     << ms.count();
  return ss.str();
}

void TritonClient::drawAndSaveResults(const std::vector<cv::Mat> &images,
                                      const std::vector<std::vector<Detection>> &detections,
                                      const std::string &outputDir) {
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(outputDir);

  // Process each image
  for (size_t i = 0; i < images.size(); ++i) {
    // Clone the image to avoid modifying the original
    cv::Mat resultImg = images[i].clone();

    // Draw all detections for this image
    for (const auto &det : detections[i]) {
      // Define color based on class ID (for variety)
      cv::Scalar color(255, 0, 0);
      if (det.classId == 1)
        color = cv::Scalar(0, 0, 255);
      // Draw bounding box
      cv::rectangle(resultImg, det.box, color, 2);

      // Prepare label text with class ID and confidence
      std::stringstream ss;
      ss << "Class:" << det.classId << " " << std::fixed << std::setprecision(2) << det.score;
      std::string label = ss.str();

      // Get text size for better positioning
      int baseline = 0;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      // Draw label background
      cv::rectangle(resultImg, cv::Point(det.box.x, det.box.y - textSize.height - 5),
                    cv::Point(det.box.x + textSize.width, det.box.y), color, -1);
    }

    // Generate a unique filename with timestamp
    std::string timestamp = getCurrentTimestamp();
    std::string filename = outputDir + "/detection_" + timestamp + "_" + std::to_string(i) + ".jpg";

    // Save the image
    cv::imwrite(filename, resultImg);

    Logger::info("TRITON CLIENT", "Saved annotated image: " + filename);
  }
}
