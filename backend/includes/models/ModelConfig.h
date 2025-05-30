#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <string>
#include <vector>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

const std::string METADATA_MODEL_PATH = "../assets/models/metadata.onnx";
const std::string METADATA_ENGINE_PATH = "../assets/models/metadata_model/1/model.plan";
const int METADATA_BATCH_SIZE = 2;
const int METADATA_TOP_K = 300;
const int METADATA_MAX_OUTPUT_BOXES = 90;

const std::string CONTENT_MODEL_PATH  = "../assets/models/content.onnx";
const std::string CONTENT_ENGINE_PATH  = "../assets/models/content_model/1/model.plan";
const int CONTENT_BATCH_SIZE = 8;
const int CONTENT_TOP_K = 100;
const int CONTENT_MAX_OUTPUT_BOXES = 30;

const float SCORE_THRESHOLD = 0.5f;
const float IOU_THRESHOLD = 0.5f;

#endif // MODEL_CONFIG_H