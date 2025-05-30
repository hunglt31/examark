#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  float score;
  int classId;
  float avgGray;
};

#endif // DETECTION_H