#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  float score;
  int classId;
  float avgGray;
  Detection() : box(), score(0.0f), classId(0), avgGray(0.0f) {}
  Detection(const cv::Rect &b, float s, int c, float a) : box(b), score(s), classId(c), avgGray(a) {}
};

#endif // DETECTION_H