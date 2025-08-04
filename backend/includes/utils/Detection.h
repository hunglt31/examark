#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  float score;
  int classId;
  float avg_gray;
  float percent_below_threshold;
  Detection() : box(), score(0.0f), classId(0), avg_gray(0.0f), percent_below_threshold(0.0f) {}
  Detection(const cv::Rect &b, float s, int c, float a, float p)
      : box(b), score(s), classId(c), avg_gray(a), percent_below_threshold(p) {}
};

#endif // DETECTION_H