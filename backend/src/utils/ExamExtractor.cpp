#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/Detection.h"
#include "utils/ExamConfig.h"
#include "utils/ExamExtractor.h"

// Constants for matrix sizes
const int STUDENT_ID_NUM_CENTER_X = 9;
const int STUDENT_ID_NUM_CENTER_Y = 10;
const int EXAM_ID_NUM_CENTER_X = 3;
const int EXAM_ID_NUM_CENTER_Y = 10;

const int PART_1_NUM_CENTER_X = 4;
const int PART_1_NUM_CENTER_Y = 4;
const int PART_2_NUM_CENTER_X = 4;
const int PART_2_NUM_CENTER_Y = 6;

const int NUM_SUBMATRICES_ON_IMG = 2;
const int NUM_SUBMATRIX_COLS = 2;

cv::Mat ExamExtractor::createMetadataMatrix(const std::vector<Detection> &detections, int numRows, int numCols) {
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);

  if (detections.empty()) {
    return matrix;
  }

  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto &detection : detections) {
    centers.emplace_back(
        cv::Point2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2));
  }

  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto &center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }

  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));

  std::vector<std::vector<std::pair<int, Detection>>> columnCandidates(numCols);

  // Group detections by column using K-means assignment
  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;

    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;

    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));

    columnCandidates[col].push_back({row, detections[i]});
  }

  // Process each column to select candidates
  for (int col = 0; col < numCols; ++col) {
    if (!columnCandidates[col].empty()) {
      std::vector<std::pair<int, Detection>> class1Candidates;
      for (const auto &candidate : columnCandidates[col]) {
        if (candidate.second.classId == 1) {
          class1Candidates.emplace_back(candidate);
        }
      }
      std::pair<int, Detection> selected;
      if (class1Candidates.empty()) {
        // No class 1 candidates—select the one with the minimum avgGray
        if (columnCandidates[col].size() > 1) {
          auto minCandidate = columnCandidates[col][0];
          float sumGray = 0.0f;
          for (const auto &candidate : columnCandidates[col]) {
            if (candidate.second.avgGray < minCandidate.second.avgGray)
              minCandidate = candidate;
            sumGray += candidate.second.avgGray;
          }
          sumGray -= minCandidate.second.avgGray;
          float avgGrayAll = sumGray / (columnCandidates[col].size() - 1);
          bool hasSelected = false;
          if (minCandidate.second.avgGray < avgGrayAll * 0.9f) {
            selected = minCandidate;
            hasSelected = true;
          }
          if (hasSelected) {
            matrix.at<uchar>(selected.first, col) = 1;
          }
        }
      } else if (class1Candidates.size() == 1) {
        // Only one class 1 candidate—select it
        matrix.at<uchar>(class1Candidates[0].first, col) = 1;
      } else {
        // Multiple class 1 candidates—filter light ones
        std::vector<std::pair<int, Detection>> filtered;
        for (const auto &cand : class1Candidates) {
          bool removeCand = false;
          for (const auto &other : class1Candidates) {
            if (cand.second.avgGray > other.second.avgGray * 1.1f) {
              removeCand = true;
              break;
            }
          }
          if (!removeCand) {
            filtered.emplace_back(cand);
          }
        }
        if (filtered.size() == 1) {
          matrix.at<uchar>(filtered[0].first, col) = 1;
        } else if (filtered.empty()) {
          selected = class1Candidates[0];
          for (const auto &cand : class1Candidates) {
            if (cand.second.avgGray < selected.second.avgGray)
              selected = cand;
          }
          matrix.at<uchar>(selected.first, col) = 1;
        } else {
          for (const auto &cand : filtered) {
            matrix.at<uchar>(cand.first, col) = 2;
          }
        }
      }
    }
  }
  return matrix;
}

cv::Mat ExamExtractor::createPart1Matrix(const std::vector<Detection> &detections, int numRows, int numCols) {
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);

  if (detections.empty()) {
    return matrix;
  }

  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto &detection : detections) {
    centers.emplace_back(
        cv::Point2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2));
  }

  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto &center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }

  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));

  std::vector<std::vector<std::pair<int, Detection>>> rowCandidates(numRows);

  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;

    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;

    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));

    rowCandidates[row].push_back({col, detections[i]});
  }

  // Process each row to select candidates
  for (int row = 0; row < numRows; ++row) {
    if (!rowCandidates[row].empty()) {
      std::vector<std::pair<int, Detection>> class1Candidates;
      for (const auto &candidate : rowCandidates[row]) {
        if (candidate.second.classId == 1) {
          class1Candidates.emplace_back(candidate);
        }
      }
      std::pair<int, Detection> selected;
      if (class1Candidates.empty()) {
        // No class 1 candidates—select the one with the minimum avgGray
        if (rowCandidates[row].size() > 1) {
          auto minCandidate = rowCandidates[row][0];
          float sumGray = 0.0f;
          for (const auto &candidate : rowCandidates[row]) {
            if (candidate.second.avgGray < minCandidate.second.avgGray)
              minCandidate = candidate;
            sumGray += candidate.second.avgGray;
          }
          sumGray -= minCandidate.second.avgGray;
          float avgGrayAll = sumGray / (rowCandidates[row].size() - 1);
          bool hasSelected = false;
          if (minCandidate.second.avgGray < avgGrayAll * 0.9f) {
            selected = minCandidate;
            hasSelected = true;
          }
          if (hasSelected) {
            matrix.at<uchar>(row, selected.first) = 1;
          }
        }
      } else if (class1Candidates.size() == 1) {
        // Only one class 1 candidate—select it directly
        matrix.at<uchar>(row, class1Candidates[0].first) = 1;
      } else {
        // Multiple class 1 candidates—filter out those that are significantly
        // lighter
        std::vector<std::pair<int, Detection>> filtered;
        for (const auto &cand : class1Candidates) {
          bool removeCand = false;
          for (const auto &other : class1Candidates) {
            if (cand.second.avgGray > other.second.avgGray * 1.1f) {
              removeCand = true;
              break;
            }
          }
          if (!removeCand) {
            filtered.emplace_back(cand);
          }
        }
        if (filtered.size() == 1) {
          matrix.at<uchar>(row, filtered[0].first) = 2;
        } else if (filtered.empty()) {
          selected = class1Candidates[0];
          for (const auto &cand : class1Candidates) {
            if (cand.second.avgGray < selected.second.avgGray)
              selected = cand;
          }
          matrix.at<uchar>(row, selected.first) = 2;
        } else {
          for (const auto &cand : filtered) {
            matrix.at<uchar>(row, cand.first) = 3;
          }
        }
      }
    }
  }
  return matrix;
}

cv::Mat ExamExtractor::createPart2Matrix(const std::vector<Detection> &detections, int numRows, int numCols) {
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);

  if (detections.empty()) {
    return matrix;
  }

  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto &detection : detections) {
    centers.emplace_back(
        cv::Point2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2));
  }

  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto &center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }

  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));

  std::vector<std::vector<std::pair<int, Detection>>> rowCandidates(numRows);

  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;

    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;

    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));

    rowCandidates[row].push_back({col, detections[i]});
  }

  // Process each row to select candidates
  for (int sm = 0; sm < NUM_SUBMATRICES_ON_IMG; ++sm) {
    int colStart = sm * NUM_SUBMATRIX_COLS;
    for (int row = 0; row < numRows; ++row) {
      std::vector<std::pair<int, Detection>> subRowCandidates;
      for (const auto &cand : rowCandidates[row]) {
        if (cand.first >= colStart && cand.first < colStart + NUM_SUBMATRIX_COLS)
          subRowCandidates.emplace_back(cand);
      }
      if (!subRowCandidates.empty()) {
        std::vector<std::pair<int, Detection>> class1Candidates;
        for (const auto &candidate : subRowCandidates) {
          if (candidate.second.classId == 1) {
            class1Candidates.emplace_back(candidate);
          }
        }
        std::pair<int, Detection> selected;
        if (class1Candidates.empty()) {
          // No class 1 candidates—select the one with the minimum avgGray
          if (subRowCandidates.size() > 1) {
            auto minCandidate = subRowCandidates[0];
            float sumGray = 0.0f;
            for (const auto &candidate : subRowCandidates) {
              if (candidate.second.avgGray < minCandidate.second.avgGray)
                minCandidate = candidate;
              sumGray += candidate.second.avgGray;
            }
            sumGray -= minCandidate.second.avgGray;
            float avgGrayAll = sumGray / (subRowCandidates.size() - 1);
            bool hasSelected = false;
            if (minCandidate.second.avgGray < avgGrayAll * 0.9f) {
              selected = minCandidate;
              hasSelected = true;
            }
            if (hasSelected) {
              matrix.at<uchar>(row, selected.first) = 1;
            }
          }
        } else if (class1Candidates.size() == 1) {
          // Only one class 1 candidate—select it directly.
          matrix.at<uchar>(row, class1Candidates[0].first) = 1;
        } else {
          // Multiple class 1 candidates—filter out those that are significantly
          // lighter.
          std::vector<std::pair<int, Detection>> filtered;
          for (const auto &cand : class1Candidates) {
            bool removeCand = false;
            for (const auto &other : class1Candidates) {
              if (cand.second.avgGray > other.second.avgGray * 1.1f) {
                removeCand = true;
                break;
              }
            }
            if (!removeCand) {
              filtered.emplace_back(cand);
            }
          }
          if (filtered.size() == 1) {
            matrix.at<uchar>(row, filtered[0].first) = 2;
          } else if (filtered.empty()) {
            selected = class1Candidates[0];
            for (const auto &cand : class1Candidates) {
              if (cand.second.avgGray < selected.second.avgGray)
                selected = cand;
            }
            matrix.at<uchar>(row, selected.first) = 2;
          } else {
            for (const auto &cand : filtered) {
              matrix.at<uchar>(row, cand.first) = 3;
            }
          }
        }
      }
    }
  }
  return matrix;
}

std::string ExamExtractor::getStudentId(const cv::Mat &matrix) {
  std::string studentId;
  if (matrix.rows != 10 || matrix.cols != 9) {
    return studentId;
  }
  for (int col = 0; col < matrix.cols; col++) {
    int digit = -1;
    bool hasMultiple = false;
    for (int row = 0; row < matrix.rows; row++) {
      if (matrix.at<uchar>(row, col) == 1) {
        digit = row;
        break;
      } else if (matrix.at<uchar>(row, col) == 2) {
        hasMultiple = true;
      }
    }
    if (digit >= 0) {
      studentId.push_back(static_cast<char>('0' + digit));
    } else if (hasMultiple) {
      studentId.push_back('X');
    } else {
      studentId.push_back('_');
    }
  }
  return studentId;
}

std::string ExamExtractor::getExamId(const cv::Mat &matrix) {
  std::string examId;
  if (matrix.rows != 10 || matrix.cols != 3) {
    return examId;
  }
  for (int col = 0; col < matrix.cols; col++) {
    int digit = -1;
    bool hasMultiple = false;
    for (int row = 0; row < matrix.rows; row++) {
      if (matrix.at<uchar>(row, col) == 1) {
        digit = row;
        break;
      } else if (matrix.at<uchar>(row, col) == 2) {
        hasMultiple = true;
      }
    }
    if (digit >= 0) {
      examId.push_back(static_cast<char>('0' + digit));
    } else if (hasMultiple) {
      examId.push_back('X');
    } else {
      examId.push_back('_');
    }
  }
  return examId;
}

std::vector<std::string> ExamExtractor::processContentPart1(const cv::Mat &array) {
  std::vector<std::string> answers;
  if (array.rows != (PART_1_NUM_QUESTIONS + 2) || array.cols != PART_1_NUM_CENTER_Y) {
    return answers;
  }

  for (int i = 0; i < PART_1_NUM_QUESTIONS; i++) {
    int selectedIdx = -1;
    bool hasMultiple = false;
    bool isSuggested = false;
    for (int j = 0; j < PART_1_NUM_CENTER_Y; j++) {
      if (array.at<uchar>(i, j) == 1) {
        selectedIdx = j;
        break;
      } else if (array.at<uchar>(i, j) == 2) {
        isSuggested = true;
        selectedIdx = j;
        break;
      } else if (array.at<uchar>(i, j) == 3) {
        hasMultiple = true;
      }
    }

    if (selectedIdx != -1) {
      if (isSuggested) {
        answers.push_back(std::string(1, static_cast<char>('a' + selectedIdx)));
      } else {
        answers.push_back(std::string(1, static_cast<char>('A' + selectedIdx)));
      }
    } else if (hasMultiple) {
      answers.push_back("X");
    } else {
      answers.push_back("_");
    }
  }
  return answers;
}

std::vector<std::string> ExamExtractor::processContentPart2(const cv::Mat &array) {
  std::vector<std::string> overallAnswers;
  int numCols = PART_2_NUM_QUESTIONS + 1;
  if (array.rows != PART_2_NUM_CENTER_Y || array.cols != (3 * PART_2_NUM_CENTER_X) || !array.isContinuous()) {
    return overallAnswers;
  }

  const uchar *data = array.data;
  int step = static_cast<int>(array.step[0]);

  for (int i = 0; i < PART_2_NUM_QUESTIONS; i++) {
    std::string eachAnswer;
    int colStart = i * NUM_SUBMATRIX_COLS;

    for (int r = 0; r < PART_2_NUM_CENTER_Y; r++) {
      const uchar *rowPtr = data + r * step;
      uchar val0 = rowPtr[colStart];
      uchar val1 = rowPtr[colStart + 1];

      bool isSuggested = (val0 == 2 || val1 == 2);
      // bool hasMultiple = (val0 == 3 || val1 == 3);

      if (val0 == 1) {
        eachAnswer.push_back('D');
      } else if (val1 == 1) {
        eachAnswer.push_back('S');
      } else if (isSuggested) {
        if (val0 == 2) {
          eachAnswer.push_back('d');
        } else if (val1 == 2) {
          eachAnswer.push_back('s');
        }
        // } else if (hasMultiple) {
        //   eachAnswer.push_back('S');
      } else {
        eachAnswer.push_back('S');
      }
    }
    overallAnswers.emplace_back(eachAnswer);
  }
  return overallAnswers;
}

std::vector<std::string>
ExamExtractor::extract_answers_from_detections(const std::string &imageBasename,
                                               const std::vector<std::vector<Detection>> &metadataDetections,
                                               const std::vector<std::vector<Detection>> &contentDetections) {
  try {
    // Process metadata matrix
    cv::Mat studentIdMatrix =
        createMetadataMatrix(metadataDetections[0], STUDENT_ID_NUM_CENTER_Y, STUDENT_ID_NUM_CENTER_X);
    cv::Mat examIdMatrix = createMetadataMatrix(metadataDetections[1], EXAM_ID_NUM_CENTER_Y, EXAM_ID_NUM_CENTER_X);

    // Process content part 1 matrix
    cv::Mat content11Matrix = createPart1Matrix(contentDetections[0], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content12Matrix = createPart1Matrix(contentDetections[1], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content13Matrix = createPart1Matrix(contentDetections[2], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    // cv::Mat content14Matrix = createPart1Matrix(
    //     contentDetections[3], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);

    cv::Mat contentPart1Matrix;
    std::vector<cv::Mat> matricesPart1 = {content11Matrix, content12Matrix, content13Matrix};
    cv::vconcat(matricesPart1, contentPart1Matrix);

    // std::cout << "Content Part 1 Matrix:\n" << contentPart1Matrix << std::endl;

    // Process content part 2 matrix
    cv::Mat content21Matrix = createPart2Matrix(contentDetections[4], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content22Matrix = createPart2Matrix(contentDetections[5], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content23Matrix = createPart2Matrix(contentDetections[6], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    // cv::Mat content24Matrix = createPart2Matrix(
    //     contentDetections[7], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);

    cv::Mat contentPart2Matrix;
    std::vector<cv::Mat> matricesPart2 = {content21Matrix, content22Matrix, content23Matrix};
    cv::hconcat(matricesPart2, contentPart2Matrix);

    // std::cout << "Content Part 2 Matrix:\n" << contentPart2Matrix << std::endl;

    // Extract answers
    std::string studentId = getStudentId(studentIdMatrix);
    std::string examId = getExamId(examIdMatrix);
    std::vector<std::string> contentPart1Answers = processContentPart1(contentPart1Matrix);
    std::vector<std::string> contentPart2Answers = processContentPart2(contentPart2Matrix);

    std::vector<std::string> result;
    result.emplace_back(imageBasename);
    result.emplace_back(studentId);
    result.emplace_back(examId);

    result.emplace_back("Answers");
    result.insert(result.end(), contentPart1Answers.begin(), contentPart1Answers.end());
    result.insert(result.end(), contentPart2Answers.begin(), contentPart2Answers.end());

    result.emplace_back("N/A");
    result.emplace_back("N/A");
    result.emplace_back("N/A");

    return result;

  } catch (const std::exception &e) {
    Logger::error("EXAM EXTRACTOR", "Answer extraction failed for " + imageBasename + ": " + std::string(e.what()));
    return {};
  }
}
