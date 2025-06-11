#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cctype>
#include "utils/ExamGrader.h"
#include "utils/Detection.h"
#include "utils/ExamConfig.h"

// Constants for matrix sizes
const int STUDENT_ID_NUM_CENTER_X = 9;
const int STUDENT_ID_NUM_CENTER_Y = 10;
const int EXAM_ID_NUM_CENTER_X = 3;
const int EXAM_ID_NUM_CENTER_Y = 10;

const int PART_1_NUM_CENTER_X = 4;
const int PART_1_NUM_CENTER_Y = 4;
const int PART_2_NUM_CENTER_X = 4;
const int PART_2_NUM_CENTER_Y = 6;

const int NUM_SUBMATRICES_PART_2 = 8;
const int NUM_SUBMATRICES_ON_IMG = 2;
const int NUM_SUBMATRIX_COLS = 2;

cv::Mat ExamGrader::createMetadataMatrix(const std::vector<Detection>& detections,
                                         int numRows, int numCols) {
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);
  
  if (detections.empty()) {
    return matrix;
  }
  
  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
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
      for (const auto& candidate : columnCandidates[col]) {
        if (candidate.second.classId == 1) {
          class1Candidates.push_back(candidate);
        }
      }
      std::pair<int, Detection> selected;
      if (class1Candidates.empty()) {
        // No class 1 candidates—select the one with the minimum avgGray
        if (columnCandidates[col].size() > 1) {
          auto minCandidate = columnCandidates[col][0];
          float sumGray = 0.0f;
          for (const auto& candidate : columnCandidates[col]) {
            if (candidate.second.avgGray < minCandidate.second.avgGray)
              minCandidate = candidate;
            sumGray += candidate.second.avgGray;
          }
          sumGray -= minCandidate.second.avgGray;
          float avgGrayAll = sumGray / (columnCandidates[col].size() - 1); 
          bool hasSelected = false;
          if (minCandidate.second.avgGray < avgGrayAll * 0.8f) {
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
            filtered.push_back(cand);
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
            matrix.at<uchar>(col, cand.first) = 2;
          }
        } 
      }
    }
  }
  return matrix;
}

cv::Mat ExamGrader::createPart1Matrix(const std::vector<Detection>& detections,
                                      int numRows, int numCols) {  
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);
  
  if (detections.empty()) {
    return matrix;
  }
  
  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
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
      for (const auto& candidate : rowCandidates[row]) {
        if (candidate.second.classId == 1) {
          class1Candidates.push_back(candidate);
        }
      }
      std::pair<int, Detection> selected;
      if (class1Candidates.empty()) {
        // Keep 0 values
      } else if (class1Candidates.size() == 1) {
        // Only one class 1 candidate—select it directly
        matrix.at<uchar>(row, class1Candidates[0].first) = 1;
      } else {
        // Multiple class 1 candidates—filter out those that are significantly lighter
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
            filtered.push_back(cand);
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

cv::Mat ExamGrader::createPart2Matrix(const std::vector<Detection>& detections,
                                      int numRows, int numCols) {       
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);
  
  if (detections.empty()) {
    return matrix;
  }
  
  // Use K-means to assign detections to grid positions
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
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
      for (const auto& cand : rowCandidates[row]) {
        if (cand.first >= colStart && cand.first < colStart + NUM_SUBMATRIX_COLS)
          subRowCandidates.push_back(cand);
      }
      // Keep your existing logic for processing each subrow
      if (!subRowCandidates.empty()) {
        std::vector<std::pair<int, Detection>> class1Candidates;
        for (const auto& candidate : subRowCandidates) {
          if (candidate.second.classId == 1) {
            class1Candidates.push_back(candidate);
          }
        }
        std::pair<int, Detection> selected;
        if (class1Candidates.empty()) {
          // Keep 0 values
        } else if (class1Candidates.size() == 1) {
          // Only one class 1 candidate—select it directly.
          matrix.at<uchar>(row, class1Candidates[0].first) = 1;
        } else {
          // Multiple class 1 candidates—filter out those that are significantly lighter.
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
              filtered.push_back(cand);
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

std::string ExamGrader::getStudentId(const cv::Mat& matrix) {
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

std::string ExamGrader::getExamId(const cv::Mat& matrix) {
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

std::vector<std::string> ExamGrader::processContentPart1(const cv::Mat& array) {
  std::vector<std::string> answers;
  if(array.rows != 16 || array.cols != 4) {
    return answers; 
  }
    
  for (int i = 0; i < array.rows; i++) {
    int selectedIdx = -1;
    bool hasMultiple = false;
    bool isSuggested = false;
    for (int j = 0; j < array.cols; j++) {
      if(array.at<uchar>(i, j) == 1) {
        selectedIdx = j;
        break;
      } else if(array.at<uchar>(i, j) == 2) {
        isSuggested = true;
        selectedIdx = j;
        break;
      } else if(array.at<uchar>(i, j) == 3) {
        hasMultiple = true;
      }
    }
    
    if(selectedIdx != -1) {
      if(isSuggested) {
        answers.push_back(std::string(1, static_cast<char>('a' + selectedIdx)));
      } else {
        answers.push_back(std::string(1, static_cast<char>('A' + selectedIdx)));
      }
    } else if(hasMultiple) {
      answers.push_back("X");  
    } else {
      answers.push_back("_");
    }
  }  
  return answers;
}

std::vector<std::string> ExamGrader::processContentPart2(const cv::Mat& array) {
  std::vector<std::string> overallAnswers;
  if (array.rows != 6 || array.cols != 16 || !array.isContinuous()) {
    return overallAnswers;
  }
  
  const uchar* data = array.data;
  int step = static_cast<int>(array.step[0]);
    
  for (int i = 0; i < NUM_SUBMATRICES_PART_2; i++) {
    std::string eachAnswer;
    int colStart = i * NUM_SUBMATRIX_COLS;
    
    for (int r = 0; r < PART_2_NUM_CENTER_Y; r++) {
      const uchar* rowPtr = data + r * step;
      uchar val0 = rowPtr[colStart];
      uchar val1 = rowPtr[colStart + 1];
      
      bool isSuggested = (val0 == 2 || val1 == 2);
      bool hasMultiple = (val0 == 3 || val1 == 3);
      
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
      } else if (hasMultiple) {
        eachAnswer.push_back('X'); 
      } else {
        eachAnswer.push_back('_'); 
      }
    }
    overallAnswers.push_back(eachAnswer);
  }
  return overallAnswers;
}

std::vector<std::string> ExamGrader::extractAnswersAndGradeExam(
  const std::string& imageBasename,
  const std::vector<std::vector<Detection>>& metadataDetections,
  const std::vector<std::vector<Detection>>& contentDetections,
  const std::map<std::string, std::vector<std::string>>& examAnswerKeys) 
{
  try {
    // Process metadata matrix
    cv::Mat studentIdMatrix = createMetadataMatrix(metadataDetections[0], STUDENT_ID_NUM_CENTER_Y, STUDENT_ID_NUM_CENTER_X);
    cv::Mat examIdMatrix = createMetadataMatrix(metadataDetections[1], EXAM_ID_NUM_CENTER_Y, EXAM_ID_NUM_CENTER_X);

    // Process content part 1 matrix
    cv::Mat content11Matrix = createPart1Matrix(contentDetections[0], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content12Matrix = createPart1Matrix(contentDetections[1], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content13Matrix = createPart1Matrix(contentDetections[2], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content14Matrix = createPart1Matrix(contentDetections[3], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);

    cv::Mat contentPart1Matrix;
    std::vector<cv::Mat> matricesPart1 = { content11Matrix, content12Matrix, content13Matrix, content14Matrix };
    cv::vconcat(matricesPart1, contentPart1Matrix);

    // Process content part 2 matrix
    cv::Mat content21Matrix = createPart2Matrix(contentDetections[4], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content22Matrix = createPart2Matrix(contentDetections[5], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content23Matrix = createPart2Matrix(contentDetections[6], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content24Matrix = createPart2Matrix(contentDetections[7], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    
    cv::Mat contentPart2Matrix;
    std::vector<cv::Mat> matricesPart2 = { content21Matrix, content22Matrix, content23Matrix, content24Matrix };
    cv::hconcat(matricesPart2, contentPart2Matrix);
    
    // Extract answers
    std::string studentId = getStudentId(studentIdMatrix);
    std::string examId = getExamId(examIdMatrix);
    std::vector<std::string> contentPart1Answers = processContentPart1(contentPart1Matrix);
    std::vector<std::string> contentPart2Answers = processContentPart2(contentPart2Matrix);

    std::vector<std::string> result;
    result.push_back(imageBasename);
    result.push_back(studentId);
    result.push_back(examId);
    result.push_back("Answers");
    result.insert(result.end(), contentPart1Answers.begin(), contentPart1Answers.end());
    result.insert(result.end(), contentPart2Answers.begin(), contentPart2Answers.end());

    // Grade exam
    auto answerKeyIt = examAnswerKeys.find(examId);
    if (answerKeyIt != examAnswerKeys.end()) {
      const std::vector<std::string>& correctAnswers = answerKeyIt->second;
      ExamGradingResult gradingResult = gradeStudentExam(result, correctAnswers);
      
      result.push_back(std::to_string(gradingResult.part1CorrectCount));  
      result.push_back(std::to_string(gradingResult.part2CorrectCount)); 
      result.push_back(std::to_string(gradingResult.totalPoints));       
    } else {
      result.push_back("N/A");  
      result.push_back("N/A");  
      result.push_back("N/A"); 
    }

    return result;
  } catch (const std::exception& e) {
    Logger::error("EXAM GRADER", "Answer extraction and grading failed for " + imageBasename + ": " + std::string(e.what()));
    return {};
  }
}

std::vector<std::string> ExamGrader::regradeExamFromCsv(
  const std::string& imageBasename,
  const std::vector<std::string>& studentAnswers,
  const std::map<std::string, std::vector<std::string>>& examAnswerKeys) 
{
  try {
    // Extract exam ID from student answers (should be at index 2)
    if (studentAnswers.size() < 3) {
      return {};
    }
    
    std::string examId = studentAnswers[2]; 
    
    // Find the answer key for this exam ID
    auto answerKeyIt = examAnswerKeys.find(examId);
    if (answerKeyIt == examAnswerKeys.end()) {
      std::vector<std::string> result = studentAnswers;
      if (result.size() >= 3) {
        // Add or update grading results
        while (result.size() < studentAnswers.size() + 3) {
          result.push_back("N/A");
        }
        result[result.size() - 3] = "N/A"; 
        result[result.size() - 2] = "N/A"; 
        result[result.size() - 1] = "N/A";  
      }
      return result;
    }
    
    const std::vector<std::string>& correctAnswers = answerKeyIt->second;
    
    // Perform grading using existing grading logic
    ExamGradingResult gradingResult = gradeStudentExam(studentAnswers, correctAnswers);
    
    // Prepare result with updated scores
    std::vector<std::string> result = studentAnswers;
    
    // Ensure result has space for grading scores
    while (result.size() < studentAnswers.size() + 3) {
      result.push_back("0");
    }
    
    // Update the last 3 elements with new grading results
    result[result.size() - 3] = std::to_string(gradingResult.part1CorrectCount); 
    result[result.size() - 2] = std::to_string(gradingResult.part2CorrectCount);  
    result[result.size() - 1] = std::to_string(gradingResult.totalPoints);        
    
    return result;
    
  } catch (const std::exception& e) {
    return studentAnswers;
  }
}

ExamGrader::ExamGradingResult ExamGrader::gradeStudentExam(
  const std::vector<std::string>& studentAnswers,
  const std::vector<std::string>& correctAnswers) 
{
  ExamGradingResult result;
  result.part1CorrectCount = 0;
  result.part2CorrectCount = 0;
  result.totalPoints = 0;
  
  // Grade Part 1 
  for (int i = 0; i < PART_1_NUM_QUESTIONS; i++) {
    QuestionResult qResult;
    
    int studentAnswerIndex = PART_1_START_INDEX + i;
    std::string studentAns = studentAnswers[studentAnswerIndex];
    std::string correctAns = correctAnswers[i];
    
    qResult.studentAnswer = studentAns;
    qResult.correctAnswer = correctAns;
    qResult.isCorrect = false;
    
    if (studentAns != "_" && studentAns != "X" && !correctAns.empty()) {
      std::string studentUpper = studentAns;
      std::transform(studentUpper.begin(), studentUpper.end(), studentUpper.begin(), ::toupper);
      
      if (studentUpper == correctAns) {
        qResult.isCorrect = true;
        result.part1CorrectCount++;
      }
    }
    result.part1Results.push_back(qResult);
  }
  
  // Grade Part 2 
  for (int i = 0; i < PART_2_NUM_QUESTIONS; i++) {
    QuestionResult qResult;
    
    int studentAnswerIndex = PART_2_START_INDEX + i; 
    int correctAnswerIndex = PART_1_NUM_QUESTIONS + i;
    
    std::string studentAns = studentAnswers[studentAnswerIndex];
    std::string correctAns = correctAnswers[correctAnswerIndex];
    
    qResult.studentAnswer = studentAns;
    qResult.correctAnswer = correctAns;
    qResult.isCorrect = false;
    
    if (!studentAns.empty() && !correctAns.empty() &&
      studentAns.find('_') == std::string::npos &&
      studentAns.find('X') == std::string::npos) {
      
      std::string studentUpper = studentAns;
      std::transform(studentUpper.begin(), studentUpper.end(), studentUpper.begin(), ::toupper);
      
      if (studentUpper == correctAns) {
        qResult.isCorrect = true;
        result.part2CorrectCount++;
      }
    }
    result.part2Results.push_back(qResult);
  }
  
  result.totalPoints = result.part1CorrectCount + result.part2CorrectCount;
  return result;
}