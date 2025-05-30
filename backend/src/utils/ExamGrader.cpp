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

// Constants for matrix sizes
const int STUDENT_ID_NUM_CENTER_X = 9;
const int STUDENT_ID_NUM_CENTER_Y = 10;
const int EXAM_ID_NUM_CENTER_X = 3;
const int EXAM_ID_NUM_CENTER_Y = 10;

const int PART_1_NUM_CENTER_X = 4;
const int PART_1_NUM_CENTER_Y = 4;
const int PART_2_NUM_CENTER_X = 4;
const int PART_2_NUM_CENTER_Y = 6;

// Constants for grading
const int PART_1_START_INDEX = 4;
const int PART_2_START_INDEX = 20;
const int PART_1_NUM_QUESTIONS = 16;
const int PART_1_MAX_SCORE = 16;
const int PART_2_NUM_QUESTIONS = 8;
const int PART_2_MAX_SCORE = 8;
const int TOTAL_NUM_QUESTIONS = PART_1_NUM_QUESTIONS + PART_2_NUM_QUESTIONS;
const int TOTAL_MAX_SCORE = PART_1_MAX_SCORE + PART_2_MAX_SCORE;

cv::Mat ExamGrader::createMetadataMatrix(const std::vector<Detection>& detections,
                                         int numRows, int numCols) {
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);
  
  if (detections.empty()) {
    return matrix;
  }
  
  // Use K-means to assign detections to grid positions instead of sorting
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  // Find bounding box of all detections
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }
  
  // Calculate grid step sizes
  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));
  
  // Group detections by column using K-means assignment
  std::vector<std::vector<std::pair<int, Detection>>> columnCandidates(numCols);
  
  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;
    
    // Find nearest grid position
    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;
    
    // Clamp to valid range
    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));
    
    columnCandidates[col].push_back({row, detections[i]});
  }
  
  // Keep your existing logic for processing each column
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
  
  // Use K-means to assign detections to grid positions instead of sorting
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  // Find bounding box of all detections
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }
  
  // Calculate grid step sizes
  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));
  
  // Group detections by row using K-means assignment
  std::vector<std::vector<std::pair<int, Detection>>> rowCandidates(numRows);
  
  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;
    
    // Find nearest grid position
    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;
    
    // Clamp to valid range
    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));
    
    rowCandidates[row].push_back({col, detections[i]});
  }
  
  // Keep your existing logic for processing each row
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
  return matrix;
}

cv::Mat ExamGrader::createPart2Matrix(const std::vector<Detection>& detections,
                                      int numRows, int numCols) {       
  cv::Mat matrix = cv::Mat::zeros(numRows, numCols, CV_8UC1);
  
  if (detections.empty()) {
    return matrix;
  }
  
  // Use K-means to assign detections to grid positions instead of sorting
  std::vector<cv::Point2f> centers;
  for (const auto& detection : detections) {
    centers.push_back(cv::Point2f(detection.box.x + detection.box.width/2, 
                                  detection.box.y + detection.box.height/2));
  }
  
  // Find bounding box of all detections
  float minX = centers[0].x, maxX = centers[0].x;
  float minY = centers[0].y, maxY = centers[0].y;
  for (const auto& center : centers) {
    minX = std::min(minX, center.x);
    maxX = std::max(maxX, center.x);
    minY = std::min(minY, center.y);
    maxY = std::max(maxY, center.y);
  }
  
  // Calculate grid step sizes
  float stepX = (maxX - minX) / std::max(1.0f, (float)(numCols - 1));
  float stepY = (maxY - minY) / std::max(1.0f, (float)(numRows - 1));
  
  // Group detections by row using K-means assignment
  std::vector<std::vector<std::pair<int, Detection>>> rowCandidates(numRows);
  
  for (int i = 0; i < detections.size(); i++) {
    float x = centers[i].x;
    float y = centers[i].y;
    
    // Find nearest grid position
    int col = stepX > 0 ? std::round((x - minX) / stepX) : 0;
    int row = stepY > 0 ? std::round((y - minY) / stepY) : 0;
    
    // Clamp to valid range
    col = std::max(0, std::min(col, numCols - 1));
    row = std::max(0, std::min(row, numRows - 1));
    
    rowCandidates[row].push_back({col, detections[i]});
  }
  
  // Keep your existing submatrix processing logic
  int numSubmatrices = 2;
  int subCols = 2;
  for (int sm = 0; sm < numSubmatrices; ++sm) {
    int colStart = sm * subCols;
    for (int row = 0; row < numRows; ++row) {
      std::vector<std::pair<int, Detection>> subRowCandidates;
      for (const auto& cand : rowCandidates[row]) {
        if (cand.first >= colStart && cand.first < colStart + subCols)
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
  
  const int numSubmatrices = 8;
  const int submatrixCols = 2;
  const int rows = array.rows;
  const uchar* data = array.data;
  int step = static_cast<int>(array.step[0]);
    
  for (int i = 0; i < numSubmatrices; i++) {
    std::string eachAnswer;
    int colStart = i * submatrixCols;
    for (int r = 0; r < rows; r++) {
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

std::vector<std::string> ExamGrader::extractAnswers(
  const std::string& imageBasename,
  const std::vector<std::vector<Detection>>& metadataDetections,
  const std::vector<std::vector<Detection>>& contentDetections) 
{
  try {
    // Process metadata 
    cv::Mat studentIdMatrix = createMetadataMatrix(metadataDetections[0], STUDENT_ID_NUM_CENTER_Y, STUDENT_ID_NUM_CENTER_X);
    cv::Mat examIdMatrix = createMetadataMatrix(metadataDetections[1], EXAM_ID_NUM_CENTER_Y, EXAM_ID_NUM_CENTER_X);

    // Process content part 1 to a 16x4 matrix
    cv::Mat content11Matrix = createPart1Matrix(contentDetections[0], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content12Matrix = createPart1Matrix(contentDetections[1], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content13Matrix = createPart1Matrix(contentDetections[2], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);
    cv::Mat content14Matrix = createPart1Matrix(contentDetections[3], PART_1_NUM_CENTER_Y, PART_1_NUM_CENTER_X);

    cv::Mat contentPart1Matrix;
    std::vector<cv::Mat> matricesPart1 = { content11Matrix, content12Matrix, content13Matrix, content14Matrix };
    cv::vconcat(matricesPart1, contentPart1Matrix);

    // Process content part 2 to a 6x16 matrix
    cv::Mat content21Matrix = createPart2Matrix(contentDetections[4], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content22Matrix = createPart2Matrix(contentDetections[5], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content23Matrix = createPart2Matrix(contentDetections[6], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    cv::Mat content24Matrix = createPart2Matrix(contentDetections[7], PART_2_NUM_CENTER_Y, PART_2_NUM_CENTER_X);
    
    cv::Mat contentPart2Matrix;
    std::vector<cv::Mat> matricesPart2 = { content21Matrix, content22Matrix, content23Matrix, content24Matrix };
    cv::hconcat(matricesPart2, contentPart2Matrix);
    
    // Process exam to a vector of strings
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

    return result;
  } catch (const std::exception& e) {
    Logger::error("EXAM GRADER", "Answer extraction failed for " + imageBasename + ": " + std::string(e.what()));
    return {};
  }
}

ExamGrader::ExamGradingResult ExamGrader::gradeStudentExam(
  const std::vector<std::string>& studentAnswers,
  const std::vector<std::string>& correctAnswers,
  const std::vector<int>& pointValues
) {
  ExamGradingResult result;
  
  // Initialize counters
  result.part1CorrectCount = 0;
  result.part1TotalPoints = 0;
  result.part2CorrectCount = 0;
  result.part2TotalPoints = 0;
  result.totalCorrectCount = 0;
  result.totalPoints = 0;
  
  // Student answers start from index 3 (skip image name, student ID, exam ID)
  const int answerStartIndex = 4;
  
  // Grade Part 1 questions (16 questions, multiple choice A-D)
  for (int i = 0; i < 16; i++) {
    QuestionResult qResult;
    
    int studentAnswerIndex = answerStartIndex + i;
    std::string studentAns = (studentAnswerIndex < studentAnswers.size()) ? 
                           studentAnswers[studentAnswerIndex] : "_";
    std::string correctAns = (i < correctAnswers.size()) ? correctAnswers[i] : "";
    int points = (i < pointValues.size()) ? pointValues[i] : 1;
    
    qResult.studentAnswer = studentAns;
    qResult.correctAnswer = correctAns;
    qResult.pointsEarned = 0;
    qResult.isCorrect = false;
    
    // Skip grading if student didn't answer or marked multiple/invalid
    if (studentAns != "_" && studentAns != "X" && !correctAns.empty()) {
      // Convert to uppercase for comparison (handles suggested answers in lowercase)
      std::string studentUpper = studentAns;
      std::string correctUpper = correctAns;
      std::transform(studentUpper.begin(), studentUpper.end(), studentUpper.begin(), ::toupper);
      std::transform(correctUpper.begin(), correctUpper.end(), correctUpper.begin(), ::toupper);
      
      if (studentUpper == correctUpper) {
        qResult.isCorrect = true;
        qResult.pointsEarned = points;
        result.part1CorrectCount++;
        result.part1TotalPoints += points;
      }
    }
    
    result.part1Results.push_back(qResult);
  }
  
  // Grade Part 2 questions (8 questions, D/S format)
  for (int i = 0; i < 8; i++) {
    QuestionResult qResult;
    
    int studentAnswerIndex = answerStartIndex + 16 + i; // After Part 1 questions
    int correctAnswerIndex = 16 + i; // After Part 1 in answer key
    
    std::string studentAns = (studentAnswerIndex < studentAnswers.size()) ? 
                           studentAnswers[studentAnswerIndex] : "_";
    std::string correctAns = (correctAnswerIndex < correctAnswers.size()) ? 
                           correctAnswers[correctAnswerIndex] : "";
    int points = (correctAnswerIndex < pointValues.size()) ? pointValues[correctAnswerIndex] : 1;
    
    qResult.studentAnswer = studentAns;
    qResult.correctAnswer = correctAns;
    qResult.pointsEarned = 0;
    qResult.isCorrect = false;
    
    // Part 2 answers are strings like "DSDSDS" - must match exactly
    if (!studentAns.empty() && !correctAns.empty() && 
        studentAns != "_" && studentAns != "X") {
      
      // Convert both to uppercase for comparison
      std::string studentUpper = studentAns;
      std::string correctUpper = correctAns;
      std::transform(studentUpper.begin(), studentUpper.end(), studentUpper.begin(), ::toupper);
      std::transform(correctUpper.begin(), correctUpper.end(), correctUpper.begin(), ::toupper);
      
      // For Part 2, check if the length matches and all characters match
      if (studentUpper.length() == correctUpper.length()) {
        bool allMatch = true;
        for (size_t j = 0; j < studentUpper.length(); j++) {
          // Allow '_' in student answer (unfilled) but don't count as wrong
          if (studentUpper[j] != correctUpper[j] && studentUpper[j] != '_') {
            allMatch = false;
            break;
          }
        }
        
        if (allMatch) {
          qResult.isCorrect = true;
          qResult.pointsEarned = points;
          result.part2CorrectCount++;
          result.part2TotalPoints += points;
        }
      }
    }
    
    result.part2Results.push_back(qResult);
  }
  
  // Calculate totals
  result.totalCorrectCount = result.part1CorrectCount + result.part2CorrectCount;
  result.totalPoints = result.part1TotalPoints + result.part2TotalPoints;
  
  return result;
}