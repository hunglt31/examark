#ifndef EXAM_GRADER_H
#define EXAM_GRADER_H

#include "Detection.h"
#include "utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class ExamGrader {
public:
  /**
   * @brief Processes the exam by extracting student ID, exam ID, and answers.
   *
   * The function takes metadata and content detections, processes them into matrices,
   * and extracts the student ID, exam ID, and answers for both parts of the assignment.
   *
   * @param imageBasename The base name of the image file (without extension).
   * @param metadataDetections A vector of metadata detections.
   * @param contentDetections A vector of content detections.
   * @return std::vector<std::string> A vector containing the student ID, exam ID, answers and score.
   */
  std::vector<std::string>
  extract_answers_from_detections(const std::string &imageBasename,
                                  const std::vector<std::vector<Detection>> &metadataDetections,
                                  const std::vector<std::vector<Detection>> &contentDetections);

  struct QuestionResult {
    std::string studentAnswer;
    std::string correctAnswer;
    bool isCorrect;
  };

  struct ExamGradingResult {
    std::vector<QuestionResult> part1Results;
    std::vector<QuestionResult> part2Results;

    int part1CorrectCount;
    int part2CorrectCount;
    int totalPoints;
  };

  /**
   * @brief Grades the exam based on student answers and the answer key.
   *
   * The function compares the student's answers with the answer key and calculates
   * the total score, maximum score, and correctness for each part of the exam.
   *
   * @param studentAnswers A vector of strings representing the student's answers.
   * @param answerKey A vector of strings representing the correct answers.
   * @return GradingResult A struct containing the grading results.
   */
  ExamGradingResult gradeStudentExam(const std::vector<std::string> &studentAnswers,
                                     const std::vector<std::string> &answerKey);

  /**
   * @brief Re-grades an exam using existing CSV data without extracting answers
   *
   * Takes pre-extracted student answers from CSV and re-grades them against
   * the answer key. This is used for re-grading functionality where answers
   * have already been extracted and potentially modified.
   *
   * @param studentAnswers Vector containing the student's answers from CSV
   * @param examAnswerKeys Map of exam IDs to their corresponding answer keys
   * @return Vector of strings containing the re-graded results
   */
  std::vector<std::string>
  extractAnswersAndRegradeExam(const std::vector<std::string> &studentAnswers,
                               const std::map<std::string, std::vector<std::string>> &examAnswerKeys);

private:
  /**
   * @brief Create matrix for student ID and exam ID.
   *
   * The matrix is of size (numCenterY, numCenterX) and is filled with 0s.
   * For each detection, the center coordinates are calculated and the
   * corresponding cell in the matrix is set to 1 if the classId is 0.
   *
   * @param detections A vector of Detection objects.
   * @param numCenterX Number of columns in the matrix.
   * @param numCenterY Number of rows in the matrix.
   * @return cv::Mat A binary matrix of size (numCenterY, numCenterX).
   */
  cv::Mat createMetadataMatrix(const std::vector<Detection> &detections, int numRows, int numCols);

  /**
   * @brief Create matrix for content part 1.
   *
   * The matrix is of size (numCenterY, numCenterX) and is filled with 0s.
   * For each detection, the center coordinates are calculated and the
   * corresponding cell in the matrix is set to 1 if the classId is 0.
   *
   * @param detections A vector of Detection objects.
   * @param numCenterX Number of columns in the matrix.
   * @param numCenterY Number of rows in the matrix.
   * @return cv::Mat A binary matrix of size (numCenterY, numCenterX).
   */
  cv::Mat createPart1Matrix(const std::vector<Detection> &detections, int numRows, int numCols);

  /**
   * @brief Create matrix for content part 2.
   *
   * The matrix is of size (numCenterY, numCenterX) and is filled with 0s.
   * For each detection, the center coordinates are calculated and the
   * corresponding cell in the matrix is set to 1 if the classId is 0.
   *
   * @param detections A vector of Detection objects.
   * @param numCenterX Number of columns in the matrix.
   * @param numCenterY Number of rows in the matrix.
   * @return cv::Mat A binary matrix of size (numCenterY, numCenterX).
   */
  cv::Mat createPart2Matrix(const std::vector<Detection> &detections, int numRows, int numCols);

  /**
   * @brief Extracts the student ID from a 10x9 matrix.
   *
   * The matrix is assumed to be of type CV_8UC1 with 10 rows (digits 0-9) and 9 columns.
   * For each column (representing one digit of the student ID), the function scans its rows
   * in order (0 to 9) to find a cell with value 1. The row index is interpreted as the digit.
   * If no cell is found with value 1 in a column, "X" is appended to the result.
   *
   * @param matrix A cv::Mat of size (10, 9) of type CV_8UC1.
   * @return std::string The extracted student ID.
   */
  std::string getStudentId(const cv::Mat &matrix);

  /**
   * @brief Extracts the exam ID from a 10x3 binary matrix.
   *
   * The matrix is assumed to be of type CV_8UC1 with 10 rows and 3 columns.
   * For each column (representing one digit of the exam ID), the function scans its rows
   * in order (0 to 9) to find a cell with value 1. The row index is interpreted as the digit.
   * If no cell is found with value 1 in a column, "X" is appended to the result.
   *
   * @param matrix A cv::Mat of size (10, 3) of type CV_8UC1.
   * @return std::string The extracted exam ID.
   */
  std::string getExamId(const cv::Mat &matrix);

  /**
   * @brief Processes a 16x4 binary matrix for part 1 of the assignment.
   *
   * Each row represents a question and each column represents an answer option (A, B, C, D).
   * If exactly one option is marked as 1, the corresponding answer (A, B, C, or D) is recorded.
   * Otherwise, "X" is marked for invalid answers.
   *
   * @param array A cv::Mat of size (16, 4) of type CV_8UC1.
   * @return std::vector<std::string> A vector of answers, e.g., {"A", "B", "C", ...}.
   */
  std::vector<std::string> processContentPart1(const cv::Mat &array);

  /**
   * @brief Processes a 6x16 binary matrix for part 2 of the assignment.
   *
   * The matrix is divided into 8 submatrices of size 6x2.
   * Each submatrix represents a question and each row represents an answer option.
   * For each row in a submatrix:
   *   - If the value in the first column is 1, append "Đ" to the answer string.
   *   - Else if the value in the second column is 1, append "S" to the answer string.
   *   - Otherwise, append "X" (invalid).
   *
   * @param array A cv::Mat of size (6, 16) of type CV_8UC1.
   * @return std::vector<std::string> A vector with the processed answer strings for each submatrix.
   */
  std::vector<std::string> processContentPart2(const cv::Mat &array);
};
#endif // EXAM_GRADER_H
