#include <utils.hpp>
#include <set>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


// Function to find good matches using Lowe's ratio test
static void loweMatch(const std::vector<std::vector<cv::DMatch>>& knn_matches,
                      std::set<cv::DMatch>& good_matches,
                      float threshold) {
    for (const auto& match_pair : knn_matches) {
        if (match_pair[0].distance < threshold * match_pair[1].distance) {
            good_matches.insert(match_pair[0]);
        }
    }
}

cv::Mat alignImages(const cv::Mat& scanned_img, const cv::Mat& ref_img_gray) {
    cv::Mat scan_img_gray;
    cv::cvtColor(scanned_img, scan_img_gray, cv::COLOR_BGR2GRAY);

    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> scan_kp, ref_kp;
    cv::Mat scan_desc, ref_desc;

    sift->detectAndCompute(scan_img_gray, cv::noArray(), scan_kp, scan_desc);
    sift->detectAndCompute(ref_img_gray, cv::noArray(), ref_kp, ref_desc);

    if (scan_desc.empty() || ref_desc.empty()) {
        throw std::runtime_error("Failed to compute SIFT descriptors.");
    }

    cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                                cv::makePtr<cv::flann::SearchParams>(50));
    std::vector<std::vector<cv::DMatch>> knn_matches;
    flann.knnMatch(scan_desc, ref_desc, knn_matches, 2);

    std::set<cv::DMatch> good_matches;
    float threshold = 0.3f;
    while (good_matches.size() < 4) {
        good_matches.clear();
        loweMatch(knn_matches, good_matches, threshold);
        if (threshold < 1.0f) {
            threshold += 0.1f;
        } else {
            throw std::runtime_error("Cannot find enough good matches.");
        }
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(scan_kp[m.queryIdx].pt);
        pts2.push_back(ref_kp[m.trainIdx].pt);
    }

    cv::Mat h = cv::findHomography(pts1, pts2, cv::RANSAC, 5.0);
    if (h.empty()) {
        throw std::runtime_error("Homography estimation failed.");
    }

    cv::Mat aligned;
    cv::warpPerspective(scanned_img, aligned, h, cv::Size(2480, 3508));
    return aligned;
}