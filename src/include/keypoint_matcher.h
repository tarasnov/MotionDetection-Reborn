#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <utility>

class KeypointMatcher {
public:
    // matchTo performs KNN matching between current and previous frame descriptors.
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint> > MatchTo(
        const std::vector<cv::KeyPoint> &from_keypoints,
        const cv::Mat &from_descriptors,
        const std::vector<cv::KeyPoint> &to_keypoints,
        const cv::Mat &to_descriptors) const;

private:
    // BFMatcher object for matching descriptors.
    cv::BFMatcher bf_{cv::BFMatcher(cv::NORM_HAMMING)};

    // Storage for the last frame's keypoints and descriptors.
    bool has_last_{false};
    std::vector<cv::KeyPoint> last_keypoints_;
    cv::Mat last_descriptors_;
};
