#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <utility>

class KeypointMatcher {
public:
    // Constructor initializes BFMatcher with NORM_HAMMING.
    KeypointMatcher() : bf(cv::NORM_HAMMING), hasLast(false) {}

    // forward takes keypoints and descriptors from the current frame,
    // matches them with the previous frame if available, stores the current
    // frame as the "last" frame, and returns valid keypoint matches.
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> Forward(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& descriptors);

    // matchTo performs KNN matching between current and previous frame descriptors.
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> MatchTo(
        const std::vector<cv::KeyPoint>& from_keypoints,
        const cv::Mat& from_descriptors,
        const std::vector<cv::KeyPoint>& to_keypoints,
        const cv::Mat& to_descriptors) const;

private:
    // BFMatcher object for matching descriptors.
    cv::BFMatcher bf;

    // Storage for the last frame's keypoints and descriptors.
    bool hasLast;
    std::vector<cv::KeyPoint> lastKeypoints;
    cv::Mat lastDescriptors;
};
