#pragma once

#include <string>
#include <chrono>
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include "feature_extractor.h"
#include "keypoint_matcher.h"
#include "model.h"


class MovementDetection {
public:
    using Keypoints = std::vector<cv::KeyPoint>;
    using TimePoint = std::chrono::steady_clock::time_point;

    MovementDetection(const std::string& model_path, std::chrono::milliseconds window_duration, float threshold = 0.5);
    
    cv::Mat PushFrame(const cv::Mat& frame, TimePoint timestamp, bool skip_detection = false);
private:
    std::vector<cv::Point2d> Process() const;

    struct BufferEntry {
        TimePoint timestamp;
        cv::Mat frame;
        Keypoints keypoints;
        cv::Mat descriptors;

        BufferEntry(TimePoint timestamp, const cv::Mat& frame, Keypoints keypoints, cv::Mat descriptors);
    };

    std::vector<cv::Mat> GetAlignedFrames() const;
    cv::Mat CalcTransform(const Keypoints& keypoints, const cv::Mat& descriptors, const Keypoints& to_keypoints, const cv::Mat& to_descriptors) const;
    std::vector<cv::Point2d> MotionDetection(cv::Mat&& x_input) const;

    static constexpr int kMaxMatchFails = 3; // max number of matcher fails before emptying the buffer
    static constexpr int kNnInputFrames = 5; // number of frames that NN requires to perform movement detection
    static constexpr int kMinMatches = 15; // min matches between frames

    const std::chrono::milliseconds window_duration_;
    const float threshold_;

    std::deque<BufferEntry> buffer_;

    const FeatureExtractor extractor_;
    const KeypointMatcher matcher_;
    const Model model_;
};
