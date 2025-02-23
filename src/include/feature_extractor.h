#pragma once

#include <opencv2/opencv.hpp>


class FeatureExtractor {
public:
    FeatureExtractor();

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> Forward(const cv::Mat &frame) const;

private:
    static constexpr int kMaxCorners = 1000;
    static constexpr float kQualityLvl = 0.025;
    static constexpr float kMinDistance = 5.0;
    static constexpr int kBlockSize = 7;

    cv::Ptr<cv::ORB> orb_;
};
