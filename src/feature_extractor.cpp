#include "include/feature_extractor.h"

FeatureExtractor::FeatureExtractor() {
    orb_ = cv::ORB::create(1,         // nfeatures
        1.2f,      // scaleFactor (default)
        8,         // nlevels (default)
        31,        // edgeThreshold (default)
        0,         // firstLevel (default)
        2,         // WTA_K (default)
        cv::ORB::FAST_SCORE, // scoreType
        31,        // patchSize (default)
        20);       // fastThreshold (default)
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> FeatureExtractor::Forward(const cv::Mat& frame) const {
    // Check that the image depth is CV_8U (8-bit unsigned)
    if (frame.depth() != CV_8U) {
        throw std::invalid_argument("Frame depth is not CV_8U");
    }

    cv::Mat frame_gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    } else {
        frame_gray = frame;
    }

    // Detect good features to track in the grayscale image.
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(frame_gray,
                            points,
                            kMaxCorners,
                            kQualityLvl,
                            kMinDistance,
                            cv::Mat(),     // no mask
                            kBlockSize);

    // If no points were found, return empty results.
    if (points.empty()) {
        return std::make_pair(std::vector<cv::KeyPoint>(), cv::Mat());
    }

    // Convert detected points into KeyPoints with a fixed size (30).
    std::vector<cv::KeyPoint> keypoints;
    for (const auto& pt : points) {
        keypoints.emplace_back(pt, 30.0f);
    }

    cv::Mat descriptors;
    // Compute ORB descriptors for the detected keypoints.
    orb_->compute(frame, keypoints, descriptors);

    return std::make_pair(keypoints, descriptors);
}