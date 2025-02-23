#include "include/keypoint_matcher.h"


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <utility>

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> KeypointMatcher::Forward(
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors)
{
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> valid_matches;

    // If input keypoints or descriptors are empty, clear the stored last frame.
    if (keypoints.empty() || descriptors.empty()) {
        hasLast = false;
        lastKeypoints.clear();
        lastDescriptors.release();
        return valid_matches;
    }

    // If we have a previous frame, perform matching.
    if (hasLast) {
        valid_matches = MatchTo(keypoints, descriptors, lastKeypoints, lastDescriptors);
    }

    // Save the current frame's keypoints and descriptors for the next call.
    lastKeypoints = keypoints;                // Copy keypoints
    descriptors.copyTo(lastDescriptors);      // Copy descriptors
    hasLast = true;

    return valid_matches;
}
    

// matchTo performs KNN matching between current and previous frame descriptors.
std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> KeypointMatcher::MatchTo(
    const std::vector<cv::KeyPoint>& from_keypoints,
    const cv::Mat& from_descriptors,
    const std::vector<cv::KeyPoint>& to_keypoints,
    const cv::Mat& to_descriptors) const
{
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> valid_matches;

    // Check for empty inputs.
    if (from_keypoints.empty() || from_descriptors.empty() ||
        to_keypoints.empty() || to_descriptors.empty())
    {
        return valid_matches;
    }

    // Perform knnMatch with k=2.
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(from_descriptors, to_descriptors, knn_matches, 2);

    // Iterate over all matches.
    for (const auto& match_pair : knn_matches) {
        // Skip if there are not exactly 2 matches.
        if (match_pair.size() != 2)
            continue;

        const cv::DMatch& m = match_pair[0];
        const cv::DMatch& n = match_pair[1];

        // Apply the ratio test.
        if (m.distance < 0.75f * n.distance) {
            const cv::KeyPoint& kpt1 = from_keypoints[m.queryIdx];
            const cv::KeyPoint& kpt2 = to_keypoints[m.trainIdx];

            // Check Euclidean distance between the matched keypoints.
            if (cv::norm(kpt1.pt - kpt2.pt) < 160.0)
                valid_matches.emplace_back(kpt1, kpt2);
        }
    }

    return valid_matches;
}