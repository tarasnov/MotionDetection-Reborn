#include <utility>

#include "include/motion_detection.h"


MovementDetection::BufferEntry::BufferEntry(TimePoint timestamp, cv::Mat frame, Keypoints keypoints,
                                            cv::Mat descriptors)
    : timestamp(timestamp)
      , frame(std::move(frame))
      , keypoints(std::move(keypoints))
      , descriptors(std::move(descriptors)) {
}

MovementDetection::MovementDetection(const std::string &model_path, std::chrono::milliseconds window_duration,
                                     float threshold)
    : window_duration_(window_duration)
      , threshold_(threshold)
      , model_(model_path) {
}


std::vector<cv::Point2d> MovementDetection::Process() const {
    if (buffer_.size() < kNnInputFrames) {
        return {};
    }

    const auto aligned_frames = GetAlignedFrames();

    if (aligned_frames.empty()) {
        return {};
    }

    static cv::Mat x_input; 
    cv::dnn::blobFromImages(aligned_frames, x_input, 1.0, cv::Size(), cv::Scalar(), false, false, CV_8U);
    const auto detections = MotionDetection(std::move(x_input));

    return {};
}

std::vector<cv::Mat> MovementDetection::GetAlignedFrames() const {
    std::vector aligned_frames = {buffer_.back().frame};

    int height = buffer_[0].frame.rows;
    int width = buffer_[0].frame.cols;

    int step = buffer_.size() / kNnInputFrames;
    for (int i = buffer_.size() - step - 1; aligned_frames.size() < kNnInputFrames; i -= step) {
        auto M = CalcTransform(buffer_[i].keypoints, buffer_[i].descriptors, buffer_.back().keypoints,
                               buffer_.back().descriptors);

        if (M.empty()) {
            return {};
        }

        const auto &frame = buffer_[i].frame;
        auto &warped_frame = aligned_frames.emplace_back();
        cv::warpAffine(frame, warped_frame, M, cv::Size(width, height));
    }

    return aligned_frames;
}

std::vector<cv::Point2d> MovementDetection::MotionDetection(cv::Mat &&x_input) const {
    // x_input is assumed to be a 4D blob with shape [N, C, H, W].
    // Get the original dimensions.
    CV_Assert(x_input.dims == 4);
    int N = x_input.size[0];
    int C = x_input.size[1];
    int H = x_input.size[2];
    int W = x_input.size[3];

    // Reshape to [1, (N * C), H, W]:
    int new_channels = N * C;
    int new_sizes[4] = {1, new_channels, H, W};
    // Use 0 for the type parameter to keep the original type.
    cv::Mat reshaped = x_input.reshape(0, 4, new_sizes);

    // Normalize: convert to float and scale by 1/255.
    static cv::Mat x_normalized;
    reshaped.convertTo(x_normalized, CV_32F, 1.0 / 255.0);

    return model_.Forward(std::move(x_normalized));
}


cv::Mat MovementDetection::CalcTransform(const Keypoints &keypoints, const cv::Mat &descriptors,
                                         const Keypoints &to_keypoints, const cv::Mat &to_descriptors) const {
    // Obtain matches between the current frame and the "to" frame.
    // The matchTo function should behave similar to your Python version.
    auto matches = matcher_.MatchTo(keypoints, descriptors, to_keypoints, to_descriptors);

    if (matches.size() < static_cast<size_t>(kMinMatches)) {
        return {};
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    points1.reserve(matches.size());
    points2.reserve(matches.size());

    // Unzip the matches into two vectors of point coordinates.
    for (const auto &match: matches) {
        points1.push_back(match.first.pt);
        points2.push_back(match.second.pt);
    }

    // Estimate an affine partial 2D transformation using RANSAC.
    cv::Mat inliers;
    cv::Mat M = cv::estimateAffinePartial2D(points1, points2, inliers, cv::RANSAC);

    // Check if a valid transformation was found and that the number of inliers is sufficient.
    if (!M.empty() && cv::countNonZero(inliers) >= static_cast<int>(kMinMatches * 0.5)) {
        return M;
    }

    return {};
}


cv::Mat MovementDetection::PushFrame(const cv::Mat &frame, TimePoint timestamp, bool skip_detection) {
    while (!buffer_.empty() && buffer_.front().timestamp + window_duration_ < buffer_.back().timestamp) {
        buffer_.pop_front();
    }

    auto [keypoints, descriptors] = extractor_.Forward(frame);
    buffer_.emplace_back(timestamp, frame, std::move(keypoints), std::move(descriptors));

    std::vector<cv::Point2d> detections;
    if (!skip_detection && buffer_.back().timestamp - buffer_.front().timestamp >= window_duration_) {
        detections = Process();
    }

    return {};
}
