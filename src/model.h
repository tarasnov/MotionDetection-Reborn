#pragma once

#include <string>
#include <opencv2/opencv.hpp>


class Model {
public:
    explicit Model(std::string model_path);

    std::vector<cv::Point2d> Forward(cv::Mat&& x_input) const;
private:
};
