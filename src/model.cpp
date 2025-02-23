#include "include/model.h"

Model::Model(std::string) {
}

std::vector<cv::Point2d> Model::Forward(cv::Mat &&x_input) const {
  for (int i = 0; i < x_input.dims; ++i) {
    std::cout << x_input.size[i] << (i < x_input.dims - 1 ? " x " : "\n");
  }

  printf("\n");
  return {};
}
