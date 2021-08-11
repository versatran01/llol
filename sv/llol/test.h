#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

inline cv::Mat MakeScan(cv::Size size) {
  cv::Mat sweep = cv::Mat::zeros(size, CV_32FC4);

  const float azim_delta = M_PI * 2 / size.width;
  const float elev_max = M_PI_4 / 2;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < sweep.rows; ++i) {
    for (int j = 0; j < sweep.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = M_PI * 2 - j * azim_delta;

      auto& xyzr = sweep.at<cv::Vec4f>(i, j);
      xyzr[0] = std::cos(elev) * std::cos(azim);
      xyzr[1] = std::cos(elev) * std::sin(azim);
      xyzr[2] = std::sin(elev);
      xyzr[3] = 1;
    }
  }

  return sweep;
}

}  // namespace sv
