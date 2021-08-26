#include "sv/llol/scan.h"

#include <fmt/core.h>
#include <glog/logging.h>

namespace sv {

LidarScan::LidarScan(double t0,
                     double dt,
                     const cv::Mat& xyzr,
                     const cv::Range& col_range)
    : time{t0}, dt{dt}, xyzr{xyzr}, col_rg{col_range} {
  CHECK_GE(t0, 0);
  CHECK_GT(dt, 0);
  CHECK_EQ(xyzr.type(), kDtype);
  CHECK_EQ(xyzr.cols, col_range.size());
}

void LidarScan::MeanCovarAt(const cv::Point& px,
                            int width,
                            MeanCovar3f& mc) const {
  mc.Reset();

  // NOTE (chao): for now only take first row of cell due to staggered scan
  for (int c = 0; c < width; ++c) {
    const auto& xyzr = XyzrAt({px.x + c, px.y});
    if (std::isnan(xyzr[0])) continue;
    mc.Add({xyzr[0], xyzr[1], xyzr[2]});
  }
}

float LidarScan::CurveAt(const cv::Point& px, int width) const {
  static constexpr float kValidCellRatio = 0.8;

  // compute sum of range in cell
  int num = 0;
  float sum = 0.0F;

  const int half = width / 2;
  const auto left = RangeAt({px.x + half - 1, px.y});
  const auto right = RangeAt({px.x + half, px.y});
  const auto mid = std::min(left, right);
  if (std::isnan(mid)) return kNaNF;

  for (int c = 0; c < width; ++c) {
    const auto rg = RangeAt({px.x + c, px.y});
    if (std::isnan(rg)) continue;
    sum += rg;
    ++num;
  }

  // Discard if there are too many nans in this cell
  if (num < kValidCellRatio * width) return kNaNF;
  return std::abs(sum / mid / num - 1.0F);
}

/// Test Related ===============================================================
cv::Mat MakeTestXyzr(const cv::Size& size) {
  cv::Mat xyzr = cv::Mat::zeros(size, LidarScan::kDtype);

  const float azim_delta = M_PI * 2 / size.width;
  const float elev_max = M_PI_4;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < xyzr.rows; ++i) {
    for (int j = 0; j < xyzr.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = M_PI * 2 - j * azim_delta;

      auto& p = xyzr.at<cv::Vec4f>(i, j);
      p[0] = std::cos(elev) * std::cos(azim);
      p[1] = std::cos(elev) * std::sin(azim);
      p[2] = std::sin(elev);
      p[3] = 1;
    }
  }

  return xyzr;
}

LidarScan MakeTestScan(const cv::Size& size) {
  return {0, 0.1 / size.width, MakeTestXyzr(size), {0, size.width}};
}

}  // namespace sv
