#include "sv/llol/scan.h"

#include <fmt/core.h>
#include <glog/logging.h>

namespace sv {

/// ScanBase ===================================================================
ScanBase::ScanBase(const cv::Size& size, int dtype) : mat{size, dtype} {
  tfs.resize(size.width);
}

ScanBase::ScanBase(double t0,
                   double dt,
                   const cv::Mat& mat,
                   const cv::Range& curr)
    : t0{t0}, dt{dt}, mat{mat}, curr{curr} {
  CHECK_GE(t0, 0) << "Time cannot be negative";
  CHECK_GT(dt, 0) << "Delta time must be positive";
  CHECK_EQ(cols(), curr.size()) << "Mat width mismatch";
}

void ScanBase::UpdateTime(double new_t0, double new_dt) {
  CHECK_LE(t0, new_t0);
  t0 = new_t0;
  t0 = new_t0;
  if (dt == 0) {
    dt = new_dt;
  } else {
    CHECK_EQ(dt, new_dt);
  }
}

void ScanBase::UpdateView(const cv::Range& new_curr) {
  const int width = new_curr.size();
  CHECK_EQ(new_curr.start, curr.end % cols());
  CHECK_LE(width, cols());

  // Update curr
  curr = new_curr;
  if (full()) {
    // If span full increment both
    span.start += width;
    span.end += width;
  } else {
    // If span not full, only increment end
    span.end += width;
  }
  CHECK_LE(span.size(), cols());
}

/// LidarScan ==================================================================
LidarScan::LidarScan(double t0,
                     double dt,
                     const cv::Mat& xyzr,
                     const cv::Range& curr)
    : ScanBase{t0, dt, xyzr, curr} {
  CHECK_EQ(xyzr.type(), kDtype) << "Mat type mismatch";
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
