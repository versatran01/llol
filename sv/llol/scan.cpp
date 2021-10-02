#include "sv/llol/scan.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <opencv2/core.hpp>

namespace sv {

/// ScanBase ===================================================================
ScanBase::ScanBase(const cv::Size& size, int dtype) : mat{size, dtype} {
  tfs.resize(size.width);
}

ScanBase::ScanBase(double time,
                   double dt,
                   const cv::Mat& scan,
                   const cv::Range& curr)
    : time{time}, dt{dt}, mat{scan}, curr{curr} {
  CHECK_GE(time, 0) << "Time cannot be negative";
  CHECK_GT(dt, 0) << "Delta time must be positive";
  CHECK_EQ(cols(), curr.size()) << "Mat width mismatch";
}

void ScanBase::UpdateTime(double new_time, double new_dt) {
  CHECK_LE(time, new_time);
  time = new_time;
  if (dt == 0) {
    dt = new_dt;
  } else {
    CHECK_EQ(dt, new_dt);
  }
}

cv::Mat ScanBase::ExtractRange() const {
  static cv::Mat range;
  cv::Mat image(size(), CV_16UC(8), mat.data);
  cv::extractChannel(image, range, 6);
  return range;
}

void ScanBase::UpdateView(const cv::Range& new_curr) {
  const int width = new_curr.size();
  CHECK_EQ(new_curr.start, curr.end % cols());
  CHECK_LE(width, cols());
  curr = new_curr;
}

/// LidarScan ==================================================================
LidarScan::LidarScan(const cv::Size& size) : ScanBase{size, kDtype} {
  mat.setTo(kNaNF);
}

LidarScan::LidarScan(double time,
                     double dt,
                     double scale,
                     const cv::Mat& scan,
                     const cv::Range& curr)
    : ScanBase{time, dt, scan, curr}, scale{scale} {
  CHECK_EQ(scan.type(), kDtype) << "Mat type mismatch";
  CHECK_GT(scale, 0) << "Scale must be positive";
}

void LidarScan::CalcMeanCovar(const cv::Rect& rect, MeanCovar3f& mc) const {
  mc.Reset();

  // NOTE (chao): for now only take first row of cell due to staggered scan
  for (int c = 0; c < rect.width; ++c) {
    const auto& xyzr = PixelAt({rect.x + c, rect.y});
    if (xyzr.Ok()) mc.Add(xyzr.Vec3fMap());
  }
}

cv::Vec2f LidarScan::CalcScore(const cv::Point& px, int width) const {
  cv::Vec2f score(kNaNF, kNaNF);

  // compute sum of range in cell
  int n = 0;
  float sum = 0.0F;
  float sq_sum = 0.0F;

  const int half = width / 2;
  const auto left = RangeAt({px.x + half - 1, px.y});
  const auto right = RangeAt({px.x + half, px.y});
  const auto mid = std::min(left, right);
  if (mid == 0) return score;

  for (int c = 0; c < width; ++c) {
    const auto& pixel = PixelAt({px.x + c, px.y});
    if (!pixel.Ok()) continue;
    const float rg = pixel.range_raw / scale;

    sum += rg;
    sq_sum += rg * rg;
    ++n;
  }

  // Discard if there are fewer than 8 points
  if (n < 8) return score;

  // Check variance
  // https://www.johndcook.com/blog/standard_deviation/
  score[0] = std::abs(sum / mid / n - 1.0F);
  score[1] = 1.0 / (n * (n - 1)) * (n * sq_sum - sum * sum) / mid;
  return score;
}

/// Test Related ===============================================================
cv::Mat MakeTestMat(const cv::Size& size) {
  cv::Mat xyzr = cv::Mat::zeros(size, LidarScan::kDtype);

  const float azim_delta = kPiF * 2 / size.width;
  const float elev_max = kPiF / 4;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < xyzr.rows; ++i) {
    for (int j = 0; j < xyzr.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = kTauF - j * azim_delta;

      auto& p = xyzr.at<ScanPixel>(i, j);
      p.x = std::cos(elev) * std::cos(azim);
      p.y = std::cos(elev) * std::sin(azim);
      p.z = std::sin(elev);
      p.range_raw = 1024;
    }
  }

  return xyzr;
}

LidarScan MakeTestScan(const cv::Size& size) {
  return {0, 0.1 / size.width, 512.0, MakeTestMat(size), {0, size.width}};
}

}  // namespace sv
