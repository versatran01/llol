#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"  // MeanCovar

namespace sv {

/// @struct Lidar Scan like an image, with pixel (x,y,z,r)
struct LidarScan {
  using PixelT = cv::Vec4f;
  static constexpr int kDtype = CV_32FC4;

  double time{};       // time stamp of the first column
  double dt{};         // delta time between two columns
  cv::Mat xyzr{};      // scan data (x,y,z,r)
  cv::Range col_rg{};  // indicates scan range within a sweep

  LidarScan() = default;

  /// @brief Ctor for allocating storage
  explicit LidarScan(const cv::Size& size) : xyzr{size, kDtype} {}

  /// @brief Ctor for incoming lidar scan
  LidarScan(double t0,
            double dt,
            const cv::Mat& xyzr,
            const cv::Range& col_range);

  virtual ~LidarScan() noexcept = default;

  /// @brief At
  float RangeAt(const cv::Point& px) const { return xyzr.at<PixelT>(px)[3]; }
  const auto& XyzrAt(const cv::Point& px) const { return xyzr.at<PixelT>(px); }

  /// @brief Info
  int total() const { return xyzr.total(); }
  bool empty() const { return xyzr.empty(); }
  cv::Size size() const noexcept { return {xyzr.cols, xyzr.rows}; }

  void MeanCovarAt(const cv::Point& px, int width, MeanCovar3f& mc) const;
  float CurveAt(const cv::Point& px, int width) const;
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);

}  // namespace sv
