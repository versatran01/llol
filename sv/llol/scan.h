#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"  // MeanCovar

namespace sv {

struct ScanBase {
  cv::Mat mat;       // underlying mat
  cv::Range curr{};  // current working range

  ScanBase() = default;
  ScanBase(const cv::Size& size, int dtype) : mat{size, dtype} {}
  ScanBase(const cv::Mat& mat, const cv::Range& curr) : mat{mat}, curr{curr} {}
  virtual ~ScanBase() noexcept = default;

  /// @brief Info
  int rows() const { return mat.rows; }
  int cols() const { return mat.cols; }
  int total() const { return mat.total(); }
  bool empty() const { return mat.empty(); }
  cv::Size size() const noexcept { return {mat.cols, mat.rows}; }
};

/// @struct Lidar Scan like an image, with pixel (x,y,z,r)
struct LidarScan : public ScanBase {
  using PixelT = cv::Vec4f;
  static constexpr int kDtype = CV_32FC4;

  double t0{};  // time of column 0
  double dt{};  // time between each column

  LidarScan() = default;

  /// @brief Ctor for allocating storage
  explicit LidarScan(const cv::Size& size) : ScanBase{size, kDtype} {}

  /// @brief Ctor for incoming lidar scan
  LidarScan(double t0, double dt, const cv::Mat& xyzr, const cv::Range& curr);

  /// @brief At
  float RangeAt(const cv::Point& px) const { return mat.at<PixelT>(px)[3]; }
  const auto& XyzrAt(const cv::Point& px) const { return mat.at<PixelT>(px); }

  void MeanCovarAt(const cv::Point& px, int width, MeanCovar3f& mc) const;
  float CurveAt(const cv::Point& px, int width) const;
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);

}  // namespace sv
