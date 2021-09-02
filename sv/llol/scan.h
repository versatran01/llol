#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "sv/util/math.h"  // MeanCovar

namespace sv {

inline int ColMod(int c, int cols) { return c < 0 ? c + cols : c; }

struct ScanBase {
  // start and delta time
  double time{};  // time of the last column
  double dt{};    // delta time between two columns

  // Scan related data
  cv::Mat mat;                    // storage
  cv::Range curr;                 // current range
  std::vector<Sophus::SE3f> tfs;  // tfs of each col to some frame

  ScanBase() = default;
  ScanBase(const cv::Size& size, int dtype);
  ScanBase(double time, double dt, const cv::Mat& mat, const cv::Range& curr);
  virtual ~ScanBase() noexcept = default;

  /// @brief Info
  int rows() const { return mat.rows; }
  int cols() const { return mat.cols; }
  int type() const { return mat.type(); }
  int total() const { return mat.total(); }
  bool empty() const { return mat.empty(); }
  int channels() const { return mat.channels(); }
  cv::Size size() const { return {mat.cols, mat.rows}; }

  double time_begin() const { return time - dt * cols(); }
  double time_end() const { return time; }
  const Sophus::SE3f& TfAt(int c) const { return tfs.at(c); }

  /// @brief Update view (curr and span) given new curr
  void UpdateView(const cv::Range& new_curr);
  /// @brief Update time (time and dt) given new time
  void UpdateTime(double new_time, double new_dt);
};

/// @struct Lidar Scan like an image, with pixel (x,y,z,r)
struct LidarScan : public ScanBase {
  using PixelT = cv::Vec4f;
  static constexpr int kDtype = CV_32FC4;

  LidarScan() = default;
  /// @brief Ctor for allocating storage
  explicit LidarScan(const cv::Size& size);
  /// @brief Ctor for incoming lidar scan
  LidarScan(double time, double dt, const cv::Mat& xyzr, const cv::Range& curr);

  /// @brief At
  float RangeAt(const cv::Point& px) const { return mat.at<PixelT>(px)[3]; }
  const auto& XyzrAt(const cv::Point& px) const { return mat.at<PixelT>(px); }

  void MeanCovarAt(const cv::Point& px, int width, MeanCovar3f& mc) const;
  cv::Vec2f ScoreAt(const cv::Point& px, int width) const;
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);

}  // namespace sv
