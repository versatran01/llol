#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

namespace sv {

/// @struct Lidar Scan
struct LidarScan {
  using PixelT = cv::Vec4f;
  static constexpr int kDtype = CV_32FC4;

  LidarScan() = default;

  /// @brief Ctor for allocating storage
  LidarScan(const cv::Size& size) : xyzr{size, kDtype} {}

  /// @brief Ctor for incoming lidar scan
  LidarScan(double t0,
            double dt,
            const cv::Mat& xyzr,
            const cv::Range& col_range);

  virtual ~LidarScan() noexcept = default;

  /// @brief At
  float RangeAt(const cv::Point& px) const { return xyzr.at<PixelT>(px)[3]; }
  const auto& XyzrAt(const cv::Point& px) const { return xyzr.at<PixelT>(px); }

  /// @brief Mat
  int total() const { return xyzr.total(); }
  cv::Size size() const noexcept { return {xyzr.cols, xyzr.rows}; }

  double t0{};
  double dt{};
  cv::Range col_range{};  // working range
  cv::Mat xyzr{};
};

/// @struct Lidar Sweep is a Lidar Scan that covers 360 degree horizontal fov
struct LidarSweep final : public LidarScan {
  /// Data
  int id{-1};
  std::vector<Sophus::SE3f> tfs;

  LidarSweep() = default;
  explicit LidarSweep(const cv::Size& size) : LidarScan{size} {}

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Add a scan to this sweep
  /// @return Number of points added
  int AddScan(const LidarScan& scan);

  /// @brief Info
  int width() const noexcept { return col_range.end; }
  bool full() const noexcept { return width() == xyzr.cols; }
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);
LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
