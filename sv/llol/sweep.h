#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

int CalcScanCurve(const cv::Mat& scan, cv::Mat& grid, bool tbb = false);

/// @class Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep {
  LidarSweep() = default;
  LidarSweep(cv::Size sweep_size, cv::Size cell_size);

  /// @brief basic info
  bool empty() const { return sweep_.empty(); }
  bool full() const noexcept { return !empty() && width() == sweep_.cols; }

  size_t total() const { return sweep_.total(); }
  size_t grid_total() const { return grid_.total(); }

  cv::Size size() const { return {sweep_.cols, sweep_.rows}; }
  cv::Size grid_size() const { return {grid_.cols, grid_.rows}; }

  int width() const noexcept { return range_.end; }
  int grid_width() const noexcept { return width() / cell_size_.width; }

  cv::Range range() const noexcept { return range_; }
  cv::Range grid_range() const {
    return {range_.start / cell_size_.width, range_.end / cell_size_.width};
  }

  /// @brief getters
  cv::Size cell_size() const noexcept { return cell_size_; }
  const cv::Mat& grid() const noexcept { return grid_; }
  const cv::Mat& sweep() const noexcept { return sweep_; }

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const cv::Mat& scan, cv::Range scan_range, bool tbb = false);

  const auto& XyzrAt(cv::Point px) const { return sweep_.at<cv::Vec4f>(px); }

  cv::Point PixelToCell(cv::Point px_s) const {
    return {px_s.x / cell_size_.width, px_s.y / cell_size_.height};
  }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);

  /// incrementally stores scan into sweep
  cv::Range range_;
  cv::Mat sweep_;

  /// stores curvature scores of each cell in sweep
  cv::Size cell_size_;
  cv::Mat grid_;
};

}  // namespace sv
