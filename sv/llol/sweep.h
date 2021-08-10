#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

/// @class Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep {
  /// Data
  /// incrementally stores scan into sweep
  cv::Range range_;
  cv::Mat sweep_;

  /// stores curvature scores of each cell in sweep
  cv::Size cell_size_;
  cv::Mat grid_;

  std::vector<int> offsets;  // not used

  /// @brief Ctors
  LidarSweep() = default;
  LidarSweep(cv::Size sweep_size, cv::Size cell_size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const cv::Mat& scan,
              const cv::Range& scan_range,
              bool tbb = false);

  int CalcScanCurve(const cv::Mat& scan, cv::Mat grid, bool tbb = false);
  int CalcScanCurveRow(const cv::Mat& scan, cv::Mat& grid, int r);

  /// @brief Reset range to [0,0), sweep and grid to nan, keep cell_size
  void Reset();

  /// @brief whether sweep is full
  bool full() const noexcept { return width() == sweep_.cols; }

  const auto& XyzrAt(const cv::Point& px) const {
    return sweep_.at<cv::Vec4f>(px);
  }
  cv::Point PixelToCell(const cv::Point& px_s) const;

  /// @brief getters
  cv::Range range() const noexcept { return range_; }
  const cv::Mat& sweep() const noexcept { return sweep_; }
  const cv::Mat& grid() const noexcept { return grid_; }
  cv::Size cell_size() const noexcept { return cell_size_; }

  /// @brief basic info
  int width() const noexcept { return range_.end; }
  int grid_width() const noexcept { return width() / cell_size_.width; }

  cv::Mat CellAt(const cv::Point& grid_px) const;
};

}  // namespace sv
