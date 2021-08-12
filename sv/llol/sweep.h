#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

/// @class Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep {
  /// Data
  /// incrementally stores scan into sweep
  cv::Range col_range_;
  cv::Mat xyzr_;

  /// stores curvature scores of each cell in sweep
  cv::Size cell_size_;  // only use first row
  cv::Mat grid_;

  cv::Mat offsets_;     // pixel offsets per row
  cv::Mat transforms_;  // Nx7 [qx,qy,qz,qw,x,y,z]

  /// @brief Ctors
  LidarSweep() = default;
  LidarSweep(const cv::Size& sweep_size, const cv::Size& cell_size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const cv::Mat& scan,
              const cv::Range& scan_range,
              bool tbb = false);

  /// @brief Reset range to [0,0), sweep and grid to nan, keep cell_size
  void Reset();

  /// @brief whether sweep is full
  bool IsFull() const noexcept { return width() == xyzr_.cols; }

  cv::Rect CellAt(const cv::Point& px_g) const;
  cv::Point Pixel2CellInd(const cv::Point& px_sweep) const;
  const auto& XyzrAt(const cv::Point& px_sweep) const {
    return xyzr_.at<cv::Vec4f>(px_sweep);
  }

  /// @brief getters
  cv::Range range() const noexcept { return col_range_; }
  cv::Size cell_size() const noexcept { return cell_size_; }
  cv::Size grid_size() const noexcept { return {grid_.cols, grid_.rows}; }
  cv::Size xyzr_size() const noexcept { return {xyzr_.cols, xyzr_.rows}; }
  const cv::Mat& grid() const noexcept { return grid_; }
  const cv::Mat& xyzr() const noexcept { return xyzr_; }

  /// @brief basic info
  int width() const noexcept { return col_range_.end; }
  int grid_width() const noexcept { return width() / cell_size_.width; }
};

cv::Mat MakeTestScan(const cv::Size& size);

}  // namespace sv
