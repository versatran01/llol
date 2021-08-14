#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

/// @struct Lidar Scan
struct LidarScan {
  LidarScan() = default;
  LidarScan(const cv::Mat& xyzr, const cv::Range& col_range);

  cv::Mat xyzr;
  cv::Range col_range{0, 0};
};

/// @struct Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep {
  /// Data
  int id{-1};
  /// incrementally stores scan into sweep
  cv::Range col_range;  // range of last added scan
  cv::Mat xyzr_;        // [x,y,z,r] from driver

  /// stores curvature scores of each cell in sweep
  cv::Size cell_size;  // only use first row
  cv::Mat grid_;       // stores curvature, could be nan

  std::vector<uint8_t> offsets;  // pixel offsets per row
  void SetOffsets(const std::vector<double>& offsets_in);

  /// @brief Ctors
  LidarSweep() = default;
  LidarSweep(const cv::Size& sweep_size, const cv::Size& cell_size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const LidarScan& scan, bool tbb = false);

  /// @brief whether sweep is full
  bool IsFull() const noexcept { return width() == xyzr_.cols; }

  cv::Rect CellAt(const cv::Point& px_grid) const;
  const auto& XyzrAt(const cv::Point& px_sweep) const {
    return xyzr_.at<cv::Vec4f>(px_sweep);
  }
  cv::Point Pix2Cell(const cv::Point& px_sweep) const;

  /// @brief getters
  cv::Size grid_size() const noexcept { return {grid_.cols, grid_.rows}; }
  cv::Size xyzr_size() const noexcept { return {xyzr_.cols, xyzr_.rows}; }
  const cv::Mat& grid() const noexcept { return grid_; }
  const cv::Mat& xyzr() const noexcept { return xyzr_; }

  /// @brief basic info
  int width() const noexcept { return col_range.end; }
  int grid_width() const noexcept { return width() / cell_size.width; }
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);
LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
