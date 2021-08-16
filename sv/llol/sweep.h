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

  double t0{};
  double dt{};
  cv::Mat xyzr{};
  cv::Range col_range{};
};

/// @struct Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep final : public LidarScan {
  /// Data
  int id{-1};
  std::vector<Sophus::SE3f> tfs;

  /// stores curvature scores of each cell in sweep
  cv::Mat grid{};        // stores curvature, could be nan
  cv::Size cell_size{};  // only use first row

  LidarSweep() = default;
  /// @brief Ctors for allocating storage
  LidarSweep(const cv::Size& sweep_size, const cv::Size& cell_size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const LidarScan& scan, bool tbb = false);

  /// @brief Indexing related
  cv::Rect CellAt(const cv::Point& px_grid) const;
  cv::Point Pix2Cell(const cv::Point& px_sweep) const;
  cv::Point Cell2Pix(const cv::Point& px_grid) const;
  const auto& PixAt(const cv::Point& px_sweep) const {
    return xyzr.at<PixelT>(px_sweep);
  }

  /// @brief basic info
  int width() const noexcept { return col_range.end; }
  bool full() const noexcept { return width() == xyzr.cols; }
  cv::Size size() const noexcept { return {xyzr.cols, xyzr.rows}; }
  int grid_width() const noexcept { return width() / cell_size.width; }
  cv::Size grid_size() const noexcept { return {grid.cols, grid.rows}; }
  cv::Range grid_range() const noexcept {
    return {col_range.start / cell_size.width, col_range.end / cell_size.width};
  }
};

cv::Mat MakeTestXyzr(const cv::Size& size);
LidarScan MakeTestScan(const cv::Size& size);
LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
