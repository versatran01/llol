#pragma once

#include <iosfwd>
#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

std::string Repr(const cv::Mat& mat);
std::string Repr(const cv::Size& size);
std::string Repr(const cv::Range& range);

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
  float CellAt(int gr, int gc) const { return grid_.at<float>(gr, gc); }

  cv::Point PixelToCell(cv::Point px_s) const {
    return {px_s.x / cell_size_.width, px_s.y / cell_size_.height};
  }

  /// @brief Compute corresponding subgrid given scan range
  cv::Mat GetSubgrid(cv::Range scan_range);

  /// @brief For now compute curvature of each cell
  int ReduceScan(const cv::Mat& scan, cv::Mat& subgrid, bool tbb = false);
  int ReduceScanRow(const cv::Mat& scan, int gr, cv::Mat& subgrid);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);

  /// incrementally stores scan into sweep
  cv::Range range_;
  cv::Mat sweep_;

  /// stores curvature scores of each cell in sweep
  cv::Size cell_size_;
  cv::Mat grid_;
};

/// @struct LidarModel
struct LidarModel {
  LidarModel() = default;
  LidarModel(cv::Size size, float hfov);

  /// @brief xyzr to image, bad result is {-1, -1}
  cv::Point2i Forward(float x, float y, float z, float r) const;
  cv::Point3f Backward(int r, int c, float rg = 1.0) const;

  /// @brief compute row and col given xyzr
  int ToRow(float z, float r) const;
  int ToCol(float x, float y) const;

  /// @brief Check if r/c inside image
  bool RowInside(int r) const noexcept { return 0 <= r && r < size_.height; }
  bool ColInside(int c) const noexcept { return 0 <= c && c < size_.width; }

  /// @brief width / height
  double WidthHeightRatio() const { return size_.aspectRatio(); }
  float ElevAzimRatio() const { return elev_delta_ / azim_delta_; }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarModel& rhs);

  cv::Size size_{};
  float elev_max_{};
  float elev_delta_{};
  float azim_delta_{};
  std::vector<SinCosF> elevs_{};
  std::vector<SinCosF> azims_{};
};

/// @class Depth Panorama
class DepthPano {
 public:
  static constexpr float kScale = 256.0F;
  static constexpr float kMaxRange = 65536.0F / kScale;

  DepthPano() = default;
  DepthPano(cv::Size size, float hfov = 0);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs);

  bool empty() const { return buf_.empty(); }
  size_t total() const { return buf_.total(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return model_.size_; }

  float GetRange(cv::Point pt) const { return buf_.at<ushort>(pt) / kScale; }
  void SetRange(cv::Point pt, float rg, cv::Mat& mat) {
    mat.at<ushort>(pt) = rg * kScale;
  }

  cv::Rect WinCenterAt(cv::Point pt, cv::Size size) const;
  cv::Rect BoundWinCenterAt(cv::Point pt, cv::Size size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const LidarSweep& sweep, bool tbb);
  int AddSweep(const cv::Mat& sweep, bool tbb);
  int AddSweepRow(const cv::Mat& sweep, int row);

  /// @brief Render pano at a new location
  int Render(bool tbb);
  int RenderRow(int row1);

  /// @brief Computes mean and covar of points in window at
  void CalcMeanCovar(cv::Rect win, MeanCovar3f& mc) const;

  int num_sweeps_{0};
  cv::Mat buf_;
  cv::Mat buf2_;
  LidarModel model_;
};

}  // namespace sv
