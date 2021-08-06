#pragma once

#include <iosfwd>
#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

/// @class Lidar Sweep covers 360 degree horizontal fov
struct LidarSweep {
  LidarSweep() = default;
  LidarSweep(cv::Size sweep_size, cv::Size cell_size);

  /// @brief basic info
  bool empty() const { return sweep_.empty(); }
  size_t total() const { return sweep_.total(); }
  int width() const noexcept { return range_.end; }
  cv::Range range() const noexcept { return range_; }
  cv::Size cell_size() const noexcept { return cell_size_; }
  cv::Size grid_size() const { return {grid_.cols, grid_.rows}; }
  cv::Size sweep_size() const { return {sweep_.cols, sweep_.rows}; }
  bool full() const noexcept { return !empty() && width() == sweep_.cols; }

  /// @brief getters
  const cv::Mat& grid() const noexcept { return grid_; }
  const cv::Mat& sweep() const noexcept { return sweep_; }

  /// @brief Add a scan to this sweep (tbb makes this slower)
  /// @return num of valid cells
  int AddScan(const cv::Mat& scan, cv::Range scan_range, bool tbb = false);

  const cv::Vec4f& XyzrAt(int sweep_row, int sweep_col) const {
    return sweep_.at<cv::Vec4f>(sweep_row, sweep_col);
  }

  float ScoreAt(int grid_row, int grid_col) const {
    return grid_.at<float>(grid_row, grid_col);
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

/// @class Depth Panorama
class DepthPano {
 public:
  static constexpr float kScale = 256.0;
  static constexpr float kMaxRange = 65536.0 / kScale;

  DepthPano() = default;
  DepthPano(cv::Size size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs);

  bool empty() const { return mat_.empty(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return {mat_.cols, mat_.rows}; }
  float wh_ratio() const noexcept { return wh_ratio_; }

  /// @brief At
  ushort& RawAt(int r, int c) { return mat_.at<ushort>(r, c); }
  ushort RawAt(int r, int c) const { return mat_.at<ushort>(r, c); }
  float MetricAt(int r, int c) const { return RawAt(r, c) / kScale; }

  /// @brief WinAt
  cv::Rect WinAt(const cv::Point& pt, const cv::Size& half_size) const;
  cv::Rect BoundedWinAt(const cv::Point& pt, const cv::Size& half_size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const LidarSweep& sweep, bool tbb);
  int AddSweep(const cv::Mat& sweep, bool tbb);
  int AddSweepRow(const cv::Mat& sweep, int row);

  /// @brief Render pano at a new location
  void Render(bool tbb);
  void RenderRow(int row1);

  int ToRow(float z, float r) const;
  int ToCol(float x, float y) const;
  cv::Point3f To3d(int r, int c, float rg) const noexcept;

  bool RowInside(int r) const noexcept { return 0 <= r && r < mat_.rows; }
  bool ColInside(int c) const noexcept { return 0 <= c && c < mat_.cols; }

  cv::Mat mat_;
  cv::Mat mat2_;

  float wh_ratio_;
  float elev_max_;
  float elev_delta_;
  float azim_delta_;
  std::vector<SinCosF> elevs_;
  std::vector<SinCosF> azims_;

  int num_sweeps_{0};
};

/// @struct Match
struct NormalMatch {
  int src_col{};    // src col (change to time?)
  MeanCovar src{};  // sweep
  MeanCovar dst{};  // pano
};

/// @class Feature Matcher
struct MatcherParams {
  float max_score{0.01};
  bool nms{true};
  int half_rows{2};
};

// struct DataMatcher {
//  DataMatcher() = default;
//  DataMatcher(int max_matches, const MatcherParams& params);

//  std::string Repr() const;
//  friend std::ostream& operator<<(std::ostream& os, const DataMatcher& rhs);

//  const auto& matches() const noexcept { return matches_; }

//  /// @brief Match features in sweep to pano
//  void Match(const LidarSweep& sweep,
//             const PointGrid& grid,
//             const DepthPano& pano);

//  bool IsGoodFeature(const PointGrid& grid, int r, int c) const;

//  MatcherParams params_;
//  std::vector<NormalMatch> matches_;
//};

}  // namespace sv
