#pragma once

#include <iosfwd>
#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

/// @class Mat + Range
struct RangeMat {
  RangeMat() = default;
  RangeMat(cv::Size size, int type) : mat_{size, type} {}
  virtual ~RangeMat() noexcept = default;

  int width() const noexcept { return range_.end; }
  bool empty() const noexcept { return mat_.empty(); }
  bool full() const noexcept { return !empty() && (width() == mat_.cols); }
  cv::Size size() const noexcept { return {mat_.cols, mat_.rows}; }

  const cv::Mat& mat() const noexcept { return mat_; }
  /// @brief Current range of the mat at this point
  cv::Range curr_range() const noexcept { return range_; }
  /// @brief Full range of the mat up to this point
  cv::Range full_range() const noexcept { return {0, width()}; }

  void ResetRange() { range_ = {}; }

  virtual std::string Repr() const;

  cv::Mat mat_;
  cv::Range range_;
};

/// @class Lidar Sweep covers 360
struct LidarSweep : public RangeMat {
  LidarSweep() = default;
  LidarSweep(cv::Size size) : RangeMat{size, CV_32FC4} {}

  /// @brief Add a scan to this sweep
  void AddScan(const cv::Mat& scan, cv::Range scan_range);

  std::string Repr() const override;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);
};

/// @class Feature Grid computes smoothness score within each cell
struct FeatureGrid : public RangeMat {
  FeatureGrid() = default;
  FeatureGrid(cv::Size sweep_size, cv::Size win_size);

  std::string Repr() const override;
  friend std::ostream& operator<<(std::ostream& os, const FeatureGrid& rhs);

  /// @brief Detect feature from sweep
  void Detect(const LidarSweep& sweep, bool tbb);
  void Detect(const cv::Mat& sweep, cv::Range sweep_range, bool tbb);
  void DetectRow(const cv::Mat& sweep, int row);

  /// @brief Count valid features in range, if range empty then count to width
  int NumCells(cv::Range range = {}) const noexcept;

  cv::Size win{};
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

  bool empty() const noexcept { return mat_.empty(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return {mat_.cols, mat_.rows}; }

  /// @brief Add a sweep to the pano
  void AddSweep(const LidarSweep& sweep, bool tbb);
  void AddSweep(const cv::Mat& sweep, bool tbb);
  void AddSweepRow(const cv::Mat& sweep, int row);

  /// @brief Render pano at a new location
  void Render(/* Transform */ bool tbb);
  void RenderRow(int row);

  int ToRow(float z, float r) const noexcept;
  int ToCol(float x, float y) const noexcept;

  bool RowInside(int r) const noexcept { return 0 <= r && r < mat_.rows; }
  bool ColInside(int c) const noexcept { return 0 <= c && c < mat_.cols; }

  cv::Mat mat_;
  cv::Mat mat2_;

  float elev_max_;
  float elev_delta_;
  float azim_delta_;
  std::vector<SinCosF> elevs_;
  std::vector<SinCosF> azims_;

  int num_sweeps_{0};
};

/// @struct Match
struct Match {
  int src_col{};    // src col (change to time?)
  MeanCovar src{};  // sweep
  MeanCovar dst{};  // pano
};

/// @class Projective Matcher
struct ProjectiveMatcher {
  ProjectiveMatcher() = default;
  ProjectiveMatcher(int size);

  std::vector<Match> matches_;
};

}  // namespace sv
