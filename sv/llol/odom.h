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

  bool empty() const { return mat_.empty(); }
  size_t total() const { return mat_.total(); }
  int width() const noexcept { return range_.end; }
  bool full() const { return !empty() && (width() == mat_.cols); }
  cv::Size size() const noexcept { return {mat_.cols, mat_.rows}; }

  const cv::Mat& mat() const noexcept { return mat_; }
  /// @brief Current range of the mat at this point
  cv::Range curr_range() const noexcept { return range_; }
  /// @brief Full range of the mat up to this point
  cv::Range full_range() const { return {0, width()}; }

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
  void AddScan(const cv::Mat& scan, const cv::Range& scan_range);

  /// @brief XyzrAt (row, col)
  const cv::Vec4f& XyzrAt(int r, int c) const {
    return mat().at<cv::Vec4f>(r, c);
  }

  std::string Repr() const override;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs);
};

/// @class Point Grid divide data into different cells and compute smoothness
struct PointGrid : public RangeMat {
  PointGrid() = default;
  PointGrid(const cv::Size& sweep_size, const cv::Size& win_size);

  std::string Repr() const override;
  friend std::ostream& operator<<(std::ostream& os, const PointGrid& rhs);

  float ScoreAt(int r, int c) const { return mat_.at<float>(r, c); }
  float& ScoreAt(int r, int c) { return mat_.at<float>(r, c); }

  /// @brief Detect feature from sweep
  /// @return Number of valid cells in the current range
  int Detect(const LidarSweep& sweep, bool tbb);
  int DetectRow(const cv::Mat& sweep, int row);

  /// @brief Count valid features in range, if range empty then count to width
  int NumValid(cv::Range range = {}) const;

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

struct DataMatcher {
  DataMatcher() = default;
  DataMatcher(int max_matches, const MatcherParams& params);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DataMatcher& rhs);

  const auto& matches() const noexcept { return matches_; }

  /// @brief Match features in sweep to pano
  void Match(const LidarSweep& sweep,
             const PointGrid& grid,
             const DepthPano& pano);

  bool IsGoodFeature(const PointGrid& grid, int r, int c) const;

  MatcherParams params_;
  std::vector<NormalMatch> matches_;
};

}  // namespace sv
