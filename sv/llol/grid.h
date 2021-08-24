#pragma once

#include "sv/llol/scan.h"  // LidarScan
#include "sv/util/math.h"  // MeanCovar

namespace sv {

/// @struct Match
struct IcpMatch {
  static constexpr int kBad = -100;

  cv::Point px_s{kBad, kBad};  // 8 sweep pixel coord
  MeanCovar3f mc_s{};          // 52 sweep mean covar
  cv::Point px_p{kBad, kBad};  // 8 pano pixel coord
  MeanCovar3f mc_p{};          // 52 pano mean covar
  Eigen::Matrix3f U{};         // 36 sqrt of info

  /// @brief Whether this match is good
  bool Ok() const noexcept { return SweepOk() && PanoOk(); }
  bool SweepOk() const { return px_s.x >= 0 && mc_s.ok(); }
  bool PanoOk() const { return px_p.x >= 0 && mc_p.ok(); }

  void ResetSweep() {
    px_s = {kBad, kBad};
    mc_s.Reset();
  }
  void ResetPano() {
    px_p = {kBad, kBad};
    mc_p.Reset();
  }
  void Reset() {
    ResetSweep();
    ResetPano();
    U.setZero();
  }

  void SqrtInfo(float lambda = 0.0F);
};

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.01F};  // score > max_score will be discarded
  bool nms{true};          // non-minimum suppression in Filter()
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid {
  /// Params
  cv::Size cell_size{16, 2};
  float max_score{0.01};
  bool nms{true};

  /// Data
  cv::Mat score;                     // smoothness score, smaller is smoother
  cv::Range col_rg{};                // working range in this grid
  std::vector<Sophus::SE3f> tf_p_s;  // transforms from sweep to pano (nominal)
  std::vector<IcpMatch> matches;

  /// Disp
  cv::Mat mask_filter;
  cv::Mat mask_match;

  SweepGrid() = default;
  explicit SweepGrid(const cv::Size& sweep_size, const GridParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const SweepGrid& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Score, Filter and Reduce
  std::pair<int, int> Add(const LidarScan& scan, int gsize = 0);
  void Check(const LidarScan& scan) const;

  /// @brief Score each cell of the incoming scan
  /// @param gsize is number of rows per task, <=0 means single thread
  /// @return Number of valid cells
  int Score(const LidarScan& scan, int gsize = 0);
  int ScoreRow(const LidarScan& scan, int r);

  /// @brief
  int Reduce(const LidarScan& scan, int gisze = 0);
  int ReduceRow(const LidarScan& scan, int r);
  /// @brief Check whether this cell is good or not for Filter()
  bool IsCellGood(const cv::Point& px) const;

  /// @brief Rect in sweep corresponding to a cell
  cv::Rect SweepCell(const cv::Point& px) const;

  /// @brief At
  float& ScoreAt(const cv::Point& px) { return score.at<float>(px); }
  float ScoreAt(const cv::Point& px) const { return score.at<float>(px); }
  IcpMatch& MatchAt(const cv::Point& px) { return matches[Grid2Ind(px)]; }
  const IcpMatch& MatchAt(const cv::Point& px) const {
    return matches[Grid2Ind(px)];
  }

  /// @brief Pxiel coordinates conversion (sweep <-> grid)
  cv::Point Sweep2Grid(const cv::Point& px_sweep) const;
  cv::Point Grid2Sweep(const cv::Point& px_grid) const;
  int Grid2Ind(const cv::Point& px_grid) const;

  /// @brief Info
  int total() const { return score.total(); }
  bool empty() const { return score.empty(); }
  int width() const noexcept { return col_rg.end; }
  bool full() const noexcept { return width() == score.cols; }
  cv::Size size() const noexcept { return {score.cols, score.rows}; }

  /// @brief Draw
  const cv::Mat& FilterMask();
  const cv::Mat& MatchMask();
};

/// @brief Draw match, valid pixel is percentage of pano points in window
cv::Mat DrawMatches(const SweepGrid& grid);

}  // namespace sv
