#pragma once

#include "sv/llol/scan.h"  // LidarScan
#include "sv/util/math.h"  // MeanCovar

namespace sv {

/// @struct Match
struct NormalMatch {
  static constexpr int kBad = -100;

  cv::Point px_s{kBad, kBad};  // 8
  MeanCovar3f mc_s{};          // 52 sweep
  cv::Point px_p{kBad, kBad};  // 8
  MeanCovar3f mc_p{};          // 52 pano
  Eigen::Matrix3f U{};         // 36

  /// @brief Whether this match is good
  bool ok() const noexcept {
    return px_s.x >= 0 && px_p.x >= 0 && mc_s.ok() && mc_p.ok();
  }

  void SqrtInfo(float lambda = 0.0F);
  void ResetPano() {
    px_p = {kBad, kBad};
    mc_p.Reset();
  }
};

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.01};  // score > max_score will be discarded
  bool nms{true};         // non-minimum suppression in Filter()
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid {
  /// Params
  cv::Size cell_size{16, 2};
  float max_score{0.01};
  bool nms{true};

  /// Data
  cv::Range col_rg{};                // working range in this grid
  cv::Mat score;                     // smoothness score, smaller is smoother
  cv::Mat mask;                      // binary mask of match candidates
  std::vector<Sophus::SE3f> tf_p_s;  // transforms from sweep to pano
  std::vector<NormalMatch> matches;

  SweepGrid() = default;
  explicit SweepGrid(const cv::Size& sweep_size, const GridParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const SweepGrid& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Clear all matches, must be called on a new sweep
  void ResetMatches();

  /// @brief Reduce and Filter
  std::pair<int, int> Reduce(const LidarScan& scan, int gsize = 0);
  void Check(const LidarScan& scan);

  /// @brief Reduce scan into score grid
  /// @param gsize is number of rows per task, <=0 means single thread
  /// @return Number of valid cells (not NaN)
  int Score(const LidarScan& scan, int gsize = 0);
  int ScoreRow(const LidarScan& scan, int r);

  /// @brief Filter score grid given max_score and nms
  /// @return Number of remaining cells
  int Filter();
  /// @brief Check whether this cell is good or not for Filter()
  bool IsCellGood(const cv::Point& px) const;

  /// @brief Rect in sweep corresponding to a cell
  cv::Rect SweepCell(const cv::Point& px) const;

  /// @brief At
  float& ScoreAt(const cv::Point& px) { return score.at<float>(px); }
  float ScoreAt(const cv::Point& px) const { return score.at<float>(px); }
  uint8_t& MaskAt(const cv::Point& px) { return mask.at<uint8_t>(px); }
  uint8_t MaskAt(const cv::Point& px) const { return mask.at<uint8_t>(px); }
  NormalMatch& MatchAt(const cv::Point& px) { return matches[Grid2Ind(px)]; }
  const NormalMatch& MatchAt(const cv::Point& px) const {
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
};

/// @brief Draw match, valid pixel is percentage of pano points in window
cv::Mat DrawMatches(const SweepGrid& grid);

}  // namespace sv
