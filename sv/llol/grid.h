#pragma once

#include "sv/llol/match.h"  // GicpMatch
#include "sv/llol/pano.h"   // DepthPano
#include "sv/llol/scan.h"   // LidarScan

namespace sv {

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.01F};   // score > max_score will be discarded
  bool nms{true};           // non-minimum suppression in Filter()
  int half_rows{2};         // half rows when match in pano
  float cov_lambda{1e-6F};  // lambda added to diagonal of cov
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid {
  /// Params
  cv::Size cell_size;
  float max_score{};
  bool nms{};
  float cov_lambda{};      // lambda added to diagonal of covar
  cv::Size pano_win_size;  // win size in pano used to compute mean covar
  cv::Size max_dist_size;  // max dist size to resue pano mc
  int min_pts{};           // min pts in pano win for a valid match

  /// Data
  cv::Mat score;                  // smoothness score, smaller is smoother
  cv::Range col_rg{};             // working range in this grid
  std::vector<Sophus::SE3f> tfs;  // transforms from edge of cell to pano
  std::vector<GicpMatch> matches;

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
  int Filter(const LidarScan& scan, int gisze = 0);
  int FilterRow(const LidarScan& scan, int r);
  /// @brief Check whether this cell is good or not for Filter()
  bool IsCellGood(const cv::Point& px) const;

  /// @brief Match features in sweep to pano using mask
  /// @return Number of final matches
  int Match(const DepthPano& pano, int gsize = 0);
  int MatchRow(const DepthPano& pano, int gr);
  int MatchCell(const DepthPano& pano, const cv::Point& gpx);

  /// @brief At
  float& ScoreAt(const cv::Point& px) { return score.at<float>(px); }
  float ScoreAt(const cv::Point& px) const { return score.at<float>(px); }
  GicpMatch& MatchAt(const cv::Point& px) { return matches.at(Grid2Ind(px)); }
  const GicpMatch& MatchAt(const cv::Point& px) const {
    return matches.at(Grid2Ind(px));
  }
  Sophus::SE3f CellTfAt(int c) const;

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
  cv::Mat DispFilter() const;
  cv::Mat DispMatch() const;

  void InterpSweep(LidarSweep& sweep, int gsize = 0) const;
};

void InterpPosesImpl(const std::vector<Sophus::SE3f>& tf_cell,
                     int cell_width,
                     std::vector<Sophus::SE3f>& tf_col,
                     int gsize = 0);

}  // namespace sv
