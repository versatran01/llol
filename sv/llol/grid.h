#pragma once

#include <sophus/se3.hpp>

#include "sv/llol/match.h"
#include "sv/llol/sweep.h"

namespace sv {

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.01F};  // score > max_score will be discarded
  bool nms{true};          // non-minimum suppression in Filter()
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid {
  /// Params
  cv::Size cell_size;
  float max_score{};
  bool nms{};

  /// Data
  cv::Mat score;                   // smoothness score, smaller is smoother
  cv::Range col_rg{};              // working range in this grid
  std::vector<Sophus::SE3f> tfs;   // transforms from edge of cell to pano
  std::vector<GicpMatch> matches;  // all matches

  SweepGrid() = default;
  explicit SweepGrid(const cv::Size& sweep_size, const GridParams& params = {});

  /// @brief Repr / <<
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

  /// @brief At
  float& ScoreAt(const cv::Point& px) { return score.at<float>(px); }
  float ScoreAt(const cv::Point& px) const { return score.at<float>(px); }
  GicpMatch& MatchAt(const cv::Point& px) { return matches.at(Px2Ind(px)); }
  const GicpMatch& MatchAt(const cv::Point& px) const {
    return matches.at(Px2Ind(px));
  }
  Sophus::SE3f CellTfAt(int c) const;

  /// @brief Pxiel coordinates conversion (sweep <-> grid)
  cv::Point Sweep2Grid(const cv::Point& px_sweep) const;
  cv::Point Grid2Sweep(const cv::Point& px_grid) const;
  int Px2Ind(const cv::Point& px_grid) const;

  /// @brief Info
  int total() const { return score.total(); }
  bool empty() const { return score.empty(); }
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
