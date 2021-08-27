#pragma once

#include <sophus/se3.hpp>

#include "sv/llol/match.h"
#include "sv/llol/scan.h"

namespace sv {

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.01F};  // score > max_score will be discarded
  bool nms{true};          // non-minimum suppression in Filter()
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid final : public ScanBase {
  /// Params
  cv::Size cell_size;
  float max_score{};
  bool nms{};

  /// Data
  std::vector<Sophus::SE3f> tfs;    // transforms of each column
  std::vector<PointMatch> matches;  // all matches

  SweepGrid() = default;
  explicit SweepGrid(const cv::Size& sweep_size, const GridParams& params = {});

  /// @brief Repr / <<
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const SweepGrid& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Score, Filter and Reduce
  cv::Vec2i Add(const LidarScan& scan, int gsize = 0);
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
  float& ScoreAt(const cv::Point& px) { return mat.at<float>(px); }
  float ScoreAt(const cv::Point& px) const { return mat.at<float>(px); }
  PointMatch& MatchAt(const cv::Point& px) { return matches.at(Px2Ind(px)); }
  const PointMatch& MatchAt(const cv::Point& px) const {
    return matches.at(Px2Ind(px));
  }

  /// @brief Pxiel coordinates conversion (sweep <-> grid)
  cv::Point Sweep2Grid(const cv::Point& px) const {
    return {px.x / cell_size.width, px.y / cell_size.height};
  }
  cv::Point Grid2Sweep(const cv::Point& px) const {
    return {px.x * cell_size.width, px.y * cell_size.height};
  }
  int Px2Ind(const cv::Point& px) const { return px.y * cols() + px.x; }

  /// @brief Interpolate poses of each col (cell)
  void Interp(const std::vector<Sophus::SE3f>& traj);

  /// @brief Draw
  cv::Mat DrawFilter() const;
  cv::Mat DrawMatch() const;
};

}  // namespace sv
