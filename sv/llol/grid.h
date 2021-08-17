#pragma once

#include "sv/llol/scan.h"

namespace sv {

struct GridParams {
  int cell_rows{2};
  int cell_cols{16};
  float max_score{0.1};  // cell score larger will be discarded
  bool nms{true};        // non-minimum suppression in grid after score thresh

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const GridParams& rhs) {
    return os << rhs.Repr();
  }
};

/// @struct Sweep Grid summarizes sweep into reduced-sized grid
struct SweepGrid {
  /// Data
  GridParams params{};
  cv::Range col_rg{};                // working range in this grid
  cv::Mat score;                     // smoothness score, smaller the better
  cv::Mat mask;                      // 1 means potential good match
  std::vector<Sophus::SE3f> tf_p_s;  // transforms from sweep to pano

  SweepGrid() = default;
  explicit SweepGrid(const cv::Size& sweep_size, const GridParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const SweepGrid& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Reduce scan into score grid
  /// @return Number of valid cells (not NaN)
  int Reduce(const LidarScan& scan, bool tbb = false);
  int ReduceRow(const LidarScan& scan, int r);

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
  cv::Size cell_size() const noexcept {
    return {params.cell_cols, params.cell_rows};
  }
};

}  // namespace sv
