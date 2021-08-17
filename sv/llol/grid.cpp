#include "sv/llol/grid.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

static constexpr float kValidCellRatio = 0.8;

/// @brief Compute scan curvature starting from px with size (width, 1)
float CalcCellCurve(const LidarScan& scan, const cv::Point& px, int width) {
  // compute sum of range in cell
  int num = 0;
  float sum = 0.0F;

  const int half = width / 2;
  const auto mid = scan.RangeAt({px.x + half, px.y});
  if (std::isnan(mid)) return kNaNF;

  for (int c = 0; c < width; ++c) {
    const auto rg = scan.RangeAt({px.x + c, px.y});
    if (std::isnan(rg)) continue;
    sum += rg;
    ++num;
  }

  // Discard if there are too many nans in this cell
  if (num < kValidCellRatio * width) return kNaNF;
  return std::abs(sum / mid / num - 1);
}

std::string GridParams::Repr() const {
  return fmt::format("(cell_rows={}, cell_cols={}, max_score={}, nms={})",
                     cell_rows,
                     cell_cols,
                     max_score,
                     nms);
}

SweepGrid::SweepGrid(const cv::Size& sweep_size, const GridParams& params)
    : params{params},
      score{sweep_size / cell_size(), CV_32FC1},
      mask{sweep_size / cell_size(), CV_8UC1} {
  CHECK_GT(params.max_score, 0);
  CHECK_EQ(params.cell_cols * score.cols, sweep_size.width);
  CHECK_EQ(params.cell_rows * score.rows, sweep_size.height);
  tf_p_s.resize(score.cols);
}

std::string SweepGrid::Repr() const {
  return fmt::format("SweepGrid(size={}, col_range={}, params={})",
                     sv::Repr(size()),
                     sv::Repr(col_rg),
                     params.Repr());
}

int SweepGrid::Reduce(const LidarScan& scan, bool tbb) {
  // Check
  CHECK_EQ(scan.col_rg.start, full() ? 0 : col_rg.end * params.cell_cols);
  CHECK_EQ(score.rows * params.cell_rows, scan.xyzr.rows);
  col_rg = scan.col_rg / params.cell_cols;
  CHECK_LE(col_rg.end, score.cols);

  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, score.rows),
        0,
        [&](const auto& block, int total) {
          for (int r = block.begin(); r < block.end(); ++r) {
            total += ReduceRow(scan, r);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int r = 0; r < score.rows; ++r) {
      n += ReduceRow(scan, r);
    }
  }
  return n;
}

int SweepGrid::ReduceRow(const LidarScan& scan, int r) {
  int n = 0;
  for (int c = col_rg.start; c < col_rg.end; ++c) {
    const cv::Point px_g{c, r};
    const auto px_s = Grid2Sweep(px_g);
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    const auto curve = CalcCellCurve(scan, px_s, params.cell_cols);
    ScoreAt(px_g) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int SweepGrid::Filter() {
  int n = 0;
  const int pad = static_cast<int>(params.nms);
  for (int gr = 0; gr < mask.rows; ++gr) {
    for (int gc = col_rg.start + pad; gc < col_rg.end - pad; ++gc) {
      const cv::Point px{gc, gr};
      const int good = static_cast<int>(IsCellGood(px));
      MaskAt(px) = good;
      n += good;
    }
  }
  return n;
}

cv::Rect SweepGrid::SweepCell(const cv::Point& px) const {
  const int sr = px.y * params.cell_rows;
  const int sc = px.x * params.cell_cols;
  return {sc, sr, params.cell_cols, 1};
}

bool SweepGrid::IsCellGood(const cv::Point& px) const {
  // curve could be nan
  // Threshold check
  const auto& m = ScoreAt(px);
  if (!(m < params.max_score)) return false;

  // NMS check, nan neighbor is considered as inf
  if (params.nms) {
    const auto& l = ScoreAt({px.x - 1, px.y});
    const auto& r = ScoreAt({px.x + 1, px.y});
    if (m > l || m > r) return false;
  }

  return true;
}

cv::Point SweepGrid::Sweep2Grid(const cv::Point& px_sweep) const {
  return {px_sweep.x / params.cell_cols, px_sweep.y / params.cell_rows};
}

cv::Point SweepGrid::Grid2Sweep(const cv::Point& px_grid) const {
  return {px_grid.x * params.cell_cols, px_grid.y * params.cell_rows};
}

int SweepGrid::Grid2Ind(const cv::Point& px_grid) const {
  return px_grid.y * score.cols + px_grid.x;
}

}  // namespace sv
