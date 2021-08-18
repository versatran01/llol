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

SweepGrid::SweepGrid(const cv::Size& sweep_size, const GridParams& params)
    : cell_size{params.cell_cols, params.cell_rows},
      max_score{params.max_score},
      nms{params.nms},
      score{sweep_size / cell_size, CV_32FC1},
      mask{sweep_size / cell_size, CV_8UC1} {
  CHECK_GT(max_score, 0);
  CHECK_EQ(cell_size.width * score.cols, sweep_size.width);
  CHECK_EQ(cell_size.height * score.rows, sweep_size.height);
  tf_p_s.resize(score.cols);
}

std::string SweepGrid::Repr() const {
  return fmt::format(
      "SweepGrid(cell_size={}, max_score={}, nms={}, col_range={}, score={}, "
      "mask={})",
      sv::Repr(cell_size),
      max_score,
      nms,
      sv::Repr(col_rg),
      sv::Repr(score),
      sv::Repr(mask));
}

int SweepGrid::Reduce(const LidarScan& scan, int gsize) {
  // Check
  CHECK_EQ(scan.col_rg.start, full() ? 0 : col_rg.end * cell_size.width);
  CHECK_EQ(score.rows * cell_size.height, scan.xyzr.rows);

  col_rg = scan.col_rg / cell_size.width;
  CHECK_LE(col_rg.end, score.cols);

  gsize = gsize <= 0 ? score.rows : gsize;
  CHECK_GE(gsize, 1);

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, score.rows, gsize),
      0,
      [&](const auto& block, int n) {
        for (int r = block.begin(); r < block.end(); ++r) {
          n += ReduceRow(scan, r);
        }
        return n;
      },
      std::plus<>{});
}

int SweepGrid::ReduceRow(const LidarScan& scan, int r) {
  int n = 0;
  for (int c = col_rg.start; c < col_rg.end; ++c) {
    const cv::Point px_g{c, r};
    const auto px_s = Grid2Sweep(px_g);
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    const auto curve = CalcCellCurve(scan, px_s, cell_size.width);
    ScoreAt(px_g) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int SweepGrid::Filter() {
  int n = 0;
  const int pad = static_cast<int>(nms);
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
  const int sr = px.y * cell_size.height;
  const int sc = px.x * cell_size.width;
  return {sc, sr, cell_size.width, 1};
}

bool SweepGrid::IsCellGood(const cv::Point& px) const {
  // curve could be nan
  // Threshold check
  const auto& m = ScoreAt(px);
  if (!(m < max_score)) return false;

  // NMS check, nan neighbor is considered as inf
  if (nms) {
    const auto& l = ScoreAt({px.x - 1, px.y});
    const auto& r = ScoreAt({px.x + 1, px.y});
    if (m > l || m > r) return false;
  }

  return true;
}

cv::Point SweepGrid::Sweep2Grid(const cv::Point& px_sweep) const {
  return {px_sweep.x / cell_size.width, px_sweep.y / cell_size.height};
}

cv::Point SweepGrid::Grid2Sweep(const cv::Point& px_grid) const {
  return {px_grid.x * cell_size.width, px_grid.y * cell_size.height};
}

int SweepGrid::Grid2Ind(const cv::Point& px_grid) const {
  return px_grid.y * score.cols + px_grid.x;
}

}  // namespace sv
