#include "sv/llol/match.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"

namespace sv {

bool PointInSize(const cv::Point& p, const cv::Size& size) {
  return std::abs(p.x) <= size.width && std::abs(p.y) << size.height;
}

/// @brief Compute mean covar of cell in sweep
void SweepCellMeanCovar(const LidarSweep& sweep,
                        const cv::Rect& cell,
                        MeanCovar3f& mc) {
  // TODO (chao): for now only take first row of cell no matte what
  //  for (int r = 0; r < cell.height; ++r) {
  for (int c = 0; c < cell.width; ++c) {
    const auto& xyzr = sweep.XyzrAt({cell.x + c, cell.y});
    if (std::isnan(xyzr[0])) continue;
    mc.Add({xyzr[0], xyzr[1], xyzr[2]});
  }
  //  }
}

/// @brief Compute mean covar of win in pano
void PanoWinMeanCovar(const DepthPano& pano,
                      const cv::Rect& win,
                      float rg_p,
                      MeanCovar3f& mc) {
  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point px_w{wc + win.x, wr + win.y};
      const auto& dp = pano.PixelAt(px_w);
      const auto rg_w = dp.GetMeter();
      // TODO (chao): check if cnt is old enough
      if (rg_w == 0 || (std::abs(rg_w - rg_p) / rg_p) > pano.range_ratio) {
        continue;
      }
      const auto p = pano.model.Backward(px_w.y, px_w.x, rg_w);
      mc.Add({p.x, p.y, p.z});
    }
  }
}

/// ProjMatcher ================================================================
ProjMatcher::ProjMatcher(const MatcherParams& params)
    : cov_lambda{params.cov_lambda} {
  // Pano win size, for now width is twice the height
  pano_win_size.height = params.half_rows * 2 + 1;
  pano_win_size.width = pano_win_size.height * 2 + 1;
  min_pts = (params.half_rows + 1) * pano_win_size.width;
  max_dist_size = pano_win_size / 4;
}

std::string ProjMatcher::Repr() const {
  return fmt::format(
      "ProjMatcher(cov_lamda={}, min_pts={}, pano_win_size={}, "
      "max_dist_size={})",
      cov_lambda,
      min_pts,
      sv::Repr(pano_win_size),
      sv::Repr(max_dist_size));
}

int ProjMatcher::Match(SweepGrid& grid, const DepthPano& pano, int gsize) {
  const auto rows = grid.size().height;
  gsize = gsize <= 0 ? rows : gsize;
  CHECK_GE(gsize, 1);

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows, gsize),
      0,
      [&](const auto& blk, int n) {
        for (int gr = blk.begin(); gr < blk.end(); ++gr) {
          n += MatchRow(grid, pano, gr);
        }
        return n;
      },
      std::plus<>{});
}

int ProjMatcher::MatchRow(SweepGrid& grid, const DepthPano& pano, int gr) {
  int n = 0;
  // Note that here we use width instead of col_range, which means we will
  // revisit earlier matches in the current sweep
  for (int gc = 0; gc < grid.width(); ++gc) {
    n += MatchCell(grid, pano, {gc, gr});
  }
  return n;
}

int ProjMatcher::MatchCell(SweepGrid& grid,
                           const DepthPano& pano,
                           const cv::Point& px_g) {
  if (grid.MaskAt(px_g) == 0) return 0;

  auto& match = grid.MatchAt(px_g);
  CHECK(match.SweepOk());

  // Record sweep px
  //  match.px_s = grid.Grid2Sweep(px_g);
  //  match.px_s.x += grid.cell_size.width / 2;  // TODO (chao): hide cell_size?

  // Compute normal dist around sweep cell (if it is not already computed)
  // TODO: maybe move this to grid
  //  if (!match.mc_s.ok()) {
  //    const auto cell = grid.SweepCell(px_g);
  //    SweepCellMeanCovar(sweep, cell, match.mc_s);
  //  }

  // Transform to pano frame
  const Eigen::Vector3f pt_p = grid.tf_p_s.at(px_g.x) * match.mc_s.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) {
    // Bad projection, reset and return
    match.ResetPano();
    return 0;
  }

  // Check distance between new pix and old pix
  if (PointInSize(px_p - match.px_p, max_dist_size) && match.mc_p.ok()) {
    // If new and old are close and pano match is ok
    return 1;
  }

  // Compute normal dist around pano point
  match.px_p = px_p;
  match.mc_p.Reset();
  const auto win = pano.BoundWinCenterAt(px_p, pano_win_size);
  PanoWinMeanCovar(pano, win, rg_p, match.mc_p);

  // if we don't have enough points also reset and return
  if (match.mc_p.n < min_pts) {
    match.ResetPano();
    return 0;
  }
  // Otherwise compute U'U = inv(C + lambda * I)
  match.SqrtInfo(cov_lambda);
  return 1;
}

}  // namespace sv
