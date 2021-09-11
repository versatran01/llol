#include "sv/llol/gicp.h"

#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/llol/cost.h"
#include "sv/util/ocv.h"

namespace sv {

bool PointInSize(const cv::Point& p, const cv::Size& size) {
  return std::abs(p.x) <= size.width && std::abs(p.y) <= size.height;
}

/// GicpSolver =================================================================
GicpSolver::GicpSolver(const GicpParams& params)
    : iters{params.outer, params.inner},
      cov_lambda{params.cov_lambda},
      pano_win{params.half_rows * 2 + 1, params.half_cols * 2 + 1},
      imu_weight{params.imu_weight},
      min_eigval{params.min_eigval} {
  pano_min_pts = std::max(pano_win.height, pano_win.width) * 2;
}

std::string GicpSolver::Repr() const {
  return fmt::format(
      "GicpSolver(outer={}, inner={}, cov_lambda={}, min_pano_pts={}, "
      "imu_weight={}, pano_win={})",
      iters.first,
      iters.second,
      cov_lambda,
      pano_min_pts,
      imu_weight,
      sv::Repr(pano_win));
}

int GicpSolver::Match(SweepGrid& grid, const DepthPano& pano, int gsize) {
  const auto rows = grid.rows();
  gsize = gsize <= 0 ? rows : gsize;

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

int GicpSolver::MatchRow(SweepGrid& grid, const DepthPano& pano, int gr) {
  int n = 0;
  for (int gc = 0; gc < grid.cols(); ++gc) {
    n += MatchCell(grid, pano, {gc, gr});
  }
  return n;
}

int GicpSolver::MatchCell(SweepGrid& grid,
                          const DepthPano& pano,
                          const cv::Point& px_g) {
  auto& match = grid.MatchAt(px_g);
  if (!match.GridOk()) return 0;

  // Transform to pano frame
  const auto& T_p_g = grid.TfAt(px_g.x);
  const auto pt_g = T_p_g * match.mc_g.mean;  // grid point in pano frame
  const auto rg_g = pt_g.norm();  // range of grid point in pano frame

  // Project to pano
  const auto px_p = pano.model.Forward(pt_g.x(), pt_g.y(), pt_g.z(), rg_g);
  if (px_p.x < 0) {
    // Bad projection, reset pano and return
    match.ResetPano();
    return 0;
  }

  // Check distance between new pix and old pix (allow 1 pix in azim direction)
  if (match.PanoOk()) {
    if (px_p == match.px_p) {
      // If new and old are the same and pano match is good we reuse this match
      return 1;
    }
  }

  // Compute mean covar around pano point
  const auto weight = pano.MeanCovarAt(px_p, pano_win, rg_g, match.mc_p);
  //  pano.UpdateMean(px_p, rg_p, match.mc_p.mean);

  // if we don't have enough points also reset and return 0
  if (match.mc_p.n < pano_min_pts) {
    match.ResetPano();
    return 0;
  }

  // Now this is a good match
  match.px_p = px_p;
  // Otherwise compute U'U = inv(C + lambda * I) and we have a good match
  //  match.CalcSqrtInfo(cov_lambda);
  match.CalcSqrtInfo(T_p_g.rotationMatrix());
  // Although scale could be subsumed by U, we kept it for visualization
  // weight / pano_area is in [0, 1], but if it is too small, then imu cost will
  // dominate and drift. So we make this scale [0.5, 1]
  //  match.scale = std::sqrt(weight / pano_win.area() / 2 + 0.5);
  match.scale = std::sqrt(static_cast<float>(match.mc_p.n) / pano_win.area());
  return 1;
}

}  // namespace sv
