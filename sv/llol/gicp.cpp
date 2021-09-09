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
      max_dist{params.half_rows / 2, params.half_rows / 2},
      imu_weight{params.imu_weight},
      min_eigval{params.min_eigval} {
  pano_win.height = params.half_rows * 2 + 1;
  pano_win.width = params.half_cols * 2 + 1;
  pano_min_pts = pano_win.area() / 2;
}

std::string GicpSolver::Repr() const {
  return fmt::format(
      "GicpSolver(outer={}, inner={}, cov_lambda={}, min_pano_pts={}, "
      "imu_weight={}, pano_win={}, max_dist={})",
      iters.first,
      iters.second,
      cov_lambda,
      pano_min_pts,
      imu_weight,
      sv::Repr(pano_win),
      sv::Repr(max_dist));
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
  const auto pt_p = T_p_g * match.mc_g.mean;  // grid point in pano frame
  const auto rg_p = pt_p.norm();  // range of grid point in pano frame

  // Project to pano
  const auto px_p = pano.model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) {
    // Bad projection, reset pano and return
    match.ResetPano();
    return 0;
  }

  // Check distance between new pix and old pix (allow 1 pix in azim direction)
  //  if (std::abs(px_p.x - match.px_p.x) <= 1 && px_p.y == match.px_p.y &&
  //      match.PanoOk()) {
  if (px_p == match.px_p && match.PanoOk()) {
    // If new and old are the same and pano match is good we reuse this match

    // but we set mean to the new match point if it is valid and assume the
    // structure (covariance) stays the same
    // pano.UpdateMean(px_p, rg_p, match.mc_p.mean);
    return 1;
  }

  // Compute mean covar around pano point
  match.px_p = px_p;
  const auto weight = pano.MeanCovarAt(px_p, pano_win, rg_p, match.mc_p);
  //  pano.UpdateMean(px_p, rg_p, match.mc_p.mean);

  // if we don't have enough points also reset and return 0
  if (match.mc_p.n < pano_min_pts) {
    match.ResetPano();
    return 0;
  }
  // Otherwise compute U'U = inv(C + lambda * I) and we have a good match
  //  match.CalcSqrtInfo(cov_lambda);
  match.CalcSqrtInfo(T_p_g.rotationMatrix());
  // Although scale could be subsumed by U, we kept it for visualization
  match.scale = weight / pano_win.area();
  match.U *= std::sqrt(match.scale);
  return 1;
}

}  // namespace sv
