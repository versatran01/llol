#include "sv/llol/gicp.h"

#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"

namespace sv {

bool PointInSize(const cv::Point& p, const cv::Size& size) {
  return std::abs(p.x) <= size.width && std::abs(p.y) <= size.height;
}

/// GicpSolver =================================================================
GicpSolver::GicpSolver(const GicpParams& params)
    : iters{params.outer, params.inner}, cov_lambda{params.cov_lambda} {
  pano_win.height = params.half_rows * 2 + 1;
  pano_win.width = params.half_rows * 4 + 1;
  pano_min_pts = (params.half_rows + 1) * pano_win.width;
  max_dist = pano_win / 4;
}

std::string GicpSolver::Repr() const {
  return fmt::format(
      "GicpSolver(outer={}, inner={}, cov_lambda={}, min_pano_pts={}, "
      "pano_win={}, max_dist={})",
      iters.first,
      iters.second,
      cov_lambda,
      pano_min_pts,
      sv::Repr(pano_win),
      sv::Repr(max_dist));
}

int GicpSolver::Match(SweepGrid& grid, const DepthPano& pano, int gsize) {
  const auto rows = grid.size().height;
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
  for (int gc = 0; gc < grid.size().width; ++gc) {
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
  // TODO (chao): move this transform somewhere else
  const Eigen::Vector3f pt_p = grid.tfs.at(px_g.x) * match.mc_g.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) {
    // Bad projection, reset and return
    match.ResetPano();
    return 0;
  }

  // Check distance between new pix and old pix
  if (PointInSize(px_p - match.px_p, max_dist) && match.PanoOk()) {
    //  if (px_p == match.px_p && match.PanoOk()) {
    // If new and old are close and pano match is ok
    // we reuse this match and there is no need to recompute
    return 1;
  }

  // Compute mean covar around pano point
  match.px_p = px_p;
  pano.MeanCovarAt(px_p, pano_win, rg_p, match.mc_p);

  // if we don't have enough points also reset and return
  if (match.mc_p.n < pano_min_pts) {
    match.ResetPano();
    return 0;
  }
  // Otherwise compute U'U = inv(C + lambda * I) and we have a good match
  match.SqrtInfo(cov_lambda);
  return 1;
}

}  // namespace sv
