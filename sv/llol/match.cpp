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

void PointMatch::SqrtInfo(float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  cov.diagonal().array() += lambda;
  U = MatrixSqrtUtU(cov.inverse().eval());
}

/// @brief Check if a point is a good candidate for matching
bool IsCellGood(const cv::Mat& grid,
                const cv::Point& px,
                double max_curve,
                bool nms) {
  // curve could be nan
  // Threshold check
  const auto& m = grid.at<float>(px);
  if (!(m < max_curve)) return false;

  // NMS check, nan neighbor is considered as inf
  if (nms) {
    const auto& l = grid.at<float>(px.y, px.x - 1);
    const auto& r = grid.at<float>(px.y, px.x + 1);
    if (m > l || m > r) return false;
  }

  return true;
}

/// @brief Compute mean covar of cell in sweep
void SweepCellMeanCovar(const LidarSweep& sweep,
                        const cv::Rect& cell,
                        MeanCovar3f& mc) {
  for (int r = 0; r < cell.height; ++r) {
    for (int c = 0; c < cell.width; ++c) {
      const auto& xyzr = sweep.XyzrAt({cell.x + c, cell.y + r});
      if (std::isnan(xyzr[0])) continue;
      mc.Add({xyzr[0], xyzr[1], xyzr[2]});
    }
  }
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
ProjMatcher::ProjMatcher(const cv::Size& grid_size, const MatcherParams& params)
    : size{grid_size}, cov_lambda{params.cov_lambda} {
  // Pano win size, for now width is twice the height
  pano_win_size.height = params.half_rows * 2 + 1;
  pano_win_size.width = pano_win_size.height * 2 + 1;
  min_pts = (params.half_rows + 1) * pano_win_size.width;
  max_dist_size = pano_win_size / 4;

  matches.resize(grid_size.area());
}

std::string ProjMatcher::Repr() const {
  return fmt::format(
      "ProjMatcher(max_matches={}, grid_size={}, cov_lamda={}, min_pts={}, "
      "pano_win_size={}, max_dist_size={})",
      matches.size(),
      sv::Repr(size),
      cov_lambda,
      min_pts,
      sv::Repr(pano_win_size),
      sv::Repr(max_dist_size));
}

int ProjMatcher::Match(const LidarSweep& sweep,
                       const SweepGrid& grid,
                       const DepthPano& pano,
                       int gsize) {
  CHECK_EQ(matches.size(), grid.size().area());
  CHECK_EQ(size.width, grid.size().width);
  CHECK_EQ(size.height, grid.size().height);

  const auto rows = grid.size().height;
  gsize = gsize <= 0 ? rows : gsize;
  CHECK_GE(gsize, 1);

  // cache width
  width = grid.width();

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows, gsize),
      0,
      [&](const auto& blk, int n) {
        for (int gr = blk.begin(); gr < blk.end(); ++gr) {
          n += MatchRow(sweep, grid, pano, gr);
        }
        return n;
      },
      std::plus<>{});
}

int ProjMatcher::MatchRow(const LidarSweep& sweep,
                          const SweepGrid& grid,
                          const DepthPano& pano,
                          int gr) {
  int n = 0;
  // Note that here we use width instead of col_range, which means we will
  // revisit earlier matches in the current sweep
  for (int gc = 0; gc < grid.width(); ++gc) {
    if (grid.MaskAt({gc, gr}) == 0) continue;
    n += MatchCell(sweep, grid, pano, {gc, gr});
  }
  return n;
}

int ProjMatcher::MatchCell(const LidarSweep& sweep,
                           const SweepGrid& grid,
                           const DepthPano& pano,
                           const cv::Point& px_g) {
  const int mi = grid.Grid2Ind(px_g);
  auto& match = matches.at(mi);

  // Record sweep px
  match.px_s = grid.Grid2Sweep(px_g);
  match.px_s.x += grid.cell_size.width / 2;  // TODO (chao): hide cell_size?

  // Compute normal dist around sweep cell (if it is not already computed)
  if (!match.mc_s.ok()) {
    const auto cell = grid.SweepCell(px_g);
    SweepCellMeanCovar(sweep, cell, match.mc_s);
  }

  // Transform to pano frame
  const Eigen::Vector3f pt_p = grid.tf_p_s.at(px_g.x) * match.mc_s.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  // Bad projection, reset and return
  if (px_p.x < 0) {
    match.mc_p.Reset();
    return 0;
  }

  // Check distance between new pix and old pix
  const auto px_diff = px_p - match.px_p;

  // If new and old are too far or mc not ok, recompute
  if (!PointInSize(px_diff, max_dist_size) || !match.mc_p.ok()) {
    // Compute normal dist around pano point
    match.px_p = px_p;
    match.mc_p.Reset();
    const auto win = pano.BoundWinCenterAt(px_p, pano_win_size);
    PanoWinMeanCovar(pano, win, rg_p, match.mc_p);

    // if we don't have enough points also reset and return
    if (match.mc_p.n < min_pts) {
      match.mc_p.Reset();
      return 0;
    }
    // Otherwise compute U'U = inv(C + lambda * I)
    match.SqrtInfo(cov_lambda);
  }

  return 1;
}

int ProjMatcher::NumMatches() const {
  int k = 0;
  for (int r = 0; r < size.height; ++r) {
    for (int c = 0; c < width; ++c) {
      k += matches[Grid2Ind({c, r})].ok();
    }
  }
  return k;
}

void ProjMatcher::Reset() {
  matches.clear();
  matches.resize(size.area());
}

cv::Mat DrawMatches(const SweepGrid& grid,
                    const std::vector<PointMatch>& matches) {
  cv::Mat disp(grid.size(), CV_32FC1, kNaNF);

  for (const auto& match : matches) {
    const auto px_g = grid.Sweep2Grid(match.px_s);
    if (px_g.x >= grid.width()) continue;
    disp.at<float>(px_g) = match.mc_p.n;
  }
  return disp;
}

}  // namespace sv
