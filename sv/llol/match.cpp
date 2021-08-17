#include "sv/llol/match.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"

namespace sv {

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

// TODO (chao): improve this by considering distance to center
float CalcRangeDiffRel(float rg1, float rg2) {
  return std::abs(rg1 - rg2) / std::max(rg1, rg2);
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
                      float rg_ratio,
                      MeanCovar3f& mc) {
  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point px_w{wc + win.x, wr + win.y};
      const float rg_w = pano.RangeAt(px_w);
      if (rg_w == 0 || CalcRangeDiffRel(rg_w, rg_p) > rg_ratio) continue;
      const auto p = pano.model_.Backward(px_w.y, px_w.x, rg_w);
      mc.Add({p.x, p.y, p.z});
    }
  }
}

std::string MatcherParams::Repr() const {
  return fmt::format("half_rows={}, min_dist={}, range_ratio={}, cov_lambda={}",
                     half_rows,
                     min_dist,
                     range_ratio,
                     cov_lambda);
}

/// ProjMatcher ================================================================
ProjMatcher::ProjMatcher(const cv::Size& grid_size, const MatcherParams& params)
    : grid_size{grid_size}, params{params} {
  pano_win_size.height = params.half_rows * 2 + 1;
  pano_win_size.width = pano_win_size.height * 2 + 1;
  min_pts = params.half_rows * pano_win_size.width;

  matches.resize(grid_size.area());
}

std::string ProjMatcher::Repr() const {
  return fmt::format(
      "ProjMatcher(max_matches={}, win_size={}, min_pts={}, params=({}))",
      matches.capacity(),
      sv::Repr(pano_win_size),
      min_pts,
      params.Repr());
}

int ProjMatcher::Match(const LidarSweep& sweep,
                       const SweepGrid& grid,
                       const DepthPano& pano,
                       bool tbb) {
  CHECK_EQ(matches.size(), grid.size().area());
  CHECK_EQ(grid_size.width, grid.size().width);
  CHECK_EQ(grid_size.height, grid.size().height);

  const auto rows = grid.size().height;

  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, rows),
        0,
        [&](const auto& block, int total) {
          for (int gr = block.begin(); gr < block.end(); ++gr) {
            total += MatchRow(sweep, grid, pano, gr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int gr = 0; gr < rows; ++gr) {
      n += MatchRow(sweep, grid, pano, gr);
    }
  }

  return n;
}

int ProjMatcher::MatchRow(const LidarSweep& sweep,
                          const SweepGrid& grid,
                          const DepthPano& pano,
                          int gr) {
  int n = 0;
  // Note that here we use width instead of col_range, which means we will redo
  // earlier matches
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
  const int mi = grid.Pix2Ind(px_g);
  auto& match = matches.at(mi);

  // Record sweep px
  match.px_s = grid.Grid2Sweep(px_g);
  match.px_s.x += grid.cell_size().width / 2;

  // Compute normal dist around sweep cell (if it is not already computed)
  if (!match.mc_s.ok()) {
    const auto cell = grid.SweepCell(px_g);
    SweepCellMeanCovar(sweep, cell, match.mc_s);
  }

  // Transform to pano frame
  const Eigen::Vector3f pt_p = grid.tfs[px_g.x] * match.mc_s.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  // Bad projection, clear dst
  if (px_p.x < 0) {
    match.mc_p.Reset();
    return 0;
  }

  // Check distance between new pix and old pix
  const auto px_diff = px_p - match.px_p;
  const auto px_dist2 = px_diff.x * px_diff.x + px_diff.y * px_diff.y;

  // If new and old are too far and mc not ok, recompute
  if (px_dist2 > params.min_dist * params.min_dist || !match.mc_p.ok()) {
    // Compute normal dist around pano point
    match.px_p = px_p;
    match.mc_p.Reset();
    const auto win = pano.BoundWinCenterAt(px_p, pano_win_size);
    PanoWinMeanCovar(pano, win, rg_p, params.range_ratio, match.mc_p);
  }

  // if we don't have enough points then reset pano
  if (match.mc_p.n < min_pts) {
    match.mc_p.Reset();
    return 0;
  }

  return 1;
}

void ProjMatcher::Reset() {
  matches.clear();
  matches.resize(grid_size.area());
}

cv::Mat DrawMatches(const SweepGrid& grid,
                    const std::vector<PointMatch>& matches) {
  cv::Mat disp(grid.size(), CV_32FC1, kNaNF);

  for (const auto& match : matches) {
    disp.at<float>(grid.Sweep2Grid(match.px_s)) = match.mc_p.n;
  }
  return disp;
}

}  // namespace sv
