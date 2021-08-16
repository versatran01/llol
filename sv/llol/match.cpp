#include "sv/llol/match.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"
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
      const auto& xyzr = sweep.PixAt({cell.x + c, cell.y + r});
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
  return fmt::format("nms={}, half_rows={}, max_score={}, range_ratio={}",
                     nms,
                     half_rows,
                     max_curve,
                     range_ratio);
}

/// ProjMatcher ================================================================
ProjMatcher::ProjMatcher(const cv::Size& grid_size, const MatcherParams& params)
    : params{params}, mask{grid_size, CV_8UC1} {
  pano_win_size.height = params.half_rows * 2 + 1;
  pano_win_size.width = pano_win_size.height * 2 + 1;
  min_pts = params.half_rows * pano_win_size.width;

  matches.resize(mask.total());
  tfs.resize(mask.cols);
}

std::string ProjMatcher::Repr() const {
  return fmt::format(
      "ProjMatcher(max_matches={}, win_size={}, min_pts={}, params=({}))",
      matches.capacity(),
      sv::Repr(pano_win_size),
      min_pts,
      params.Repr());
}

int ProjMatcher::Filter(const LidarSweep& sweep) {
  const auto& grid = sweep.grid;
  const auto grid_range = sweep.grid_range();

  CHECK_EQ(mask.rows, grid.rows);
  CHECK_LE(mask.cols, grid.cols);
  // Check that the new grid start right after
  CHECK_EQ(grid_range.start, full() ? 0 : col_range.end);

  // Update internal
  id = sweep.id;
  col_range = grid_range;

  int n = 0;
  const int pad = static_cast<int>(params.nms);
  for (int gr = 0; gr < grid.rows; ++gr) {
    for (int gc = col_range.start + pad; gc < col_range.end - pad; ++gc) {
      bool good = IsCellGood(grid, {gc, gr}, params.max_curve, params.nms);
      mask.at<uint8_t>(gr, gc) = good;
      n += good;
    }
  }

  return n;
}

int ProjMatcher::Match(const LidarSweep& sweep,
                       const DepthPano& pano,
                       bool tbb) {
  // sweep id same as internal id
  CHECK_EQ(id, sweep.id);
  // width same as grid width
  CHECK_EQ(width(), sweep.grid_width());

  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, mask.rows),
        0,
        [&](const auto& block, int total) {
          for (int gr = block.begin(); gr < block.end(); ++gr) {
            total += MatchRow(sweep, pano, gr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int gr = 0; gr < mask.rows; ++gr) {
      n += MatchRow(sweep, pano, gr);
    }
  }

  return n;
}

int ProjMatcher::MatchRow(const LidarSweep& sweep,
                          const DepthPano& pano,
                          int gr) {
  int n = 0;
  // Note that here we use width instead of col_range, which means we will redo
  // earlier matches
  for (int gc = 0; gc < width(); ++gc) {
    if (mask.at<uint8_t>(gr, gc) == 0) continue;
    n += MatchCell(sweep, pano, {gc, gr});
  }
  return n;
}

int ProjMatcher::MatchCell(const LidarSweep& sweep,
                           const DepthPano& pano,
                           const cv::Point& px_g) {
  const int mi = px_g.y * mask.cols + px_g.x;
  auto& match = matches.at(mi);

  // Record sweep px
  match.px_s.x = (px_g.x + 0.5) * sweep.cell_size.width;
  match.px_s.y = px_g.y * sweep.cell_size.height;

  // Compute normal dist around sweep cell (if it is not already computed)
  if (!match.mc_s.ok()) {
    const auto cell = sweep.CellAt(px_g);
    SweepCellMeanCovar(sweep, cell, match.mc_s);
  }

  // Transform to pano frame
  const Eigen::Vector3f pt_p = tfs[px_g.x] * match.mc_s.mean;
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
  if (px_dist2 > params.min_dist2 || !match.mc_p.ok()) {
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
  matches.resize(mask.total());
}

cv::Mat DrawMatches(const LidarSweep& sweep,
                    const std::vector<PointMatch>& matches) {
  cv::Mat disp(sweep.grid_size(), CV_32FC1, kNaNF);

  for (const auto& match : matches) {
    disp.at<float>(sweep.Pix2Cell(match.px_s)) = match.mc_p.n;
  }
  return disp;
}

}  // namespace sv
