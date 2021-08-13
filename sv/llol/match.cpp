#include "sv/llol/match.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <Eigen/Geometry>

#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"
#include "sv/util/ocv.h"

namespace sv {

/// @brief Check if a point is a good candidate for matching
bool IsCellGood(const cv::Mat& grid,
                const cv::Point& px,
                double max_curve,
                bool nms);

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
  return fmt::format("nms={}, half_rows={}, max_score={}, range_ratio={}",
                     nms,
                     half_rows,
                     max_curve,
                     range_ratio);
}

/// PointMatcher ===============================================================
PointMatcher::PointMatcher(const cv::Size& grid_size,
                           const MatcherParams& params)
    : params_{params} {
  pano_win_size_.height = params_.half_rows * 2 + 1;
  pano_win_size_.width = pano_win_size_.height * 2 + 1;
  matches_.resize(grid_size.area());
}

std::string PointMatcher::Repr() const {
  return fmt::format("PointMatcher(max_matches={}, win_size={}, params=({}))",
                     matches_.capacity(),
                     sv::Repr(pano_win_size_),
                     params_.Repr());
}

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

void PointMatcher::Match(const LidarSweep& sweep,
                         const DepthPano& pano,
                         bool tbb) {
  const auto& grid = sweep.grid();
  matches_.clear();
  matches_.resize(grid.total());

  const int pad = static_cast<int>(params_.nms);

  if (tbb) {
    tbb::parallel_for(tbb::blocked_range<int>(0, grid.rows),
                      [&](const auto& block) {
                        for (int gr = block.begin(); gr < block.end(); ++gr) {
                          for (int gc = pad; gc < grid.cols - pad; ++gc) {
                            MatchSingle(sweep, pano, {gc, gr});
                          }
                        }
                      });
  } else {
    for (int gr = 0; gr < grid.rows; ++gr) {
      for (int gc = pad; gc < grid.cols - pad; ++gc) {
        MatchSingle(sweep, pano, {gc, gr});
      }
    }
  }

  // Clean up
  // TODO (chao): keep matches_ intact, for multiple rounds for match
  // Instead generate a compact version by copying valid ones and return
  const int min_pts = 0.5 * pano_win_size_.area();
  const auto it = std::remove_if(
      matches_.begin(), matches_.end(), [min_pts](const PointMatch& m) {
        return m.dst.n < 5;
      });
  matches_.erase(it, matches_.end());
}

void PointMatcher::MatchSingle(const LidarSweep& sweep,
                               const DepthPano& pano,
                               const cv::Point& px_g) {
  // Check if grid cell is good
  if (!IsCellGood(sweep.grid(), px_g, params_.max_curve, params_.nms)) {
    return;
  }

  // TODO (chao): project first and then compute mean and covar
  // Also don't compute src again
  const int i = px_g.y * sweep.grid().cols + px_g.x;
  auto& match = matches_.at(i);

  // Record sweep px
  match.px_s.x = (px_g.x + 0.5) * sweep.cell_size.width;
  match.px_s.y = px_g.y * sweep.cell_size.height;

  // Compute normal dist around sweep cell (if it is not already computed)
  if (!match.src.ok()) {
    const auto cell = sweep.CellAt(px_g);
    SweepCellMeanCovar(sweep, cell, match.src);
  }

  // Transform to pano frame
  const Eigen::Vector3f pt_p = Eigen::Matrix3f::Identity() * match.src.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) return;

  // Compute normal dist around pano point
  // TODO (chao): Check if px_p is close enough to match.px_p
  if (!match.dst.ok()) {
    match.px_p = px_p;
    const auto win = pano.BoundWinCenterAt(px_p, pano_win_size_);
    PanoWinMeanCovar(pano, win, rg_p, params_.range_ratio, match.dst);

    // if we have enough points then compute U
    if (match.dst.n > pano_win_size_.area() * 0.4) {
      match.U = MatrixSqrtUtU(match.dst.Covar().inverse().eval());
    }
  }
}

cv::Mat DrawMatches(const LidarSweep& sweep,
                    const std::vector<PointMatch>& matches) {
  cv::Mat disp(sweep.grid_size(), CV_32FC1, kNaNF);

  for (const auto& match : matches) {
    if (!match.ok()) continue;
    disp.at<float>(sweep.Pix2Cell(match.px_s)) = match.dst.n;
  }
  return disp;
}

}  // namespace sv
