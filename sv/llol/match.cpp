#include "sv/llol/match.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include "sv/util/ocv.h"

namespace sv {

void MatXyzr2MeanCovar(const cv::Mat& mat, MeanCovar3f& mc) {
  for (int r = 0; r < mat.rows; ++r) {
    for (int c = 0; c < mat.cols; ++c) {
      const auto& xyzr = mat.at<cv::Vec4f>(r, c);
      if (std::isnan(xyzr[0])) continue;
      mc.Add({xyzr[0], xyzr[1], xyzr[2]});
    }
  }
}

/// PointMatcher ===============================================================
PointMatcher::PointMatcher(int max_matches, const MatcherParams& params)
    : params_{params},
      win_size_{params.half_rows * 8 + 1, params.half_rows * 2 + 1} {
  matches_.reserve(max_matches);
}

std::string PointMatcher::Repr() const {
  return fmt::format(
      "PointMatcher(max_matches={}, max_score={}, nms={}, win_size={})",
      matches_.capacity(),
      params_.max_curve,
      params_.nms,
      sv::Repr(win_size_));
}

std::ostream& operator<<(std::ostream& os, const PointMatcher& rhs) {
  return os << rhs.Repr();
}

bool IsCellGood(const cv::Mat& grid, cv::Point px, double max_curve, bool nms) {
  // curve could be nan
  // Threshold check
  const auto& curve = grid.at<float>(px);
  if (!(curve < max_curve)) return false;

  // NMS check, nan neighbor is considered as inf
  if (nms) {
    const auto& curve_l = grid.at<float>(px.y, px.x - 1);
    const auto& curve_r = grid.at<float>(px.y, px.x + 1);
    if (curve > curve_l || curve > curve_r) return false;
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

  //  if (tbb) {
  //    tbb::parallel_for(tbb::blocked_range<int>(0, matches_.size()),
  //                      [&](const auto& block) {
  //                        for (int i = block.begin(); i < block.end(); ++i) {
  //                          MatchSingle(sweep, pano, i);
  //                        }
  //                      });
  //  } else {
  //    for (int i = 0; i < matches_.size(); ++i) {
  //      MatchSingle(sweep, pano, i);
  //    }
  //  }

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
  const int min_pts = 0.5 * win_size_.area();
  const auto it =
      std::remove_if(matches_.begin(), matches_.end(), [](const PointMatch& m) {
        return m.dst.n < 9;
      });
  matches_.erase(it, matches_.end());
}

void PointMatcher::MatchSingle(const LidarSweep& sweep,
                               const DepthPano& pano,
                               const cv::Point& gpx) {
  // Check if grid cell is good
  if (!IsCellGood(sweep.grid(), gpx, params_.max_curve, params_.nms)) {
    return;
  }

  const int i = gpx.y * sweep.grid().cols + gpx.x;
  auto& match = matches_.at(i);

  match.pt.x = (gpx.x + 0.5) * sweep.cell_size().width;
  match.pt.y = gpx.y * sweep.cell_size().height;

  // Compute normal dist around sweep cell
  const auto cell = sweep.CellAt(gpx);
  MatXyzr2MeanCovar(cell, match.src);

  // Transform to pano frame
  const Eigen::Vector3f pt_p = Eigen::Matrix3f::Identity() * match.src.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) return;

  // Compute normal dist around pano point
  const auto win = pano.BoundWinCenterAt(px_p, win_size_);

  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point px_w{wc + win.x, wr + win.y};
      const float rg_w = pano.GetRange(px_w);
      if (rg_w == 0 || CalcRangeDiffRel(rg_w, rg_p) > params_.range_ratio) {
        continue;
      }
      const auto p = pano.model_.Backward(px_w.y, px_w.x, rg_w);
      match.dst.Add({p.x, p.y, p.z});
    }
  }
}

cv::Mat PointMatcher::Draw(const LidarSweep& sweep) const {
  cv::Mat disp(cv::Size{sweep.grid().cols, sweep.grid().rows}, CV_32FC1, kNaNF);
  float max_pts = win_size_.area();
  for (const auto& match : matches_) {
    disp.at<float>(sweep.PixelToCell(match.pt)) = match.dst.n / max_pts;
  }
  return disp;
}

}  // namespace sv
