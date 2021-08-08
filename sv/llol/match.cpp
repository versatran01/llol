#include "sv/llol/match.h"

#include <fmt/core.h>

#include "sv/util/ocv.h"

namespace sv {

/// PointMatcher ===============================================================
PointMatcher::PointMatcher(int max_matches, const MatcherParams& params)
    : params_{params},
      win_size_{params.half_rows * 4 + 1, params.half_rows * 2 + 1} {
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

bool PointMatcher::IsCellGood(const cv::Mat& grid, cv::Point px) const {
  // curve could be nan
  // Threshold check
  const auto& curve = grid.at<float>(px);
  if (!(curve < params_.max_curve)) return false;

  // NMS check, nan neighbor is considered as inf
  if (params_.nms) {
    const auto& curve_l = grid.at<float>(px.y, px.x - 1);
    const auto& curve_r = grid.at<float>(px.y, px.x + 1);
    if (curve > curve_l || curve > curve_r) return false;
  }

  return true;
}

void PointMatcher::Match(const LidarSweep& sweep, const DepthPano& pano) {
  matches_.clear();

  const auto& grid = sweep.grid();
  const int pad = static_cast<int>(params_.nms);

  for (int gr = pad; gr < grid.rows - pad; ++gr) {
    for (int gc = pad; gc < sweep.grid_width() - pad; ++gc) {
      if (!IsCellGood(grid, {gc, gr})) continue;

      // Get the mid point in sweep
      const int sr = gr * sweep.cell_size().height;
      const int sc = (gc + 0.5) * sweep.cell_size().width;
      const cv::Point px_s{sc, sr};
      const auto& xyzr_s = sweep.XyzrAt(px_s);
      const auto rg_s = xyzr_s[3];
      if (!(rg_s > 0)) continue;

      // Transform xyz from sweep to pano frame
      Eigen::Map<const Eigen::Vector3f> pt_s(&xyzr_s[0]);
      const Eigen::Vector3f pt_p = Eigen::Matrix3f::Identity() * pt_s;
      const float rg_p = pt_p.norm();

      // Check viewpoint close
      const float cos = pt_p.dot(pt_p) / (rg_s * rg_p);
      if (cos < 0) continue;

      // Project to pano
      const auto px_p = pano.model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
      if (px_p.x < 0) continue;

      PointMatch match;
      // take a window around that pixel in pano and compute its mean
      const auto win = pano.BoundWinCenterAt(px_p, win_size_);
      pano.CalcMeanCovar(win, match.dst);
      if (match.dst.n < 9) continue;

      // TODO (chao): use point for now, consider using mean and cov
      match.src.mean = pt_s;
      match.pt = px_s;

      matches_.push_back(match);
    }
  }
}

cv::Mat PointMatcher::Draw(const LidarSweep& sweep) const {
  cv::Mat disp(sweep.grid_size(), CV_32FC1, kNaNF);
  float max_pts = win_size_.area();
  for (const auto& match : matches_) {
    disp.at<float>(sweep.PixelToCell(match.pt)) = match.dst.n / max_pts;
  }
  return disp;
}

}  // namespace sv
