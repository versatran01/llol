#include "sv/llol/match.h"

#include <fmt/core.h>

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

void PointMatcher::Match(const LidarSweep& sweep, const DepthPano& pano) {
  matches_.clear();

  const auto& grid = sweep.grid();
  const int pad = static_cast<int>(params_.nms);

  for (int gr = pad; gr < grid.rows - pad; ++gr) {
    for (int gc = pad; gc < sweep.grid_width() - pad; ++gc) {
      if (!IsCellGood(grid, {gc, gr}, params_.max_curve, params_.nms)) {
        continue;
      }

      // Get the mid point in cell
      const int sr = gr * sweep.cell_size().height;
      const int sc = gc * sweep.cell_size().width;
      const int sc_mid = sc + sweep.cell_size().width / 2;
      const cv::Point px_s{sc_mid, sr};
      const auto& xyzr_s = sweep.XyzrAt(px_s);
      const auto rg_s = xyzr_s[3];
      if (!(rg_s > 0)) continue;

      // Transform xyz from sweep to pano frame
      Eigen::Map<const Eigen::Vector3f> pt_s(&xyzr_s[0]);
      const Eigen::Vector3f pt_p = Eigen::Matrix3f::Identity() * pt_s;
      const float rg_p = pt_p.norm();

      // Project to pano
      const auto px_p = pano.model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
      if (px_p.x < 0) continue;

      PointMatch match;

      // Compute normal distribution around sweep point

      // Compute normal distribution around pano point
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

      if (match.dst.n < 9) continue;

      // TODO (chao): use point for now, consider using mean and cov
      // Get cell
      const cv::Mat cell =
          sweep.sweep().row(sr).colRange(sc, sc + sweep.cell_size().width);

      MatXyzr2MeanCovar(cell, match.src);

      match.pt = px_s;
      matches_.push_back(match);
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
