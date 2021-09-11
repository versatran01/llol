#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <opencv2/core.hpp>

#include "sv/util/ocv.h"  // Repr

namespace sv {

DepthPano::DepthPano(const cv::Size& size, const PanoParams& params)
    : max_cnt{params.max_cnt},
      min_range{params.min_range},
      win_ratio{params.win_ratio},
      fuse_ratio{params.fuse_ratio},
      align_gravity{params.align_gravity},
      min_match_ratio{params.min_match_ratio},
      max_translation{params.max_translation},
      model{size, params.vfov},
      dbuf{size, CV_16UC2},
      dbuf2{size, CV_16UC2} {}

std::string DepthPano::Repr() const {
  return fmt::format(
      "DepthPano(max_cnt={}, min_range={}, win_ratio={}, fuse_ratio={}, "
      "align_gravity={}, min_match_ratio={}, max_translation={}, model={}, "
      "dbuf={}, pixel=(scale={}, max_range={})",
      max_cnt,
      min_range,
      win_ratio,
      fuse_ratio,
      align_gravity,
      min_match_ratio,
      max_translation,
      model.Repr(),
      sv::Repr(dbuf),
      DepthPixel::kScale,
      DepthPixel::kMaxRange);
}

cv::Rect DepthPano::BoundWinCenterAt(const cv::Point& pt,
                                     const cv::Size& win_size) const {
  const cv::Rect bound{cv::Point{}, size()};
  return WinCenterAt(pt, win_size) & bound;
}

int DepthPano::Add(const LidarSweep& sweep, const cv::Range& curr, int gsize) {
  gsize = gsize <= 0 ? sweep.rows() : gsize;

  // increment added sweep
  num_sweeps += static_cast<float>(curr.size()) / sweep.cols();

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, sweep.rows(), gsize),
      0,
      [&](const auto& blk, int n) {
        for (int sr = blk.begin(); sr < blk.end(); ++sr) {
          n += AddRow(sweep, curr, sr);
        }
        return n;
      },
      std::plus<>{});
}

int DepthPano::AddRow(const LidarSweep& sweep, const cv::Range& curr, int sr) {
  int n = 0;

  for (int sc = curr.start; sc < curr.end; ++sc) {
    const auto& xyzr = sweep.XyzrAt({sc, sr});
    const float rg_s = xyzr[3];  // precomputed range
    if (!(rg_s > 0)) continue;   // filter out nan

    // Transform into pano frame
    Eigen::Map<const Vector3f> pt_s(&xyzr[0]);
    const auto pt_p = sweep.TfAt(sc) * pt_s;
    const auto rg_p = pt_p.norm();

    // Project to pano
    const auto px_p = model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
    if (px_p.x < 0 || px_p.y < 0) continue;

    n += static_cast<int>(FuseDepth(px_p, rg_p));
  }

  return n;
}

bool DepthPano::FuseDepth(const cv::Point& px, float rg) {
  // TODO (chao): should we increment count based on the distance?

  // Ignore too far and too close stuff
  if (rg < min_range || rg >= DepthPixel::kMaxRange) return false;

  auto& p = PixelAt(px);
  // If current pixel is empty, just use this range and give it half of max cnt
  if (p.raw == 0) {
    p.SetRangeCount(rg, max_cnt / 2);
    return true;
  }

  // If cnt is 0 while range is not, this means there exists enough evidence
  // that differ from previous measurement, for example some new object just
  // entered and stayed long enough. In this case, we use the new range, but
  // only increment count by 1
  if (p.cnt == 0) {
    p.SetRange(rg);
    ++p.cnt;
    return true;
  }

  // Otherwise we have a valid depth with some evidence (cnt > 0)
  const auto rg0 = p.GetRange();

  // Check if new and old are close enough
  if ((std::abs(rg - rg0) / rg0) < fuse_ratio) {
    // close enough, do a weighted update
    const auto rg1 = (rg0 * p.cnt + rg) / (p.cnt + 1);
    p.SetRange(rg1);
    // And increment cnt but do not exceed max
    if (p.cnt < max_cnt) ++p.cnt;
    return true;
  } else {
    // not close, keep old but decrement its cnt
    if (p.cnt > 0) --p.cnt;
    return false;
  }
}

bool DepthPano::ShouldRender(const Sophus::SE3d& tf_p2_p1,
                             double match_ratio) const {
  if (num_sweeps < max_cnt) return false;

  // match ratio is the most important criteria
  if (match_ratio < min_match_ratio) return true;

  // Otherwise we have enough match, then we check translation
  if (max_translation > 0) {
    const auto trans = tf_p2_p1.translation().norm();
    if (trans > max_translation) return true;
  }

  // Note that it is highly unlikely that we will reach here because match ratio
  // would already decide to re-render

  // cos_rp is just col z of rotation dot with e_z, which is just R22
  const auto R22 = tf_p2_p1.rotationMatrix()(2, 2);
  const auto cos_max_rp = std::cos(model.elev_max * 2.0 / 3.0);
  return R22 < cos_max_rp;
}

int DepthPano::Render(Sophus::SE3f tf_p2_p1, int gsize) {
  // clear pano2
  dbuf2.setTo(0);
  gsize = gsize <= 0 ? rows() : gsize;

  const int n = tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows(), gsize),
      0,
      [&](const auto& blk, int n) {
        for (int r = blk.begin(); r < blk.end(); ++r) {
          n += RenderRow(tf_p2_p1, r);
        }
        return n;
      },
      std::plus<>{});

  cv::swap(dbuf, dbuf2);

  // Need at least 2 for pano to be ready
  // for a well update pano, we will set it to max_cnt / 4.
  // for one that is not sufficiently updated, we set it to num_sweeps / 4 but
  // make sure it is more than 2
  num_sweeps =
      std::max(2.0F, std::min(num_sweeps, static_cast<float>(max_cnt)) / 4);

  return n;
}

int DepthPano::RenderRow(const Sophus::SE3f& tf_p2_p1, int r1) {
  int n = 0;

  for (int c1 = 0; c1 < cols(); ++c1) {
    const auto& dp1 = PixelAt({c1, r1});
    const auto rg1 = dp1.GetRange();
    if (rg1 == 0) continue;

    // px1 -> xyz1
    const auto pt1 = model.Backward(r1, c1, rg1);
    Eigen::Map<const Vector3f> pt1_map(&pt1.x);

    // xyz1 -> xyz2
    const auto pt2 = tf_p2_p1 * pt1_map;
    const auto rg2 = pt2.norm();

    // xyz2 -> px2
    const auto px2 = model.Forward(pt2.x(), pt2.y(), pt2.z(), rg2);
    if (px2.x < 0) continue;

    // Check for occlusion
    n += UpdateBuffer(px2, rg2, dp1.cnt);
  }

  return n;
}

bool DepthPano::UpdateBuffer(const cv::Point& px, float rg, int cnt) {
  if (rg < min_range || rg >= DepthPixel::kMaxRange) return false;

  auto& dp2 = dbuf2.at<DepthPixel>(px);
  // if the destination pixel is empty, then just set it to range
  if (dp2.raw == 0) {
    // We set cnt to a low value so that it can be cleared quickly
    dp2.SetRangeCount(rg, 2);
    return true;
  }

  // Depth buffer update, handles occlusion by taking the closer one
  const auto rg2 = dp2.GetRange();
  if (rg < rg2) {
    dp2.SetRange(rg);
    return true;
  }

  return false;
}

float DepthPano::MeanCovarAt(const cv::Point& px,
                             const cv::Size& size,
                             float rg,
                             MeanCovar3f& mc) const {
  mc.Reset();
  const auto win = BoundWinCenterAt(px, size);
  float weight = 0.0;

  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point px_w{wc + win.x, wr + win.y};
      const auto& dp = PixelAt(px_w);
      const auto rg_w = dp.GetRange();

      // Check for validity and range similarity
      if (rg_w == 0 || (std::abs(rg_w - rg) / rg) > win_ratio) continue;

      // Add 3d point
      const auto pt = model.Backward(px_w.y, px_w.x, rg_w);
      mc.Add({pt.x, pt.y, pt.z});
      weight += dp.cnt;
    }
  }

  return weight / max_cnt;
}

void DepthPano::UpdateMean(const cv::Point& px,
                           float rg,
                           Eigen::Vector3f& mean) const {
  const auto& dp = PixelAt(px);
  const auto rg_w = dp.GetRange();

  if (rg_w == 0 || (std::abs(rg_w - rg) / rg) > win_ratio) return;
  const auto pt = model.Backward(px.y, px.x, rg_w);
  mean.x() = pt.x;
  mean.y() = pt.y;
  mean.z() = pt.z;
}

const std::vector<cv::Mat>& DepthPano::DrawRangeCount() const {
  static std::vector<cv::Mat> disp;
  cv::split(dbuf, disp);
  return disp;
}

const std::vector<cv::Mat>& DepthPano::DrawRangeCount2() const {
  static std::vector<cv::Mat> disp;
  cv::split(dbuf2, disp);
  return disp;
}

}  // namespace sv
