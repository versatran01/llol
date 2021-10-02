#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <opencv2/core.hpp>

#include "sv/util/ocv.h"  // Repr

namespace sv {

using Vector3f = Eigen::Vector3f;

DepthPano::DepthPano(const cv::Size& size, const PanoParams& params)
    : max_cnt{params.max_cnt},
      min_sweeps{params.min_sweeps},
      min_range{params.min_range},
      max_range{params.max_range},
      win_ratio{params.win_ratio},
      fuse_ratio{params.fuse_ratio},
      align_gravity{params.align_gravity},
      min_match_ratio{params.min_match_ratio},
      max_translation{params.max_translation},
      model{size, params.vfov},
      dbuf{size, CV_16UC2},
      dbuf2{size, CV_16UC2} {
  if (max_range <= 0) max_range = DepthPixel::kMaxRange;
  CHECK_LE(0, min_range);
  CHECK_LT(min_range, max_range);
  CHECK_LE(max_range, DepthPixel::kMaxRange);
}

std::string DepthPano::Repr() const {
  return fmt::format(
      "DepthPano(max_cnt={}, min_sweeps={}, min_range={}, max_range={}, "
      "win_ratio={}, fuse_ratio={}, match_ratio={}, align_gravity={}, "
      "max_translation={}, model={}, dbuf={}, pixel=(scale={}, max_range={})",
      max_cnt,
      min_sweeps,
      min_range,
      max_range,
      win_ratio,
      fuse_ratio,
      min_match_ratio,
      align_gravity,
      max_translation,
      model.Repr(),
      sv::Repr(dbuf),
      DepthPixel::kScale,
      DepthPixel::kMaxRange);
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
    const auto& pixel_s = sweep.PixelAt({sc, sr});
    if (!pixel_s.Ok()) continue;

    // Transform into pano frame
    const auto pt_p = sweep.TfAt(sc) * pixel_s.Vec3fMap();
    const auto rg_p = pt_p.norm();

    // Ignore too far and too close stuff
    if (rg_p < min_range || rg_p > max_range) continue;

    // Project to pano
    const auto px_p = model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
    if (px_p.x < 0 || px_p.y < 0) continue;

    n += static_cast<int>(FuseDepth(px_p, rg_p));
  }

  return n;
}

bool DepthPano::FuseDepth(const cv::Point& px, float rg) {
  auto& pixel = PixelAt(px);

  // If depth is 0, this is a new point and we give it a relatively large cnt
  if (pixel.raw == 0) {
    pixel.SetRangeCount(rg, max_cnt / 2);
    return true;
  }

  // If cnt is empty (no matter what the depth value is, since it doesn't
  // matter), we just set the range to the new one.
  // This could happend at the beginning of the odom, or when a pixel is cleared
  // by new evidence, or right after a new rendering
  if (pixel.cnt == 0) {
    pixel.SetRangeCount(rg, 2);
    return true;
  }

  // Otherwise we have a valid depth with some evidence (cnt > 0)
  const auto rg0 = pixel.GetRange();

  // Check if new and old are close enough
  if ((std::abs(rg - rg0) / rg0) < fuse_ratio) {
    // close enough, do a weighted update
    const auto rg1 = (rg0 * pixel.cnt + rg) / (pixel.cnt + 1);
    pixel.SetRange(rg1);
    // And increment cnt but do not exceed max
    if (pixel.cnt < max_cnt) ++pixel.cnt;
    return true;
  } else {
    // not close, keep old but decrement its cnt
    if (pixel.cnt > 0) --pixel.cnt;
    return false;
  }
}

bool DepthPano::ShouldRender(const Sophus::SE3d& tf_p2_p1,
                             double match_ratio) const {
  // This is to prevent too frequent render
  if (num_sweeps <= min_sweeps) return false;

  // match ratio is the most important criteria
  if (match_ratio < min_match_ratio) return true;

  // Otherwise we have enough match, then we check translation
  // TODO (chao): Replace this with vel * time
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

  const int total = tbb::parallel_reduce(
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

  // set number of sweeps to 1
  num_sweeps = 1;
  return total;
}

int DepthPano::RenderRow(const Sophus::SE3f& tf_p2_p1, int r1) {
  int n = 0;

  for (int c1 = 0; c1 < cols(); ++c1) {
    const auto& dp1 = PixelAt({c1, r1});
    // We skip pixel that is empty or uncertainy
    if (dp1.raw == 0 || dp1.cnt < max_cnt / 4) continue;

    // px1 -> xyz1
    const auto pt1 = model.Backward(r1, c1, dp1.GetRange());
    Eigen::Map<const Vector3f> pt1_map(&pt1.x);

    // xyz1 -> xyz2
    const auto pt2 = tf_p2_p1 * pt1_map;
    const auto rg2 = pt2.norm();

    if (rg2 < min_range || rg2 > max_range) continue;

    // xyz2 -> px2
    const auto px2 = model.Forward(pt2.x(), pt2.y(), pt2.z(), rg2);
    if (px2.x < 0) continue;

    // Check for occlusion
    n += UpdateBuffer(px2, rg2, dp1.cnt);
  }

  return n;
}

bool DepthPano::UpdateBuffer(const cv::Point& px, float rg, int cnt) {
  auto& pixel = dbuf2.at<DepthPixel>(px);

  // if the destination pixel is empty, or the new rg is smaller than the old
  // one, we update the depth
  if (pixel.raw == 0 || rg < pixel.GetRange()) {
    // When rendering a new depth pano, if the original pixel is well estimated
    // (high cnt), this means that it also has good visibility from the current
    // viewpoint. On the other hand, if it has low cnt, this means that it was
    // probably occluded. Therefore, we simply half the original cnt and make it
    // the new one
    pixel.SetRangeCount(rg, cnt / 2);
    return true;
  }

  return false;
}

float DepthPano::CalcMeanCovar(cv::Rect win, float rg, MeanCovar3f& mc) const {
  mc.Reset();

  // Make sure window is within bound
  win = win & cv::Rect{cv::Point{}, size()};

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
