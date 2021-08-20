#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"  // Repr

namespace sv {

DepthPano::DepthPano(const cv::Size& size, const PanoParams& params)
    : max_cnt{params.max_cnt},
      range_ratio{params.range_ratio},
      min_range{params.min_range},
      model{size, params.hfov},
      dbuf{size, CV_16UC2},
      dbuf2{size, CV_16UC2} {}

std::string DepthPano::Repr() const {
  return fmt::format(
      "DepthPano(max_cnt={}, range_ratio={}, model={}, dbuf={}, "
      "pixel=(scale={}, max_range={})",
      max_cnt,
      range_ratio,
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

int DepthPano::Add(const LidarSweep& sweep, int gsize) {
  CHECK(sweep.full());

  const int sweep_rows = sweep.size().height;
  gsize = gsize <= 0 ? sweep_rows : gsize;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, sweep_rows, gsize),
      0,
      [&](const auto& block, int n) {
        for (int sr = block.begin(); sr < block.end(); ++sr) {
          n += AddRow(sweep, sr);
        }
        return n;
      },
      std::plus<>{});
}

int DepthPano::AddRow(const LidarSweep& sweep, int sr) {
  int n = 0;

  const int sweep_cols = sweep.xyzr.cols;
  for (int sc = 0; sc < sweep_cols; ++sc) {
    const auto& xyzr = sweep.XyzrAt({sc, sr});
    const float rg_s = xyzr[3];  // precomputed range
    if (!(rg_s > 0)) continue;   // filter out nan

    Eigen::Map<const Eigen::Vector3f> pt_s(&xyzr[0]);
    const Eigen::Vector3f pt_p = sweep.tf_p_s.at(sc) * pt_s;
    const auto rg_p = pt_p.norm();

    // Project to pano
    const auto px_p = model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
    if (px_p.x < 0 || px_p.y < 0) continue;

    n += static_cast<int>(FuseDepth(px_p, rg_p));
  }

  return n;
}

bool DepthPano::FuseDepth(const cv::Point& px, float rg) {
  // TODO (chao): when adding consider distance, weigh far points less
  // Ignore too far and too close stuff
  if (rg < min_range || rg > DepthPixel::kMaxRange) return false;

  auto& p = dbuf.at<DepthPixel>(px);
  // If current pixel is empty, just use this range and give it half of max cnt
  if (p.raw == 0) {
    p.SetMeter(rg);
    p.cnt = max_cnt / 2;
    return true;
  }

  // If cnt is 0, this means there exists enough evidence that differ from
  // previous measurement, for example some new object just moved into view and
  // stayed long enough. In this case, we use the new range, but only increment
  // count by 1
  if (p.cnt == 0) {
    p.SetMeter(rg);
    ++p.cnt;
    return true;
  }

  // Otherwise we have a valid depth with cnt
  const auto rg0 = p.GetMeter();

  // Check if new and old are close enough
  if ((std::abs(rg - rg0) / rg0) < range_ratio) {
    // close, do a weighted update
    const auto rg1 = (rg0 * p.cnt + rg) / (p.cnt + 1);
    p.SetMeter(rg1);
    if (p.cnt < max_cnt) ++p.cnt;
    return true;
  } else {
    // not close, keep old but decrement its cnt
    if (p.cnt > 0) --p.cnt;
    return false;
  }
}

int DepthPano::Render(const Sophus::SE3f& tf_2_1, int gsize) {
  // clear pano2
  dbuf2.setTo(0);

  gsize = gsize <= 0 ? dbuf.rows : gsize;
  CHECK_GE(gsize, 1);

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, dbuf.rows, gsize),
      0,
      [&](const tbb::blocked_range<int>& blk, int n) {
        for (int r = blk.begin(); r < blk.end(); ++r) {
          n += RenderRow(tf_2_1, r);
        }
        return n;
      },
      std::plus<>{});
}

int DepthPano::RenderRow(const Sophus::SE3f& tf_2_1, int r1) {
  int n = 0;
  for (int c1 = 0; c1 < dbuf.cols; ++c1) {
    const float rg1 = RangeAt({c1, r1});
    if (rg1 == 0) continue;

    // px1 -> xyz1
    const auto xyz1 = model.Backward(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 = tf_2_1 * xyz1_map;
    const auto rg2 = xyz2.norm();

    // xyz2 -> px2
    const auto px2 = model.Forward(xyz2.x(), xyz2.y(), xyz2.z(), rg2);
    if (px2.x < 0) continue;

    // check max range
    if (rg2 >= DepthPixel::kMaxRange) continue;

    // Check occlusion
    n += UpdateBuffer(px2, rg2);
  }
  return n;
}

bool DepthPano::UpdateBuffer(const cv::Point& px, float rg) {
  auto& p = dbuf2.at<DepthPixel>(px);
  if (p.raw == 0) {
    p.SetMeter(rg);
    return true;
  }

  // Depth buffer update, handles occlusion
  const auto rg0 = p.GetMeter();
  if (rg < rg0) {
    p.SetMeter(rg);
    return true;
  }

  return false;
}

}  // namespace sv
