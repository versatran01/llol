#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"  // Repr

namespace sv {

bool SetBufAt(cv::Mat& buf, const cv::Point& px, float rg) {
  const uint16_t rg_raw = rg * Pixel::kScale;
  auto& tgt = buf.at<Pixel>(px);
  if (tgt.raw == 0) {
    tgt.raw = rg_raw;
    return true;
  }

  const auto tgt_meter = tgt.Metric();
  if (std::abs(rg - tgt_meter) / tgt_meter < 0.1) {
    tgt.SetMetric(rg);
    return true;
  }

  return false;
}

DepthPano::DepthPano(const cv::Size& size, float hfov)
    : model_{size, hfov}, dbuf_{size, CV_16UC1}, dbuf2_{size, CV_16UC1} {}

std::string DepthPano::Repr() const {
  return fmt::format("DepthPano({}, model={}, pixel=(scale={}, max_range={})",
                     sv::Repr(dbuf_),
                     model_.Repr(),
                     Pixel::kScale,
                     Pixel::kMaxRange);
}

cv::Rect DepthPano::BoundWinCenterAt(const cv::Point& pt,
                                     const cv::Size& win_size) const {
  const cv::Rect bound{cv::Point{}, size()};
  return WinCenterAt(pt, win_size) & bound;
}

int DepthPano::AddSweep(const LidarSweep& sweep, bool tbb) {
  CHECK(sweep.full());

  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, sweep.xyzr.rows),
        0,
        [&](const auto& block, int total) {
          for (int sr = block.begin(); sr < block.end(); ++sr) {
            total += AddSweepRow(sweep, sr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int sr = 0; sr < sweep.xyzr.rows; ++sr) {
      n += AddSweepRow(sweep, sr);
    }
  }

  ++num_sweeps_;
  return n;
}

int DepthPano::AddSweepRow(const LidarSweep& sweep, int sr) {
  int n = 0;

  for (int sc = 0; sc < sweep.xyzr.cols; ++sc) {
    const auto& xyzr = sweep.XyzrAt({sc, sr});
    const float rg_s = xyzr[3];  // precomputed range
    if (!(rg_s > 0)) continue;   // filter out nan

    // TODO (chao): transform xyz to pano frame
    Eigen::Map<const Eigen::Vector3f> pt_s(&xyzr[0]);
    const Eigen::Vector3f pt_p = sweep.tf_p_s.at(sc) * pt_s;
    const auto rg_p = pt_p.norm();

    // Project to pano
    const auto px_p = model_.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
    if (px_p.x < 0 || px_p.y < 0) continue;

    n += static_cast<int>(UpdateBuffer(px_p, rg_p));
  }

  return n;
}

bool DepthPano::UpdateBuffer(const cv::Point& px, float rg) {
  auto& tgt = dbuf_.at<Pixel>(px);
  if (tgt.raw == 0) {
    tgt.SetMetric(rg);
    return true;  // add new
  }

  const auto tgt_meter = tgt.Metric();
  if (std::abs(rg - tgt_meter) / tgt_meter < 0.1) {
    tgt.SetMetric(rg);
    return true;  // updated
  }

  return false;
}

int DepthPano::Render(bool tbb) {
  // clear pano2
  dbuf2_.setTo(0);

  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, dbuf_.rows),
        0,
        [&](const tbb::blocked_range<int>& blk, int total) {
          for (int r = blk.begin(); r < blk.end(); ++r) {
            total += RenderRow(r);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int r = 0; r < dbuf_.rows; ++r) {
      n += RenderRow(r);
    }
  }
  return n;
}

int DepthPano::RenderRow(int r1) {
  int n = 0;
  for (int c1 = 0; c1 < dbuf_.cols; ++c1) {
    const float rg1 = RangeAt({c1, r1});
    if (rg1 == 0) continue;

    // px1 -> xyz1
    const auto xyz1 = model_.Backward(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 =
        Eigen::Matrix3f::Identity() * xyz1_map + Eigen::Vector3f::Zero();
    const auto rg2 = xyz2.norm();

    // xyz2 -> px2
    const auto px2 = model_.Forward(xyz2.x(), xyz2.y(), xyz2.z(), rg2);
    if (px2.x < 0) continue;

    // check max range
    if (rg2 >= Pixel::kMaxRange) continue;

    // Check occlusion
    n += SetBufAt(dbuf2_, px2, rg2);
  }
  return n;
}

}  // namespace sv
