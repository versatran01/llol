#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"  // Repr

namespace sv {

bool SetRange(cv::Mat& buf, const cv::Point& px, float rg) {
  const uint16_t rg_raw = rg * DepthPano::kScale;
  auto& tgt = buf.at<ushort>(px);
  if (tgt == 0) {
    tgt = rg_raw;
    return true;
  } else {
    tgt = std::min(rg_raw, tgt);
    return false;
  }
}

int PanoAddSweepRow(const LidarModel& model,
                    const cv::Mat& sweep,
                    cv::Mat& dbuf,
                    int sr) {
  int n = 0;

  for (int sc = 0; sc < sweep.cols; ++sc) {
    const auto& xyzr = sweep.at<cv::Vec4f>(sr, sc);
    const float rg_s = xyzr[3];  // precomputed range
    if (!(rg_s > 0)) continue;   // filter out nan

    // TODO (chao): transform xyz to pano frame
    Eigen::Map<const Eigen::Vector3f> pt_s(&xyzr[0]);
    const Eigen::Vector3f pt_p = Eigen::Matrix3f::Identity() * pt_s;
    const auto rg_p = pt_p.norm();

    // Project to pano
    const auto px_p = model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
    if (px_p.x < 0) continue;

    n += SetRange(dbuf, px_p, rg_p);
  }

  return n;
}

int PanoAddSweep(const LidarModel& model,
                 const cv::Mat& sweep,
                 cv::Mat& dbuf,
                 bool tbb) {
  int n = 0;

  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, sweep.rows),
        0,
        [&](const tbb::blocked_range<int>& block, int total) {
          for (int sr = block.begin(); sr < block.end(); ++sr) {
            total += PanoAddSweepRow(model, sweep, dbuf, sr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int sr = 0; sr < sweep.rows; ++sr) {
      n += PanoAddSweepRow(model, sweep, dbuf, sr);
    }
  }

  return n;
}

int RenderPanoRow(const LidarModel& model,
                  const cv::Mat& dbuf1,
                  int r1,
                  cv::Mat& dbuf2) {
  int n = 0;
  for (int c1 = 0; c1 < dbuf1.cols; ++c1) {
    const float rg1 = dbuf1.at<ushort>(r1, c1) / DepthPano::kScale;
    if (rg1 == 0) continue;

    // px1 -> xyz1
    const auto xyz1 = model.Backward(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 =
        Eigen::Matrix3f::Identity() * xyz1_map + Eigen::Vector3f::Zero();
    const auto rg2 = xyz2.norm();

    // xyz2 -> px2
    const auto px2 = model.Forward(xyz2.x(), xyz2.y(), xyz2.z(), rg2);
    if (px2.x < 0) continue;

    // check max range
    if (rg2 >= DepthPano::kMaxRange) continue;

    // Check occlusion
    n += SetRange(dbuf2, px2, rg2);
  }
  return n;
}

int PanoRender(const LidarModel& model,
               const cv::Mat& dbuf1,
               cv::Mat& dbuf2,
               bool tbb) {
  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, dbuf1.rows),
        0,
        [&](const tbb::blocked_range<int>& blk, int total) {
          for (int r = blk.begin(); r < blk.end(); ++r) {
            total += RenderPanoRow(model, dbuf1, r, dbuf2);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int r = 0; r < dbuf1.rows; ++r) {
      n += RenderPanoRow(model, dbuf1, r, dbuf2);
    }
  }
  return n;
}

DepthPano::DepthPano(cv::Size size, float hfov)
    : model_{size, hfov}, dbuf_{size, CV_16UC1}, dbuf2_{size, CV_16UC1} {}

std::string DepthPano::Repr() const {
  return fmt::format("DepthPano({}, model={}, scale={}, max_range={})",
                     sv::Repr(dbuf_),
                     model_.Repr(),
                     kScale,
                     kMaxRange);
}

std::ostream& operator<<(std::ostream& os, const DepthPano& rhs) {
  return os << rhs.Repr();
}

cv::Rect DepthPano::WinCenterAt(cv::Point pt, cv::Size win_size) const {
  return {cv::Point{pt.x - win_size.width / 2, pt.y - win_size.height / 2},
          win_size};
}

cv::Rect DepthPano::BoundWinCenterAt(cv::Point pt, cv::Size win_size) const {
  const cv::Rect bound{cv::Point{}, size()};
  return WinCenterAt(pt, win_size) & bound;
}

int DepthPano::AddSweep(const cv::Mat& sweep, bool tbb) {
  ++num_sweeps_;
  return PanoAddSweep(model_, sweep, dbuf_, tbb);
}

int DepthPano::Render(bool tbb) {
  // clear pano2
  dbuf2_.setTo(0);
  return PanoRender(model_, dbuf_, dbuf2_, tbb);
}

void DepthPano::CalcMeanCovar(cv::Rect win, MeanCovar3f& mc) const {
  // Compute mean and covar within window
  for (int r = win.y; r < win.y + win.height; ++r) {
    for (int c = win.x; c < win.x + win.width; ++c) {
      const float rg = GetRange({c, r});
      if (rg == 0) continue;
      const auto p = model_.Backward(r, c, rg);
      mc.Add({p.x, p.y, p.z});
    }
  }
}

}  // namespace sv
