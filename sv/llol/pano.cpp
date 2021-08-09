#include "sv/llol/pano.h"

#include <fmt/core.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/ocv.h"

namespace sv {

DepthPano::DepthPano(cv::Size size, float hfov)
    : model_{size, hfov},
      dbuf_{size, CV_16UC1},
      pbuf_{size, CV_32FC3},
      dbuf2_{size, CV_16UC1} {}

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
  int num_added = 0;

  if (tbb) {
    num_added = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, sweep.rows),
        0,
        [&](const tbb::blocked_range<int>& block, int total) {
          for (int sr = block.begin(); sr < block.end(); ++sr) {
            total += AddSweepRow(sweep, sr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int sr = 0; sr < sweep.rows; ++sr) {
      num_added += AddSweepRow(sweep, sr);
    }
  }

  ++num_sweeps_;
  return num_added;
}

int DepthPano::AddSweepRow(const cv::Mat& sweep, int sr) {
  int num_added = 0;

  for (int sc = 0; sc < sweep.cols; ++sc) {
    const auto& xyzr = sweep.at<cv::Vec4f>(sr, sc);
    const float rg_s = xyzr[3];  // precomputed range
    if (!(rg_s > 0)) continue;   // filter out nan

    // TODO (chao): transform xyz to pano frame
    Eigen::Map<const Eigen::Vector3f> xyz_s(&xyzr[0]);
    const Eigen::Vector3f xyz_p = Eigen::Matrix3f::Identity() * xyz_s;
    const auto rg_p = xyz_p.norm();

    // Check viewpoint close
    //    const float dot = xyz_p.dot(xyz_p) / (rg_s * rg_p);
    //    if (dot < 0) continue;

    // Project to pano
    const auto pt = model_.Forward(xyz_p.x(), xyz_p.y(), xyz_p.z(), rg_p);
    if (pt.x < 0) continue;

    // Update pano
    SetRange(pt, rg_p, dbuf_);
    ++num_added;
  }

  return num_added;
}

int DepthPano::Render(bool tbb) {
  // clear pano2
  dbuf2_.setTo(0);

  int num_rendered = 0;

  if (tbb) {
    num_rendered = tbb::parallel_reduce(
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
      num_rendered += RenderRow(r);
    }
  }

  return num_rendered;
}

int DepthPano::RenderRow(int r1) {
  int num_rendered = 0;

  for (int c1 = 0; c1 < dbuf_.cols; ++c1) {
    const float rg1 = dbuf_.at<ushort>(r1, c1) / kScale;
    if (rg1 == 0) continue;

    // pano -> xyz1
    const auto xyz1 = model_.Backward(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 = Eigen::Matrix3f::Identity() * xyz1_map;
    const auto rg2 = xyz2.norm();

    // Check view point close
    //    const float cos = xyz2.dot(xyz2) / (rg1 * rg2);
    //    if (cos < 0) continue;

    // Project to mat2
    const auto pt2 = model_.Forward(xyz2.x(), xyz2.y(), xyz2.z(), rg2);
    if (pt2.x < 0) continue;

    SetRange(pt2, rg2, dbuf2_);
    ++num_rendered;
  }

  return num_rendered;
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

// int DepthPano::DbufToPBuf(bool tbb) { return 0; }

// int DepthPano::DbufToPBufRow(int r) {
//  int n = 0;
//  for (int c = 0; c < dbuf_.cols; ++c) {
//    const float rg = dbuf_.at<ushort>(r, c) / kScale;
//    if (rg == 0) continue;
//  }

//  return n;
//}

}  // namespace sv
