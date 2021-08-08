#include "sv/llol/lidar.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

float CellCurvature(const cv::Mat& cell) {
  // compute sum of range in cell
  float range_sum = 0.0F;

  for (int c = 0; c < cell.cols; ++c) {
    const auto& rg = cell.at<cv::Vec4f>(0, c)[3];
    if (std::isnan(rg)) return rg;  // early return nan
    range_sum += rg;
  }
  // range of mid point
  const float range_mid = cell.at<cv::Vec4f>(cell.cols / 2)[3];
  return std::abs(range_sum / cell.cols / range_mid - 1);
}

/// LidarSweep =================================================================
LidarSweep::LidarSweep(cv::Size sweep_size, cv::Size cell_size)
    : sweep_{sweep_size, CV_32FC4},
      cell_size_{cell_size},
      grid_{cv::Size{sweep_size.width / cell_size.width,
                     sweep_size.height / cell_size.height},
            CV_32FC1} {}

int LidarSweep::AddScan(const cv::Mat& scan, cv::Range scan_range, bool tbb) {
  // Check scan type is compatible
  CHECK_EQ(scan.type(), sweep_.type());
  // Check rows match between scan and mat
  CHECK_EQ(scan.rows, sweep_.rows);
  // Check scan width is not bigger than sweep
  CHECK_LE(scan.cols, sweep_.cols);
  // Check that the new scan start right after
  CHECK_EQ(scan_range.start, full() ? 0 : range_.end);

  // Save range and copy to storage
  range_ = scan_range;
  scan.copyTo(sweep_.colRange(range_));  // x,y,w,h

  // Compute curvature of scan
  auto subgrid = GetSubgrid(scan_range);
  return ReduceScan(scan, subgrid, tbb);
}

cv::Mat LidarSweep::GetSubgrid(cv::Range scan_range) {
  // compute corresponding grid block given scan range
  return grid_.colRange(scan_range.start / cell_size_.width,
                        scan_range.end / cell_size_.width);
}

int LidarSweep::ReduceScan(const cv::Mat& scan, cv::Mat& subgrid, bool tbb) {
  int num_valid = 0;

  if (tbb) {
    num_valid = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, subgrid.rows),
        0,
        [&](const tbb::blocked_range<int>& block, int total) {
          for (int gr = block.begin(); gr < block.end(); ++gr) {
            total += ReduceScanRow(scan, gr, subgrid);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int gr = 0; gr < subgrid.rows; ++gr) {
      num_valid += ReduceScanRow(scan, gr, subgrid);
    }
  }
  return num_valid;
}

int LidarSweep::ReduceScanRow(const cv::Mat& scan, int gr, cv::Mat& subgrid) {
  int num_valid = 0;
  for (int gc = 0; gc < subgrid.cols; ++gc) {
    const int sr = gr * cell_size_.height;
    const int sc = gc * cell_size_.width;
    const auto cell = scan.row(sr).colRange(sc, sc + cell_size_.width);
    const auto curve = CellCurvature(cell);
    subgrid.at<float>(gr, gc) = curve;
    num_valid += !std::isnan(curve);
  }
  return num_valid;
}

std::string LidarSweep::Repr() const {
  using sv::Repr;
  return fmt::format("LidarSweep(sweep={}, range={}, grid={}, cell_size={})",
                     Repr(sweep_),
                     Repr(range_),
                     Repr(grid_),
                     Repr(cell_size_));
}

std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
  return os << rhs.Repr();
}

/// DepthPano ==================================================================
LidarModel::LidarModel(cv::Size size, float hfov) : size_{size} {
  if (hfov <= 0) {
    hfov = kTauF / size.aspectRatio();
  }

  elev_max_ = hfov / 2.0F;
  elev_delta_ = hfov / (size.height - 1);
  azim_delta_ = kTauF / size.width;
  elevs_.resize(size_.height);
  for (int i = 0; i < size_.height; ++i) {
    elevs_[i] = SinCosF{elev_max_ - i * elev_delta_};
  }
  azims_.resize(size_.width);
  for (int i = 0; i < size_.width; ++i) {
    azims_[i] = SinCosF{kTauF - i * azim_delta_};
  }
}

cv::Point2i sv::LidarModel::Forward(float x, float y, float z, float r) const {
  cv::Point2i px{-1, -1};

  const int row = ToRow(z, r);
  if (!RowInside(row)) return px;

  const int col = ToCol(x, y);
  if (!ColInside(col)) return px;

  px.x = col;
  px.y = row;

  return px;
}

cv::Point3f LidarModel::Backward(int r, int c, float rg) const {
  const auto& elev = elevs_[r];
  const auto& azim = azims_[c];
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

int LidarModel::ToRow(float z, float r) const {
  const float elev = std::asin(z / r);
  return (elev_max_ - elev) / elev_delta_ + 0.5F;
}

int LidarModel::ToCol(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta_ + 0.5F;
}

std::string LidarModel::Repr() const {
  return fmt::format(
      "LidarModel(size={}, elev_max={:.4f}[deg], elev_delta={:.4f}[deg], "
      "azim_delta={:.4f}[deg], width_height_ratio={:.4f}, "
      "elev_azim_ratio={:.4f})",
      sv::Repr(size_),
      Rad2Deg(elev_max_),
      Rad2Deg(elev_delta_),
      Rad2Deg(azim_delta_),
      WidthHeightRatio(),
      ElevAzimRatio());
}

std::ostream& operator<<(std::ostream& os, const LidarModel& rhs) {
  return os << rhs;
}

/// DepthPano ==================================================================
DepthPano::DepthPano(cv::Size size, float hfov)
    : buf_{size, CV_16UC1}, buf2_{size, CV_16UC1}, model_{size, hfov} {}

std::string DepthPano::Repr() const {
  return fmt::format("DepthPano({}, model={}, scale={}, max_range={})",
                     sv::Repr(buf_),
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

int DepthPano::AddSweep(const LidarSweep& sweep, bool tbb) {
  CHECK(sweep.full());
  return AddSweep(sweep.sweep(), tbb);
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
    const float dot = xyz_p.dot(xyz_p) / (rg_s * rg_p);
    if (dot < 0) continue;

    // Project to pano
    const auto pt = model_.Forward(xyz_p.x(), xyz_p.y(), xyz_p.z(), rg_p);
    if (pt.x < 0) continue;

    // Update pano
    SetRange(pt, rg_p, buf_);
    ++num_added;
  }

  return num_added;
}

int DepthPano::Render(bool tbb) {
  // clear pano2
  buf2_.setTo(0);

  int num_rendered = 0;

  if (tbb) {
    num_rendered = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, buf_.rows),
        0,
        [&](const tbb::blocked_range<int>& blk, int total) {
          for (int r = blk.begin(); r < blk.end(); ++r) {
            total += RenderRow(r);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int r = 0; r < buf_.rows; ++r) {
      num_rendered += RenderRow(r);
    }
  }

  return num_rendered;
}

int DepthPano::RenderRow(int r1) {
  int num_rendered = 0;

  for (int c1 = 0; c1 < buf_.cols; ++c1) {
    const float rg1 = buf_.at<ushort>(r1, c1) / kScale;
    if (rg1 == 0) continue;

    // pano -> xyz1
    const auto xyz1 = model_.Backward(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 = Eigen::Matrix3f::Identity() * xyz1_map;
    const auto rg2 = xyz2.norm();

    // Check view point close
    const float cos = xyz2.dot(xyz2) / (rg1 * rg2);
    if (cos < 0) continue;

    // Project to mat2
    const auto pt2 = model_.Forward(xyz2.x(), xyz2.y(), xyz2.z(), rg2);
    if (pt2.x < 0) continue;

    SetRange(pt2, rg2, buf2_);
    ++num_rendered;
  }

  return num_rendered;
}

void DepthPano::CalcMeanCovar(cv::Rect win, MeanCovar3f& mc) const {
  // Compute mean and covar within window
  for (int wr = win.y; wr < win.y + win.height; ++wr) {
    for (int wc = win.x; wc < win.x + win.width; ++wc) {
      const float rg = GetRange({wc, wr});
      if (rg == 0) continue;
      const auto wp = model_.Backward(wr, wc, rg);
      mc.Add({wp.x, wp.y, wp.z});
    }
  }
}

}  // namespace sv
