#include "sv/llol/odom.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>

namespace sv {

std::string Repr(const cv::Mat& mat) {
  return fmt::format("(hwc=({},{},{}), depth={})",
                     mat.rows,
                     mat.cols,
                     mat.channels(),
                     mat.depth());
}

std::string Repr(const cv::Range& range) {
  return fmt::format("[{},{})", range.start, range.end);
}

std::string Repr(const cv::Size& size) {
  return fmt::format("(h={}, w={})", size.height, size.width);
}

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
    : cell_size_{cell_size},
      sweep_{sweep_size, CV_32FC4},
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
DepthPano::DepthPano(cv::Size size)
    : mat_{size, CV_16UC1},
      mat2_{size, CV_16UC1},
      azim_delta_{kTauF / size.width} {
  // assumes equal aspect ratio
  wh_ratio_ = static_cast<float>(size.width) / size.height;
  // assume horizontal fov is centered at 0
  const float hfov = kTauF / wh_ratio_;
  elev_max_ = hfov / 2.0F;
  elev_delta_ = hfov / (size.height - 1);
  azim_delta_ = kTauF / size.width;

  // Precompute elevs and azims sin and cos
  elevs_.resize(size.height);
  for (int i = 0; i < size.height; ++i) {
    elevs_[i] = SinCosF{elev_max_ - i * elev_delta_};
  }
  azims_.resize(size.width);
  for (int i = 0; i < size.width; ++i) {
    azims_[i] = SinCosF{kTauF - i * azim_delta_};
  }
}

std::string DepthPano::Repr() const {
  using sv::Repr;
  return fmt::format(
      "DepthPano({}, wh_ratio={}, elev_max={}[deg], elev_delta={}[deg], "
      "azim_delta={}[deg], scale={}, max_range={})",
      Repr(mat_),
      wh_ratio_,
      Rad2Deg(elev_max_),
      Rad2Deg(elev_delta_),
      Rad2Deg(azim_delta_),
      kScale,
      kMaxRange);
}

cv::Rect DepthPano::WinAt(const cv::Point& pt,
                          const cv::Size& half_size) const {
  return {cv::Point{pt.x - half_size.width, pt.y - half_size.height},
          cv::Size{half_size.width * 2 + 1, +half_size.height * 2 + 1}};
}

cv::Rect DepthPano::BoundedWinAt(const cv::Point& pt,
                                 const cv::Size& half_size) const {
  const cv::Rect bound{cv::Point{}, size()};
  return WinAt(pt, half_size) & bound;
}

std::ostream& operator<<(std::ostream& os, const DepthPano& rhs) {
  return os << rhs.Repr();
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
        [&](const tbb::blocked_range<int>& blk, int total) {
          for (int sr = blk.begin(); sr < blk.end(); ++sr) {
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

int DepthPano::AddSweepRow(const cv::Mat& sweep, int row) {
  int num_added = 0;
  for (int col = 0; col < sweep.cols; ++col) {
    const auto& xyzr = sweep.at<cv::Vec4f>(row, col);
    if (!(xyzr[3] > 0)) continue;

    Eigen::Map<const Eigen::Vector3f> xyz_s(&xyzr[0]);
    // TODO (chao): transform xyz to pano frame
    const Eigen::Vector3f xyz_p = Eigen::Matrix3f::Identity() * xyz_s;

    const auto rg_p = xyz_p.norm();
    const int pr = ToRow(xyz_p.z(), rg_p);
    if (!RowInside(pr)) continue;

    const int pc = ToCol(xyz_p.x(), xyz_p.y());
    if (!ColInside(pc)) continue;

    // TODO (chao): Need to check if view point is similar

    RawAt(pr, pc) = rg_p * kScale;
    ++num_added;
  }

  return num_added;
}

void DepthPano::Render(bool tbb) {
  // clear pano2
  mat2_.setTo(0);

  if (tbb) {
    tbb::parallel_for(tbb::blocked_range<int>(0, mat_.rows),
                      [&](const tbb::blocked_range<int>& blk) {
                        for (int pr = blk.begin(); pr < blk.end(); ++pr) {
                          RenderRow(pr);
                        }
                      });
  } else {
    for (int pr = 0; pr < mat_.rows; ++pr) {
      RenderRow(pr);
    }
  }

  // set num_sweeps back to 1
  num_sweeps_ = 1;
}

void DepthPano::RenderRow(int r1) {
  for (int c1 = 0; c1 < mat_.cols; ++c1) {
    const float rg1 = MetricAt(r1, c1);
    if (rg1 == 0) continue;

    // pano -> xyz1
    const auto xyz1 = To3d(r1, c1, rg1);
    Eigen::Map<const Eigen::Vector3f> xyz1_map(&xyz1.x);

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 = Eigen::Matrix3f::Identity() * xyz1_map;
    const auto rg2 = xyz2.norm();

    // compute row and col into mat2
    const int r2 = ToRow(xyz2.z(), rg2);
    if (!RowInside(r2)) continue;
    const int c2 = ToCol(xyz2.x(), xyz2.y());
    if (!ColInside(c2)) continue;

    mat2_.at<ushort>(r2, c2) = rg2 * kScale;
  }
}

int DepthPano::ToRow(float z, float r) const {
  const float elev = std::asin(z / r);
  return (elev_max_ - elev) / elev_delta_ + 0.5F;
}

int DepthPano::ToCol(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta_ + 0.5F;
}

cv::Point3f DepthPano::To3d(int r, int c, float rg) const noexcept {
  const auto& elev = elevs_[r];
  const auto& azim = azims_[c];
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

/// FeatureMatcher =============================================================
// DataMatcher::DataMatcher(int max_matches, const MatcherParams& params)
//    : params_{params} {
//  matches_.reserve(max_matches);
//}

// std::string DataMatcher::Repr() const {
//  return fmt::format("DataMatcher(max_matches={}, max_score={}, nms={})",
//                     matches_.capacity(),
//                     params_.max_score,
//                     params_.nms);
//}

// std::ostream& operator<<(std::ostream& os, const DataMatcher& rhs) {
//  return os << rhs.Repr();
//}

// void DataMatcher::Match(const LidarSweep& sweep,
//                        const PointGrid& grid,
//                        const DepthPano& pano) {
//  matches_.clear();

//  const auto& feats = grid.mat();
//  const cv::Size half_size(params_.half_rows * pano.wh_ratio(),
//                           params_.half_rows);

//  const int pad = static_cast<int>(params_.nms);

//  for (int fr = pad; fr < feats.rows - pad; ++fr) {
//    for (int fc = pad; fc < grid.width() - pad; ++fc) {
//      const auto& score = grid.ScoreAt(fr, fc);
//      if (!(score < params_.max_score)) continue;

//      // NMS
//      if (params_.nms) {
//        const auto& score_l = feats.at<float>(fr, fc - 1);
//        const auto& score_r = feats.at<float>(fr, fc + 1);
//        if (score > score_l || score > score_r) continue;
//      }

//      // Get the point in sweep
//      const int sr = fr * grid.win.height;
//      const int sc = (fc + 0.5) * grid.win.width;
//      const auto& xyzr = sweep.XyzrAt(sr, sc);

//      // Transform xyz to pano frame
//      Eigen::Map<const Eigen::Vector3f> xyz_s(&xyzr[0]);
//      const Eigen::Vector3f xyz_p = Eigen::Matrix3f::Identity() * xyz_s;
//      const float rg = xyz_p.norm();

//      const int pr = pano.ToRow(xyz_p.z(), rg);
//      if (!pano.RowInside(pr)) continue;
//      const int pc = pano.ToCol(xyz_p.x(), xyz_p.y());
//      if (!pano.ColInside(pc)) continue;

//      NormalMatch match;
//      // take a window around that pixel in pano and compute its mean
//      const auto win = pano.BoundedWinAt({pc, pr}, half_size);

//      for (int wr = win.y; wr < win.br().y; ++wr) {
//        for (int wc = win.x; wc < win.br().x; ++wc) {
//          const float rg = pano.MetricAt(wr, wc);
//          if (rg == 0) continue;
//          const auto p = pano.To3d(wr, wc, rg);
//          match.dst.Add({p.x, p.y, p.z});
//        }
//      }

//      if (match.dst.n < 10) continue;

//      match.src.mean.x() = xyzr[0];
//      match.src.mean.y() = xyzr[1];
//      match.src.mean.z() = xyzr[2];

//      matches_.push_back(match);
//    }
//  }
//}

// bool DataMatcher::IsGoodFeature(const PointGrid& grid, int r, int c) const {
//  // Threshold
//  const auto& score = grid.ScoreAt(r, c);
//  if (!(score < params_.max_score)) return false;

//  // NMS
//  if (params_.nms) {
//    const auto& score_l = grid.ScoreAt(r, c - 1);
//    const auto& score_r = grid.ScoreAt(r, c + 1);
//    if (score > score_l || score > score_r) return false;
//  }

//  return true;
//}

}  // namespace sv
