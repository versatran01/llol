#include "sv/llol/odom.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>

namespace sv {

std::string MatRepr(const cv::Mat& mat) {
  return fmt::format("hwc=({},{},{}), depth={}",
                     mat.rows,
                     mat.cols,
                     mat.channels(),
                     mat.depth());
}

std::string RangeMat::Repr() const {
  return fmt::format(
      "{}, range=({},{})", MatRepr(mat_), range_.start, range_.end);
}

/// LidarSweep =================================================================
void LidarSweep::AddScan(const cv::Mat& scan, cv::Range scan_range) {
  // Check scan type is compatible
  CHECK_EQ(mat_.type(), scan.type());
  // Check rows match between scan and mat
  CHECK_EQ(mat_.rows, scan.rows);

  range_ = scan_range;
  scan.copyTo(mat_.colRange(range_));  // x,y,w,h
}

std::string LidarSweep::Repr() const {
  return fmt::format("LidarSweep({})", RangeMat::Repr());
}

std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
  return os << rhs.Repr();
}

/// FeatureGrid ================================================================
float CellScore(const cv::Mat& cell) {
  float range_sum = 0.0F;
  for (int cc = 0; cc < cell.cols; ++cc) {
    const auto& crg = cell.at<cv::Vec4f>(cc)[3];
    if (std::isnan(crg)) return crg;  // early return nan
    range_sum += crg;
  }
  const float mid = cell.at<cv::Vec4f>(cell.cols / 2)[3];
  return std::abs(range_sum / cell.cols / mid - 1);
}

FeatureGrid::FeatureGrid(cv::Size sweep_size, cv::Size win_size)
    : RangeMat{{sweep_size.width / win_size.width,
                sweep_size.height / win_size.height},
               CV_32FC1},
      win{win_size} {
  mat_.setTo(std::numeric_limits<float>::quiet_NaN());
}

std::string FeatureGrid::Repr() const {
  return fmt::format(
      "FeatureGrid({}, win=())", RangeMat::Repr(), win.height, win.width);
}

std::ostream& operator<<(std::ostream& os, const FeatureGrid& rhs) {
  return os << rhs.Repr();
}

void FeatureGrid::Detect(const LidarSweep& sweep, bool tbb) {
  Detect(sweep.mat(), sweep.curr_range(), tbb);
}

void FeatureGrid::Detect(const cv::Mat& sweep,
                         cv::Range sweep_range,
                         bool tbb) {
  // Update range of detector
  range_ =
      cv::Range{sweep_range.start / win.width, sweep_range.end / win.width};

  if (tbb) {
    tbb::parallel_for(tbb::blocked_range<int>(0, mat_.rows),
                      [&](const tbb::blocked_range<int>& blk) {
                        for (int fr = blk.begin(); fr < blk.end(); ++fr) {
                          DetectRow(sweep, fr);
                        }
                      });
  } else {
    for (int fr = 0; fr < mat_.rows; ++fr) {
      DetectRow(sweep, fr);
    }
  }
}

void FeatureGrid::DetectRow(const cv::Mat& sweep, int row) {
  for (int fc = range_.start; fc < range_.end; ++fc) {
    const int sr = row * win.height;
    const int sc = fc * win.width;
    const cv::Mat cell = sweep.row(sr).colRange(sc, sc + win.width);
    mat_.at<float>(row, fc) = CellScore(cell);
  }
}

int FeatureGrid::NumCells(cv::Range range) const noexcept {
  if (range.size() == 0) range = cv::Range{0, width()};

  int n = 0;
  for (int i = 0; i < mat_.rows; ++i) {
    for (int j = range.start; j < range.end; ++j) {
      n += mat_.at<float>(i, j) > 0;
    }
  }
  return n;
}

/// DepthPano ==================================================================
DepthPano::DepthPano(cv::Size size)
    : mat_{size, CV_16UC1},
      mat2_{size, CV_16UC1},
      azim_delta_{kTauF / size.width} {
  // assumes equal aspect ratio
  const float wh_ratio = static_cast<float>(size.width) / size.height;
  // assume horizontal fov is centered at 0
  const float hfov = kTauF / wh_ratio;
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
  return fmt::format(
      "DepthPano({}, elev_max={}[deg], elev_delta={}[deg], azim_delta={}[deg], "
      "scale={}, max_range={})",
      MatRepr(mat_),
      Rad2Deg(elev_max_),
      Rad2Deg(elev_delta_),
      Rad2Deg(azim_delta_),
      kScale,
      kMaxRange);
}

std::ostream& operator<<(std::ostream& os, const DepthPano& rhs) {
  return os << rhs.Repr();
}

void DepthPano::AddSweep(const LidarSweep& sweep, bool tbb) {
  CHECK(sweep.full());
  AddSweep(sweep.mat(), tbb);
}

void DepthPano::AddSweep(const cv::Mat& sweep, bool tbb) {
  if (tbb) {
    tbb::parallel_for(tbb::blocked_range<int>(0, sweep.rows),
                      [&](const tbb::blocked_range<int>& blk) {
                        for (int sr = blk.begin(); sr < blk.end(); ++sr) {
                          AddSweepRow(sweep, sr);
                        }
                      });
  } else {
    for (int sr = 0; sr < sweep.rows; ++sr) {
      AddSweepRow(sweep, sr);
    }
  }

  ++num_sweeps_;
}

void DepthPano::AddSweepRow(const cv::Mat& sweep, int row) {
  for (int sc = 0; sc < sweep.cols; ++sc) {
    const auto& xyzr = sweep.at<cv::Vec4f>(row, sc);
    if (!(xyzr[3] > 0)) continue;

    Eigen::Map<const Eigen::Vector3f> xyz(&xyzr[0]);
    // TODO (chao): transform xyz
    const Eigen::Vector3f xyz_t = Eigen::Matrix3f::Identity() * xyz;

    const auto rg = xyz_t.norm();
    const int pr = ToRow(xyz_t.z(), rg);
    if (!RowInside(pr)) continue;

    const int pc = ToCol(xyz_t.x(), xyz_t.y());
    if (!ColInside(pc)) continue;

    mat_.at<ushort>(pr, pc) = rg * kScale;
  }
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

void DepthPano::RenderRow(int row) {
  const auto elev = elevs_[row];
  for (int c1 = 0; c1 < mat_.cols; ++c1) {
    const auto rg1_raw = mat_.at<ushort>(row, c1);
    if (rg1_raw == 0) continue;

    const float rg1 = rg1_raw / kScale;

    const auto azim = azims_[c1];
    // pano -> xyz1
    const Eigen::Vector3f xyz1{
        elev.cos * azim.cos * rg1, elev.cos * azim.sin * rg1, elev.sin * rg1};

    // xyz1 -> xyz2
    const Eigen::Vector3f xyz2 = Eigen::Matrix3f::Identity() * xyz1;
    const auto rg2 = xyz2.norm();

    // compute row and col into mat2
    const int r2 = ToRow(xyz2.z(), rg2);
    if (!RowInside(r2)) continue;

    const int c2 = ToCol(xyz2.x(), xyz2.y());
    if (!ColInside(c2)) continue;

    mat2_.at<ushort>(r2, c2) = rg2 * kScale;
  }
}

int DepthPano::ToRow(float z, float r) const noexcept {
  const float elev = std::asin(z / r);
  return (elev_max_ - elev) / elev_delta_ + 0.5F;
}

int DepthPano::ToCol(float x, float y) const noexcept {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta_ + 0.5F;
}

}  // namespace sv
