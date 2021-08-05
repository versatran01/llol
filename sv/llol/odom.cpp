#include "sv/llol/odom.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <Eigen/Core>

namespace sv {

/// RangeMat ===================================================================

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
void LidarSweep::AddScan(const cv::Mat& scan, const cv::Range& scan_range) {
  // Check scan type is compatible
  CHECK_EQ(mat_.type(), scan.type());
  // Check rows match between scan and mat
  CHECK_EQ(mat_.rows, scan.rows);

  // Save range and copy to storage
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
  // compute sum of range in cell
  float range_sum = 0.0F;
  for (int cc = 0; cc < cell.cols; ++cc) {
    const auto& crg = cell.at<cv::Vec4f>(cc)[3];
    if (std::isnan(crg)) return crg;  // early return nan
    range_sum += crg;
  }
  // range of mid ponit
  const float mid = cell.at<cv::Vec4f>(cell.cols / 2)[3];
  return std::abs(range_sum / cell.cols / mid - 1);
}

FeatureGrid::FeatureGrid(const cv::Size& sweep_size, const cv::Size& win_size)
    : RangeMat{{sweep_size.width / win_size.width,
                sweep_size.height / win_size.height},
               CV_32FC1},
      win{win_size} {
  mat_.setTo(std::numeric_limits<float>::quiet_NaN());
}

std::string FeatureGrid::Repr() const {
  return fmt::format(
      "FeatureGrid({}, win=({}, {}))", RangeMat::Repr(), win.height, win.width);
}

std::ostream& operator<<(std::ostream& os, const FeatureGrid& rhs) {
  return os << rhs.Repr();
}

void FeatureGrid::Detect(const LidarSweep& sweep, bool tbb) {
  // Update range of detector
  range_ = cv::Range{sweep.curr_range().start / win.width,
                     sweep.curr_range().end / win.width};

  if (tbb) {
    tbb::parallel_for(tbb::blocked_range<int>(0, mat_.rows),
                      [&](const tbb::blocked_range<int>& blk) {
                        for (int fr = blk.begin(); fr < blk.end(); ++fr) {
                          DetectRow(sweep.mat(), fr);
                        }
                      });
  } else {
    for (int fr = 0; fr < mat_.rows; ++fr) {
      DetectRow(sweep.mat(), fr);
    }
  }
}

void FeatureGrid::DetectRow(const cv::Mat& sweep, int row) {
  for (int col = range_.start; col < range_.end; ++col) {
    const int sr = row * win.height;
    const int sc = col * win.width;
    const cv::Mat cell = sweep.row(sr).colRange(sc, sc + win.width);
    At(row, col) = CellScore(cell);
  }
}

int FeatureGrid::NumCells(cv::Range range) const noexcept {
  if (range.empty()) range = cv::Range{0, width()};

  int n = 0;
  for (int i = 0; i < mat_.rows; ++i) {
    for (int j = range.start; j < range.end; ++j) {
      n += static_cast<int>(At(i, j) > 0);
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

cv::Rect DepthPano::WinAt(const cv::Point& pt,
                          const cv::Size& half_size) const noexcept {
  return {cv::Point{pt.x - half_size.width, pt.y - half_size.height},
          cv::Size{half_size.width * 2 + 1, +half_size.height * 2 + 1}};
}

cv::Rect DepthPano::BoundedWinAt(const cv::Point& pt,
                                 const cv::Size& half_size) const noexcept {
  const cv::Rect bound{cv::Point{}, size()};
  return WinAt(pt, half_size) & bound;
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

    At(pr, pc) = rg_p * kScale;
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

int DepthPano::ToRow(float z, float r) const noexcept {
  const float elev = std::asin(z / r);
  return (elev_max_ - elev) / elev_delta_ + 0.5F;
}

int DepthPano::ToCol(float x, float y) const noexcept {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta_ + 0.5F;
}

cv::Point3f DepthPano::To3d(int r, int c, float rg) const noexcept {
  const auto& elev = elevs_[r];
  const auto& azim = azims_[c];
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

/// FeatureMatcher =============================================================
FeatureMatcher::FeatureMatcher(int max_matches, const MatcherParams& params)
    : params_{params} {
  matches_.reserve(max_matches);
}

std::string FeatureMatcher::Repr() const {
  return fmt::format("FeatureMatcher(max_matches={}, max_score={}, nms={})",
                     matches_.capacity(),
                     params_.max_score,
                     params_.nms);
}

std::ostream& operator<<(std::ostream& os, const FeatureMatcher& rhs) {
  return os << rhs.Repr();
}

void FeatureMatcher::Match(const LidarSweep& sweep,
                           const FeatureGrid& grid,
                           const DepthPano& pano) {
  matches_.clear();

  const auto& feats = grid.mat();
  const cv::Size half_size{params_.half_size, params_.half_size};

  for (int fr = 1; fr < feats.rows - 1; ++fr) {
    for (int fc = 1; fc < grid.width() - 1; ++fc) {
      const auto& score = feats.at<float>(fr, fc);
      if (!(score < params_.max_score)) continue;

      // NMS
      if (params_.nms) {
        const auto& score_l = feats.at<float>(fr, fc - 1);
        const auto& score_r = feats.at<float>(fr, fc + 1);
        if (score > score_l || score > score_r) continue;
      }

      // Get the point in sweep
      const int sr = fr * grid.win.height;
      const int sc = (fc + 0.5) * grid.win.width;
      const auto& xyzr = sweep.mat().at<cv::Vec4f>(sr, sc);

      // Transform xyz to pano frame
      Eigen::Map<const Eigen::Vector3f> xyz_s(&xyzr[0]);
      const Eigen::Vector3f xyz_p = Eigen::Matrix3f::Identity() * xyz_s;
      const float rg = xyz_p.norm();

      const int pr = pano.ToRow(xyz_p.z(), rg);
      if (!pano.RowInside(pr)) continue;
      const int pc = pano.ToCol(xyz_p.x(), xyz_p.y());
      if (!pano.ColInside(pc)) continue;

      NormalMatch match;
      // take a window around that pixel in pano and compute its mean
      const auto win = pano.BoundedWinAt({pc, pr}, half_size);

      for (int wr = win.y; wr < win.br().y; ++wr) {
        for (int wc = win.x; wc < win.br().x; ++wc) {
          const float rg = pano.MetricAt(wr, wc);
          if (rg == 0) continue;
          const auto p = pano.To3d(wr, wc, rg);
          match.dst.Add({p.x, p.y, p.z});
        }
      }

      if (match.dst.n < 10) continue;

      match.src.mean.x() = xyzr[0];
      match.src.mean.y() = xyzr[1];
      match.src.mean.z() = xyzr[2];

      matches_.push_back(match);
    }
  }
}

}  // namespace sv
