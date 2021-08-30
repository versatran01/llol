#include "sv/llol/grid.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <sophus/interpolate.hpp>

#include "sv/util/ocv.h"

namespace sv {

bool PointInSize(const cv::Point& p, const cv::Size& size) {
  return std::abs(p.x) <= size.width && std::abs(p.y) <= size.height;
}

SweepGrid::SweepGrid(const cv::Size& sweep_size, const GridParams& params)
    : ScanBase{sweep_size / cv::Size{params.cell_cols, params.cell_rows},
               CV_32FC1},
      nms{params.nms},
      max_score{params.max_score},
      cell_size{params.cell_cols, params.cell_rows} {
  CHECK_EQ(cell_size.width * cols(), sweep_size.width);
  CHECK_EQ(cell_size.height * rows(), sweep_size.height);

  mat.setTo(kNaNF);
  matches.resize(total());
}

std::string SweepGrid::Repr() const {
  return fmt::format("SweepGrid(size={}, cell_size={}, max_score={}, nms={})",
                     sv::Repr(size()),
                     sv::Repr(cell_size),
                     max_score,
                     nms);
}

cv::Vec2i SweepGrid::Add(const LidarScan& scan, int gsize) {
  CHECK_EQ(scan.rows(), rows() * cell_size.height);
  const int num_valid_cells = Score(scan, gsize);
  const int num_match_dells = Filter(scan, gsize);
  return {num_valid_cells, num_match_dells};
}

int SweepGrid::Score(const LidarScan& scan, int gsize) {
  // Note that we udpate in Score() instead of Add(), and check consistency in
  // Filter()
  UpdateTime(scan.t0, scan.dt * cell_size.width);
  UpdateView(scan.curr / cell_size.width);

  gsize = gsize <= 0 ? rows() : gsize;
  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows(), gsize),
      0,
      [&](const auto& block, int n) {
        for (int r = block.begin(); r < block.end(); ++r) {
          n += ScoreRow(scan, r);
        }
        return n;
      },
      std::plus<>{});
}

int SweepGrid::ScoreRow(const LidarScan& scan, int r) {
  int n = 0;
  for (int c = 0; c < curr.size(); ++c) {
    // c starts from 0 in scan
    const auto px_s = Grid2Sweep({c, r});
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle destaggering it later
    const auto curve = scan.CurveAt(px_s, cell_size.width);
    // but the corresponding cell is within a sweep so need to offset
    ScoreAt({c + curr.start, r}) = curve;  // could be nan
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int SweepGrid::Filter(const LidarScan& scan, int gsize) {
  // Check scan curr matches stored curr, this makes sure that Filter() is
  // called after Score()
  const auto new_curr = scan.curr / cell_size.width;
  CHECK_EQ(new_curr.start, curr.start);
  CHECK_EQ(new_curr.end, curr.end);
  gsize = gsize <= 0 ? rows() : gsize;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows(), gsize),
      0,
      [&](const auto& blk, int n) {
        for (int r = blk.begin(); r < blk.end(); ++r) {
          n += FilterRow(scan, r);
        }
        return n;
      },
      std::plus<>{});
}

int SweepGrid::FilterRow(const LidarScan& scan, int r) {
  int n = 0;

  // nms will look at left and right neighbor so need to skip first and last col
  const int pad = static_cast<int>(nms);

  for (int c = 0; c < curr.size(); ++c) {
    // Need offset for px grid
    const cv::Point px_g{c + curr.start, r};
    auto& match = MatchAt(px_g);

    // Handle pad for nms
    if (pad <= c && c < curr.size() - pad && IsCellGood(px_g)) {
      // No need to offset for px scan
      scan.MeanCovarAt(Grid2Sweep({c, r}), cell_size.width, match.mc_g);
      match.px_g = px_g;
      ++n;
    } else {
      match.Reset();
    }
  }
  return n;
}

bool SweepGrid::IsCellGood(const cv::Point& px) const {
  // Note that score could be nan
  // Threshold check
  const auto& m = ScoreAt(px);
  if (!(m < max_score)) return false;

  // NMS check, nan neighbor is considered as inf
  if (nms) {
    const auto& l = ScoreAt({px.x - 1, px.y});
    const auto& r = ScoreAt({px.x + 1, px.y});
    if (m > l || m > r) return false;
  }

  return true;
}

cv::Mat SweepGrid::DrawFilter() const {
  static cv::Mat disp;
  if (disp.empty()) disp.create(size(), CV_32FC1);

  for (int r = 0; r < disp.rows; ++r) {
    for (int c = 0; c < disp.cols; ++c) {
      const cv::Point px_g{c, r};
      const auto& match = MatchAt(px_g);
      disp.at<float>(px_g) = match.GridOk() ? ScoreAt(px_g) : kNaNF;
    }
  }
  return disp;
}

cv::Mat SweepGrid::DrawMatch() const {
  static cv::Mat disp;
  if (disp.empty()) disp.create(size(), CV_32FC1);

  for (int r = 0; r < disp.rows; ++r) {
    for (int c = 0; c < disp.cols; ++c) {
      const cv::Point px_g{c, r};
      const auto& match = MatchAt(px_g);
      disp.at<float>(px_g) = match.Ok() ? match.mc_p.n : kNaNF;
    }
  }
  return disp;
}

void SweepGrid::Interp(const ImuTrajectory& traj) {
  CHECK_EQ(tfs.size() + 1, traj.size());

  for (int gc = 0; gc < tfs.size(); ++gc) {
    // Note that the starting point of traj is where curr ends, so we need to
    // offset by curr.end to find the corresponding traj segment
    const int tc = ColMod(gc - curr.end, cols());
    const auto& st0 = traj.StateAt(tc);
    const auto& st1 = traj.StateAt(tc + 1);

    Sophus::SE3d tf_p_i;
    tf_p_i.so3() = Sophus::interpolate(st0.rot, st1.rot, 0.5);
    tf_p_i.translation() = (st0.pos + st1.pos) / 2.0;
    tfs.at(gc) = (tf_p_i * traj.T_imu_lidar).cast<float>();
  }
}

}  // namespace sv
