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
      cell_size{params.cell_cols, params.cell_rows},
      max_score{params.max_score},
      nms{params.nms} {
  CHECK_EQ(cell_size.width * mat.cols, sweep_size.width);
  CHECK_EQ(cell_size.height * mat.rows, sweep_size.height);

  mat.setTo(kNaNF);
  tfs.resize(mat.cols);
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
  Check(scan);

  // Reset matches at start of sweep
  const int n1 = Score(scan, gsize);
  const int n2 = Filter(scan, gsize);
  return {n1, n2};
}

void SweepGrid::Check(const LidarScan& scan) const {
  // scans row must match grid rows
  CHECK_EQ(scan.mat.rows, mat.rows * cell_size.height);
  // scan start must match current end
  CHECK_EQ(scan.curr.start, (curr.end * cell_size.width) % mat.cols);
  // scan end must not excced grid cols
  CHECK_LE(scan.curr.end, mat.cols * cell_size.width);
}

int SweepGrid::Score(const LidarScan& scan, int gsize) {
  gsize = gsize <= 0 ? mat.rows : gsize;

  // update col_rg
  curr = scan.curr / cell_size.width;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, mat.rows, gsize),
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

  // Note that scan is not sweep, so we need to start from 0
  for (int c = 0; c < curr.size(); ++c) {
    // this is in scan so c start from 0
    const auto px_s = Grid2Sweep({c, r});
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    const auto curve = scan.CurveAt(px_s, cell_size.width);
    ScoreAt({c + curr.start, r}) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int SweepGrid::Filter(const LidarScan& scan, int gsize) {
  // Check scan col_rg matches stored col_rg, this makes sure that Reduce() is
  // called after Score()
  const auto new_rg = scan.curr / cell_size.width;
  CHECK_EQ(new_rg.start, curr.start);
  CHECK_EQ(new_rg.end, curr.end);
  gsize = gsize <= 0 ? mat.rows : gsize;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, mat.rows, gsize),
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

  // Note that scan is not sweep, so we need to start from 0
  // nms will look at left and right neighbor so need to skip first and last
  const int pad = static_cast<int>(nms);

  for (int c = 0; c < curr.size(); ++c) {
    // px_g is for grid, so offset by col_rg
    const cv::Point px_g{c + curr.start, r};
    auto& match = MatchAt(px_g);

    // Handle pad for nms
    if (pad <= c && c < curr.size() - pad && IsCellGood(px_g)) {
      // scan starts from 0 so use {c, r}
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
  // curve could be nan
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

void SweepGrid::Interp(const std::vector<Sophus::SE3f>& traj) {
  CHECK_EQ(tfs.size() + 1, traj.size());

  for (int c = 0; c < tfs.size(); ++c) {
    const auto& tf0 = traj.at(c);
    const auto& tf1 = traj.at(c + 1);
    auto& tf = tfs.at(c);
    tf.so3() = Sophus::interpolate(tf0.so3(), tf1.so3(), 0.5);
    tf.translation() = (tf0.translation() + tf1.translation()) / 2;
  }
}

}  // namespace sv
