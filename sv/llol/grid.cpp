#include "sv/llol/grid.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <sophus/interpolate.hpp>

#include "sv/util/ocv.h"

namespace sv {

bool PointInSize(const cv::Point& p, const cv::Size& size) {
  return std::abs(p.x) <= size.width && std::abs(p.y) <= size.height;
}

SweepGrid::SweepGrid(const cv::Size& sweep_size, const GridParams& params)
    : cell_size{params.cell_cols, params.cell_rows},
      max_score{params.max_score},
      nms{params.nms},
      cov_lambda{params.cov_lambda},
      score{sweep_size / cell_size, CV_32FC1, kNaNF} {
  CHECK_GT(max_score, 0);
  CHECK_EQ(cell_size.width * score.cols, sweep_size.width);
  CHECK_EQ(cell_size.height * score.rows, sweep_size.height);

  tfs.resize(score.cols + 1);  // one more to cover both ends
  matches.resize(total());

  pano_win_size.height = params.half_rows * 2 + 1;
  pano_win_size.width = params.half_rows * 4 + 1;
  min_pts = (params.half_rows + 1) * pano_win_size.width;
  max_dist_size = pano_win_size / 4;
}

std::string SweepGrid::Repr() const {
  return fmt::format(
      "SweepGrid(cell_size={}, max_score={}, nms={}, cov_lambda={}, "
      "min_pts={}, pano_win_size={}, max_dist_size={})",
      sv::Repr(cell_size),
      max_score,
      nms,
      cov_lambda,
      min_pts,
      sv::Repr(pano_win_size),
      sv::Repr(max_dist_size));
}

std::pair<int, int> SweepGrid::Add(const LidarScan& scan, int gsize) {
  Check(scan);

  // Reset matches at start of sweep
  const int n1 = Score(scan, gsize);
  const int n2 = Filter(scan, gsize);
  return {n1, n2};
}

void SweepGrid::Check(const LidarScan& scan) const {
  // scans row must match grid rows
  CHECK_EQ(scan.xyzr.rows, score.rows * cell_size.height);
  // scan start must match current end
  CHECK_EQ(scan.col_rg.start, full() ? 0 : width() * cell_size.width);
  // scan end must not excced grid cols
  CHECK_LE(scan.col_rg.end, score.cols * cell_size.width);
}

int SweepGrid::Score(const LidarScan& scan, int gsize) {
  gsize = gsize <= 0 ? score.rows : gsize;

  // update col_rg
  col_rg = scan.col_rg / cell_size.width;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, score.rows, gsize),
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
  for (int c = 0; c < col_rg.size(); ++c) {
    // this is in scan so c start from 0
    const auto px_s = Grid2Sweep({c, r});
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    const auto curve = scan.CurveAt(px_s, cell_size.width);
    ScoreAt({c + col_rg.start, r}) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int SweepGrid::Filter(const LidarScan& scan, int gsize) {
  // Check scan col_rg matches stored col_rg, this makes sure that Reduce() is
  // called after Score()
  const auto g_rg = scan.col_rg / cell_size.width;
  CHECK_EQ(g_rg.start, col_rg.start);
  CHECK_EQ(g_rg.end, col_rg.end);
  gsize = gsize <= 0 ? score.rows : gsize;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, score.rows, gsize),
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

  for (int c = 0; c < col_rg.size(); ++c) {
    // px_g is for grid, so offset by col_rg
    const cv::Point px_g{c + col_rg.start, r};
    auto& match = MatchAt(px_g);

    // Handle pad for nms
    if (pad <= c && c < col_rg.size() - pad && IsCellGood(px_g)) {
      // scan starts from 0 so use {c, r}
      scan.MeanCovarAt(Grid2Sweep({c, r}), cell_size.width, match.mc_g);
      match.px_g = px_g;
      // Set px_s to sweep px, so use px_g
      //      match.px_g = Grid2Sweep(px_g);
      //      match.px_g.x += cell_size.width / 2;
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

int SweepGrid::Match(const DepthPano& pano, int gsize) {
  const auto rows = score.rows;
  gsize = gsize <= 0 ? rows : gsize;
  CHECK_GE(gsize, 1);

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, rows, gsize),
      0,
      [&](const auto& blk, int n) {
        for (int gr = blk.begin(); gr < blk.end(); ++gr) {
          n += MatchRow(pano, gr);
        }
        return n;
      },
      std::plus<>{});
}

int SweepGrid::MatchRow(const DepthPano& pano, int gr) {
  int n = 0;
  // TODO (chao): for now we assume full sweep and match everying
  for (int gc = 0; gc < score.cols; ++gc) {
    n += MatchCell(pano, {gc, gr});
  }
  return n;
}

int SweepGrid::MatchCell(const DepthPano& pano, const cv::Point& px_g) {
  auto& match = MatchAt(px_g);
  if (!match.GridOk()) return 0;

  // Transform to pano frame
  const Eigen::Vector3f pt_p = CellTfAt(px_g.x) * match.mc_g.mean;
  const float rg_p = pt_p.norm();

  // Project to pano
  const auto px_p = pano.model.Forward(pt_p.x(), pt_p.y(), pt_p.z(), rg_p);
  if (px_p.x < 0) {
    // Bad projection, reset and return
    match.ResetPano();
    return 0;
  }

  // Check distance between new pix and old pix
  if (PointInSize(px_p - match.px_p, max_dist_size) && match.PanoOk()) {
    //  if (px_p == match.px_p && match.PanoOk()) {
    // If new and old are close and pano match is ok
    // we reuse this match and there is no need to recompute
    return 1;
  }

  // Compute mean covar around pano point
  match.px_p = px_p;
  pano.MeanCovarAt(px_p, pano_win_size, rg_p, match.mc_p);

  // if we don't have enough points also reset and return
  if (match.mc_p.n < min_pts) {
    match.ResetPano();
    return 0;
  }
  // Otherwise compute U'U = inv(C + lambda * I) and we have a good match
  match.SqrtInfo(cov_lambda);
  return 1;
}

Sophus::SE3f SweepGrid::CellTfAt(int c) const {
  const auto& Tc0 = tfs.at(c);
  const auto& Tc1 = tfs.at(c + 1);
  //  return Sophus::interpolate(tf0, tf1, 0.5);
  Sophus::SE3f Tc;
  Tc.so3() = Sophus::interpolate(Tc0.so3(), Tc1.so3(), 0.5);
  Tc.translation() = (Tc0.translation() + Tc1.translation()) / 2;
  return Tc;
}

cv::Point SweepGrid::Sweep2Grid(const cv::Point& px_sweep) const {
  return {px_sweep.x / cell_size.width, px_sweep.y / cell_size.height};
}

cv::Point SweepGrid::Grid2Sweep(const cv::Point& px_grid) const {
  return {px_grid.x * cell_size.width, px_grid.y * cell_size.height};
}

int SweepGrid::Grid2Ind(const cv::Point& px_grid) const {
  return px_grid.y * score.cols + px_grid.x;
}

cv::Mat SweepGrid::DispFilter() const {
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

cv::Mat SweepGrid::DispMatch() const {
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

void SweepGrid::InterpSweep(LidarSweep& sweep, int gsize) const {
  InterpPosesImpl(tfs, cell_size.width, sweep.tfs, gsize);
}

void InterpPosesImpl(const std::vector<Sophus::SE3f>& tf_grid,
                     int cell_width,
                     std::vector<Sophus::SE3f>& tf_sweep,
                     int gsize) {
  CHECK_EQ((tf_grid.size() - 1) * cell_width, tf_sweep.size());

  const int ncells = tf_grid.size() - 1;
  gsize = gsize <= 0 ? ncells : gsize;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, ncells, gsize), [&](const auto& blk) {
        for (int i = blk.begin(); i < blk.end(); ++i) {
          // interpolate rotation and translation separately
          const auto& T0 = tf_grid.at(i);
          const auto& T1 = tf_grid.at(i + 1);
          const auto& R0 = T0.so3();
          const auto& R1 = T1.so3();
          const auto dR = (R0.inverse() * R1).log();

          const auto& t0 = T0.translation();
          const auto& t1 = T1.translation();
          const Eigen::Vector3f dt = t1 - t0;

          for (int j = 0; j < cell_width; ++j) {
            // which column
            const int col = i * cell_width + j;
            const float s = static_cast<float>(j) / cell_width;
            tf_sweep.at(col).so3() = R0 * Sophus::SO3f::exp(s * dR);
            tf_sweep.at(col).translation() = t0 + s * dt;
          }
        }
      });
}

}  // namespace sv
