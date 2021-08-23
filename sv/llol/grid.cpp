#include "sv/llol/grid.h"

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

void ScanCellMeanCovar(const LidarScan& scan,
                       const cv::Rect& cell,
                       MeanCovar3f& mc) {
  // NOTE (chao): for now only take first row of cell due to staggered scan
  //  for (int r = 0; r < cell.height; ++r) {
  for (int c = 0; c < cell.width; ++c) {
    const auto& xyzr = scan.XyzrAt({cell.x + c, cell.y});
    if (std::isnan(xyzr[0])) continue;
    mc.Add({xyzr[0], xyzr[1], xyzr[2]});
  }
  //  }
}

void NormalMatch::SqrtInfo(float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  cov.diagonal().array() += lambda;
  U = MatrixSqrtUtU(cov.inverse().eval());
}

static constexpr float kValidCellRatio = 0.8;

/// @brief Compute scan curvature starting from px with size (width, 1)
float CalcCellCurve(const LidarScan& scan, const cv::Point& px, int width) {
  // compute sum of range in cell
  int num = 0;
  float sum = 0.0F;

  const int half = width / 2;
  const auto mid = std::min(scan.RangeAt({px.x + half - 1, px.y}),
                            scan.RangeAt({px.x + half, px.y}));
  if (std::isnan(mid)) return kNaNF;

  for (int c = 0; c < width; ++c) {
    const auto rg = scan.RangeAt({px.x + c, px.y});
    if (std::isnan(rg)) continue;
    sum += rg;
    ++num;
  }

  // Discard if there are too many nans in this cell
  if (num < kValidCellRatio * width) return kNaNF;
  return std::abs(sum / mid / num - 1);
}

SweepGrid::SweepGrid(const cv::Size& sweep_size, const GridParams& params)
    : cell_size{params.cell_cols, params.cell_rows},
      max_score{params.max_score},
      nms{params.nms},
      score{sweep_size / cell_size, CV_32FC1, kNaNF} {
  CHECK_GT(max_score, 0);
  CHECK_EQ(cell_size.width * score.cols, sweep_size.width);
  CHECK_EQ(cell_size.height * score.rows, sweep_size.height);
  tf_p_s.resize(score.cols + 1);  // one more to cover both ends
  matches.resize(total());
}

std::string SweepGrid::Repr() const {
  return fmt::format(
      "SweepGrid(cell_size={}, max_score={}, nms={}, score={}, col_range={})",
      sv::Repr(cell_size),
      max_score,
      nms,
      sv::Repr(score),
      sv::Repr(col_rg));
}

void SweepGrid::ResetMatches() {
  matches.clear();
  matches.resize(total());
}

std::pair<int, int> SweepGrid::Add(const LidarScan& scan, int gsize) {
  Check(scan);

  // Reset matches at start of sweep
  if (scan.col_rg.start == 0) {
    ResetMatches();
  }
  const int n1 = Score(scan, gsize);
  const int n2 = Reduce(scan, gsize);
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
    const auto px_s = Grid2Sweep({c, r});
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    CHECK_LT(px_s.x, scan.xyzr.cols);
    CHECK_LT(px_s.y, scan.xyzr.rows);

    const auto curve = CalcCellCurve(scan, px_s, cell_size.width);
    ScoreAt({c + col_rg.start, r}) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

cv::Rect SweepGrid::SweepCell(const cv::Point& px) const {
  const int sr = px.y * cell_size.height;
  const int sc = px.x * cell_size.width;
  return {{sc, sr}, cell_size};
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

int SweepGrid::Reduce(const LidarScan& scan, int gsize) {
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
          n += ReduceRow(scan, r);
        }
        return n;
      },
      std::plus<>{});
}

int SweepGrid::ReduceRow(const LidarScan& scan, int r) {
  int n = 0;

  // Note that scan is not sweep, so we need to start from 0
  // nms will look at left and right neighbor so need to skip first and last
  const int pad = static_cast<int>(nms);
  for (int c = pad; c < col_rg.size() - pad; ++c) {
    // px_g is for grid, so is offset by col_rg
    const cv::Point px_g{c + col_rg.start, r};
    // Skip if cell is not good
    if (!IsCellGood(px_g)) continue;

    auto& match = MatchAt(px_g);
    const auto cell = SweepCell({c, r});
    ScanCellMeanCovar(scan, cell, match.mc_s);

    // Set px_s to sweep px, so use px_g
    match.px_s = Grid2Sweep(px_g);
    match.px_s.x += cell_size.width / 2;
    ++n;
  }
  return n;
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

cv::Mat DrawMatches(const SweepGrid& grid) {
  cv::Mat disp(grid.size(), CV_32FC1, kNaNF);

  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.width(); ++c) {
      const cv::Point px_g{c, r};
      const auto i = grid.Grid2Ind(px_g);
      const auto& match = grid.matches.at(i);
      if (!match.Ok()) continue;
      disp.at<float>(px_g) = match.mc_p.n;
    }
  }

  return disp;
}

}  // namespace sv
