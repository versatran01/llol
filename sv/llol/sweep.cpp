#include "sv/llol/sweep.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>

#include "sv/util/ocv.h"

namespace sv {

float CalcCurvature(const cv::Mat& cell) {
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

int CalcScanCurveRow(const cv::Mat& scan,
                     cv::Mat& grid,
                     cv::Size cell_size,
                     int gr) {
  int n = 0;
  for (int gc = 0; gc < grid.cols; ++gc) {
    const int sr = gr * cell_size.height;
    const int sc = gc * cell_size.width;
    const auto cell = scan.row(sr).colRange(sc, sc + cell_size.width);
    const auto curve = CalcCurvature(cell);
    grid.at<float>(gr, gc) = curve;
    n += !std::isnan(curve);
  }
  return n;
}

// TODO (chao): tbb ersion is slower, need to check for false sharing
int CalcScanCurve(const cv::Mat& scan, cv::Mat& grid, bool tbb) {
  CHECK_EQ(scan.type(), CV_32FC4) << CvTypeStr(scan.type());
  CHECK_EQ(grid.type(), CV_32FC1) << CvTypeStr(grid.type());

  int n = 0;
  const cv::Size cell_size{scan.cols / grid.cols, scan.rows / grid.rows};

  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, grid.rows),
        0,
        [&](const tbb::blocked_range<int>& block, int total) {
          for (int gr = block.begin(); gr < block.end(); ++gr) {
            total += CalcScanCurveRow(scan, grid, cell_size, gr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int gr = 0; gr < grid.rows; ++gr) {
      n += CalcScanCurveRow(scan, grid, cell_size, gr);
    }
  }
  return n;
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

  // Compute curvature of scan and save to grid
  auto subgrid = grid_.colRange(scan_range.start / cell_size_.width,
                                scan_range.end / cell_size_.width);
  return CalcScanCurve(scan, subgrid, tbb);
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

}  // namespace sv
