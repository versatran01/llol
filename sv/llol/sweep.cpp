#include "sv/llol/sweep.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

// cell is (1,c, 32FC4), return nan if any point is nan
float CalcCellCurve(const cv::Mat& cell) {
  // compute sum of range in cell
  const int n = cell.cols;

  int num = 0;
  float sum = 0.0F;
  for (int c = 0; c < n; ++c) {
    const auto& rg = cell.at<cv::Vec4f>(0, c)[3];
    //    if (std::isnan(rg)) return rg;  // early return nan
    if (std::isnan(rg)) continue;
    sum += rg;
    ++num;
  }

  if (num < 0.75 * n) return kNaNF;
  // range of mid point
  const auto mid =
      (cell.at<cv::Vec4f>(n / 2)[3] + cell.at<cv::Vec4f>(n / 2 - 1)[3]) / 2;
  return std::abs(sum / mid / num - 1);
}

/// LidarSweep =================================================================
LidarSweep::LidarSweep(cv::Size sweep_size, cv::Size cell_size)
    : sweep_{sweep_size, CV_32FC4},
      cell_size_{cell_size},
      grid_{cv::Size{sweep_size.width / cell_size.width,
                     sweep_size.height / cell_size.height},
            CV_32FC1} {}

int LidarSweep::AddScan(const cv::Mat& scan,
                        const cv::Range& scan_range,
                        bool tbb) {
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
  auto grid = grid_.colRange(scan_range / cell_size().width);
  return CalcScanCurve(scan, grid, tbb);
}

int LidarSweep::CalcScanCurve(const cv::Mat& scan, cv::Mat grid, bool tbb) {
  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, grid_.rows),
        0,
        [&](const tbb::blocked_range<int>& block, int total) {
          for (int gr = block.begin(); gr < block.end(); ++gr) {
            total += CalcScanCurveRow(scan, grid, gr);
          }
          return total;
        },
        std::plus<>{});
  } else {
    for (int gr = 0; gr < grid.rows; ++gr) {
      n += CalcScanCurveRow(scan, grid, gr);
    }
  }
  return n;
}

int LidarSweep::CalcScanCurveRow(const cv::Mat& scan, cv::Mat& grid, int gr) {
  int n = 0;
  for (int gc = 0; gc < grid.cols; ++gc) {
    const int sr = gr * cell_size_.height;
    const int sc = gc * cell_size_.width;
    const auto cell = scan.row(sr).colRange(sc, sc + cell_size_.width);
    const auto curve = CalcCellCurve(cell);
    grid.at<float>(gr, gc) = curve;
    n += static_cast<int>(std::isnan(curve));
  }
  return n;
}

void LidarSweep::Reset() {
  range_ = {0, 0};
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  sweep_.setTo(nan);
  grid_.setTo(nan);
}

cv::Point LidarSweep::PixelToCell(const cv::Point& px_s) const {
  return {px_s.x / cell_size_.width, px_s.y / cell_size_.height};
}

cv::Mat LidarSweep::CellAt(const cv::Point& grid_px) const {
  const int sr = grid_px.y * cell_size_.height;
  const int sc = grid_px.x * cell_size_.width;
  return sweep_.row(sr).colRange(sc, sc + cell_size_.width);
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
