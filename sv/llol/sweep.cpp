#include "sv/llol/sweep.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

/// @brief Compute scan curvature and store to grid
int CalcScanCurve(const cv::Mat& scan, cv::Mat& grid, bool tbb = false);
int CalcScanCurveRow(const cv::Mat& scan, cv::Mat& grid, int r);
float CalcCellCurve(const cv::Mat& cell);

float CalcCellCurve(const cv::Mat& scan, int row, const cv::Range& col_range) {
  using T = cv::Vec4f;
  // compute sum of range in cell
  const int n = col_range.size();

  int num = 0;
  float sum = 0.0F;
  for (int c = col_range.start; c < col_range.end; ++c) {
    const auto& rg = scan.at<T>(row, c)[3];
    if (std::isnan(rg)) continue;
    sum += rg;
    ++num;
  }

  if (num < 0.75 * n) return kNaNF;

  // range of mid point
  const auto mid = (scan.at<T>(row, col_range.start + n / 2 - 1)[3] +
                    scan.at<T>(row, col_range.start + n / 2)[3]) /
                   2.0F;
  return std::abs(sum / mid / num - 1);
}

int CalcScanCurveRow(const cv::Mat& scan, cv::Mat& grid, int gr) {
  int n = 0;
  const int cell_rows = scan.rows / grid.rows;
  const int cell_cols = scan.cols / grid.cols;
  for (int gc = 0; gc < grid.cols; ++gc) {
    const int sr = gr * cell_rows;
    const int sc = gc * cell_cols;
    // Note that we only take the first row regardless of row size
    // this is because ouster lidar image is staggered.
    // Maybe we will handle it later
    const auto curve = CalcCellCurve(scan, sr, {sc, sc + cell_cols});
    grid.at<float>(gr, gc) = curve;
    n += static_cast<int>(!std::isnan(curve));
  }
  return n;
}

int CalcScanCurve(const cv::Mat& scan, cv::Mat& grid, bool tbb) {
  int n = 0;
  if (tbb) {
    n = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, grid.rows),
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

/// LidarSweep =================================================================
LidarSweep::LidarSweep(const cv::Size& sweep_size, const cv::Size& cell_size)
    : xyzr_(sweep_size, CV_32FC4),
      cell_size_(cell_size),
      grid_(sweep_size / cell_size, CV_32FC1),
      offsets_(1, sweep_size.width, CV_8UC1),
      transforms_(sweep_size.width, 7, CV_64FC1) {}

int LidarSweep::AddScan(const cv::Mat& scan,
                        const cv::Range& scan_range,
                        bool tbb) {
  // Check scan type is compatible
  CHECK_EQ(scan.type(), xyzr_.type());
  // Check rows match between scan and mat
  CHECK_EQ(scan.rows, xyzr_.rows);
  // Check scan width is not bigger than sweep
  CHECK_LE(scan.cols, xyzr_.cols);
  // Check that the new scan start right after
  CHECK_EQ(scan_range.start, IsFull() ? 0 : col_range_.end);

  // Save range and copy to storage
  col_range_ = scan_range;
  scan.copyTo(xyzr_.colRange(col_range_));  // x,y,w,h

  // Compute curvature of scan and save to grid
  auto grid = grid_.colRange(scan_range / cell_size().width);
  return CalcScanCurve(scan, grid, tbb);
}

void LidarSweep::Reset() {
  col_range_ = {0, 0};
  xyzr_.setTo(kNaNF);
  grid_.setTo(kNaNF);
}

cv::Point LidarSweep::Pixel2CellInd(const cv::Point& px_sweep) const {
  return {px_sweep.x / cell_size_.width, px_sweep.y / cell_size_.height};
}

cv::Rect LidarSweep::CellAt(const cv::Point& grid_px) const {
  const int sr = grid_px.y * cell_size_.height;
  const int sc = grid_px.x * cell_size_.width;
  return {sc, sr, cell_size_.width, 1};
  //  return xyzr_.row(sr).colRange(sc, sc + cell_size_.width);
}

std::string LidarSweep::Repr() const {
  using sv::Repr;
  return fmt::format("LidarSweep(xyzr={}, col_range={}, grid={}, cell_size={})",
                     Repr(xyzr_),
                     Repr(col_range_),
                     Repr(grid_),
                     Repr(cell_size_));
}

std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
  return os << rhs.Repr();
}

cv::Mat MakeTestScan(const cv::Size& size) {
  cv::Mat sweep = cv::Mat::zeros(size, CV_32FC4);

  const float azim_delta = M_PI * 2 / size.width;
  const float elev_max = M_PI_4 / 2;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < sweep.rows; ++i) {
    for (int j = 0; j < sweep.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = M_PI * 2 - j * azim_delta;

      auto& xyzr = sweep.at<cv::Vec4f>(i, j);
      xyzr[0] = std::cos(elev) * std::cos(azim);
      xyzr[1] = std::cos(elev) * std::sin(azim);
      xyzr[2] = std::sin(elev);
      xyzr[3] = 1;
    }
  }

  return sweep;
}

}  // namespace sv
