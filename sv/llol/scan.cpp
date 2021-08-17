#include "sv/llol/scan.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include "sv/util/ocv.h"

namespace sv {

/// LidarScan ==================================================================
LidarScan::LidarScan(double t0,
                     double dt,
                     const cv::Mat& xyzr,
                     const cv::Range& col_range)
    : t0{t0}, dt{dt}, xyzr{xyzr}, col_range{col_range} {
  CHECK_GE(t0, 0);
  CHECK_GT(dt, 0);
  CHECK_EQ(xyzr.type(), kDtype);
  CHECK_EQ(xyzr.cols, col_range.size());
}

/// LidarSweep =================================================================
int LidarSweep::AddScan(const LidarScan& scan) {
  // Check scan type compatible
  CHECK_EQ(scan.xyzr.type(), xyzr.type());
  // Check rows match between scan and mat
  CHECK_EQ(scan.xyzr.rows, xyzr.rows);
  // Check scan width is not bigger than sweep
  CHECK_LE(scan.xyzr.cols, xyzr.cols);
  CHECK_LE(scan.col_range.end, xyzr.cols);
  // Check that the new scan start right after
  CHECK_EQ(scan.col_range.start, full() ? 0 : col_range.end);
  // Check dt is consistent, assume it stays the same
  if (dt == 0) dt = scan.dt;
  CHECK_EQ(dt, scan.dt);
  CHECK_GT(dt, 0);

  // Increment id when we got a new sweep (indicated by the starting col of the
  // incoming scan being 0)
  if (scan.col_range.start == 0) {
    ++id;
    t0 = scan.t0;  // update time
  }

  // Save range and copy to storage
  col_range = scan.col_range;
  scan.xyzr.copyTo(xyzr.colRange(col_range));  // x,y,w,h
  return scan.xyzr.total();
}

LidarSweep::LidarSweep(const cv::Size& size) : LidarScan{size} {
  tfs.resize(size.width);
}

std::string LidarSweep::Repr() const {
  using sv::Repr;
  return fmt::format("LidarSweep(id={}, t0={}, dt={}, xyzr={}, col_range={})",
                     id,
                     t0,
                     dt,
                     Repr(xyzr),
                     Repr(col_range));
}

/// Test Related ===============================================================
cv::Mat MakeTestXyzr(const cv::Size& size) {
  cv::Mat xyzr = cv::Mat::zeros(size, LidarScan::kDtype);

  const float azim_delta = M_PI * 2 / size.width;
  const float elev_max = M_PI_4;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < xyzr.rows; ++i) {
    for (int j = 0; j < xyzr.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = M_PI * 2 - j * azim_delta;

      auto& p = xyzr.at<cv::Vec4f>(i, j);
      p[0] = std::cos(elev) * std::cos(azim);
      p[1] = std::cos(elev) * std::sin(azim);
      p[2] = std::sin(elev);
      p[3] = 1;
    }
  }

  return xyzr;
}

LidarScan MakeTestScan(const cv::Size& size) {
  return {0, 0.1 / size.width, MakeTestXyzr(size), {0, size.width}};
}

LidarSweep MakeTestSweep(const cv::Size& size) {
  LidarSweep sweep(size);
  LidarScan scan = MakeTestScan(size);
  sweep.AddScan(scan);
  return sweep;
}

}  // namespace sv
