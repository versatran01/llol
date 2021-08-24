#include "sv/llol/scan.h"

#include <fmt/core.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <opencv2/core/core.hpp>

#include "sv/util/ocv.h"

namespace sv {

/// LidarScan ==================================================================
LidarScan::LidarScan(double t0,
                     double dt,
                     const cv::Mat& xyzr,
                     const cv::Range& col_range)
    : time{t0}, dt{dt}, xyzr{xyzr}, col_rg{col_range} {
  CHECK_GE(t0, 0);
  CHECK_GT(dt, 0);
  CHECK_EQ(xyzr.type(), kDtype);
  CHECK_EQ(xyzr.cols, col_range.size());
}

/// LidarSweep =================================================================
int LidarSweep::Add(const LidarScan& scan) {
  if (dt == 0) dt = scan.dt;
  Check(scan);

  // Increment id when we got a new sweep (indicated by the starting col of the
  // incoming scan being 0)
  if (scan.col_rg.start == 0) {
    ++id;
    time = scan.time;  // update time
  }

  // Save range and copy to storage
  col_rg = scan.col_rg;
  scan.xyzr.copyTo(xyzr.colRange(col_rg));  // x,y,w,h
  // TODO (chao): return number of valid points?
  return scan.xyzr.total();
}

void LidarSweep::Check(const LidarScan& scan) const {
  // Check scan type compatible
  CHECK_EQ(scan.xyzr.type(), xyzr.type());
  // Check rows match between scan and mat
  CHECK_EQ(scan.xyzr.rows, xyzr.rows);
  // Check scan width is not bigger than sweep
  CHECK_LE(scan.xyzr.cols, xyzr.cols);
  CHECK_LE(scan.col_rg.end, xyzr.cols);
  // Check that the new scan start right after
  CHECK_EQ(scan.col_rg.start, full() ? 0 : width());

  // Check dt is consistent, assume it stays the same
  CHECK_EQ(dt, scan.dt);
  CHECK_GT(dt, 0);
}

const cv::Mat& LidarSweep::ExtractRange() {
  cv::extractChannel(xyzr, disp, 3);
  return disp;
}

LidarSweep::LidarSweep(const cv::Size& size)
    : LidarScan{size}, disp{size, CV_32FC1} {
  tf_p_s.resize(size.width);
}

std::string LidarSweep::Repr() const {
  return fmt::format("LidarSweep(id={}, t0={}, dt={}, xyzr={}, col_range={})",
                     id,
                     time,
                     dt,
                     sv::Repr(xyzr),
                     sv::Repr(col_rg));
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
  sweep.Add(scan);
  return sweep;
}

}  // namespace sv
