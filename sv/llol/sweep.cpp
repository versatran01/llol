#include "sv/llol/sweep.h"

#include <glog/logging.h>

#include <opencv2/core/core.hpp>

#include "sv/util/ocv.h"

namespace sv {

int LidarSweep::Add(const LidarScan& scan) {
  if (dt == 0) dt = scan.dt;
  Check(scan);

  // Increment id when we got a new sweep (indicated by the starting col of the
  // incoming scan being 0)
  if (scan.col_rg.start == 0) {
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
  CHECK_EQ(scan.col_rg.start, col_rg.end % xyzr.cols);

  // Check dt is consistent, assume it stays the same
  CHECK_EQ(dt, scan.dt);
  CHECK_GT(dt, 0);
}

cv::Mat LidarSweep::DrawRange() const {
  static cv::Mat disp;
  cv::extractChannel(xyzr, disp, 3);
  return disp;
}

LidarSweep::LidarSweep(const cv::Size& size) : LidarScan{size} {
  tfs.resize(size.width);
}

std::string LidarSweep::Repr() const {
  return fmt::format("LidarSweep( t0={}, dt={}, xyzr={}, col_range={})",
                     time,
                     dt,
                     sv::Repr(xyzr),
                     sv::Repr(col_rg));
}

LidarSweep MakeTestSweep(const cv::Size& size) {
  LidarSweep sweep(size);
  LidarScan scan = MakeTestScan(size);
  sweep.Add(scan);
  return sweep;
}

}  // namespace sv
