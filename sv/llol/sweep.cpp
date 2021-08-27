#include "sv/llol/sweep.h"

#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <opencv2/core/core.hpp>

#include "sv/util/ocv.h"

namespace sv {

int LidarSweep::Add(const LidarScan& scan) {
  if (dt == 0) dt = scan.dt;
  Check(scan);

  // Increment id when we got a new sweep (indicated by the starting col of the
  // incoming scan being 0)
  if (scan.curr.start == 0) {
    t0 = scan.t0;  // update time
  }

  // Save range and copy to storage
  curr = scan.curr;
  scan.mat.copyTo(mat.colRange(curr));  // x,y,w,h
  // TODO (chao): return number of valid points?
  return scan.mat.total();
}

void LidarSweep::Check(const LidarScan& scan) const {
  // Check scan type compatible
  CHECK_EQ(scan.mat.type(), mat.type());
  // Check rows match between scan and mat
  CHECK_EQ(scan.mat.rows, mat.rows);
  // Check scan width is not bigger than sweep
  CHECK_LE(scan.mat.cols, mat.cols);
  CHECK_LE(scan.curr.end, mat.cols);
  // Check that the new scan start right after
  CHECK_EQ(scan.curr.start, curr.end % mat.cols);

  // Check dt is consistent, assume it stays the same
  CHECK_EQ(dt, scan.dt);
  CHECK_GT(dt, 0);
}

void LidarSweep::Interp(const std::vector<Sophus::SE3f>& traj, int gsize) {
  const int num_cells = traj.size() - 1;
  const int cell_width = mat.cols / num_cells;
  // TODO (chao): hack check
  CHECK_EQ(num_cells, 64);
  CHECK_EQ(cell_width, 16);

  gsize = gsize <= 0 ? num_cells : gsize;

  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, gsize),
                    [&](const auto& blk) {
                      for (int i = blk.begin(); i < blk.end(); ++i) {
                        // interpolate rotation and translation separately
                        const auto& T0 = traj.at(i);
                        const auto& T1 = traj.at(i + 1);
                        const auto& R0 = T0.so3();
                        const auto& R1 = T1.so3();
                        const auto dR = (R0.inverse() * R1).log();

                        const auto& t0 = T0.translation();
                        const auto& t1 = T1.translation();
                        const auto dt = (t1 - t0).eval();

                        for (int j = 0; j < cell_width; ++j) {
                          // which column
                          const int col = i * cell_width + j;
                          const float s = static_cast<float>(j) / cell_width;
                          auto& tf = tfs.at(col);
                          tf.so3() = R0 * Sophus::SO3f::exp(s * dR);
                          tf.translation() = t0 + s * dt;
                        }
                      }
                    });
}

cv::Mat LidarSweep::DrawRange() const {
  static cv::Mat disp;
  cv::extractChannel(mat, disp, 3);
  return disp;
}

LidarSweep::LidarSweep(const cv::Size& size) : LidarScan{size} {
  tfs.resize(size.width);
}

std::string LidarSweep::Repr() const {
  return fmt::format("LidarSweep( t0={}, dt={}, xyzr={}, col_range={})",
                     t0,
                     dt,
                     sv::Repr(mat),
                     sv::Repr(curr));
}

LidarSweep MakeTestSweep(const cv::Size& size) {
  LidarSweep sweep(size);
  LidarScan scan = MakeTestScan(size);
  sweep.Add(scan);
  return sweep;
}

}  // namespace sv
