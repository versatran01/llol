#include "sv/llol/sweep.h"

#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <opencv2/core/core.hpp>

#include "sv/util/ocv.h"

namespace sv {

int LidarSweep::Add(const LidarScan& scan) {
  CHECK_EQ(scan.type(), type());
  CHECK_EQ(scan.rows(), rows());
  CHECK_LE(scan.cols(), cols());

  UpdateTime(scan.t0, scan.dt);
  UpdateView(scan.curr);

  // copy to storage
  scan.mat.copyTo(mat.colRange(curr));  // x,y,w,h
  // TODO (chao): return number of valid points?
  return scan.total();
}

void LidarSweep::Interp(const ImuTrajectory& traj, int gsize) {
  const int num_cells = traj.size() - 1;
  const int cell_width = cols() / num_cells;
  gsize = gsize <= 0 ? num_cells : gsize;

  tbb::parallel_for(tbb::blocked_range<int>(0, num_cells, gsize),
                    [&](const auto& blk) {
                      for (int i = blk.begin(); i < blk.end(); ++i) {
                        // interpolate rotation and translation separately
                        const auto& T0 = traj.StateAt(i);
                        const auto& T1 = traj.StateAt(i + 1);

                        const auto dR = (T0.rot.inverse() * T1.rot).log();
                        const auto dt = (T0.pos - T1.pos).eval();

                        for (int j = 0; j < cell_width; ++j) {
                          // which column
                          const int col = i * cell_width + j;
                          const float s = static_cast<float>(j) / cell_width;
                          Sophus::SE3d tf;
                          tf.so3() = T0.rot * Sophus::SO3d::exp(s * dR);
                          tf.translation() = T0.pos + s * dt;
                          tfs.at(col) = (tf * traj.T_imu_lidar).cast<float>();
                        }
                      }
                    });
}

cv::Mat LidarSweep::DrawRange() const {
  static cv::Mat disp;
  cv::extractChannel(mat, disp, 3);
  return disp;
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
