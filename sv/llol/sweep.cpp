#include "sv/llol/sweep.h"

#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <opencv2/core.hpp>

#include "sv/util/ocv.h"

namespace sv {

int LidarSweep::Add(const LidarScan& scan) {
  CHECK_EQ(scan.type(), type());
  CHECK_EQ(scan.rows(), rows());
  CHECK_LE(scan.cols(), cols());

  UpdateTime(scan.time, scan.dt);
  UpdateView(scan.curr);

  static cv::Mat range;
  cv::extractChannel(scan.mat, range, 3);

  // copy to storage
  scan.mat.copyTo(mat.colRange(curr));  // x,y,w,h
  // TODO (chao): return number of valid points?
  return cv::countNonZero(range == range);
}

void LidarSweep::Interp(const Trajectory& traj, int gsize) {
  const int num_cells = traj.size() - 1;
  const int cell_width = cols() / num_cells;
  const auto curr_g = curr / cell_width;
  gsize = gsize <= 0 ? num_cells : gsize;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_cells, gsize), [&](const auto& blk) {
        for (int gc = blk.begin(); gc < blk.end(); ++gc) {
          // Note that the starting point of traj is where curr
          // ends, so we need to offset by curr.end to find the
          // corresponding traj segment

          const int tc = WrapCols(gc - curr_g.end, num_cells);
          const auto& st0 = traj.At(tc);
          const auto& st1 = traj.At(tc + 1);

          const auto dr = (st0.rot.inverse() * st1.rot).log();
          const auto dp = (st0.pos - st1.pos).eval();

          for (int j = 0; j < cell_width; ++j) {
            // which column
            const int col = gc * cell_width + j;
            const float s = static_cast<float>(j) / cell_width;
            Sophus::SE3d tf_p_i;
            tf_p_i.so3() = st0.rot * Sophus::SO3d::exp(s * dr);
            tf_p_i.translation() = st0.pos + s * dp;
            tfs.at(col) = (tf_p_i * traj.T_imu_lidar).cast<float>();
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
                     time,
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
