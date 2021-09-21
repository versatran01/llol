#include "sv/llol/traj.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(TrajTest, TestPredict) {
  Trajectory traj(4);
  ImuQueue imuq;

  for (int i = 0; i < 5; ++i) {
    ImuData imu;
    imu.time = i;
    imuq.Add(imu);
  }

  const int n = traj.PredictNew(imuq, 0.5, 1, traj.size() - 1);
  EXPECT_EQ(n, 4);
  EXPECT_EQ(traj.front().time, 0.5);
  EXPECT_EQ(traj.back().time, 3.5);
}

}  // namespace
}  // namespace sv
