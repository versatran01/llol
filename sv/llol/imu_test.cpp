#include "sv/llol/imu.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sophus/interpolate.hpp>

namespace sv {
namespace {

TEST(ImuTest, TestImuNoise) {
  ImuNoise noise(0.1, 1, 2, 3, 4);
  std::cout << noise << std::endl;

  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNa), 10);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNw), 40);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kBa), 0.9);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kBw), 1.6);
}

TEST(ImuTest, TestFindNextImu) {
  ImuBuffer buffer(10);
  for (int i = 0; i < 5; ++i) {
    ImuData d;
    d.time = i;
    buffer.push_back(d);
  }

  EXPECT_EQ(FindNextImu(buffer, 15), -1);
  EXPECT_EQ(FindNextImu(buffer, 0), 1);
  EXPECT_EQ(FindNextImu(buffer, 0.5), 1);
}

TEST(ImuTest, TestImuTrajectoryPredict) {
  ImuTrajectory traj(4);
  for (int i = 0; i < 5; ++i) {
    ImuData imu;
    imu.time = i;
    traj.Add(imu);
  }

  const int n = traj.Predict(0.5, 1, traj.size());
  EXPECT_EQ(n, 4);
  EXPECT_EQ(traj.states.front().time, 0.5);
  EXPECT_EQ(traj.states.back().time, 3.5);
}

TEST(ImuTest, TestImuPreintegration) {
  ImuTrajectory traj(4);
  for (int i = 0; i < 5; ++i) {
    ImuData imu;
    imu.time = i;
    traj.Add(imu);
  }

  // Have imu that is later then end of traj
  ImuPreintegration preint;
  traj.states.front().time = 0.5;
  traj.states.back().time = 3.5;
  preint.Compute(traj);
  EXPECT_EQ(preint.n, 4);
  EXPECT_EQ(preint.duration, 3);

  // Imu ends earlier in time
  traj.states.back().time = 5.5;
  preint.Reset();
  preint.Compute(traj);
  EXPECT_EQ(preint.n, 5);
  EXPECT_EQ(preint.duration, 5);
}

TEST(ImuTest, TestImuPreintegration2) {
  ImuTrajectory traj(4);
  for (int i = 0; i < 5; ++i) {
    ImuData imu;
    imu.acc = Eigen::Vector3d::Random() * 0.01;
    imu.gyr = Eigen::Vector3d::Random() * 0.01;
    imu.time = i * 0.01;
    traj.Add(imu);
  }

  traj.noise = ImuNoise(0.01, 1e-3, 1e-4, 1e-4, 1e-5);

  ImuPreintegration preint;
  traj.states.front().time = 0;
  traj.states.back().time = 0.05;
  preint.Compute(traj);
  EXPECT_EQ(preint.n, 5);
  EXPECT_EQ(preint.duration, 0.05);
  LOG(INFO) << "\n" << preint.P;
  LOG(INFO) << "\n" << preint.F;
  LOG(INFO) << "\n" << preint.U;
}

void BM_InterpRot(benchmark::State& state) {
  const auto size = state.range(0);
  std::vector<Sophus::SO3d> Rs(size + 1);
  for (int i = 0; i < size; ++i) {
    Rs[i + 1] = Rs[i] * Sophus::SO3d::exp(0.01 * Eigen::Vector3d::Random());
  }

  std::vector<Sophus::SO3d> Rs_interp(size);

  for (auto _ : state) {
    for (int i = 0; i < size; ++i) {
      Rs_interp[i] = Sophus::interpolate(Rs[i], Rs[i + 1], 0.5);
    }
    benchmark::DoNotOptimize(Rs_interp);
  }
}
BENCHMARK(BM_InterpRot)->Arg(64)->Arg(128);

}  // namespace
}  // namespace sv
