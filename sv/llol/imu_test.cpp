#include "sv/llol/imu.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sophus/interpolate.hpp>

namespace sv {
namespace {

TEST(ImuTest, TestImuNoise) {
  ImuNoise noise(10, 1, 2, 3, 4);
  std::cout << noise << std::endl;

  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNa), 10);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNw), 40);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNba), 0.9);
  EXPECT_DOUBLE_EQ(noise.sigma2(ImuNoise::kNbw), 1.6);
}

TEST(ImuTest, TestFindNextImu) {
  ImuBuffer buffer(10);
  for (int i = 0; i < 5; ++i) {
    ImuData d;
    d.time = i;
    buffer.push_back(d);
  }

  EXPECT_EQ(GetImuIndexAfterTime(buffer, 0), 1);
  EXPECT_EQ(GetImuIndexAfterTime(buffer, 0.5), 1);
  EXPECT_EQ(GetImuIndexAfterTime(buffer, 1.5), 2);
  EXPECT_EQ(GetImuIndexAfterTime(buffer, 15), -1);
}

TEST(ImuTest, TestImuPreintegration) {
  ImuQueue imuq;
  for (int i = 0; i < 5; ++i) {
    ImuData imu;
    imu.time = i;
    imuq.Add(imu);
  }

  // Have imu that is later then end of traj
  ImuPreintegration preint;
  preint.Compute(imuq, 0.5, 3.5);
  EXPECT_EQ(preint.n, 4);
  EXPECT_EQ(preint.duration, 3);

  // Imu ends earlier in time
  preint.Reset();
  preint.Compute(imuq, 0.5, 5.5);
  EXPECT_EQ(preint.n, 5);
  EXPECT_EQ(preint.duration, 5);
}

TEST(ImuTest, TestImuPreintegrationPrint) {
  ImuQueue imuq;
  for (int i = 0; i < 10; ++i) {
    ImuData imu;
    imu.acc = Eigen::Vector3d::Ones() * 0.1;
    imu.gyr = Eigen::Vector3d::Ones() * 0.02;
    imu.time = i * 0.01;
    imuq.Add(imu);
  }

  imuq.noise = ImuNoise(100.0, 1e-3, 1e-4, 1e-4, 1e-5);
  LOG(INFO) << imuq.noise;

  ImuPreintegration preint;
  preint.Compute(imuq, 0, 0.1);
  EXPECT_EQ(preint.n, 10);
  EXPECT_EQ(preint.duration, 0.1);
  LOG(INFO) << "F\n" << preint.F;
  LOG(INFO) << "P\n" << preint.P;
  LOG(INFO) << "U\n" << preint.U;
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
