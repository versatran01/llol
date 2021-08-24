#include "sv/llol/imu.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <sophus/interpolate.hpp>

namespace sv {
namespace {

TEST(ImuTest, TestExtractImus) {
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

void BM_IntegrateRot(benchmark::State& state) {
  const auto size = state.range(0);
  std::vector<Sophus::SO3d> Rs(size + 1);
  const Eigen::Matrix3Xd gyrs = Eigen::Matrix3Xd::Random(3, size) * 0.01;

  for (auto _ : state) {
    for (int i = 0; i < size; ++i) {
      Rs[i + 1] = IntegrateRot(Rs[i], gyrs.col(i), 0.01);
    }
    benchmark::DoNotOptimize(Rs);
  }
}
BENCHMARK(BM_IntegrateRot)->Arg(64)->Arg(128);

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
