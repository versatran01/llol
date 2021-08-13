#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/util/math.h"

namespace sv {

TEST(MathTest, TestAngleConversion) {
  EXPECT_DOUBLE_EQ(Deg2Rad(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(90.0), M_PI / 2.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(180.0), M_PI);
  EXPECT_DOUBLE_EQ(Deg2Rad(360.0), M_PI * 2);
  EXPECT_DOUBLE_EQ(Deg2Rad(-180.0), -M_PI);

  EXPECT_DOUBLE_EQ(Rad2Deg(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI / 2), 90.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI), 180.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI * 2), 360.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(-M_PI), -180.0);
}

TEST(MathTest, TestMeanCovar) {
  for (int i = 3; i < 50; i += 10) {
    const auto X = Eigen::Matrix3Xd::Random(3, i).eval();
    const auto cov0 = CalCovar3d(X);
    const Eigen::Vector3d m = X.rowwise().mean();

    MeanCovar3d mc;
    for (int j = 0; j < X.cols(); ++j) mc.Add(X.col(j));
    const auto cov1 = mc.Covar();

    EXPECT_TRUE(cov0.isApprox(cov1));
    EXPECT_TRUE(mc.mean.isApprox(m));
  }
}

TEST(LinalgTest, TestMatrixSqrtUpper) {
  Eigen::Matrix3Xd X = Eigen::Matrix3Xd::Random(3, 100);
  const Eigen::Matrix3d A = X * X.transpose();
  const Eigen::Matrix3d U = MatrixSqrtUtU(A);
  const Eigen::Matrix3d UtU = U.transpose() * U;
  EXPECT_TRUE(A.isApprox(UtU));
}

void BM_Covariance(benchmark::State& state) {
  const auto X = Eigen::Matrix3Xd::Random(3, state.range(0)).eval();

  for (auto _ : state) {
    benchmark::DoNotOptimize(CalCovar3d(X));
  }
}
BENCHMARK(BM_Covariance)->Range(8, 512);

void BM_MeanCovar(benchmark::State& state) {
  const auto X = Eigen::Matrix3Xd::Random(3, state.range(0)).eval();

  for (auto _ : state) {
    MeanCovar3d mc;
    for (int i = 0; i < X.cols(); ++i) mc.Add(X.col(i));
    const auto cov = mc.Covar();
    benchmark::DoNotOptimize(cov);
  }
}
BENCHMARK(BM_MeanCovar)->Range(8, 512);
}  // namespace sv