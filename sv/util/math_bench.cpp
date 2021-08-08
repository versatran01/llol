#include <benchmark/benchmark.h>

#include "sv/util/math.h"

namespace sv {

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
    const auto cov = mc.covar();
    benchmark::DoNotOptimize(cov);
  }
}
BENCHMARK(BM_MeanCovar)->Range(8, 512);

}  // namespace sv
