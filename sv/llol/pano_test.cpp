#include "sv/llol/pano.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/scan.h"  // MakeTestScan

namespace sv {
namespace {

TEST(DepthPanoTest, TestCtor) {
  DepthPano dp{{1024, 256}};
  EXPECT_EQ(dp.cols(), 1024);
  EXPECT_EQ(dp.rows(), 256);
  EXPECT_EQ(dp.dbuf.rows, 256);
  EXPECT_EQ(dp.dbuf.cols, 1024);
  std::cout << dp << std::endl;
}

void BM_PanoAddSweep(benchmark::State& state) {
  DepthPano pano({1024, 256});
  const auto sweep = MakeTestSweep({1024, 64});
  const int gsize = state.range(0);

  for (auto _ : state) {
    pano.Add(sweep, sweep.curr, gsize);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoAddSweep)->Arg(0)->Arg(1)->Arg(2)->Arg(4);

void BM_PanoRender(benchmark::State& state) {
  DepthPano pano({1024, 256});
  pano.dbuf.setTo(1024);
  const int gsize = state.range(0);

  for (auto _ : state) {
    pano.Render({}, gsize);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoRender)->Arg(0)->Arg(1)->Arg(2)->Arg(4);

}  // namespace
}  // namespace sv
