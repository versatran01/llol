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

TEST(DepthPanoTest, TestWinAt) {
  DepthPano dp({256, 64});
  const auto win = WinCenterAt({0, 0}, {5, 7});
  EXPECT_EQ(win.x, -2);
  EXPECT_EQ(win.y, -3);
  EXPECT_EQ(win.width, 5);
  EXPECT_EQ(win.height, 7);
}

TEST(DepthPanoTest, TestBoundedWinAt) {
  DepthPano dp({256, 64});
  const auto win = dp.BoundWinCenterAt({0, 0}, {5, 7});
  EXPECT_EQ(win.x, 0);
  EXPECT_EQ(win.y, 0);
  EXPECT_EQ(win.width, 3);
  EXPECT_EQ(win.height, 4);
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
