#include "sv/llol/pano.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/scan.h"  // MakeTestScan

namespace sv {
namespace {

TEST(DepthPanoTest, TestCtor) {
  DepthPano dp{{1024, 256}};
  EXPECT_EQ(dp.size().width, 1024);
  EXPECT_EQ(dp.size().height, 256);
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

void BM_PanoAddSweepSeq(benchmark::State& state) {
  DepthPano pano({1024, 256});
  const auto sweep = MakeTestSweep({1024, 64});

  for (auto _ : state) {
    pano.AddSweep(sweep, false);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoAddSweepSeq);

void BM_PanoAddSweepTbb(benchmark::State& state) {
  DepthPano pano({1024, 256});
  const auto sweep = MakeTestSweep({1024, 64});

  for (auto _ : state) {
    pano.AddSweep(sweep, true);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoAddSweepTbb);

void BM_PanoRenderSeq(benchmark::State& state) {
  DepthPano pano({1024, 256});
  pano.dbuf.setTo(1024);

  for (auto _ : state) {
    pano.Render(false);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoRenderSeq);

void BM_PanoRenderTbb(benchmark::State& state) {
  DepthPano pano({1024, 256});
  pano.dbuf.setTo(1024);

  for (auto _ : state) {
    pano.Render(true);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoRenderTbb);

}  // namespace
}  // namespace sv
