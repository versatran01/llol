#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"  // MakeTestScan

namespace sv {
namespace {

TEST(DepthPanoTest, TestWinAt) {
  DepthPano dp({256, 64});
  const auto win = dp.WinCenterAt({0, 0}, {5, 7});
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
  const cv::Mat sweep = MakeTestScan({1024, 64});

  for (auto _ : state) {
    pano.AddSweep(sweep, false);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoAddSweepSeq);

void BM_PanoAddSweepTbb(benchmark::State& state) {
  DepthPano pano({1024, 256});
  const cv::Mat sweep = MakeTestScan({1024, 64});

  for (auto _ : state) {
    pano.AddSweep(sweep, true);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoAddSweepTbb);

void BM_PanoRenderSeq(benchmark::State& state) {
  DepthPano pano({1024, 256});
  pano.dbuf_.setTo(1024);

  for (auto _ : state) {
    pano.Render(false);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoRenderSeq);

void BM_PanoRenderTbb(benchmark::State& state) {
  DepthPano pano({1024, 256});
  pano.dbuf_.setTo(1024);

  for (auto _ : state) {
    pano.Render(true);
    benchmark::DoNotOptimize(pano);
  }
}
BENCHMARK(BM_PanoRenderTbb);

}  // namespace
}  // namespace sv
