#include "sv/llol/grid.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(GridTest, TestCtor) {
  const SweepGrid grid({1024, 64});
  EXPECT_EQ(grid.total(), 2048);
  EXPECT_EQ(grid.width(), 0);
  EXPECT_EQ(grid.full(), false);
  EXPECT_EQ(grid.size().width, 64);
  EXPECT_EQ(grid.size().height, 32);
  EXPECT_EQ(grid.col_rg.start, 0);
  EXPECT_EQ(grid.col_rg.end, 0);
  std::cout << grid << std::endl;
}

TEST(GridTest, TestConversion) {
  const SweepGrid grid({1024, 64});
  EXPECT_EQ(grid.Sweep2Grid({0, 0}), cv::Point(0, 0));
  EXPECT_EQ(grid.Sweep2Grid({1, 1}), cv::Point(0, 0));
  EXPECT_EQ(grid.Grid2Sweep({0, 0}), cv::Point(0, 0));
  EXPECT_EQ(grid.Grid2Sweep({1, 1}), cv::Point(16, 2));
}

TEST(GridTest, TestScore) {
  SweepGrid grid({1024, 64});

  auto scan = MakeTestScan({512, 64});
  scan.col_rg = {0, 512};
  const auto n1 = grid.Score(scan);
  EXPECT_EQ(n1, 1024);
  EXPECT_EQ(grid.width(), 32);
  EXPECT_TRUE(std::isnan(grid.ScoreAt({32, 0})));
  EXPECT_EQ(grid.ScoreAt({0, 0}), 0);
  EXPECT_EQ(grid.ScoreAt({31, 0}), 0);
  std::cout << grid << std::endl;

  scan.col_rg = {512, 1024};
  const auto n2 = grid.Score(scan);
  EXPECT_EQ(n2, 1024);
  EXPECT_EQ(grid.width(), 64);
  std::cout << grid << std::endl;

  scan.col_rg = {0, 512};
  const auto n3 = grid.Score(scan);
  EXPECT_EQ(n3, 1024);
  EXPECT_EQ(grid.width(), 32);
  std::cout << grid << std::endl;
}

void BM_Score(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());
  const int gsize = state.range(0);

  for (auto _ : state) {
    auto n = grid.Score(scan, gsize);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_Score)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

void BM_Filter(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());
  grid.Score(scan);

  for (auto _ : state) {
    const auto n = grid.Filter();
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_Filter);

void BM_Reduce(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());
  grid.Score(scan);
  grid.Filter();
  const int gsize = state.range(0);

  for (auto _ : state) {
    grid.Reduce(scan, gsize);
    benchmark::DoNotOptimize(grid);
  }
}
BENCHMARK(BM_Reduce)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
}  // namespace sv
