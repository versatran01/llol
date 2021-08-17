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

TEST(GridTest, TestReduce) {
  GridParams gp;
  SweepGrid grid({1024, 64}, gp);

  auto scan = MakeTestScan({512, 64});
  scan.col_rg = {0, 512};
  const auto n1 = grid.Reduce(scan);
  EXPECT_EQ(n1, 1024);
  EXPECT_EQ(grid.width(), 32);
  std::cout << grid << std::endl;

  scan.col_rg = {512, 1024};
  const auto n2 = grid.Reduce(scan);
  EXPECT_EQ(n2, 1024);
  EXPECT_EQ(grid.width(), 64);
  std::cout << grid << std::endl;

  scan.col_rg = {0, 512};
  const auto n3 = grid.Reduce(scan);
  EXPECT_EQ(n3, 1024);
  EXPECT_EQ(grid.width(), 32);
  std::cout << grid << std::endl;
}

void BM_ReduceSeq(benchmark::State& state) {
  const LidarScan scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());

  for (auto _ : state) {
    auto n = grid.Reduce(scan, false);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_ReduceSeq);

void BM_ReducePar(benchmark::State& state) {
  const LidarScan scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());

  for (auto _ : state) {
    auto n = grid.Reduce(scan, true);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_ReducePar);

void BM_Filter(benchmark::State& state) {
  const LidarScan scan = MakeTestScan({1024, 64});
  SweepGrid grid(scan.size());
  grid.Reduce(scan, true);

  for (auto _ : state) {
    const auto n = grid.Filter();
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_Filter);

}  // namespace
}  // namespace sv
