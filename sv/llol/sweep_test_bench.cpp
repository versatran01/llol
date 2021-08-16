#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/sweep.h"

namespace sv {
namespace {

TEST(LidarSweepTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.full(), true);
}

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls({8, 4}, {2, 1});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.full(), false);

  EXPECT_EQ(ls.xyzr.rows, 4);
  EXPECT_EQ(ls.xyzr.cols, 8);
  EXPECT_EQ(ls.xyzr.channels(), 4);

  EXPECT_EQ(ls.grid.rows, 4);
  EXPECT_EQ(ls.grid.cols, 4);
  EXPECT_EQ(ls.grid.channels(), 1);
}

TEST(LidarSweepTest, TestAddScan) {
  LidarSweep ls({8, 4}, {2, 1});
  LidarScan scan = MakeTestScan({4, 4});
  scan.col_range = {0, 4};

  const int n = ls.AddScan(scan, false);

  EXPECT_EQ(ls.col_range.start, 0);
  EXPECT_EQ(ls.col_range.end, 4);
  EXPECT_EQ(ls.id, 0);
  EXPECT_EQ(n, 8);

  scan.col_range = {4, 8};
  const int n2 = ls.AddScan(scan, false);
  EXPECT_EQ(ls.col_range.start, 4);
  EXPECT_EQ(ls.col_range.end, 8);
  EXPECT_EQ(ls.id, 0);
  EXPECT_EQ(ls.full(), true);
  EXPECT_EQ(n2, 8);

  scan.col_range = {0, 4};
  const int n3 = ls.AddScan(scan, false);
  EXPECT_EQ(ls.col_range.start, 0);
  EXPECT_EQ(ls.col_range.end, 4);
  EXPECT_EQ(ls.id, 1);
  EXPECT_EQ(ls.full(), false);
  EXPECT_EQ(n3, 8);
}

void BM_AddScanSeq(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  LidarScan scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.AddScan(scan, false);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_AddScanSeq);

void BM_AddScanPar(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  const auto scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.AddScan(scan, true);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_AddScanPar);

}  // namespace
}  // namespace sv
