#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/sweep.h"

namespace sv {
namespace {

TEST(LidarSweepTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.IsFull(), true);
}

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls({8, 4}, {2, 1});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.IsFull(), false);

  EXPECT_EQ(ls.xyzr().rows, 4);
  EXPECT_EQ(ls.xyzr().cols, 8);
  EXPECT_EQ(ls.xyzr().channels(), 4);

  EXPECT_EQ(ls.grid().rows, 4);
  EXPECT_EQ(ls.grid().cols, 4);
  EXPECT_EQ(ls.grid().channels(), 1);
}

TEST(LidarSweepTest, TestAddScan) {
  LidarSweep ls({8, 4}, {2, 1});
  cv::Mat scan;
  scan.create(4, 4, CV_32FC4);
  scan.setTo(1);

  const int n = ls.AddScan(scan, {0, 4}, false);

  EXPECT_EQ(ls.col_range.start, 0);
  EXPECT_EQ(ls.col_range.end, 4);
  EXPECT_EQ(n, 8);

  const int n2 = ls.AddScan(scan, {4, 8}, false);
  EXPECT_EQ(ls.col_range.start, 4);
  EXPECT_EQ(ls.col_range.end, 8);
  EXPECT_EQ(n2, 8);
  EXPECT_EQ(ls.IsFull(), true);
}

void BM_AddScanSeq(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  const auto scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.AddScan(scan, {0, size.width}, false);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_AddScanSeq);

void BM_AddScanPar(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  const auto scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.AddScan(scan, {0, size.width}, true);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_AddScanPar);

}  // namespace
}  // namespace sv
