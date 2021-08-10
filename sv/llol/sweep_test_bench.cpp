#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "sv/llol/sweep.h"
#include "sv/llol/test.h"

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

  EXPECT_EQ(ls.sweep().rows, 4);
  EXPECT_EQ(ls.sweep().cols, 8);
  EXPECT_EQ(ls.sweep().channels(), 4);

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

  EXPECT_EQ(ls.range().start, 0);
  EXPECT_EQ(ls.range().end, 4);
  EXPECT_EQ(n, 8);

  const int n2 = ls.AddScan(scan, {4, 8}, false);
  EXPECT_EQ(ls.range().start, 4);
  EXPECT_EQ(ls.range().end, 8);
  EXPECT_EQ(n2, 8);
  EXPECT_EQ(ls.full(), true);
}

void BM_AddScanSeq(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  const auto scan = MakeScan(size);

  for (auto _ : state) {
    sweep.AddScan(scan, {0, size.width}, false);
    benchmark::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_AddScanSeq);

void BM_AddScanPar(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size, {16, 2});
  const auto scan = MakeScan(size);

  for (auto _ : state) {
    sweep.AddScan(scan, {0, size.width}, true);
    benchmark::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_AddScanPar);

}  // namespace
}  // namespace sv
