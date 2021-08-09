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
  EXPECT_EQ(ls.empty(), true);
  EXPECT_EQ(ls.full(), false);
}

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls({8, 4}, {2, 1});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.empty(), false);
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

void BM_CalcScanCurveSeq(benchmark::State& state) {
  const cv::Size cell_size{16, 2};
  const cv::Mat scan = MakeScan({1024, 64});
  cv::Mat grid{
      cv::Size{scan.cols / cell_size.width, scan.rows / cell_size.height},
      CV_32FC1};

  for (auto _ : state) {
    CalcScanCurve(scan, grid, false);
    benchmark::DoNotOptimize(grid);
  }
}
BENCHMARK(BM_CalcScanCurveSeq);

void BM_CalcScanCurvePar(benchmark::State& state) {
  const cv::Size cell_size{16, 2};
  const cv::Mat scan = MakeScan({1024, 64});
  cv::Mat grid{
      cv::Size{scan.cols / cell_size.width, scan.rows / cell_size.height},
      CV_32FC1};

  for (auto _ : state) {
    CalcScanCurve(scan, grid, true);
    benchmark::DoNotOptimize(grid);
  }
}
BENCHMARK(BM_CalcScanCurvePar);

}  // namespace
}  // namespace sv
