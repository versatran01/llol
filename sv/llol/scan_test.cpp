#include "sv/llol/scan.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(LidarSweepTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.full(), true);
}

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls({8, 4});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.full(), false);

  EXPECT_EQ(ls.xyzr.rows, 4);
  EXPECT_EQ(ls.xyzr.cols, 8);
  EXPECT_EQ(ls.xyzr.channels(), 4);
}

TEST(LidarSweepTest, TestAddScan) {
  LidarSweep ls({8, 4});
  LidarScan scan = MakeTestScan({4, 4});
  scan.col_range = {0, 4};

  ls.AddScan(scan);

  EXPECT_EQ(ls.col_range.start, 0);
  EXPECT_EQ(ls.col_range.end, 4);
  EXPECT_EQ(ls.id, 0);

  scan.col_range = {4, 8};
  ls.AddScan(scan);
  EXPECT_EQ(ls.col_range.start, 4);
  EXPECT_EQ(ls.col_range.end, 8);
  EXPECT_EQ(ls.id, 0);
  EXPECT_EQ(ls.full(), true);

  scan.col_range = {0, 4};
  ls.AddScan(scan);
  EXPECT_EQ(ls.col_range.start, 0);
  EXPECT_EQ(ls.col_range.end, 4);
  EXPECT_EQ(ls.id, 1);
  EXPECT_EQ(ls.full(), false);

  std::cout << ls << "\n";
}

void BM_AddScan(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size);
  LidarScan scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.AddScan(scan);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_AddScan);

}  // namespace
}  // namespace sv
