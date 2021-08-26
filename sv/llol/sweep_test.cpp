#include "sv/llol/sweep.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(ScanTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.time, 0);
  EXPECT_EQ(ls.dt, 0);
  EXPECT_EQ(ls.xyzr.empty(), true);
}

TEST(ScanTest, TestCtor) {
  LidarSweep ls({8, 4});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.xyzr.rows, 4);
  EXPECT_EQ(ls.xyzr.cols, 8);
  EXPECT_EQ(ls.xyzr.channels(), 4);
}

TEST(ScanTest, TestAdd) {
  LidarSweep ls({8, 4});
  LidarScan scan = MakeTestScan({4, 4});
  scan.col_rg = {0, 4};

  ls.Add(scan);

  EXPECT_EQ(ls.col_rg.start, 0);
  EXPECT_EQ(ls.col_rg.end, 4);

  scan.col_rg = {4, 8};
  ls.Add(scan);
  EXPECT_EQ(ls.col_rg.start, 4);
  EXPECT_EQ(ls.col_rg.end, 8);

  scan.col_rg = {0, 4};
  ls.Add(scan);
  EXPECT_EQ(ls.col_rg.start, 0);
  EXPECT_EQ(ls.col_rg.end, 4);

  std::cout << ls << "\n";
}

void BM_SweepAdd(benchmark::State& state) {
  const cv::Size size{1024, 64};
  LidarSweep sweep(size);
  LidarScan scan = MakeTestScan(size);

  for (auto _ : state) {
    auto n = sweep.Add(scan);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_SweepAdd);

void BM_SweepInterp(benchmark::State& state) {
  LidarSweep sweep({1024, 64});
  std::vector<Sophus::SE3f> traj(64 + 1);
  int gsize = state.range(0);

  for (auto _ : state) {
    sweep.Interp(traj, gsize);
    benchmark::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepInterp)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
}  // namespace sv
