#include "sv/llol/sweep.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(ScanTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.t0, 0);
  EXPECT_EQ(ls.dt, 0);
  EXPECT_EQ(ls.empty(), true);
}

TEST(ScanTest, TestCtor) {
  LidarSweep ls({8, 4});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.rows(), 4);
  EXPECT_EQ(ls.cols(), 8);
  EXPECT_EQ(ls.channels(), 4);
}

TEST(ScanTest, TestAdd) {
  LidarSweep ls({8, 4});
  LidarScan scan = MakeTestScan({4, 4});
  scan.curr = {0, 4};

  ls.Add(scan);

  EXPECT_EQ(ls.curr.start, 0);
  EXPECT_EQ(ls.curr.end, 4);
  EXPECT_EQ(ls.span.start, 0);
  EXPECT_EQ(ls.span.end, 4);

  scan.curr = {4, 8};
  ls.Add(scan);
  EXPECT_EQ(ls.curr.start, 4);
  EXPECT_EQ(ls.curr.end, 8);
  EXPECT_EQ(ls.span.start, 0);
  EXPECT_EQ(ls.span.end, 8);

  scan.curr = {0, 4};
  ls.Add(scan);
  EXPECT_EQ(ls.curr.start, 0);
  EXPECT_EQ(ls.curr.end, 4);
  EXPECT_EQ(ls.span.start, 4);
  EXPECT_EQ(ls.span.end, 12);

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
