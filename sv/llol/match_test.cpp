#include "sv/llol/match.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(MatcherTest, TestCtor) {
  ProjMatcher pm({64, 32});

  EXPECT_EQ(pm.size.width, 64);
  EXPECT_EQ(pm.size.height, 32);
  EXPECT_EQ(pm.pano_win_size.height, 5);
  EXPECT_EQ(pm.pano_win_size.width, 11);
  EXPECT_EQ(pm.matches.size(), 64 * 32);
}

TEST(MatcherTest, TestMatch) {
  const auto scan = MakeTestScan({1024, 64});
  LidarSweep sweep({1024, 64});
  sweep.AddScan(scan);
  auto grid = SweepGrid(sweep.size());
  grid.Reduce(scan);
  grid.Filter();

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  ProjMatcher matcher(grid.size());
  const int n = matcher.Match(sweep, grid, pano);
  EXPECT_EQ(n, 1984);  // probably miss top and bottom
  EXPECT_EQ(n, matcher.NumMatches());
}

void BM_MatcherMatch(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  LidarSweep sweep({1024, 64});
  sweep.AddScan(scan);
  auto grid = SweepGrid(sweep.size());
  grid.Reduce(scan);
  grid.Filter();

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  ProjMatcher matcher(grid.size());

  const int gsize = state.range(0);
  for (auto _ : state) {
    matcher.Match(sweep, grid, pano, gsize);
  }
}
BENCHMARK(BM_MatcherMatch)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
}  // namespace sv
