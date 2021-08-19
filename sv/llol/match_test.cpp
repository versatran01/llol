#include "sv/llol/match.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(MatcherTest, TestCtor) {
  ProjMatcher pm;
  EXPECT_EQ(pm.pano_win_size.height, 5);
  EXPECT_EQ(pm.pano_win_size.width, 11);
}

TEST(MatcherTest, TestMatch) {
  const auto scan = MakeTestScan({1024, 64});
  auto grid = SweepGrid(scan.size());
  grid.Add(scan);

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  ProjMatcher matcher;
  const int n = matcher.Match(grid, pano);
  EXPECT_EQ(n, 1984);  // probably miss top and bottom
  //  EXPECT_EQ(n, matcher.NumMatches());
}

void BM_MatcherMatch(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  auto grid = SweepGrid(scan.size());
  grid.Add(scan);

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  ProjMatcher matcher;

  const int gsize = state.range(0);
  for (auto _ : state) {
    matcher.Match(grid, pano, gsize);
  }
}
BENCHMARK(BM_MatcherMatch)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
}  // namespace sv
