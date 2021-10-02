#include "sv/llol/gicp.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(GicpTest, TestCtor) {
  GicpSolver gicp;
  std::cout << gicp << "\n";

  EXPECT_EQ(gicp.outer_iters, 3);
  EXPECT_EQ(gicp.inner_iters, 3);
  EXPECT_EQ(gicp.cov_lambda, 1e-6F);
}

TEST(GicpTest, TestMatch) {
  const auto scan = MakeTestScan({1024, 64});
  auto grid = SweepGrid(scan.size());
  grid.Add(scan);

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  GicpSolver gicp;

  const auto n = gicp.Match(grid, pano);
  EXPECT_EQ(n, 1984);  // probably miss top and bottom
}

void BM_GicpMatch(benchmark::State& state) {
  const auto scan = MakeTestScan({1024, 64});
  auto grid = SweepGrid(scan.size());
  grid.Add(scan);

  DepthPano pano({1024, 256});
  pano.dbuf.setTo(DepthPixel::kScale);

  GicpSolver gicp;

  const auto gsize = state.range(0);
  for (auto _ : state) {
    const auto n = gicp.Match(grid, pano, gsize);
    benchmark::DoNotOptimize(n);
  }
}
BENCHMARK(BM_GicpMatch)->Arg(0)->Arg(1)->Arg(2)->Arg(4);

}  // namespace
}  // namespace sv
