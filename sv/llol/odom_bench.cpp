#include <benchmark/benchmark.h>

#include "sv/llol/odom.h"

namespace sv {
namespace {

cv::Mat MakeScan(cv::Size size) {
  cv::Mat sweep = cv::Mat::zeros(size, CV_32FC4);

  const float azim_delta = M_PI * 2 / size.width;
  const float elev_max = M_PI_4 / 2;
  const float elev_delta = elev_max * 2 / (size.height - 1);

  for (int i = 0; i < sweep.rows; ++i) {
    for (int j = 0; j < sweep.cols; ++j) {
      const float elev = elev_max - i * elev_delta;
      const float azim = M_PI * 2 - j * azim_delta;

      auto& xyzr = sweep.at<cv::Vec4f>(i, j);
      xyzr[0] = std::cos(elev) * std::cos(azim);
      xyzr[1] = std::cos(elev) * std::sin(azim);
      xyzr[2] = std::sin(elev);
      xyzr[3] = 1;
    }
  }

  return sweep;
}

void BM_SweepAddScanSeq(benchmark::State& state) {
  int cell = state.range(0);
  int cols = 1024;
  LidarSweep sweep({cols, 64}, {cell, 1});
  cv::Mat scan = MakeScan(sweep.sweep_size());

  for (auto _ : state) {
    sweep.AddScan(scan, {0, cols}, false);
    benchmark::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepAddScanSeq)->Arg(16)->Arg(8);

void BM_SweepAddScanTbb(benchmark::State& state) {
  int cell = state.range(0);
  int cols = 1024;
  LidarSweep sweep({cols, 64}, {cell, 1});
  cv::Mat scan = MakeScan(sweep.sweep_size());

  for (auto _ : state) {
    sweep.AddScan(scan, {0, cols}, true);
    benchmark::DoNotOptimize(sweep);
  }
}
BENCHMARK(BM_SweepAddScanTbb)->Arg(16)->Arg(8);

// void BM_PanoAddSweepSeq(benchmark::State& state) {
//  DepthPano pano({1024, 256});
//  cv::Mat sweep = MakeScan({1024, 64});

//  for (auto _ : state) {
//    pano.AddSweep(sweep, false);
//    benchmark::DoNotOptimize(pano);
//  }
//}
// BENCHMARK(BM_PanoAddSweepSeq);

// void BM_PanoAddSweepPar(benchmark::State& state) {
//  DepthPano pano({1024, 256});
//  cv::Mat sweep = MakeScan({1024, 64});

//  for (auto _ : state) {
//    pano.AddSweep(sweep, true);
//    benchmark::DoNotOptimize(pano);
//  }
//}
// BENCHMARK(BM_PanoAddSweepPar);

// void BM_PanoRenderSeq(benchmark::State& state) {
//  DepthPano pano({1024, 256});
//  pano.mat_.setTo(1024);

//  for (auto _ : state) {
//    pano.Render(false);
//    benchmark::DoNotOptimize(pano);
//  }
//}
// BENCHMARK(BM_PanoAddSweepSeq);

// void BM_PanoRenderPar(benchmark::State& state) {
//  DepthPano pano({1024, 256});
//  pano.mat_.setTo(1024);

//  for (auto _ : state) {
//    pano.Render(true);
//    benchmark::DoNotOptimize(pano);
//  }
//}
// BENCHMARK(BM_PanoAddSweepPar);

}  // namespace
}  // namespace sv
