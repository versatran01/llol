#pragma once

#include <opencv2/core/mat.hpp>

namespace sv {

struct Pose {};

struct LidarScan {
  LidarScan(const cv::Mat& xyzr, const cv::Range& col_range);
  double t;
  cv::Mat xyzr;
  cv::Mat col_range;
};

struct SweepParams {
  cv::Size size;
  double dt;
  float dazimuth;
};

struct LidarSweep {
  LidarSweep(const SweepParams& params);

  /// Add a scan to this sweep and filter
  void AddScan(const LidarScan& scan);

  int id;
  double t;
  cv::Mat xyzr;  // 32FC4
  cv::Mat col_range;
  std::vector<Pose> tfs;
};

struct GridParams {};

struct FeatureGrid {
  FeatureGrid(const cv::Size& sweep_size, const GridParams& params);

  /// Compute smoothness score of each cell and filter out bad ones
  void Filter(const LidarScan& scan);

  cv::Size cell_size;
  cv::Mat score;  // 32FC1
  cv::Mat mask;   // 8UC1
  std::vector<Pose> tfs;
};

struct SweepGrid {
  LidarSweep sweep;
  FeatureGrid grid;
};

struct DepthPano {
  cv::Mat buf;
};

struct NormalMatch {
  cv::Point px_s;
  cv::Mat mc_s;
  cv::Point px_p;
  cv::Mat mc_p;
  cv::Mat U;
};

struct PointMatcher {
  int Match(const LidarSweep& sweep, const DepthPano& pano);
  std::vector<NormalMatch> matches_;
};

}  // namespace sv
