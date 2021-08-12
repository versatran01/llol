#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

class LidarSweep;
class DepthPano;

/// @struct Match
struct PointMatch {
  cv::Point src_px{-1, -1};  // 8
  cv::Point dst_px{-1, -1};  // 8
  MeanCovar3f src{};         // sweep 52
  MeanCovar3f dst{};         // pano 52
  Eigen::Matrix3d U;         // 72
};
static_assert(sizeof(PointMatch) == 192, "PointMatch size is not 192");

/// @class Feature Matcher
struct MatcherParams {
  bool nms{true};
  int half_rows{2};
  double max_curve{0.01};
  double range_ratio{0.1};
};

// TODO: rename to GridMatcher
struct PointMatcher {
  PointMatcher() = default;
  // TODO: use grid size
  PointMatcher(const cv::Size& grid_size, const MatcherParams& params);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const PointMatcher& rhs);

  /// @brief getters
  const auto& matches() const noexcept { return matches_; }

  /// @brief Match features in sweep to pano
  void Match(const LidarSweep& sweep, const DepthPano& pano, bool tbb = false);
  void MatchSingle(const LidarSweep& sweep,
                   const DepthPano& pano,
                   const cv::Point& gpx);

  MatcherParams params_;
  cv::Size pano_win_size_;
  std::vector<PointMatch> matches_;
};

/// @brief Draw match, valid pixel is percentage of pano points in window
cv::Mat DrawMatches(const LidarSweep& sweep,
                    const std::vector<PointMatch>& matches);

}  // namespace sv
