#pragma once

#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"

namespace sv {

/// @struct Match
struct PointMatch {
  cv::Point pt{};
  MeanCovar3f src{};  // sweep
  MeanCovar3f dst{};  // pano
};

/// @class Feature Matcher
struct MatcherParams {
  bool nms{true};
  int half_rows{2};
  double max_curve{0.01};
  double range_ratio{0.1};
};

/// @brief Mat must be 32FC4
void MatXyzr2MeanCovar(const cv::Mat& mat, MeanCovar3f& mc);

float CalcRangeDiffRel(float rg1, float rg2) {
  return std::abs(rg1 - rg2) / std::max(rg1, rg2);
}

/// @brief Check if a point is a good candidate for matching
bool IsCellGood(const cv::Mat& grid, cv::Point px, double max_curve, bool nms);

struct PointMatcher {
  PointMatcher() = default;
  PointMatcher(int max_matches, const MatcherParams& params);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const PointMatcher& rhs);

  /// @brief getters
  cv::Size win_size() const noexcept { return win_size_; }
  const auto& matches() const noexcept { return matches_; }

  /// @brief Match features in sweep to pano
  void Match(const LidarSweep& sweep, const DepthPano& pano);

  /// @brief Draw match, valid pixel is percentage of pano points in window
  cv::Mat Draw(const LidarSweep& sweep) const;

  MatcherParams params_;
  cv::Size win_size_;
  std::vector<PointMatch> matches_;
};

}  // namespace sv
