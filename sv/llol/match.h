#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

class LidarSweep;
class DepthPano;

/// @struct Match
struct PointMatch {
  cv::Point px_s{-100, -100};  // 8
  MeanCovar3f src{};           // 52 sweep
  cv::Point px_p{-100, -100};  // 8
  MeanCovar3f dst{};           // 52 pano
  Eigen::Matrix3f U;           // 36

  /// @brief Wether this match is good
  bool ok() const noexcept {
    return px_s.x >= 0 && px_p.x >= 0 && src.ok() && dst.ok();
  }
};
static_assert(sizeof(PointMatch) == 156, "PointMatch size is not 192");

/// @class Feature Matcher
struct MatcherParams {
  bool nms{true};
  int half_rows{2};
  float max_curve{0.01};
  float range_ratio{0.1};  // TODO (chao): consider surface angle
  float tan_phi{std::tan(static_cast<float>(M_PI / 2.5))};

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const MatcherParams& rhs) {
    return os << rhs.Repr();
  }
};

struct PointMatcher {
  PointMatcher() = default;
  PointMatcher(const cv::Size& grid_size, const MatcherParams& params);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const PointMatcher& rhs) {
    return os << rhs.Repr();
  }

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
