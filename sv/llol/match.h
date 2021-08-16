#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "sv/util/math.h"

namespace sv {

class LidarSweep;
class DepthPano;

/// @struct Match
struct PointMatch {
  static constexpr int kBad = -100;

  cv::Point px_s{kBad, kBad};  // 8
  MeanCovar3f mc_s{};          // 52 sweep

  cv::Point px_p{kBad, kBad};  // 8
  MeanCovar3f mc_p{};          // 52 pano
  Eigen::Matrix3f U;

  Sophus::SE3f tf_p_s{};  // 14

  /// @brief Whether this match is good
  bool ok() const noexcept {
    return px_s.x >= 0 && px_p.x >= 0 && mc_s.ok() && mc_p.ok();
  }
};

/// @class Feature Matcher
struct MatcherParams {
  bool nms{true};         // non-minimum suppression
  float max_curve{0.01};  // max curvature to be considered a good match

  int half_rows{2};        // half rows of pano win
  float min_dist2{2};      // min dist^2 for recompute mc in pano
  float range_ratio{0.1};  // range ratio when computing mc in pano
  float lambda{1e-6};      // lambda added to diagonal of cov when inverting

  float tan_phi{std::tan(static_cast<float>(M_PI / 2.5))};  // not used

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const MatcherParams& rhs) {
    return os << rhs.Repr();
  }
};

struct ProjMatcher {
  /// Data
  cv::Size pano_win_size;  // win size in pano used to compute mean covar
  int min_pts;             // min pts in pano win for a valid match

  int id{-1};
  cv::Mat mask;         // binary mask indicating good (1) vs bad (0) cell
  cv::Range col_range;  // current range
  MatcherParams params;
  std::vector<Sophus::SE3f> tfs;  // transforms from sweep to pano
  std::vector<PointMatch> matches;

  /// @brief Ctors
  ProjMatcher() = default;
  ProjMatcher(const cv::Size& grid_size, const MatcherParams& params);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ProjMatcher& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Filter grid to mask
  /// @return Number of putative matches
  int Filter(const LidarSweep& sweep);

  /// @brief Match features in sweep to pano using mask
  /// @return Number of final matches
  int Match(const LidarSweep& sweep, const DepthPano& pano, bool tbb = false);
  int MatchRow(const LidarSweep& sweep, const DepthPano& pano, int gr);
  int MatchCell(const LidarSweep& sweep,
                const DepthPano& pano,
                const cv::Point& gpx);

  int width() const noexcept { return col_range.end; }
  bool full() const noexcept { return width() == mask.cols; }

  void Reset();
};

/// @brief Draw match, valid pixel is percentage of pano points in window
cv::Mat DrawMatches(const LidarSweep& sweep,
                    const std::vector<PointMatch>& matches);

}  // namespace sv
