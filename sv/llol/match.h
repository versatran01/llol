#pragma once

#include "sv/llol/grid.h"
#include "sv/llol/pano.h"
#include "sv/llol/scan.h"

namespace sv {

/// @class Feature Matcher
struct MatcherParams {
  int half_rows{2};        // half rows of pano win
  float cov_lambda{1e-6};  // lambda added to cov diagonal when inverting
};

struct ProjMatcher {
  /// Params
  float cov_lambda;        // lambda added to diagonal of covar
  cv::Size pano_win_size;  // win size in pano used to compute mean covar
  cv::Size max_dist_size;  // max dist size to resue pano mc
  int min_pts;             // min pts in pano win for a valid match

  /// @brief Ctors
  ProjMatcher(const MatcherParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ProjMatcher& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Match features in sweep to pano using mask
  /// @return Number of final matches
  int Match(SweepGrid& grid, const DepthPano& pano, int gsize = 0);
  int MatchRow(SweepGrid& grid, const DepthPano& pano, int gr);
  int MatchCell(SweepGrid& grid, const DepthPano& pano, const cv::Point& gpx);
};

}  // namespace sv
