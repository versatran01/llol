#pragma once

#include "sv/llol/grid.h"
#include "sv/llol/pano.h"

namespace sv {

struct GicpParams {
  int outer{2};
  int inner{2};
  int half_rows{2};
  int half_cols{6};
  float cov_lambda{1e-6F};
  double imu_weight{0.0};
  double min_eigval{0.0};
};

struct GicpSolver {
  explicit GicpSolver(const GicpParams& params = {});

  /// Params
  std::pair<int, int> iters;  // (outer, inner) iterations
  float cov_lambda{};         // lambda added to diagonal of covar
  cv::Size pano_win;          // win size in pano used to compute mean covar
  int pano_min_pts{};         // min pts in pano win for a valid match
  double imu_weight{};        // how much weight to put on imu cost
  double min_eigval{};        // min eigenvalues for solution remapping

  /// @brief Repr / <<
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const GicpSolver& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Match features in sweep to pano using mask
  /// @return Number of final matches
  int Match(SweepGrid& grid, const DepthPano& pano, int gsize = 0);
  int MatchRow(SweepGrid& grid, const DepthPano& pano, int gr);
  int MatchCell(SweepGrid& grid, const DepthPano& pano, const cv::Point& px_g);
};

}  // namespace sv
