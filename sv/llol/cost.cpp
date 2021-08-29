#include "sv/llol/cost.h"

namespace sv {

GicpCost::GicpCost(const SweepGrid& grid, int gsize) : pgrid{&grid} {
  // Collect all good matches
  matches.reserve(grid.total() / 4);
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }

  // we don't want to use grainsize of 1 or 2, because each residual is 3
  // doubles which is 3 * 8 = 24. However a cache line is typically 64 bytes so
  // we need at least 3 residuals (3 * 3 * 8 = 72 bytes) to fill one cache line
  gsize_ = gsize <= 0 ? matches.size() : gsize + 2;
}

ImuCost::ImuCost(const ImuTrajectory& traj) : ptraj{&traj} {
  preint.Compute(traj);
}

GicpAndImuCost::GicpAndImuCost(const SweepGrid& grid,
                               const ImuTrajectory& traj,
                               int gsize)
    : gicp_cost(grid, gsize), imu_cost(traj) {}

}  // namespace sv
