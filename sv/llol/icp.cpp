#include "sv/llol/icp.h"

#include <glog/logging.h>

namespace sv {

using SE3d = Sophus::SE3d;

GicpCostBase::GicpCostBase(const SweepGrid& grid, int size) : pgrid{&grid} {
  matches.reserve(size);
  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.size().width; ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }

  CHECK_EQ(matches.size(), size);
}

}  // namespace sv
