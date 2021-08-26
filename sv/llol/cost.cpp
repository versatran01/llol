#include "sv/llol/cost.h"

namespace sv {

GicpCostBase::GicpCostBase(const SweepGrid& grid, int gsize)
    : pgrid{&grid}, gsize{gsize} {
  // Collect all good matches
  matches.reserve(grid.total() / 4);
  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.size().width; ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }

  // Get poses of each grid col
  tfs_g.reserve(grid.size().width);
  for (int c = 0; c < grid.size().width; ++c) {
    tfs_g.push_back(grid.CellTfAt(c).cast<double>());
  }
}

}  // namespace sv
