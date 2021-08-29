#include "sv/llol/cost.h"

namespace sv {

GicpCost::GicpCost(const SweepGrid& grid, int gsize)
    : gsize{gsize}, pgrid{&grid} {
  // Collect all good matches
  matches.reserve(grid.total() / 4);
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }
}

}  // namespace sv
