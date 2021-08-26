#include "sv/temp/factor.h"

namespace sv {

using SE3d = Sophus::SE3d;

GicpFactor::GicpFactor(const PointMatch& match) {
  pt_s_ = match.mc_g.mean.cast<double>();
  pt_p_ = match.mc_p.mean.cast<double>();
  U_ = match.U.cast<double>();
}

GicpFactor3::GicpFactor3(const SweepGrid& grid, int size, int gsize)
    : pgrid{&grid}, size_{size}, gsize_{gsize} {
  matches.reserve(size);
  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.width(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }

  CHECK_EQ(matches.size(), size);
}

TinyGicpFactor::TinyGicpFactor(const SweepGrid& grid,
                               int size,
                               const Sophus::SE3d& T0)
    : size_(size), T0_(T0) {
  matches.reserve(size);
  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.width(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }

  CHECK_EQ(matches.size(), size);
}

}  // namespace sv
