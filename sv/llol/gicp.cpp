#include "sv/llol/gicp.h"

#include <glog/logging.h>

namespace sv {

using SE3d = Sophus::SE3d;

bool LocalParamSE3::Plus(const double* _T,
                         const double* _x,
                         double* _T_plus_x) const {
  Eigen::Map<const SE3d> T(_T);
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> t(_x);
  Eigen::Map<SE3d> T_plus_del(_T_plus_x);
  T_plus_del = T * SE3d::exp(t);
  return true;
}

bool LocalParamSE3::ComputeJacobian(const double* _T, double* _J) const {
  Eigen::Map<SE3d const> T(_T);
  Eigen::Map<
      Eigen::Matrix<double, SE3d::num_parameters, SE3d::DoF, Eigen::RowMajor>>
      J(_J);
  J = T.Dx_this_mul_exp_x_at_0();
  return true;
}

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

GicpCostSingle2::GicpCostSingle2(const SweepGrid& grid, int gsize)
    : pgrid{&grid}, gsize{gsize} {
  // Collect all good matches
  pmatches.reserve(grid.total() / 4);
  for (int r = 0; r < grid.size().height; ++r) {
    for (int c = 0; c < grid.size().width; ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      pmatches.push_back(&grid.matches[grid.Grid2Ind({c, r})]);
    }
  }

  // Get poses of each grid col
  // TODO (chao): this needs to be done several times
  tfs_g.reserve(grid.size().width);
  for (int c = 0; c < grid.size().width; ++c) {
    tfs_g.push_back(grid.CellTfAt(c).cast<double>());
  }
}

}  // namespace sv
