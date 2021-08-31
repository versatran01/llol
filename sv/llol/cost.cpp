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

bool GicpRigidCost::operator()(const double* _x, double* _r, double* _J) const {
  const State<double> es(_x);
  const Sophus::SE3d eT{Sophus::SO3d::exp(es.r0()), es.p0()};

  tbb::parallel_for(
      tbb::blocked_range<int>(0, matches.size(), gsize_), [&](const auto& blk) {
        for (int i = blk.begin(); i < blk.end(); ++i) {
          const auto& match = matches.at(i);
          const auto c = match.px_g.x;
          const auto U = match.U.cast<double>().eval();
          const auto pt_p = match.mc_p.mean.cast<double>().eval();
          const auto pt_g = match.mc_g.mean.cast<double>().eval();
          const auto tf_p_g = pgrid->tfs.at(c).cast<double>();
          const auto pt_p_hat = tf_p_g * pt_g;

          const int ri = kNumResiduals * i;
          Eigen::Map<Eigen::Vector3d> r(_r + ri);
          r = U * (pt_p - eT * pt_p_hat);

          if (_J) {
            Eigen::Map<Eigen::MatrixXd> J(_J, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -U;
          }
        }
      });

  return true;
}

// ImuCost::ImuCost(const Trajectory& traj) : ptraj{&traj} {
//  preint.Compute(traj);
//}

// GicpAndImuCost::GicpAndImuCost(const SweepGrid& grid,
//                               const Trajectory& traj,
//                               int gsize)
//    : gicp_cost(grid, gsize), imu_cost(traj) {}

}  // namespace sv
