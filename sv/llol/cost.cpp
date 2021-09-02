#include "sv/llol/cost.h"

namespace sv {

GicpCost::GicpCost(const SweepGrid& grid, int gsize) : pgrid{&grid} {
  // we don't want to use grainsize of 1 or 2, because each residual is 3
  // doubles which is 3 * 8 = 24. However a cache line is typically 64 bytes so
  // we need at least 3 residuals (3 * 3 * 8 = 72 bytes) to fill one cache line
  gsize_ = gsize <= 0 ? matches.size() : gsize + 2;
  matches.reserve(grid.total() / 2);
}

void GicpCost::Update() {
  // Collect all good matches

  matches.clear();
  const auto& grid = *pgrid;
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }
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
          const auto tf_p_g = pgrid->TfAt(c).cast<double>();
          const auto pt_p_hat = tf_p_g * pt_g;

          const int ri = kResidualDim * i;
          Eigen::Map<Vector3d> r(_r + ri);
          r = U * (pt_p - eT * pt_p_hat);

          if (_J) {
            Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -U;
          }
        }
      });

  return true;
}

bool GicpLinearCost::operator()(const double* _x,
                                double* _r,
                                double* _J) const {
  const State<double> es(_x);
  const auto eR = Sophus::SO3d::exp(es.r0());

  tbb::parallel_for(
      tbb::blocked_range<int>(0, matches.size(), gsize_), [&](const auto& blk) {
        for (int i = blk.begin(); i < blk.end(); ++i) {
          const auto& match = matches.at(i);
          const auto c = match.px_g.x;
          const auto U = match.U.cast<double>().eval();
          const auto pt_p = match.mc_p.mean.cast<double>().eval();
          const auto pt_g = match.mc_g.mean.cast<double>().eval();
          const auto tf_p_g = pgrid->TfAt(c).cast<double>();
          const auto pt_p_hat = tf_p_g * pt_g;
          const double s = (c + 0.5) / pgrid->cols();

          const int ri = kResidualDim * i;
          Eigen::Map<Vector3d> r(_r + ri);
          r = U * (pt_p - (eR * pt_p_hat + s * es.p0()));

          if (_J) {
            Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -s * U;
          }
        }
      });

  return true;
}

// GicpLinearCost::GicpLinearCost(const SweepGrid& grid,
//                               const Trajectory& traj,
//                               const ImuQueue& imuq,
//                               int gsize)
//    : GicpCost{grid, gsize}, ptraj{&traj} {
//  preint.Compute(imuq, traj.states.front().time, traj.states.back().time);
//}

}  // namespace sv
