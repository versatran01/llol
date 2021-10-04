#include "sv/llol/cost.h"

#include <glog/logging.h>
#include <tbb/parallel_for.h>

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using MatrixXd = Eigen::MatrixXd;
using RowMatXd = NllsSolver::RowMat;
using Vector9d = Eigen::Matrix<double, 9, 1>;
using Matrix9d = Eigen::Matrix<double, 9, 9>;

GicpCost::GicpCost(int num_params, double w_imu, int gsize)
    : imu_weight{w_imu} {
  // we don't want to use grainsize of 1 or 2, because each residual is 3
  // doubles which is 3 * 8 = 24. However a cache line is typically 64 bytes so
  // we need at least 3 residuals (3 * 3 * 8 = 72 bytes) to fill one cache line
  gsize_ = gsize <= 0 ? matches.size() : gsize + 2;
  error.resize(num_params);
  error.setZero();
}

int GicpCost::NumResiduals() const {
  return matches.size() * kResidualDim + (ptraj ? 9 : 0);
}

void GicpCost::ResetError() { error.setZero(); }

void GicpCost::UpdatePreint(const Trajectory& traj, const ImuQueue& imuq) {
  ptraj = &traj;
  preint.Reset();
  preint.Compute(imuq, traj.front().time, traj.back().time);
  // make sure preint duration is the same as traj duration
  if (preint.Ok()) {
    CHECK_EQ(preint.duration, ptraj->duration());
  }
}

void GicpCost::UpdateMatches(const SweepGrid& grid) {
  // Collect all good matches
  pgrid = &grid;

  matches.clear();
  matches.reserve(grid.total() / 2);
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (match.Ok()) matches.push_back(match);
    }
  }

  // Precompute pt_p_hat
  // This seems to make stuff slower
  pts_p_hat.resize(matches.size());
  tbb::parallel_for(
      tbb::blocked_range<int>(0, matches.size(), gsize_), [&](const auto& blk) {
        for (int i = blk.begin(); i < blk.end(); ++i) {
          const auto& match = matches.at(i);
          const auto c = match.px_g.x;
          pts_p_hat.at(i) = (pgrid->TfAt(c) * match.mc_g.mean).cast<double>();
        }
      });
}

bool GicpCostRigid::Compute(const double* px, double* pr, double* pJ) const {
  const State es(px);
  const SO3d eR = SO3d::exp(es.r0());
  const SE3d eT{eR, es.p0()};

  tbb::parallel_for(
      tbb::blocked_range<int>(0, matches.size(), gsize_), [&](const auto& blk) {
        for (int i = blk.begin(); i < blk.end(); ++i) {
          const auto& match = matches.at(i);
          const Vector3d pt_p = match.mc_p.mean.cast<double>();
          const auto& pt_p_hat = pts_p_hat.at(i);

          const int ri = kResidualDim * i;
          Eigen::Map<Vector3d> r(pr + ri);

          auto U = match.U.cast<double>().eval();
          r = U * (pt_p - eT * pt_p_hat);

          // Do a simple gating test and downweight outliers
          // https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
          double w_icp = match.scale;
          // const auto r2 = r.squaredNorm();
          // t-distribution weight from eq 22 in
          // Robust Odometry Estimation for RGB-D Cameras
          // w *= 4.0 / (3.0 + r2);

          r *= w_icp;  // scale residual
          if (pJ != nullptr) {
            Eigen::Map<RowMatXd> J(pJ, NumResiduals(), NumParameters());
            U *= w_icp;  // scale jacobian
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = U * (-1.0);
          }
        }
      });

  if (ptraj == nullptr) return true;

  double w_imu = imu_weight;
  // If preintegration failed then we just set weight to 0, this will cause all
  // the imu residual and jacobian to be 0
  if (!preint.Ok()) w_imu = 0.0;

  const auto dt = preint.duration;
  const auto dt2 = dt * dt;
  const auto& g_p = ptraj->g_pano;
  const auto& st0 = ptraj->front();
  const auto& st1 = ptraj->back();

  const auto& R0 = st0.rot;
  const auto R0_t = R0.inverse();

  // imu preint residual
  const int offset = matches.size() * kResidualDim;
  Eigen::Map<Vector9d> r_imu(pr + offset);
  // alpha residual
  // r_alpha = R0^T (p1 - p0 - v0 * dt + 0.5 * g * dt^2) - alpha
  //         = R0^T (p1 - p0 + dp) - alpha
  //      dp = 0.5 * g * dt^2 - v0 * dt
  const auto& p0 = st0.pos;
  const Vector3d p1 = eR * st1.pos + es.p0();
  const Vector3d dp = 0.5 * g_p * dt2 - st0.vel * dt;
  r_imu.segment<3>(0) = R0_t * (p1 - p0 + dp) - preint.alpha;

  // beta residual
  // r_beta = R0^T (v1 + g * dt - v0) - beta
  //        = R0^T (ep / dt + dv) - beta
  //     dv = g * dt
  const Vector3d dv = g_p * dt;
  // r_imu.segment<3>(3) = R0_t * (es.p0() / dt + dv) - preint.beta;
  r_imu.segment<3>(3).setZero();

  // gamma residual
  // r_gamma = R0' * R1 * gamma'
  const auto R1 = eR * st1.rot;
  r_imu.segment<3>(6) = (R0_t * R1 * preint.gamma.inverse()).log();

  // Premultiply by U
  const Matrix9d U = preint.U.topLeftCorner<9, 9>() * w_imu;
  r_imu.applyOnTheLeft(U);

  if (pJ != nullptr) {
    const auto R0_t_mat = R0_t.matrix();
    Eigen::Map<RowMatXd> J(pJ, NumResiduals(), NumParameters());
    // alpha jacobian
    const Vector3d q = dp - p0;
    J.block<3, 3>(offset, Block::kR0 * 3) = R0_t_mat * Hat3(q);
    J.block<3, 3>(offset, Block::kP0 * 3) = R0_t_mat;

    // beta jacobian
    // J.block<3, 3>(offset + 3, Block::kR0 * 3) = R0_t_mat * Hat3(dv);
    // J.block<3, 3>(offset + 3, Block::kP0 * 3) = R0_t_mat / dt;
    J.block<3, 6>(offset + 3, 0).setZero();

    // gamma jacobian
    J.block<3, 3>(offset + 6, Block::kR0 * 3) = R0_t_mat;
    J.block<3, 3>(offset + 6, Block::kP0 * 3).setZero();

    J.block<9, 6>(offset, 0).applyOnTheLeft(U);
  }

  return true;
}

void GicpCostRigid::UpdateTraj(Trajectory& traj) const {
  const auto dt = traj.duration();
  const State es(error.data());
  const auto eR = SO3d::exp(es.r0());

  // Only update first state, the rest will be done in repredict
  auto& st = traj.states.front();
  st.rot = eR * st.rot;
  st.pos = eR * st.pos + es.p0();
  st.vel = eR * st.vel + es.p0() / dt;
}

}  // namespace sv
