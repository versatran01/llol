#include "sv/llol/cost.h"

#include <glog/logging.h>
#include <tbb/parallel_for.h>

namespace sv {

GicpCost::GicpCost(int gsize) {
  // we don't want to use grainsize of 1 or 2, because each residual is 3
  // doubles which is 3 * 8 = 24. However a cache line is typically 64 bytes so
  // we need at least 3 residuals (3 * 3 * 8 = 72 bytes) to fill one cache line
  gsize_ = gsize <= 0 ? matches.size() : gsize + 2;
}

int GicpCost::NumResiduals() const {
  return matches.size() * kResidualDim + (ptraj ? 3 : 0);
}

void GicpCost::ResetError() {
  error.resize(NumParameters());
  error.setZero();
}

void GicpCost::UpdateMatches(const SweepGrid& grid) {
  // Collect all good matches
  pgrid = &grid;

  matches.clear();
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto& match = grid.MatchAt({c, r});
      if (!match.Ok()) continue;
      matches.push_back(match);
    }
  }
}

void GicpCost::UpdatePreint(const Trajectory& traj, const ImuQueue& imuq) {
  ptraj = &traj;
  preint.Reset();
  preint.Compute(imuq, traj.front().time, traj.back().time);
}

bool GicpRigidCost::operator()(const double* x_ptr,
                               double* r_ptr,
                               double* J_ptr) const {
  CHECK_EQ(imu_weight, 0);
  CHECK(ptraj == nullptr);

  const State es(x_ptr);
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
          Eigen::Map<Vector3d> r(r_ptr + ri);
          r = U * (pt_p - eT * pt_p_hat);

          if (J_ptr) {
            Eigen::Map<MatrixXd> J(J_ptr, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -U;
          }
        }
      });

  return true;
}

void GicpRigidCost::UpdateTraj(Trajectory& traj) const {
  const auto st1_old = traj.back();  // get a copy of the initial state

  const State es(error.data());
  const auto eR = Sophus::SO3d::exp(es.r0());
  for (auto& st : traj.states) {
    st.rot = eR * st.rot;
    st.pos = eR * st.pos + es.p0();
  }

  // only update last velocity because we need it for next round of prediction
  auto& st1 = traj.states.back();
  st1.vel += (st1.pos - st1_old.pos) / traj.duration();
}

bool GicpLinearCost::operator()(const double* x_ptr,
                                double* r_ptr,
                                double* J_ptr) const {
  const State es(x_ptr);
  const auto eR = Sophus::SO3d::exp(es.r0());
  const Vector3d ep = es.p0();

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
          // +0.5 is because we assume point is at cell center
          const double s = (c + 0.5) / pgrid->cols();

          const int ri = kResidualDim * i;
          Eigen::Map<Vector3d> r(r_ptr + ri);
          r = U * (pt_p - (eR * pt_p_hat + s * ep));

          if (J_ptr) {
            Eigen::Map<MatrixXd> J(J_ptr, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -s * U;
          }
        }
      });

  if (ptraj == nullptr) return true;

  // We don't need to zero out residual and jacobian because they are
  // initialized to zero and assuming imu_weight doesn't change, they will stay
  // zero so won't affect optimization
  if (imu_weight == 0) return true;

  const auto dt = preint.duration;
  const auto dt2 = dt * dt;
  const auto& g = ptraj->g_pano;
  const auto& st0 = ptraj->front();
  const auto& st1 = ptraj->back();

  const Vector3d p0 = eR * st0.pos;
  const Vector3d p1 = eR * st1.pos + ep;
  const auto R0 = eR * st0.rot;

  const auto R0_t = R0.inverse();
  const Vector3d dp = st0.vel * dt - 0.5 * g * dt2;
  const Vector3d alpha = R0_t * (p1 - p0 - dp);

  const int gicp_residuals = matches.size() * kResidualDim;
  Eigen::Map<Vector3d> r_alpha(r_ptr + gicp_residuals);
  const Matrix3d Ua = preint.U.topLeftCorner<3, 3>() * imu_weight;
  // alpha = R0^T (p1 - p0 - v0 * dt + 0.5 * g * dt2)
  r_alpha = Ua * (alpha - preint.alpha);

  if (J_ptr) {
    const auto R0_t_mat = R0_t.matrix();
    Eigen::Map<MatrixXd> J(J_ptr, NumResiduals(), kNumParams);
    J.block<3, 3>(gicp_residuals, Block::kR0 * 3) =
        Ua * R0_t_mat * Hat3((ep - dp).eval());
    J.block<3, 3>(gicp_residuals, Block::kP0 * 3) = Ua * R0_t_mat;
  }

  return true;
}

void GicpLinearCost::UpdateTraj(Trajectory& traj) const {
  const State es(error.data());
  const auto eR = Sophus::SO3d::exp(es.r0());

  MeanVar3d vel{};

  for (int i = 0; i < traj.size(); ++i) {
    auto& st_i = traj.At(i);
    const double s = i / (traj.size() - 1.0);
    st_i.rot = eR * st_i.rot;
    st_i.pos = eR * st_i.pos + s * es.p0();

    if (i > 1) {
      auto& st_im1 = traj.At(i - 1);
      st_im1.vel = (st_i.pos - st_im1.pos) / (st_i.time - st_im1.time);
      vel.Add(st_im1.vel);
    }
  }

  // Last vel is the average vel
  traj.states.back().vel = vel.mean;
}

}  // namespace sv
