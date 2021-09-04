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
  const int n_imus =
      preint.Compute(imuq, traj.states.front().time, traj.states.back().time);
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
          const double s = (c + 0.5) / pgrid->cols();

          const int ri = kResidualDim * i;
          Eigen::Map<Vector3d> r(_r + ri);
          r = U * (pt_p - (eR * pt_p_hat + s * ep));

          if (_J) {
            Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -s * U;
          }
        }
      });

  const int gicp_residuals = matches.size() * kResidualDim;

  const auto dt = preint.duration;
  const auto dt2 = dt * dt;
  const auto& g = ptraj->g_pano;
  const auto& st0 = ptraj->states.front();
  const auto& st1 = ptraj->states.back();

  const Vector3d p0 = eR * st0.pos;
  const Vector3d p1 = eR * st1.pos + ep;
  const auto R0 = eR * st0.rot;

  const auto R0_t = R0.inverse();
  const Vector3d dp = st0.vel * dt - 0.5 * g * dt2;
  const Vector3d alpha = R0_t * (p1 - p0 - dp);

  Eigen::Map<Vector3d> r_alpha(_r + gicp_residuals);
  const Matrix3d Ua = preint.U.topLeftCorner<3, 3>() * imu_scale;
  r_alpha = Ua * (alpha - preint.alpha);

  if (_J) {
    const auto R0_t_mat = R0_t.matrix();
    Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
    J.block<3, 3>(gicp_residuals, Block::kR0 * 3) =
        Ua * R0_t_mat * Hat3((ep - dp).eval());
    J.block<3, 3>(gicp_residuals, Block::kP0 * 3) = Ua * R0_t_mat;
  }

  return true;
}

bool GicpLinearCost2::operator()(const double* _x,
                                 double* _r,
                                 double* _J) const {
  const State<double> es(_x);
  const auto eR = Sophus::SO3d::exp(es.r0());
  const Vector3d dep = es.p1() - es.p0();

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
          r = U * (pt_p - (eR * pt_p_hat + es.p0() + s * dep));

          if (_J) {
            Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
            J.block<3, 3>(ri, Block::kR0 * 3) = U * Hat3(pt_p_hat);
            J.block<3, 3>(ri, Block::kP0 * 3) = -(1.0 - s) * U;
            J.block<3, 3>(ri, Block::kP1 * 3) = -s * U;
          }
        }
      });

  const int gicp_residuals = matches.size() * kResidualDim;

  const auto dt = preint.duration;
  const auto dt2 = dt * dt;
  const auto& g = ptraj->g_pano;
  const auto& st0 = ptraj->states.front();
  const auto& st1 = ptraj->states.back();

  const Vector3d p0 = eR * st0.pos + es.p0();
  const Vector3d p1 = eR * st1.pos + es.p1();
  const auto R0 = eR * st0.rot;

  const auto R0_t = R0.inverse();
  const Vector3d dp = st0.vel * dt - 0.5 * g * dt2;
  const Vector3d alpha = R0_t * (p1 - p0 - dp);

  Eigen::Map<Vector3d> r_alpha(_r + gicp_residuals);
  const Matrix3d Ua = preint.U.topLeftCorner<3, 3>() * imu_scale;
  r_alpha = Ua * (alpha - preint.alpha);

  if (_J) {
    const auto R0_t_mat = R0_t.matrix();
    Eigen::Map<MatrixXd> J(_J, NumResiduals(), kNumParams);
    J.block<3, 3>(gicp_residuals, Block::kR0 * 3) =
        Ua * R0_t_mat * Hat3((dep - dp).eval());
    J.block<3, 3>(gicp_residuals, Block::kP0 * 3) = -Ua * R0_t_mat;
    J.block<3, 3>(gicp_residuals, Block::kP1 * 3) = Ua * R0_t_mat;
  }

  return true;
}

}  // namespace sv
