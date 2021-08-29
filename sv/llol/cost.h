#pragma once

#include <ceres/tiny_solver_autodiff_function.h>
#include <tbb/parallel_for.h>

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

struct GicpCost {
  static constexpr int kNumResiduals = 3;

  GicpCost(const SweepGrid& grid, int gsize = 0);
  virtual ~GicpCost() noexcept = default;

  virtual int NumResiduals() const { return matches.size() * kNumResiduals; }

  int gsize{};
  const SweepGrid* const pgrid;
  std::vector<PointMatch> matches;
};

struct GicpRigidCost final : public GicpCost {
  static constexpr int kNumParams = 6;

  using GicpCost::GicpCost;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    // We now assume left multiply delta
    //    Eigen::Map<const Vec6> x(_x);
    Eigen::Map<const Vec3> dr(_x);
    Eigen::Map<const Vec3> dt(_x + 3);
    //    const SE3 dT{SO3::exp(x.template head<3>()), x.template tail<3>()};
    const SO3 dR = SO3::exp(dr);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize * 2),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const int c = match.px_g.x;
            const Eigen::Matrix3d U = match.U.cast<double>();
            const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
            const Eigen::Vector3d pt_g = match.mc_g.mean.cast<double>();
            const auto tf_g = pgrid->tfs.at(c).cast<double>();

            Eigen::Map<Vec3> r(_r + kNumResiduals * i);
            // r = U * (pt_p - dT * tf_g * pt_g);
            // SE3 tf;
            // tf.so3() = dR * tf_g.so3();
            // tf.translation() = dt + tf_g.translation();
            const auto R = dR * tf_g.so3();
            const Vec3 t = dt + tf_g.translation();
            r = U * (pt_p - (R * pt_g + t));
          }
        });

    return true;
  }
};

// Note: state x is grouped as (dr0, dt0, dr1, dt1 | v0, v1 | ba, bw)
struct ImuPreintegrationCost {
  enum Block { R0, P0, R1, P1, V0, V1, BA, BW };
  static constexpr int kNumResiduals = 15;

  ImuPreintegrationCost(const ImuTrajectory& traj);

  // alpha, beta, gamma, ba, bw
  int NumResiduals() const { return kNumResiduals; }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec15 = Eigen::Matrix<T, 15, 1>;  // Residuals
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SO3 = Sophus::SO3<T>;

    const double dt = preint.duration;
    const double dt2 = dt * dt;
    const auto& g = ptraj->gravity;
    const auto& st0 = ptraj->states.front();
    const auto& st1 = ptraj->states.back();

    Eigen::Map<const Vec3> dr0(_x + Block::R0 * 3);
    Eigen::Map<const Vec3> dp0(_x + Block::P0 * 3);
    Eigen::Map<const Vec3> dr1(_x + Block::R1 * 3);
    Eigen::Map<const Vec3> dp1(_x + Block::P1 * 3);
    Eigen::Map<const Vec3> dv0(_x + Block::V0 * 3);
    Eigen::Map<const Vec3> dv1(_x + Block::V1 * 3);
    Eigen::Map<const Vec3> dba(_x + Block::BA * 3);
    Eigen::Map<const Vec3> dbg(_x + Block::BW * 3);

    const Vec3 p0 = dp0 + st0.pos;
    const Vec3 p1 = dp1 + st1.pos;
    const Vec3 v0 = dv0 + st0.vel;
    const Vec3 v1 = dv1 + st1.vel;
    const auto R0 = SO3::exp(dr0) * st0.rot;
    const auto R1 = SO3::exp(dr1) * st1.rot;

    const auto R0_inv = R0.inverse();
    const Vec3 alpha = R0_inv * (p1 - p0 - v0 * dt + 0.5 * g * dt2);
    const Vec3 beta = R0_inv * (v1 - v0 + g * dt);
    const auto gamma = R0_inv * R1;

    using IP = ImuPreintegration::Index;
    Eigen::Map<Vec15> r(_r);
    r.segment<3>(IP::ALPHA) = alpha - preint.alpha;
    r.segment<3>(IP::BETA) = beta - preint.beta;
    r.segment<3>(IP::THETA) = (gamma * preint.gamma.inverse()).log();
    r.segment<3>(IP::BA) = dba;
    r.segment<3>(IP::BW) = dbg;
    r = preint.U * r;

    // Debug print
    if constexpr (std::is_same_v<T, double>) {
      std::cout << r.transpose() << std::endl;
    }

    return true;
  }

  const ImuTrajectory* const ptraj;
  ImuPreintegration preint;
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kNumParams>;

}  // namespace sv
