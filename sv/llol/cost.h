#pragma once

#include <ceres/tiny_solver_autodiff_function.h>
#include <tbb/parallel_for.h>

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

// Note: state x is grouped as (dr0, dt0, dr1, dt1 | dv0, dv1 | dba, dbw)
enum ErrorBlock { kR0, kP0, kR1, kP1, kV0, kV1, kBa, kBw };

struct GicpCost {
  static constexpr int kNumResiduals = 3;
  GicpCost(const SweepGrid& grid, int gsize = 0);
  virtual ~GicpCost() noexcept = default;
  virtual int NumResiduals() const { return matches.size() * kNumResiduals; }

  int gsize_{};
  const SweepGrid* const pgrid;
  std::vector<PointMatch> matches;
};

/// @brief Assume rigit transformation
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
    Eigen::Map<const Vec3> er(_x + ErrorBlock::kR0 * 3);
    Eigen::Map<const Vec3> ep(_x + ErrorBlock::kP0 * 3);
    const SE3 eT{SO3::exp(er), ep};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const int c = match.px_g.x;
            const Eigen::Matrix3d U = match.U.cast<double>();
            const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
            const Eigen::Vector3d pt_g = match.mc_g.mean.cast<double>();
            const auto tf_g = pgrid->tfs.at(c).cast<double>();

            Eigen::Map<Vec3> r(_r + kNumResiduals * i);
            r = U * (pt_p - eT * (tf_g * pt_g));
          }
        });

    return true;
  }
};

struct GicpLinearCost final : public GicpCost {
  static constexpr int kNumParams = 12;
  using GicpCost::GicpCost;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    // We now assume left multiply delta
    Eigen::Map<const Vec3> er0(_x + ErrorBlock::kR0 * 3);
    Eigen::Map<const Vec3> ep0(_x + ErrorBlock::kP0 * 3);
    Eigen::Map<const Vec3> er1(_x + ErrorBlock::kR1 * 3);
    Eigen::Map<const Vec3> ep1(_x + ErrorBlock::kP1 * 3);

    std::vector<SE3> eTs(pgrid->cols());
    const Vec3 der = er1 - er0;
    const Vec3 dep = ep1 - ep0;
    // Precompute interpolated error state fom lidar to odom
    for (int i = 0; i < eTs.size(); ++i) {
      const double s = (i + 0.5) / eTs.size();
      eTs[i].so3() = SO3::exp(er0 + s * der);
      eTs[i].translation() = ep0 + s * dep;
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const int c = match.px_g.x;
            const Eigen::Matrix3d U = match.U.cast<double>();
            const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
            const Eigen::Vector3d pt_g = match.mc_g.mean.cast<double>();
            const auto tf_g = pgrid->tfs.at(c).cast<double>();

            Eigen::Map<Vec3> r(_r + kNumResiduals * i);
            r = U * (pt_p - eTs.at(c) * (tf_g * pt_g));
          }
        });

    return true;
  }
};

struct ImuCost {
  static constexpr int kNumParams = 24;

  explicit ImuCost(const ImuTrajectory& traj);
  int NumResiduals() const { return ImuPreintegration::kDim; }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec15 = Eigen::Matrix<T, 15, 1>;  // Residuals
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SO3 = Sophus::SO3<T>;

    const auto dt = preint.duration;
    const auto dt2 = dt * dt;
    const auto& g = ptraj->gravity;
    const auto& st0 = ptraj->states.front();
    const auto& st1 = ptraj->states.back();

    Eigen::Map<const Vec3> er0(_x + ErrorBlock::kR0 * 3);
    Eigen::Map<const Vec3> ep0(_x + ErrorBlock::kP0 * 3);
    Eigen::Map<const Vec3> er1(_x + ErrorBlock::kR1 * 3);
    Eigen::Map<const Vec3> ep1(_x + ErrorBlock::kP1 * 3);
    Eigen::Map<const Vec3> ev0(_x + ErrorBlock::kV0 * 3);
    Eigen::Map<const Vec3> ev1(_x + ErrorBlock::kV1 * 3);
    Eigen::Map<const Vec3> eba(_x + ErrorBlock::kBa * 3);
    Eigen::Map<const Vec3> ebg(_x + ErrorBlock::kBw * 3);

    const auto eR0 = SO3::exp(er0);
    const auto eR1 = SO3::exp(er1);
    const Vec3 p0 = ep0 + eR0 * st0.pos;
    const Vec3 p1 = ep1 + eR1 * st1.pos;
    const auto R0 = eR0 * st0.rot;
    const auto R1 = eR1 * st1.rot;
    const Vec3 v0 = ev0 + st0.vel;
    const Vec3 v1 = ev1 + st1.vel;

    const auto R0_inv = R0.inverse();
    const Vec3 alpha = R0_inv * (p1 - p0 - v0 * dt + 0.5 * g * dt2);
    const Vec3 beta = R0_inv * (v1 - v0 + g * dt);
    const auto gamma = R0_inv * R1;

    using IP = ImuPreintegration::Index;
    Eigen::Map<Vec15> r(_r);
    r.template segment<3>(IP::kAlpha) = alpha - preint.alpha;
    r.template segment<3>(IP::kBeta) = beta - preint.beta;
    r.template segment<3>(IP::kTheta) = (gamma * preint.gamma.inverse()).log();
    r.template segment<3>(IP::kBa) = eba;
    r.template segment<3>(IP::kBw) = ebg;
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

struct GicpAndImuCost {
  static constexpr int kNumParams = ImuCost::kNumParams;

  GicpAndImuCost(const SweepGrid& grid,
                 const ImuTrajectory& traj,
                 int gsize = 0);

  int NumResiduals() const {
    return gicp_cost.NumResiduals() + imu_cost.NumResiduals();
  }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    gicp_cost(_x, _r);
    imu_cost(_x, _r + gicp_cost.NumResiduals());
    return true;
  }

  GicpLinearCost gicp_cost;
  ImuCost imu_cost;
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kNumParams>;

}  // namespace sv
