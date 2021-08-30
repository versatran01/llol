#pragma once

#include <ceres/tiny_solver_autodiff_function.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

// Note: state x is grouped as (dr0, dt0, dr1, dt1 | dv0, dv1 | dba, dbw)
enum ErrorBlock { kR0, kP0, kR1, kP1, kV0, kV1, kBa, kBw };

template <typename T>
struct ErrorState {
  static constexpr int kBlockSize = 3;
  using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
  using Vec3CMap = Eigen::Map<const Vec3>;

  ErrorState(const T* const _x) : x{_x} {}
  auto r0() const { return Vec3CMap{x + ErrorBlock::kR0 * kBlockSize}; }
  auto p0() const { return Vec3CMap{x + ErrorBlock::kP0 * kBlockSize}; }
  auto r1() const { return Vec3CMap{x + ErrorBlock::kR1 * kBlockSize}; }
  auto p1() const { return Vec3CMap{x + ErrorBlock::kP1 * kBlockSize}; }
  auto v0() const { return Vec3CMap{x + ErrorBlock::kV0 * kBlockSize}; }
  auto v1() const { return Vec3CMap{x + ErrorBlock::kV1 * kBlockSize}; }
  auto ba() const { return Vec3CMap{x + ErrorBlock::kBa * kBlockSize}; }
  auto bw() const { return Vec3CMap{x + ErrorBlock::kBw * kBlockSize}; }

  const T* const x{nullptr};
};

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

  using Scalar = double;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };

  bool operator()(const double* _x, double* _r, double* _J) const {
    const ErrorState<double> es(_x);
    const Sophus::SE3d eT{Sophus::SO3d::exp(es.r0()), es.p0()};
    //    const auto eR = Sophus::SO3d::exp(es.r0());

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const int c = match.px_g.x;
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
              // dr / dtheta
              J.block<3, 3>(ri, 0) = U * Hat3(pt_p_hat);
              J.block<3, 3>(ri, 3) = -U;
            }
          }
        });
    return true;
  }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec3 = Eigen::Vector3<T>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;
    using ES = ErrorState<T>;

    // We now assume left multiply delta
    const ES es(_x);
    const SE3 eT{SO3::exp(es.r0()), es.p0()};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const int c = match.px_g.x;
            const auto U = match.U.cast<double>().eval();
            const auto pt_p = match.mc_p.mean.cast<double>().eval();
            const auto pt_g = match.mc_g.mean.cast<double>().eval();
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
    using Vec3 = Eigen::Vector3<T>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;
    using ES = ErrorState<T>;

    // We now assume left multiply delta
    const ES es(_x);
    //    std::vector<SE3> eTs(pgrid->cols());
    std::vector<Vec3> eps(pgrid->cols());
    //    const Vec3 der = es.r1() - es.r0();
    const Vec3 dep = es.p1() - es.p0();
    const SO3 eR = SO3::exp(es.r0());
    for (int i = 0; i < eps.size(); ++i) {
      const double s = (i + 0.5) / eps.size();
      eps.at(i) = es.p0() + s * dep;
    }

    // Precompute interpolated error state fom lidar to odom
    //    for (int i = 0; i < eTs.size(); ++i) {
    //      const double s = (i + 0.5) / eTs.size();
    //      eTs[i].so3() = SO3::exp(es.r0() + s * der);
    //      eTs[i].translation() = es.p0() + s * dep;
    //    }

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const auto c = match.px_g.x;
            const auto U = match.U.cast<double>();
            const auto pt_p = match.mc_p.mean.cast<double>().eval();
            const auto pt_g = match.mc_g.mean.cast<double>().eval();
            const auto tf_g = pgrid->tfs.at(c).cast<double>();

            Eigen::Map<Vec3> r(_r + kNumResiduals * i);
            // r = U * (pt_p - eTs.at(c) * (tf_g * pt_g));
            r = U * (pt_p - (eR * (tf_g * pt_g) + eps.at(c)));
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
    using Vec3 = Eigen::Vector3<T>;
    using SO3 = Sophus::SO3<T>;
    using ES = ErrorState<T>;

    const auto dt = preint.duration;
    const auto dt2 = dt * dt;
    const auto& g = ptraj->gravity;
    const auto& st0 = ptraj->states.front();
    const auto& st1 = ptraj->states.back();

    const ES es(_x);
    const auto eR0 = SO3::exp(es.r0());
    const auto eR1 = SO3::exp(es.r1());
    const Vec3 p0 = es.p0() + eR0 * st0.pos;
    const Vec3 p1 = es.p1() + eR1 * st1.pos;
    const auto R0 = eR0 * st0.rot;
    const auto R1 = eR1 * st1.rot;
    const Vec3 v0 = es.v0() + st0.vel;
    const Vec3 v1 = es.v1() + st1.vel;

    const auto R0_inv = R0.inverse();
    const Vec3 alpha = R0_inv * (p1 - p0 - v0 * dt + 0.5 * g * dt2);
    const Vec3 beta = R0_inv * (v1 - v0 + g * dt);
    const auto gamma = R0_inv * R1;

    using IP = ImuPreintegration::Index;
    Eigen::Map<Vec15> r(_r);
    r.template segment<3>(IP::kAlpha) = alpha - preint.alpha;
    r.template segment<3>(IP::kBeta) = beta - preint.beta;
    r.template segment<3>(IP::kTheta) = (gamma * preint.gamma.inverse()).log();
    r.template segment<3>(IP::kBa) = es.ba();
    r.template segment<3>(IP::kBw) = es.bw();
    r = preint.U * r;

    // Debug print
    if constexpr (std::is_same_v<T, double>) {
      std::cout << preint.F << std::endl;
      std::cout << r.transpose() << std::endl;
      std::cout << "norm: " << r.squaredNorm() << std::endl;
    }

    return true;
  }

  const ImuTrajectory* const ptraj;
  ImuPreintegration preint;
};

struct GicpAndImuCost {
  static constexpr int kNumParams = 6;

  GicpAndImuCost(const SweepGrid& grid,
                 const ImuTrajectory& traj,
                 int gsize = 0);

  int NumResiduals() const {
    //    return gicp_cost.NumResiduals() + imu_cost.NumResiduals();
    return gicp_cost.NumResiduals();
  }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    bool ok = true;
    //    ok &= gicp_cost(_x, _r);

    // Just evaluate but dont do anything
    //    std::vector<T> s(imu_cost.NumResiduals());
    //    ok &= imu_cost(_x, s.data());

    //    const auto& pi = imu_cost.preint;

    //    LOG(INFO) << "dura: " << pi.duration;
    //    LOG(INFO) << "n: " << pi.n;
    //    LOG(INFO) << "alpha: " << pi.alpha.transpose();
    //    LOG(INFO) << "beta: " << pi.beta.transpose();
    //    LOG(INFO) << "gamma: " <<
    //    pi.gamma.unit_quaternion().coeffs().transpose();

    return ok;
  }

  GicpRigidCost gicp_cost;
  ImuCost imu_cost;
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kNumParams>;

}  // namespace sv
