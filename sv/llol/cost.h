#pragma once

#include <ceres/tiny_solver_autodiff_function.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

struct GicpCost {
  using Scalar = double;
  static constexpr int kResidualDim = 3;

  GicpCost(const SweepGrid& grid, int gsize = 0);
  virtual ~GicpCost() noexcept = default;
  virtual int NumResiduals() const { return matches.size() * kResidualDim; }

  int gsize_{};
  const SweepGrid* const pgrid;
  std::vector<PointMatch> matches;
};

/// @brief Gicp with rigid transformation
struct GicpRigidCost final : public GicpCost {
  static constexpr int kNumParams = 6;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  enum Block { kR0, kP0 };

  // Pull in base constructor
  using GicpCost::GicpCost;

  template <typename T>
  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
    using Vec3CMap = Eigen::Map<const Vec3>;

    State(const T* const _x) : x{_x} {}
    auto r0() const { return Vec3CMap{x + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x + Block::kP0 * kBlockSize}; }

    const T* const x{nullptr};
  };

  bool operator()(const double* _x, double* _r, double* _J) const;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec3 = Eigen::Vector3<T>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    const State<T> es(_x);
    const SE3 eT{SO3::exp(es.r0()), es.p0()};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const auto c = match.px_g.x;
            const auto U = match.U.cast<double>().eval();
            const auto pt_p = match.mc_p.mean.cast<double>().eval();
            const auto pt_g = match.mc_g.mean.cast<double>().eval();
            const auto tf_p_g = pgrid->tfs.at(c).cast<double>();
            const auto pt_p_hat = tf_p_g * pt_g;

            Eigen::Map<Vec3> r(_r + kResidualDim * i);
            r = U * (pt_p - eT * pt_p_hat);
          }
        });

    return true;
  }
};

/// @brief Linear interpolation in translation error state
struct GicpLinearCost final : public GicpCost {
  static constexpr int kNumParams = 6;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  enum Block { kR0, kP0 };

  //  const Trajectory* ptraj;
  //  ImuPreintegration preint;
  using GicpCost::GicpCost;

  //  virtual int NumResiduals() const { return matches.size() * kResidualDim; }

  GicpLinearCost(const SweepGrid& grid,
                 const Trajectory& traj,
                 const ImuQueue& imuq,
                 int gsize = 0);

  template <typename T>
  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
    using Vec3CMap = Eigen::Map<const Vec3>;

    State(const T* const _x) : x{_x} {}
    auto r0() const { return Vec3CMap{x + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x + Block::kP0 * kBlockSize}; }

    const T* const x{nullptr};
  };

  bool operator()(const double* _x, double* _r, double* _J) const;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec3 = Eigen::Vector3<T>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    const State<T> es(_x);
    const auto eR = SO3::exp(es.r0());

    // Precompute interpolated error state fom lidar to odom
    std::vector<Vec3> eps(pgrid->cols());
    for (int i = 0; i < eps.size(); ++i) {
      const double s = (i + 0.5) / eps.size();
      eps.at(i) = s * es.p0();
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(0, matches.size(), gsize_),
        [&](const auto& blk) {
          for (int i = blk.begin(); i < blk.end(); ++i) {
            const auto& match = matches.at(i);
            const auto c = match.px_g.x;
            const auto U = match.U.cast<double>().eval();
            const auto pt_p = match.mc_p.mean.cast<double>().eval();
            const auto pt_g = match.mc_g.mean.cast<double>().eval();
            const auto tf_p_g = pgrid->tfs.at(c).cast<double>();
            const auto pt_p_hat = tf_p_g * pt_g;

            Eigen::Map<Vec3> r(_r + kResidualDim * i);
            r = U * (pt_p - (eR * pt_p_hat + eps.at(c)));
          }
        });

    // Imu preint cost
    //    const auto dt = preint.duration;
    //    const auto dt2 = dt * dt;
    //    const auto& g = ptraj->g_pano;
    //    const auto& st0 = ptraj->states.front();
    //    const auto& st1 = ptraj->states.back();

    //    const auto R0 = eR * st0.rot;
    //    const auto R1 = eR * st1.rot;
    //    const Vec3 p0 = es.p0() + eR * st0.pos;
    //    const Vec3 p1 = es.p1() + eR * st1.pos;

    //    const auto R0_inv = R0.inverse();
    //    const Vec3 alpha = R0_inv * (p1 - p0 - st0.vel * dt + 0.5 * g * dt2);
    //    const Vec3 beta = R0_inv * (st1.vel - st0.vel + g * dt);
    //    const auto gamma = R0_inv * R1;

    //    const int offset = matches.size() * kResidualDim;
    //    const auto& V = preint.U;
    //    Eigen::Map<Vec3> r_alpha(_r + offset);
    //    r_alpha = V.topLeftCorner<3, 3>() * (alpha - preint.alpha);
    //    Eigen::Map<Vec3> r_beta(_r + offset + 3);
    //    r_beta = V.block<3, 3>(3, 3) * (beta - preint.beta);
    //    Eigen::Map<Vec3> r_gamma(_r + offset + 6);
    //    r_gamma = V.block<3, 3>(6, 6) * (gamma *
    //    preint.gamma.inverse()).log();

    // Debug print
    //    if constexpr (std::is_same_v<T, double>) {
    //      LOG(INFO) << "P: \n" << preint.P.topLeftCorner<3, 3>();
    //      LOG(INFO) << "V: \n" << preint.U.topLeftCorner<3, 3>();
    //      LOG(INFO) << "alpha: " << r_alpha.transpose();
    //      LOG(INFO) << "beta: " << r_beta.transpose();
    //      LOG(INFO) << "gamma: " << r_gamma.transpose();
    //    }

    return true;
  }
};

// struct ImuCost {
//  static constexpr int kNumParams = 24;

//  explicit ImuCost(const Trajectory& traj);
//  int NumResiduals() const { return ImuPreintegration::kDim; }

//  template <typename T>
//  bool operator()(const T* const _x, T* _r) const {
//    using Vec15 = Eigen::Matrix<T, 15, 1>;  // Residuals
//    using Vec3 = Eigen::Vector3<T>;
//    using SO3 = Sophus::SO3<T>;
//    using ES = ErrorState<T>;

//    const auto dt = preint.duration;
//    const auto dt2 = dt * dt;
//    const auto& g = ptraj->gravity;
//    const auto& st0 = ptraj->states.front();
//    const auto& st1 = ptraj->states.back();

//    const ES es(_x);
//    const auto eR0 = SO3::exp(es.r0());
//    const auto eR1 = SO3::exp(es.r1());
//    const Vec3 p0 = es.p0() + eR0 * st0.pos;
//    const Vec3 p1 = es.p1() + eR1 * st1.pos;
//    const auto R0 = eR0 * st0.rot;
//    const auto R1 = eR1 * st1.rot;
//    const Vec3 v0 = es.v0() + st0.vel;
//    const Vec3 v1 = es.v1() + st1.vel;

//    const auto R0_inv = R0.inverse();
//    const Vec3 alpha = R0_inv * (p1 - p0 - v0 * dt + 0.5 * g * dt2);
//    const Vec3 beta = R0_inv * (v1 - v0 + g * dt);
//    const auto gamma = R0_inv * R1;

//    using IP = ImuPreintegration::Index;
//    Eigen::Map<Vec15> r(_r);
//    r.template segment<3>(IP::kAlpha) = alpha - preint.alpha;
//    r.template segment<3>(IP::kBeta) = beta - preint.beta;
//    r.template segment<3>(IP::kTheta) = (gamma *
//    preint.gamma.inverse()).log(); r.template segment<3>(IP::kBa) = es.ba();
//    r.template segment<3>(IP::kBw) = es.bw();
//    r = preint.U * r;

//    // Debug print
//    if constexpr (std::is_same_v<T, double>) {
//      std::cout << preint.F << std::endl;
//      std::cout << r.transpose() << std::endl;
//      std::cout << "norm: " << r.squaredNorm() << std::endl;
//    }

//    return true;
//  }

//  const Trajectory* const ptraj;
//  ImuPreintegration preint;
//};

// struct GicpAndImuCost {
//  static constexpr int kNumParams = 6;

//  GicpAndImuCost(const SweepGrid& grid, const Trajectory& traj, int gsize =
//  0);

//  int NumResiduals() const {
//    //    return gicp_cost.NumResiduals() + imu_cost.NumResiduals();
//    return gicp_cost.NumResiduals();
//  }

//  template <typename T>
//  bool operator()(const T* const _x, T* _r) const {
//    bool ok = true;
//    ok &= gicp_cost(_x, _r);

//    return ok;
//  }

//  GicpRigidCost gicp_cost;
//  //  ImuCost imu_cost;
//};

template <typename F>
using AdCost =
    ceres::TinySolverAutoDiffFunction<F, Eigen::Dynamic, F::kNumParams>;

}  // namespace sv
