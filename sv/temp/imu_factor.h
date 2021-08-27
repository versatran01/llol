#pragma once

#include <ceres/autodiff_cost_function.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace sv {

using namespace Eigen;
using namespace Sophus;

/// Imu preintegration
/// Produce delta = [alpha, beta, gamma], Covariance P and dt
struct ImuPreintegration {
  static constexpr int num_parameters = SE3d::num_parameters + 3;  // 10
  static constexpr int DoF = SE3d::DoF + 3;                        // 9

  using Delta = VectorNd<num_parameters>;                   // 10x1
  using Fmatrix = MatrixNd<ImuState::DoF>;                  // 15x15
  using Gmatrix = MatrixMNd<ImuState::DoF, ImuNoise::DoF>;  // 15x12
  using Covariance = MatrixNd<ImuState::DoF>;               // 15x15

  using Bias = ImuState::Bias;
  using Bias2 = ImuState::Bias2;
  using AccGyr = ImuData::AccGyr;

  /// [alpha, beta, gamma]
  enum Index { ALPHA = 0, BETA = 3, GAMMA = 6, BA = 9, BW = 12 };

  /// Constructor
  ImuPreintegration(const ImuData& imu, const Bias2& baw, ImuNoise* noise);
  ImuPreintegration(const Bias2& baw, ImuNoise* noise);

  std::string repr() const;

  /// total dt
  real_t dt() const;
  real_t t_k() const { return imus_.front().t; }
  real_t t_k1() const { return imus_.back().t; }
  size_t size() const { return imus_.size(); }
  bool empty() const { return imus_.empty(); }

  /// Getters
  const Covariance& P() const { return P_; }
  const Vector3d& alpha() const { return alpha_k_i_; }
  const Vector3d& beta() const { return beta_k_i_; }
  const SO3d& gamma() const { return gamma_k_i_; }

  /// Add imu data to preintegration
  void AddImuData(const ImuData& meas);
  //  void Finish(real_t tj);  // must be called before using this

  /// Integrate using two imu data
  void IntegrateEuler(const ImuData& imu_i, const ImuData& imu_j);
  void IntegrateEuler(real_t dt,
                      const ImuData::AccGyr& aw_i,
                      const ImuData::AccGyr& aw_j);

  // private:
  void Reset_();

  Bias ba_k_, bw_k_;    // bias at t_k, assume constant
  Vector3d alpha_k_i_;  // alpha^i_j
  Vector3d beta_k_i_;   // beta^i_j
  SO3d gamma_k_i_;      // gamma^i_j
  Covariance P_;        // 15x15
  Gmatrix G_;           // 15x12

  std::vector<ImuData> imus_;
  ImuNoise* noise_{nullptr};
};

ImuPreintegration::ImuPreintegration(const ImuPreintegration::Bias2& baw,
                                     ImuNoise* noise)
    : ba_k_(baw.head<3>()), bw_k_(baw.tail<3>()), noise_(noise) {
  CHECK_NOTNULL(noise_);
  Reset_();
}

ImuPreintegration::ImuPreintegration(const ImuData& imu,
                                     const ImuPreintegration::Bias2& baw,
                                     ImuNoise* noise)
    : ImuPreintegration(baw, noise) {
  imus_.push_back(imu);
}

void ImuPreintegration::Reset_() {
  alpha_k_i_.setZero();
  beta_k_i_.setZero();
  gamma_k_i_ = SO3();
  P_.setZero();
  G_.setZero();
  G_.bottomRows<ImuNoise::DoF>() = ImuNoise::Covariance::Identity();
  imus_.clear();
}

std::string ImuPreintegration::repr() const {
  return fmt::format("ImuPreintegration(t=[{}, {}), dt={}, size={})",
                     t_k(),
                     t_k1(),
                     dt(),
                     size());
}

real_t ImuPreintegration::dt() const {
  CHECK_GT(t_k1(), t_k());
  return t_k1() - t_k();
}

void ImuPreintegration::AddImuData(const ImuData& imu) {
  if (!empty()) IntegrateEuler(imus_.back(), imu);
  imus_.push_back(imu);
}

void ImuPreintegration::IntegrateEuler(const ImuData& imu_i,
                                       const ImuData& imu_j) {
  IntegrateEuler(imu_j.t - imu_i.t, imu_i.aw, imu_j.aw);
}

void ImuPreintegration::IntegrateEuler(real_t dt,
                                       const AccGyr& aw_i,
                                       const AccGyr& aw_j) {
  // Do nothing if dt is 0, this should only happen in simulation
  if (dt == 0) return;
  // Precondition
  CHECK_GT(dt, 0);
  LOG_IF(WARNING, dt > 0.02) << "Imu rate too low: " << dt;

  const auto dt2 = dt * dt;
  // just average angular velocity
  const Vector3d w_b = (aw_i.tail<3>() + aw_j.tail<3>()) / 2.0 - bw_k_;
  // We can now compute the new gamma which will be used to rotate a1
  // r^k_j = r_i^k * R{(w - b_w) * dt}
  const SO3 gamma_i_j = SO3::exp(w_b * dt);
  const SO3 gamma_k_j = gamma_k_i_ * gamma_i_j;  // the new gamma

  // Use the gamma i and j to compute acceleration in k frame
  const Vector3d a_i = aw_i.head<3>() - ba_k_;
  const Vector3d a_k_i = gamma_k_i_ * a_i;
  const Vector3d a_k_j = gamma_k_j * (aw_j.head<3>() - ba_k_);
  // Average acceleration in k frame
  const Vector3d a_k = (a_k_i + a_k_j) / 2.0;  // average acc

  // a_j^k = a_i^k + b_i^k * dt + 0.5 * R_i^k * (a - b_a) * dt^2
  const Vector3d d_alpha = beta_k_i_ * dt + 0.5 * a_k * dt2;
  // b_j^k = b_i^k + R_i^k * (a - b_a) * dt
  const Vector3d d_beta = a_k * dt;

  // Propagate covariance
  const Matrix3d I_3x3 = Matrix3d::Identity();
  const Matrix3d R_k_i = gamma_k_i_.matrix();

  // Continuous time state transition matrix, use t = i
  Fmatrix Ft;
  Ft.setZero();
  Ft.block<3, 3>(Index::ALPHA, Index::BETA) = I_3x3;
  Ft.block<3, 3>(Index::BETA, Index::GAMMA) = -R_k_i * Hat3(a_i);
  Ft.block<3, 3>(Index::BETA, Index::BA) = -R_k_i;
  Ft.block<3, 3>(Index::GAMMA, Index::GAMMA) = -Hat3(w_b);
  Ft.block<3, 3>(Index::GAMMA, Index::BW) = -I_3x3;

  //  LOG(INFO) << "Ft: \n" << Ft;

  // Approximate matrix exponential 2nd order
  const Fmatrix Fd = Fmatrix::Identity() + Ft * dt + 0.5 * Ft * Ft * dt2;
  const ImuNoise::Covariance& Qc = noise_->Qc();

  //  LOG(INFO) << "Fd: \n" << Fd;

  // 15x15 = 15x15 * 15x15 * 15x15 * 15x15 + 15x12 * 12x12 * 12x15 * 15x15
  P_ = Fd * P_ * Fd.transpose() +
       //       G_ * Qc * G_.transpose() * dt2;  // vins
       Fd * G_ * Qc * G_.transpose() * Fd.transpose() * dt;  // rvio

  // Now update measurement
  alpha_k_i_ += d_alpha;
  beta_k_i_ += d_beta;
  gamma_k_i_ = gamma_k_j;
}

struct ImuFactor {
  ImuFactor(const ImuPreintegration* preint, real_t g)
      : preint_(preint), g_(g) {}

  static constexpr int num_residuals = ImuPreintegration::DoF + 6;

  template <typename T>
  bool operator()(T const* const _T_i,
                  T const* const _vb2_i,
                  T const* const _T_j,
                  T const* const _vb2_j,
                  T* _e) const {
    // typedefs
    using SE3T = Sophus::SE3<T>;
    using SO3T = Sophus::SO3<T>;
    using Vec3T = VectorNT<T, 3>;
    using Vec6T = VectorNT<T, 6>;
    using ResT = VectorNT<T, num_residuals>;

    // Wrap in eigen
    const EigenCMap<SE3T> T_i(_T_i);
    const EigenCMap<SE3T> T_j(_T_j);
    const EigenCMap<Vec3T> v_i(_vb2_i);
    const EigenCMap<Vec3T> v_j(_vb2_j);
    const EigenCMap<Vec6T> b2_i(_vb2_i + 3);
    const EigenCMap<Vec6T> b2_j(_vb2_j + 3);
    const Vec3T gv(T(0), T(0), T(g_));

    EigenMap<ResT> e(_e);

    // total delta time
    const auto dt = preint_->dt();

    const auto R_i_inv = T_i.so3().inverse();
    // a = R_w^k * (p_k+1^w - p_k^w + 0.5 * g * dt^2 - v_k^w * dt)
    const Vec3T a = R_i_inv * (T_j.translation() - T_i.translation() -
                               v_i * dt + 0.5 * gv * dt * dt);
    const Vec3T b = R_i_inv * (v_j - v_i + gv * dt);
    const SO3T r = R_i_inv * T_j.so3();

    // Compute residual
    e.template segment<3>(0) = a - preint_->alpha().cast<T>();  // alpha
    e.template segment<3>(3) = b - preint_->beta().cast<T>();   // beta
    e.template segment<3>(6) =
        (r * preint_->gamma().cast<T>().inverse()).log();  // gamma
    e.template segment<6>(9) = b2_j - b2_i;

    // Apply noise model
    const NoiseModel noise(preint_->P(), false);  // covariance

    //    noise.Standardize(e);

    return true;
  }

  static ceres::CostFunction* Create(const ImuPreintegration* preint,
                                     double g) {
    return (new ceres::AutoDiffCostFunction<ImuFactor,
                                            15,  // delta 9 + bias 6
                                            7,   // T_i
                                            9,   // vel+bias_i
                                            7,   // T_j
                                            9    // vel+bias_j
                                            >(new ImuFactor(preint, g)));
  }

  const ImuPreintegration* preint_;
  const double g_;
};

}  // namespace sv
