#pragma once

#include <ceres/local_parameterization.h>

#include <sophus/se3.hpp>

#include "sv/llol/match.h"

namespace sv {

// Taken from Sophus github
class LocalParamSE3 final : public ceres::LocalParameterization {
 public:
  // SE3 plus operation for Ceres
  //  T * exp(x)
  bool Plus(double const* _T,
            double const* _x,
            double* _T_plus_x) const override;

  // Jacobian of SE3 plus operation for Ceres
  // Dx T * exp(x)  with  x=0
  bool ComputeJacobian(double const* _T, double* _J) const override;

  int GlobalSize() const override { return Sophus::SE3d::num_parameters; }
  int LocalSize() const override { return Sophus::SE3d::DoF; }
};

struct IcpFactorBase {
  static constexpr int kNumResiduals = 3;
  static constexpr int kNumParams = Sophus::SE3d::num_parameters;

  template <typename T>
  using Vector3 = Eigen::Matrix<T, kNumResiduals, 1>;
};

struct GicpFactor final : public IcpFactorBase {
  GicpFactor(const NormalMatch& match);

  template <typename T>
  bool operator()(const T* const _T_p_s, T* _r) const noexcept {
    Eigen::Map<const Sophus::SE3<T>> T_p_s(_T_p_s);
    Eigen::Map<Vector3<T>> r(_r);
    r = U * (pt_p - T_p_s * pt_s);
    return true;
  }

  Eigen::Vector3d pt_s;
  Eigen::Vector3d pt_p;
  Eigen::Matrix3d U;
};

struct GicpFactor2 final : public IcpFactorBase {
  GicpFactor2(const NormalMatch& match) : pmatch{&match} {}

  template <typename T>
  bool operator()(const T* const _T_p_s, T* _r) const noexcept {
    const auto& match = *pmatch;
    Eigen::Map<const Sophus::SE3<T>> T_p_s(_T_p_s);
    Eigen::Map<Vector3<T>> r(_r);

    const Eigen::Matrix3d U = match.U.cast<double>();
    const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
    const Eigen::Vector3d pt_s = match.mc_s.mean.cast<double>();

    r = U * (pt_p - T_p_s * pt_s);
    return true;
  }

  const NormalMatch* pmatch{nullptr};
};

//struct GicpFactor3 final : public IcpFactorBase {
//  GicpFactor3(const ProjMatcher& matcher);

//  template <typename T>
//  bool operator()(const T* const _T_p_s, T* _r) const noexcept {
//    Eigen::Map<const Sophus::SE3<T>> T_p_s(_T_p_s);
//    Eigen::Map<Vector3<T>> r(_r);

//    return true;
//  }

//  const ProjMatcher* matcher;
//};

}  // namespace sv
