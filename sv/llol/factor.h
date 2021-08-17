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

struct GicpFactor {
  static constexpr int kNumResiduals = 3;
  static constexpr int kNumParams = Sophus::SE3d::num_parameters;

  template <typename T>
  using Vector3 = Eigen::Matrix<T, kNumResiduals, 1>;

  GicpFactor(const PointMatch& match);

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

}  // namespace sv
