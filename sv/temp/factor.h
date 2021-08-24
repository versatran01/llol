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
  GicpFactor(const GicpMatch& match);

  template <typename T>
  bool operator()(const T* const _T_p_s, T* _r) const noexcept {
    Eigen::Map<const Sophus::SE3<T>> T_p_s(_T_p_s);
    Eigen::Map<Vector3<T>> r(_r);
    r = U_ * (pt_p_ - T_p_s * pt_s_);
    return true;
  }

  Eigen::Vector3d pt_s_;
  Eigen::Vector3d pt_p_;
  Eigen::Matrix3d U_;
};

struct GicpFactor2 final : public IcpFactorBase {
  GicpFactor2(const GicpMatch& match) : pmatch{&match} {}

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

  const GicpMatch* pmatch{nullptr};
};

struct GicpFactor3 final : public IcpFactorBase {
  GicpFactor3(const SweepGrid& grid, int size, int gsize = 0);

  template <typename T>
  bool operator()(const T* const _T_p_s, T* _r) const noexcept {
    Eigen::Map<const Sophus::SE3<T>> T_p_s(_T_p_s);

    for (size_t i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];

      const Eigen::Matrix3d U = match.U.cast<double>();
      const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
      const Eigen::Vector3d pt_s = match.mc_s.mean.cast<double>();
      Eigen::Map<Vector3<T>> r(_r + kNumResiduals * i);
      r = U * (pt_p - T_p_s * pt_s);
    }

    return true;
  }

  const SweepGrid* pgrid;
  std::vector<GicpMatch> matches;
  int size_{};
  int gsize_{};
};

struct TinyGicpFactor final : public IcpFactorBase {
  TinyGicpFactor(const SweepGrid& grid, int size, const Sophus::SE3d& T0);

  int NumResiduals() const { return size_ * kNumResiduals; }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> x(_x);

    Sophus::SE3<T> T_p_s;
    T_p_s.so3() = T0_.so3() * Sophus::SO3<T>::exp(x.template head<3>());
    T_p_s.translation() = T0_.translation() + x.template tail<3>();

    for (int i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];

      const Eigen::Matrix3d U = match.U.cast<double>();
      const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
      const Eigen::Vector3d pt_s = match.mc_s.mean.cast<double>();
      Eigen::Map<Vector3<T>> r(_r + kNumResiduals * i);
      r = U * (pt_p - T_p_s * pt_s);
    }

    return true;
  }

  int size_{};
  Sophus::SE3d T0_;
  std::vector<GicpMatch> matches;
};

}  // namespace sv
