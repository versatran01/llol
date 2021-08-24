#pragma once

#include <ceres/tiny_solver_autodiff_function.h>

#include <sophus/se3.hpp>

#include "sv/llol/grid.h"
#include "sv/util/solver.h"

namespace sv {

struct GicpCostBase {
  static constexpr int kNumParams = 6;
  static constexpr int kNumResiduals = 3;

  GicpCostBase(const SweepGrid& grid, int size);
  virtual ~GicpCostBase() noexcept = default;

  int NumResiduals() const { return matches.size() * kNumResiduals; }

  const SweepGrid* const pgrid;
  std::vector<GicpMatch> matches;
};

struct GicpCostSingle final : public GicpCostBase {
  static constexpr int kNumPoses = 1;
  static constexpr int kTotalParams = kNumPoses * kNumParams;

  using GicpCostBase::GicpCostBase;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    return true;
  }
};

struct GicpCostLinear final : public GicpCostBase {
  static constexpr int kNumPoses = 2;
  static constexpr int kTotalParams = kNumPoses * kNumParams;

  using GicpCostBase::GicpCostBase;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    Eigen::Map<const Vec6> T0(_x);
    Eigen::Map<const Vec6> T1(_x + 6);

    // Compute dt and dR
    const Vec6 dT = T1 - T0;

    // TODO (chao): Maybe precompute these interp
    std::vector<SE3> T_p_s_vec;
    T_p_s_vec.resize(pgrid->size().width);

    // Precompute all T_p_s
    for (int i = 0; i < T_p_s_vec.size(); ++i) {
      const auto T_p_s0 = pgrid->PoseAt(i).cast<double>();
      const double s = 1.0 * i / T_p_s_vec.size();
      // Interp
      const Vec6 Ts = T0 + s * dT;
      T_p_s_vec[i].so3() = T_p_s0.so3() * SO3::exp(Ts.template head<3>());
      T_p_s_vec[i].translation() = T_p_s0.translation() + Ts.template tail<3>();
    }

    // Fill in residuals
    for (int i = 0; i < matches.size(); ++i) {
      const auto& match = matches[i];
      const Eigen::Matrix3d U = match.U.cast<double>();
      const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
      const Eigen::Vector3d pt_s = match.mc_g.mean.cast<double>();

      Eigen::Map<Vec3> r(_r + kNumResiduals * i);
      r = U * (pt_p - T_p_s_vec.at(match.px_g.x) * pt_s);
    }

    return true;
  }
};

using AdGicpCostSingle =
    ceres::TinySolverAutoDiffFunction<GicpCostSingle,
                                      Eigen::Dynamic,
                                      GicpCostSingle::kTotalParams>;

using AdGicpCostLinear =
    ceres::TinySolverAutoDiffFunction<GicpCostLinear,
                                      Eigen::Dynamic,
                                      GicpCostLinear::kTotalParams>;

struct IcpParams {
  int n_outer{2};
  int n_inner{2};
};

}  // namespace sv
