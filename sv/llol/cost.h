#pragma once

#include <ceres/tiny_solver_autodiff_function.h>
#include <tbb/parallel_for.h>

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

struct GicpCostBase {
  static constexpr int kNumParams = 6;
  static constexpr int kNumResiduals = 3;

  GicpCostBase(const SweepGrid& grid, int gsize = 0);
  virtual ~GicpCostBase() noexcept = default;

  virtual int NumResiduals() const { return matches.size() * kNumResiduals; }

  const SweepGrid* const pgrid;
  int gsize{};
  std::vector<PointMatch> matches;
};

struct GicpCostSingle final : public GicpCostBase {
  static constexpr int kNumPoses = 1;
  static constexpr int kTotalParams = kNumPoses * kNumParams;

  using GicpCostBase::GicpCostBase;

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    Eigen::Map<const Vec6> x(_x);
    const SE3 dT{SO3::exp(x.template head<3>()), x.template tail<3>()};

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
            r = U * (pt_p - tf_g * dT * pt_g);
          }
        });

    return true;
  }
};

struct GicpImuCost {
  const SweepGrid* const pgrid{nullptr};
  std::vector<PointMatch> matches;
  int gsize{};

  virtual int NumResiduals() const {
    return matches.size() * 3 + ImuPreintegration::kDim;
  }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    using Vec6 = Eigen::Matrix<T, 6, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SE3 = Sophus::SE3<T>;
    using SO3 = Sophus::SO3<T>;

    return true;
  }
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kTotalParams>;
using AdGicpCostSingle = AdCost<GicpCostSingle>;

}  // namespace sv
