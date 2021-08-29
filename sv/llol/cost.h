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
            r = U * (pt_p - dT * tf_g * pt_g);
          }
        });

    return true;
  }
};

struct ImuPreintegrationCost {
  ImuPreintegrationCost(ImuPreintegration& preint) : preint_{preint} {}

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    return true;
  }

  ImuPreintegration& preint_;
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kNumParams>;

}  // namespace sv
