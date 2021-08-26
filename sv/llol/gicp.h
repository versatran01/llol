#pragma once

#include <ceres/local_parameterization.h>
#include <ceres/tiny_solver_autodiff_function.h>
#include <glog/logging.h>
#include <tbb/parallel_for.h>

#include <sophus/se3.hpp>

#include "sv/llol/grid.h"
#include "sv/llol/pano.h"

namespace sv {

struct GicpCostBase {
  static constexpr int kNumParams = 6;
  static constexpr int kNumResiduals = 3;

  GicpCostBase(const SweepGrid& grid, int gsize = 0);
  virtual ~GicpCostBase() noexcept = default;

  virtual int NumResiduals() const { return matches.size() * kNumResiduals; }

  const SweepGrid* const pgrid;
  int gsize{};
  std::vector<GicpMatch> matches;
  std::vector<Sophus::SE3d> tfs_g;
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

            Eigen::Map<Vec3> r(_r + kNumResiduals * i);
            r = U * (pt_p - tfs_g.at(c) * dT * pt_g);
          }
        });

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

    Eigen::Map<const Vec6> x0(_x);
    Eigen::Map<const Vec6> x1(_x + 6);

    // TODO (chao): Maybe precompute these interp
    std::vector<SE3> tfs_e(tfs_g.size());

    // Precompute all interpolated error
    const Vec6 dx = x1 - x0;
    for (int i = 0; i < tfs_e.size(); ++i) {
      const double s = (i + 0.5) / tfs_e.size();  // 0.5 for center of cell

      // Interp
      const Vec6 x_s = x0 + s * dx;
      tfs_e[i].so3() = SO3::exp(x_s.template head<3>());
      tfs_e[i].translation() = x_s.template tail<3>();
    }

    // Fill in residuals
    for (int i = 0; i < matches.size(); ++i) {
      const auto& match = matches.at(i);
      const int c = match.px_g.x;
      const Eigen::Matrix3d U = match.U.cast<double>();
      const Eigen::Vector3d pt_p = match.mc_p.mean.cast<double>();
      const Eigen::Vector3d pt_g = match.mc_g.mean.cast<double>();

      Eigen::Map<Vec3> r(_r + kNumResiduals * i);
      r = U * (pt_p - tfs_g.at(c) * tfs_e.at(c) * pt_g);
    }

    return true;
  }
};

template <typename T>
using AdCost =
    ceres::TinySolverAutoDiffFunction<T, Eigen::Dynamic, T::kTotalParams>;

using AdGicpCostSingle = AdCost<GicpCostSingle>;
using AdGicpCostLinear = AdCost<GicpCostLinear>;

struct GicpParams {
  int outer{2};
  int inner{2};
  int half_rows{2};
  float cov_lambda{1e-6F};
};

struct GicpSolver {
  explicit GicpSolver(const GicpParams& params = {});

  std::pair<int, int> iters;  // (outer, inner) iterations
  float cov_lambda{};         // lambda added to diagonal of covar
  cv::Size pano_win;          // win size in pano used to compute mean covar
  cv::Size max_dist;          // max dist size to resue pano mc
  int pano_min_pts{};         // min pts in pano win for a valid match

  /// @brief Repr / <<
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const GicpSolver& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Match features in sweep to pano using mask
  /// @return Number of final matches
  int Match(SweepGrid& grid, const DepthPano& pano, int gsize = 0);
  int MatchRow(SweepGrid& grid, const DepthPano& pano, int gr);
  int MatchCell(SweepGrid& grid, const DepthPano& pano, const cv::Point& px_g);
};

}  // namespace sv
