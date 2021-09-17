#pragma once

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"
#include "sv/llol/nlls.h"

namespace sv {

struct GicpCost final : public CostBase {
 public:
  static constexpr int kNumParams = 6;
  static constexpr int kResidualDim = 3;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  using ErrorVector = Eigen::Matrix<double, kNumParams, 1>;

  /// @brief Error state
  enum Block { kR0, kP0 };
  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;

    State(const double* const x) : x_{x} {}
    auto r0() const { return Vec3CMap{x_ + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x_ + Block::kP0 * kBlockSize}; }

    const double* const x_{nullptr};
  };

  GicpCost(int gsize = 0);

  int NumResiduals() const override;
  int NumParameters() const override { return kNumParams; }
  bool Compute(const double* px, double* pr, double* pJ) const override;

  void ResetError();
  void UpdateMatches(const SweepGrid& grid);
  void UpdatePreint(const Trajectory& traj, const ImuQueue& imuq);
  void UpdateTraj(Trajectory& traj) const;

  int gsize_{};
  double imu_weight{0.0};

  const SweepGrid* pgrid{nullptr};
  std::vector<PointMatch> matches;

  const Trajectory* ptraj{nullptr};
  ImuPreintegration preint;

  ErrorVector error{};
};

/// @brief Gicp with rigid transformation
// struct GicpRigidCost final : public GicpCost {
//  using GicpCost::GicpCost;
//  bool operator()(const double* x_ptr, double* r_ptr, double* J_ptr) const;
//  void UpdateTraj(Trajectory& traj) const override;
//};

///// @brief Linear interpolation in translation error state
// struct GicpLinearCost final : public GicpCost {
//  using GicpCost::GicpCost;
//  bool operator()(const double* x_ptr, double* r_ptr, double* J_ptr) const;
//  void UpdateTraj(Trajectory& traj) const override;
//};

}  // namespace sv
