#pragma once

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"
#include "sv/util/nlls.h"

namespace sv {

struct GicpCost : public CostBase {
  static constexpr int kResidualDim = 3;

  GicpCost(int num_params, double w_imu, int gsize = 0);

  void ResetError();
  int NumResiduals() const override;
  int NumParameters() const override { return error.size(); }

  void UpdateMatches(const SweepGrid& grid);
  void UpdatePreint(const Trajectory& traj, const ImuQueue& imuq);

  virtual void UpdateTraj(Trajectory& traj) const = 0;

  int gsize_{};
  double imu_weight{0.0};

  const SweepGrid* pgrid{nullptr};
  std::vector<PointMatch> matches;
  std::vector<Eigen::Vector3d> pts_p_hat;

  const Trajectory* ptraj{nullptr};
  ImuPreintegration preint;

  Eigen::VectorXd error{};
};

struct GicpCostRigid final : public GicpCost {
  GicpCostRigid(double w_imu, int gsize = 0) : GicpCost(6, w_imu, gsize) {}

  /// @brief Error state
  enum Block { kR0, kP0 };
  struct State {
    using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;
    State(const double* const x) : x_{x} {}
    auto r0() const { return Vec3CMap{x_ + Block::kR0 * 3}; }
    auto p0() const { return Vec3CMap{x_ + Block::kP0 * 3}; }
    const double* const x_{nullptr};
  };

  bool Compute(const double* px, double* pr, double* pJ) const override;
  void UpdateTraj(Trajectory& traj) const override;
};

struct GicpCostLinear final : public GicpCost {
  GicpCostLinear(double w_imu, int gsize = 0) : GicpCost(9, w_imu, gsize) {}

  /// @brief Error state
  enum Block { kR0, kP0, kP1 };
  struct State {
    using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;
    State(const double* const x) : x_{x} {}
    auto r0() const { return Vec3CMap{x_ + Block::kR0 * 3}; }
    auto p0() const { return Vec3CMap{x_ + Block::kP0 * 3}; }
    auto p1() const { return Vec3CMap{x_ + Block::kP1 * 3}; }
    const double* const x_{nullptr};
  };

  bool Compute(const double* px, double* pr, double* pJ) const override;
  void UpdateTraj(Trajectory& traj) const override;
};

}  // namespace sv
