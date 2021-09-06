#pragma once

#include "sv/llol/grid.h"
#include "sv/llol/imu.h"

namespace sv {

struct GicpCost {
 public:
  using Scalar = double;
  static constexpr int kResidualDim = 3;

  GicpCost(int gsize = 0);
  virtual ~GicpCost() noexcept = default;

  virtual int NumResiduals() const;
  void UpdateMatches(const SweepGrid& grid);
  void UpdatePreint(const Trajectory& traj, const ImuQueue& imuq);

  int gsize_{};
  double imu_weight{};
  const SweepGrid* pgrid{nullptr};
  std::vector<PointMatch> matches;
  const Trajectory* ptraj{nullptr};
  ImuPreintegration preint;
};

/// @brief Gicp with rigid transformation
struct GicpRigidCost final : public GicpCost {
 public:
  static constexpr int kNumParams = 6;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  enum Block { kR0, kP0 };

  // Pull in base constructor
  using GicpCost::GicpCost;

  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;

    State(const double* const x_ptr) : x_ptr{x_ptr} {}
    auto r0() const { return Vec3CMap{x_ptr + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x_ptr + Block::kP0 * kBlockSize}; }

    const double* const x_ptr{nullptr};
  };

  bool operator()(const double* x_ptr, double* r_ptr, double* J_ptr) const;
};

/// @brief Linear interpolation in translation error state
struct GicpLinearCost final : public GicpCost {
 public:
  static constexpr int kNumParams = 6;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  enum Block { kR0, kP0 };
  using GicpCost::GicpCost;

  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;

    State(const double* const x_ptr) : x_ptr{x_ptr} {}
    auto r0() const { return Vec3CMap{x_ptr + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x_ptr + Block::kP0 * kBlockSize}; }

    const double* const x_ptr{nullptr};
  };

  bool operator()(const double* x_ptr, double* r_ptr, double* J_ptr) const;
};

}  // namespace sv
