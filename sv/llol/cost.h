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

  const SweepGrid* pgrid{nullptr};
  std::vector<PointMatch> matches;

  double imu_scale{10.0};
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

  template <typename T>
  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
    using Vec3CMap = Eigen::Map<const Vec3>;

    State(const T* const _x) : x{_x} {}
    auto r0() const { return Vec3CMap{x + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x + Block::kP0 * kBlockSize}; }

    const T* const x{nullptr};
  };

  bool operator()(const double* _x, double* _r, double* _J) const;
};

/// @brief Linear interpolation in translation error state
struct GicpLinearCost final : public GicpCost {
 public:
  static constexpr int kNumParams = 6;
  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
  enum Block { kR0, kP0 };
  using GicpCost::GicpCost;

  template <typename T>
  struct State {
    static constexpr int kBlockSize = 3;
    using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
    using Vec3CMap = Eigen::Map<const Vec3>;

    State(const T* const _x) : x{_x} {}
    auto r0() const { return Vec3CMap{x + Block::kR0 * kBlockSize}; }
    auto p0() const { return Vec3CMap{x + Block::kP0 * kBlockSize}; }

    const T* const x{nullptr};
  };

  bool operator()(const double* _x, double* _r, double* _J) const;
};

// struct GicpLinearCost2 final : public GicpCost {
// public:
//  static constexpr int kNumParams = 9;
//  enum { NUM_PARAMETERS = kNumParams, NUM_RESIDUALS = Eigen::Dynamic };
//  enum Block { kR0, kP0, kP1 };
//  using GicpCost::GicpCost;

//  template <typename T>
//  struct State {
//    static constexpr int kBlockSize = 3;
//    using Vec3 = Eigen::Matrix<T, kBlockSize, 1>;
//    using Vec3CMap = Eigen::Map<const Vec3>;

//    State(const T* const _x) : x{_x} {}
//    auto r0() const { return Vec3CMap{x + Block::kR0 * kBlockSize}; }
//    auto p0() const { return Vec3CMap{x + Block::kP0 * kBlockSize}; }
//    auto p1() const { return Vec3CMap{x + Block::kP1 * kBlockSize}; }

//    const T* const x{nullptr};
//  };

//  bool operator()(const double* _x, double* _r, double* _J) const;
//};

// struct ImuCost {
//  static constexpr int kNumParams = 24;

//  explicit ImuCost(const Trajectory& traj);
//  int NumResiduals() const { return ImuPreintegration::kDim; }

//  template <typename T>
//  bool operator()(const T* const _x, T* _r) const {
//    using Vec15 = Eigen::Matrix<T, 15, 1>;  // Residuals
//    using Vec3 = Eigen::Vector3<T>;
//    using SO3 = Sophus::SO3<T>;
//    using ES = ErrorState<T>;

//    const auto dt = preint.duration;
//    const auto dt2 = dt * dt;
//    const auto& g = ptraj->gravity;
//    const auto& st0 = ptraj->states.front();
//    const auto& st1 = ptraj->states.back();

//    const ES es(_x);
//    const auto eR0 = SO3::exp(es.r0());
//    const auto eR1 = SO3::exp(es.r1());
//    const Vec3 p0 = es.p0() + eR0 * st0.pos;
//    const Vec3 p1 = es.p1() + eR1 * st1.pos;
//    const auto R0 = eR0 * st0.rot;
//    const auto R1 = eR1 * st1.rot;
//    const Vec3 v0 = es.v0() + st0.vel;
//    const Vec3 v1 = es.v1() + st1.vel;

//    const auto R0_inv = R0.inverse();
//    const Vec3 alpha = R0_inv * (p1 - p0 - v0 * dt + 0.5 * g * dt2);
//    const Vec3 beta = R0_inv * (v1 - v0 + g * dt);
//    const auto gamma = R0_inv * R1;

//    using IP = ImuPreintegration::Index;
//    Eigen::Map<Vec15> r(_r);
//    r.template segment<3>(IP::kAlpha) = alpha - preint.alpha;
//    r.template segment<3>(IP::kBeta) = beta - preint.beta;
//    r.template segment<3>(IP::kTheta) = (gamma *
//    preint.gamma.inverse()).log(); r.template segment<3>(IP::kBa) = es.ba();
//    r.template segment<3>(IP::kBw) = es.bw();
//    r = preint.U * r;

//    // Debug print
//    if constexpr (std::is_same_v<T, double>) {
//      std::cout << preint.F << std::endl;
//      std::cout << r.transpose() << std::endl;
//      std::cout << "norm: " << r.squaredNorm() << std::endl;
//    }

//    return true;
//  }

//  const Trajectory* const ptraj;
//  ImuPreintegration preint;
//};

}  // namespace sv
