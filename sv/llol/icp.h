#pragma once

#include <array>
#include <sophus/se3.hpp>

#include "sv/llol/grid.h"

namespace sv {

struct GicpCost {
  static constexpr int kNumRes = 3;
  static constexpr int kNumParams = 6;

  GicpCost(const SweepGrid& grid);

  int NumResiduals() const { return kNumRes; }

  template <typename T>
  bool operator()(const T* const _x, T* _r) const {
    Eigen::Map<const Eigen::Matrix<T, kNumParams, 1>> T0(_x);
    Eigen::Map<const Eigen::Matrix<T, kNumParams, 1>> T1(_x + kNumParams);

    return true;
  }
};

struct IcpParams {
  int n_outer{2};
  int n_inner{2};
};

struct Icp {
  Icp(const IcpParams& params = {});

  void Register(const SweepGrid& grid);

  std::array<Sophus::SE3d, 2> poses;
  int n_outer;
  int n_inner;
};

}  // namespace sv
