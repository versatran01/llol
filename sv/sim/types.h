
#pragma once

#include <Eigen/Core>
#include <cmath>

namespace sv::sim {

using scalar = double;

using Vector2 = Eigen::Matrix<scalar, 2, 1>;
using Vector3 = Eigen::Matrix<scalar, 3, 1>;
using Vector4 = Eigen::Matrix<scalar, 4, 1>;

using Matrix3 = Eigen::Matrix<scalar, 3, 3>;
using Matrix4 = Eigen::Matrix<scalar, 4, 4>;
using Matrix2X = Eigen::Matrix<scalar, 2, Eigen::Dynamic>;
using Matrix3X = Eigen::Matrix<scalar, 3, Eigen::Dynamic>;

constexpr scalar kPi = M_PI;
constexpr scalar kEps = 1e-6;
constexpr scalar kHalfPi = M_PI_2;
constexpr scalar kInf = std::numeric_limits<scalar>::infinity();

template <typename T>
scalar Square(T x) noexcept {
  return x * x;
}

}  // namespace sv::sim
