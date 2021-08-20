
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

constexpr scalar kPi = M_PI;
constexpr scalar kSqrt2 = std::sqrt(2.0);
constexpr scalar kSqrt3 = std::sqrt(3.0);
constexpr scalar kEps = 1e-6;
constexpr auto kInf = std::numeric_limits<scalar>::infinity();

template <typename T>
T Square(T x) noexcept {
  return x * x;
}

}  // namespace sv::rt