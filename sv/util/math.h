#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cmath>
#include <type_traits>

namespace sv {

using Vector3f = Eigen::Vector3f;
using Matrix3f = Eigen::Matrix3f;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using MatrixXd = Eigen::MatrixXd;

static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();
static constexpr auto kPiF = static_cast<float>(M_PI);
static constexpr auto kTauF = static_cast<float>(M_PI * 2);
static constexpr auto kPiD = static_cast<double>(M_PI);
static constexpr auto kTauD = static_cast<double>(M_PI * 2);

// clang-format off
/// @brief Make skew symmetric matrix
template <typename T>
Eigen::Matrix3<T> Hat3(const Eigen::Vector3<T>& w) {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  Eigen::Matrix3<T> S;
  S << T(0.0),  -w(2),  w(1),
       w(2),    T(0.0), -w(0),
       -w(1),   w(0),   T(0.0);
  return S;
}

template <typename T>
Eigen::Matrix3<T> ExpApprox(const Eigen::Vector3<T>& w) {
  return Eigen::Matrix3<T>::Identity() + Hat3(w);
}
// clang-format on

template <typename T>
constexpr T Sq(T x) noexcept {
  return x * x;
}

template <typename T>
T Deg2Rad(T deg) {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return deg / 180.0 * M_PI;
}

template <typename T>
T Rad2Deg(T rad) {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return rad / M_PI * 180.0;
}

/// @brief Precomputed sin and cos
template <typename T>
struct SinCos {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  SinCos(T rad = 0) : sin{std::sin(rad)}, cos{std::cos(rad)} {}

  T sin{};
  T cos{};
};

using SinCosF = SinCos<float>;

/// @brief Polynomial approximation to asin
template <typename T>
T AsinApprox(T x) {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  const T x2 = x * x;
  return x * (1 + x2 * (1 / 6.0 + x2 * (3.0 / 40.0 + x2 * 5.0 / 112.0)));
}

/// @brief A faster atan2
template <typename T>
T Atan2Approx(T y, T x) {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/atan2.html
  // Volkan SALMA
  static constexpr T kPi3_4 = M_PI_4 * 3;
  static constexpr T kPi_4 = M_PI_4;

  T r, angle;
  T abs_y = fabs(y) + 1e-10;  // kludge to prevent 0/0 condition
  if (x < 0.0) {
    r = (x + abs_y) / (abs_y - x);
    angle = kPi3_4;
  } else {
    r = (x - abs_y) / (x + abs_y);
    angle = kPi_4;
  }
  angle += (0.1963 * r * r - 0.9817) * r;
  return y < 0.0 ? -angle : angle;
}

/// @struct Running mean and variance
template <typename T, int N>
struct MeanVar {
  using Vector = Eigen::Matrix<T, N, 1>;

  int n{0};
  Vector mean{Vector::Zero()};
  Vector var_sum_{Vector::Zero()};

  /// @brief compute covariance
  Vector Var() const { return var_sum_ / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) {
    ++n;
    const Vector dx = x - mean;
    const Vector dx_n = dx / n;
    mean += dx_n;
    var_sum_.noalias() += (n - 1.0) * dx_n.cwiseProduct(dx);
  }

  void Reset() {
    n = 0;
    mean.setZero();
    var_sum_.setZero();
  }
};

using MeanVar3f = MeanVar<float, 3>;
using MeanVar3d = MeanVar<double, 3>;

/// @struct Running mean and covariance
template <typename T, int N>
struct MeanCovar {
  using Matrix = Eigen::Matrix<T, N, N>;
  using Vector = Eigen::Matrix<T, N, 1>;

  int n{0};
  Vector mean{Vector::Zero()};
  Matrix covar_sum_{Matrix::Zero()};

  /// @brief compute covariance
  Matrix Covar() const { return covar_sum_ / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) {
    ++n;
    const Vector dx = x - mean;
    const Vector dx_n = dx / n;
    mean += dx_n;
    covar_sum_.noalias() += ((n - 1.0) * dx_n) * dx.transpose();
  }

  void Reset() {
    n = 0;
    mean.setZero();
    covar_sum_.setZero();
  }
};

using MeanCovar3f = MeanCovar<float, 3>;
using MeanCovar3d = MeanCovar<double, 3>;

/// @brief force the axis to be right handed for 3D
/// @details sometimes eigvecs has det -1 (reflection), this makes it a rotation
/// @ref
/// https://docs.ros.org/en/noetic/api/rviz/html/c++/covariance__visual_8cpp_source.html
void MakeRightHanded(Vector3d& eigvals, Matrix3d& eigvecs);

/// @brief Computes matrix square root using Cholesky A = LL' = U'U
template <typename T, int N>
Eigen::Matrix<T, N, N> MatrixSqrtUtU(const Eigen::Matrix<T, N, N>& A) {
  return A.template selfadjointView<Eigen::Upper>().llt().matrixU();
}

/// @brief Basically c % cols
inline int WrapCols(int c, int cols) { return c < 0 ? c + cols : c; }

}  // namespace sv
