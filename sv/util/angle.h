#pragma once

#include <cmath>
#include <type_traits>

namespace sv {

static constexpr auto kPiF = static_cast<float>(M_PI);
static constexpr auto kTauF = static_cast<float>(M_PI * 2);
static constexpr auto kPiD = static_cast<double>(M_PI);
static constexpr auto kTauD = static_cast<double>(M_PI * 2);

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

}  // namespace sv
