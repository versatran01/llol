#pragma once

#include "sv/sim/tuple.h"

namespace sv::sim {

using Transform3h = Eigen::Transform<scalar, 3, Eigen::Affine>;
using Translation3 = Eigen::Translation<scalar, 3>;
using AngleAxis = Eigen::AngleAxis<scalar>;

inline auto Scaling3(scalar x, scalar y, scalar z) noexcept {
  return Eigen::Scaling(x, y, z);
}
inline auto Scaling3(scalar s) noexcept { return Scaling3(s, s, s); }

inline AngleAxis RotX(scalar rad) noexcept { return {rad, Vector3::UnitX()}; }
inline AngleAxis RotY(scalar rad) noexcept { return {rad, Vector3::UnitY()}; }
inline AngleAxis RotZ(scalar rad) noexcept { return {rad, Vector3::UnitZ()}; }

Matrix3 Shearing3(
    scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);

Transform3h ViewTransform(const Point3h& from,
                          const Point3h& to,
                          const Vector3h& up);

template <typename T>
Transform3h MakeTransform3h(T&& t) noexcept {
  Transform3h tf;
  tf = t;
  return tf;
}

}  // namespace sv::sim