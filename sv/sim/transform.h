#pragma once

#include <sophus/se3.hpp>

#include "sv/sim/tuple.h"

namespace sv::sim {

using SO3 = Sophus::SO3<scalar>;
using SE3 = Sophus::SE3<scalar>;
using AngleAxis = Eigen::AngleAxis<scalar>;

using Affine3 = Eigen::Transform<scalar, 3, Eigen::Affine>;
using Isometry3 = Eigen::Transform<scalar, 3, Eigen::Isometry>;
using Translation3 = Eigen::Translation<scalar, 3>;

inline auto Scaling3(scalar x, scalar y, scalar z) noexcept {
  return Eigen::Scaling(x, y, z);
}
inline auto Scaling3(scalar s) noexcept { return Scaling3(s, s, s); }

inline AngleAxis RotX(scalar rad) noexcept { return {rad, Vector3::UnitX()}; }
inline AngleAxis RotY(scalar rad) noexcept { return {rad, Vector3::UnitY()}; }
inline AngleAxis RotZ(scalar rad) noexcept { return {rad, Vector3::UnitZ()}; }

Matrix3 Shearing3(
    scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy);

Affine3 ViewTransform(const Point3h& from,
                      const Point3h& to,
                      const Vector3h& up);

template <typename T>
Affine3 MakeAffine3(T&& t) noexcept {
  Affine3 af;
  af = t;
  return af;
}

/// 3d Homogeneous transformation
struct Transform3h {
  Transform3h() = default;
  Transform3h(const SO3& rot) : pose(rot, Vector3::Zero()) {}
  Transform3h(const SE3& pose_in) : pose(pose_in) {}
  Transform3h(const SE3& pose_in, const Vector3& scale_in)
      : pose(pose_in), scale(scale_in) {}

  Tuple4 operator*(const Tuple4& v) const noexcept { return affine() * v; }

  static Transform3h Identity() noexcept { return {}; }
  static Transform3h Translation(const Vector3& v) noexcept {
    return {{SO3{}, v}};
  }
  static Transform3h Translation(scalar x, scalar y, scalar z) noexcept {
    return Transform3h::Translation({x, y, z});
  }
  static Transform3h Scaling(const Vector3& s) noexcept { return {SE3{}, s}; }
  static Transform3h Scaling(scalar x, scalar y, scalar z) noexcept {
    return Transform3h::Scaling({x, y, z});
  }
  static Transform3h RotX(scalar t) noexcept { return {SO3::rotX(t)}; }
  static Transform3h RotY(scalar t) noexcept { return {SO3::rotY(t)}; }
  static Transform3h RotZ(scalar t) noexcept { return {SO3::rotZ(t)}; }

  /// convert to affine transformation
  Affine3 affine() const noexcept {
    return Eigen::Scaling(scale) * Isometry3(pose.matrix());
  }

  Transform3h inverse() const noexcept {
    return {{pose.inverse()}, scale.cwiseInverse()};
  }

  SE3 pose;
  Vector3 scale{Vector3::Ones()};
};

}  // namespace sv::sim
