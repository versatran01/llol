#pragma once

#include <Eigen/Geometry>

#include "sv/sim/types.h"

namespace sv::sim {

using Tuple4 = Vector4;

// Note: inheriting from Eigen is a bad idea, but I just don't want to implement
// all those operators
struct Point3h : public Tuple4 {
  Point3h() : Point3h(0, 0, 0) {}
  Point3h(scalar x, scalar y, scalar z) noexcept : Tuple4(x, y, z, 1.0) {}
  Point3h(const Vector3& p) : Point3h(p.x(), p.y(), p.z()) {}

  static Point3h Zero() noexcept { return {0, 0, 0}; }
  static Point3h Ones() noexcept { return {1, 1, 1}; }
  static Point3h Constant(scalar c) { return {c, c, c}; }

  // https://eigen.tuxfamily.org/dox/TopicCustomizing_InheritingMatrix.html
  template <typename Dervied>
  Point3h(const Eigen::MatrixBase<Dervied>& other) : Tuple4(other) {}

  template <typename Derived>
  Point3h& operator=(const Eigen::MatrixBase<Derived>& other) {
    Tuple4::operator=(other);
    return *this;
  }
};

struct Vector3h : public Tuple4 {
  Vector3h() : Tuple4(0, 0, 0, 0) {}
  Vector3h(scalar x, scalar y, scalar z) noexcept : Tuple4(x, y, z, 0.0) {}
  Vector3h(const Vector3& v) : Vector3h(v.x(), v.y(), v.z()) {}

  static Vector3h Constant(scalar c) { return {c, c, c}; }

  template <typename Dervied>
  Vector3h(const Eigen::MatrixBase<Dervied>& other) : Tuple4(other) {}

  template <typename Derived>
  Vector3h& operator=(const Eigen::MatrixBase<Derived>& other) {
    Tuple4::operator=(other);
    return *this;
  }
};

inline bool IsPoint(const Tuple4& t) { return t[3] == 1.0; }
inline bool IsVector(const Tuple4& t) { return t[3] == 0.0; }

inline scalar Magnitude(const Tuple4& t) noexcept { return t.norm(); }
inline Vector3h Normalized(const Vector3h& t) noexcept {
  return t.normalized();
}

inline scalar Dot(const Vector3h& t1, const Vector3h& t2) noexcept {
  return t1.dot(t2);
}

inline Vector3h Cross(const Vector3h& t1, const Vector3h& t2) noexcept {
  return t1.cross3(t2);
}

inline Vector3h Reflect(const Vector3h& in, const Vector3h& normal) noexcept {
  return in - normal * 2 * Dot(in, normal);
}

}  // namespace sv::rt