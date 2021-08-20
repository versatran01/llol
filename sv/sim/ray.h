#pragma once

#include "raytracer/transform.h"  // Transform3h
#include "sv/sim/tuple.h"      // Point3h, Vector3h

namespace sv::sim {

/// Rays from the same point
struct Ray {
  Point3h origin;
  Vector3h direction;

  Point3h Position(double t) const noexcept { return origin + direction * t; }

  Ray Transformed(const Transform3h& tf) const noexcept {
    return {tf * origin, tf * direction};
  }

  // In-place transform
  void Transform_(const Transform3h& tf) noexcept {
    origin = tf * origin;
    direction = tf * direction;
  }
};

}  // namespace sv::rt