#pragma once

#include "sv/sim/transform.h"  // Transform3h
#include "sv/sim/tuple.h"      // Point3h, Vector3h

namespace sv::sim {

/// Rays from the same point
struct Ray {
  Point3h origin;
  Vector3h direction;

  Point3h Position(double t) const { return origin + direction * t; }

  Ray Transformed(const Transform3h& tf) const {
    return {tf * origin, tf * direction};
  }

  // In-place transform
  void Transform_(const Transform3h& tf) {
    origin = tf * origin;
    direction = tf * direction;
  }
};

}  // namespace sv::sim
