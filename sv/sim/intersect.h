#pragma once

#include <vector>

#include "sv/sim/ray.h"

namespace sv::sim {

// fwd
struct Shape;

struct Intersection {
  static constexpr scalar kMiss = -1.0;
  static constexpr scalar kMin = 0.0;

  Intersection() = default;
  Intersection(scalar t, const Shape* obj) : t(t), obj(obj) {}

  bool hit() const noexcept { return t >= kMin; }
  bool missed() const noexcept { return !hit(); }

  static Intersection Miss() noexcept { return {}; }

  scalar t{kMiss};
  const Shape* obj{nullptr};
};

struct Computations {
  Computations() = default;
  Computations(const Intersection& hit, const Ray& ray) noexcept;

  scalar t{Intersection::kMiss};
  const Shape* obj{nullptr};

  Point3h light;
  Point3h point;
  Point3h over_point;
  Vector3h normal;
  bool inside{false};
};

}  // namespace sv::sim
