#pragma once

#include "sv/sim/shape.h"  // PolyShape

namespace sv::sim {

struct World {
  World() = default;
  explicit World(const std::string& filename);

  template <typename T>
  void Add(const T& shape) noexcept {
    shapes.push_back(shape);
  }

  void clear() noexcept { shapes.clear(); }
  bool empty() const noexcept { return shapes.empty(); }
  size_t size() const noexcept { return shapes.size(); }

  /// Main interface
  Matrix2X BundleCast(const Vector3& origin, const Matrix3X& directions) const;
  Vector2 RayCast(const Ray& ray) const noexcept;

  Intersection Intersect(const Ray& ray) const;
  scalar ColorAt(const Ray& ray) const;
  scalar ShadeHit(const Computations& comps) const;

  std::vector<PolyShape> shapes;
  bool shade{false};  // TODO (chao): whether to simulate intensity?
};

World DefaultWorld() noexcept;

}  // namespace sv::sim
