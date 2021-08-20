#include "sv/sim/world.h"

#include <fstream>
#include <iostream>

#include "sv/sim/shapes.h"

namespace sv::sim {

Matrix2Xd World::BundleCast(const Vector3& origin,
                            const Matrix3X& directions) const noexcept {
  Matrix2X buffer(2, directions.cols());

  for (int i = 0; i < buffer.cols(); ++i) {
    const Vector3 direction = directions.col(i);
    buffer.col(i) = RayCast({origin, direction});
  }

  return buffer;
}

Vector2 World::RayCast(const Ray& ray) const noexcept {
  const auto hit = Intersect(ray);
  if (hit.missed()) return {hit.t, 0};
  // If we disable intensity, then we use obj id as intensity
  if (!shade) return {hit.t, hit.obj->id};
  // here we simulate intensity
  const auto color = ShadeHit({hit, ray});
  return {hit.t, color};
}

Intersection World::Intersect(const Ray& ray) const noexcept {
  auto wi = Intersection::Miss();

  for (const auto& shape : shapes) {
    const Intersection si = shape.Intersect(ray);
    if (si.missed()) continue;
    if (wi.missed() || (si.t < wi.t)) wi = si;
  }

  return wi;
}

scalar World::ColorAt(const Ray& ray) const noexcept {
  const auto hit = Intersect(ray);
  if (hit.missed()) return 0;

  return ShadeHit({hit, ray});
}

scalar World::ShadeHit(const Computations& comps) const noexcept {
  // NOTE: use over_point instead of point to avoid acne effect
  // Otherwise, floating point rounding errors will make some rays originate
  // just below the surface, causing them to intersect the same surface they
  // should be reflecting from.
  const Material& m = comps.obj->material;
  return m.Lighting(comps.light, *comps.obj, comps.over_point, comps.normal);
}

/// For testing
World DefaultWorld() noexcept {
  World world;

  {
    Sphere s;
    world.Add(s);
  }

  {
    Sphere s;
    s.transform.scale = Vector3d::Constant(0.5);
    world.Add(s);
  }

  return world;
}

}  // namespace sv::sim
