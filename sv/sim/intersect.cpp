#include "sv/sim/intersect.h"

#include "sv/sim/shape.h"

namespace sv::rt {

Computations::Computations(const Intersection& hit, const Ray& ray) noexcept
    : t(hit.t),
      obj(hit.obj),
      light(ray.origin),
      point(ray.Position(t)),
      normal(obj->NormalAt(point)) {
  if (Dot(normal, -ray.direction) < 0) {
    inside = true;
    normal = -normal;
  } else {
    inside = false;
  }

  over_point = point + normal * kEps;
}

}  // namespace sv::rt
