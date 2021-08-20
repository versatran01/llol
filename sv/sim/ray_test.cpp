
#include "sv/sim/ray.h"

#include "sv/sim/doctest.h"

namespace sv::sim {
namespace {

SCENARIO("Computing a point from a distance") {
  const Ray r{{2, 3, 4}, {1, 0, 0}};

  CHECK(r.Position(0) == Point3h(2, 3, 4));
  CHECK(r.Position(1) == Point3h(3, 3, 4));
  CHECK(r.Position(-1) == Point3h(1, 3, 4));
}

SCENARIO("Translating a ray") {
  const Ray r{{1, 2, 3}, {0, 1, 0}};
  const auto t = Transform3h::Translation(3, 4, 5);
  const auto rt = r.Transformed(t);

  CHECK(rt.origin == Point3h(4, 6, 8));
  CHECK(rt.direction == Vector3h(0, 1, 0));
}

SCENARIO("Scaling a ray") {
  const Ray r{{1, 2, 3}, {0, 1, 0}};
  const auto t = Transform3h::Scaling(2, 3, 4);
  const auto rt = r.Transformed(t);

  CHECK(rt.origin == Point3h(2, 6, 12));
  CHECK(rt.direction == Vector3h(0, 3, 0));
}

}  // namespace
}  // namespace sv::sim