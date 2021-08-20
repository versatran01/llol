#include "sv/sim/material.h"

#include "sv/sim/doctest.h"
#include "sv/sim/shapes.h"

namespace sv::sim {
namespace {

SCENARIO("The default material") {
  const Material m;
  CHECK(m.reflectivity == 1.0);
  CHECK(m.ambient == 0.4);
  CHECK(m.diffuse == 0.6);
}

TEST_CASE("Lighting") {
  const Material m;
  const Point3h p(0, 0, 0);
  const Sphere s;

  SUBCASE("Look directly at a surface") {
    const Point3h l{0, 0, -10};
    const Vector3h n{0, 0, -1};
    CHECK(m.Lighting(l, s, p, n) == m.ambient + m.diffuse);
  }

  SUBCASE("Look at a surface from 45deg") {
    const Point3h l{0, 2, -2};
    const Vector3h n{0, 0, -1};
    CHECK(m.Lighting(l, s, p, n) ==
          doctest::Approx(m.ambient + m.diffuse * std::sqrt(2.0) / 2));
  }
}

}  // namespace
}  // namespace sv::sim
