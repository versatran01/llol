#include "sv/sim/tuple.h"

#include "sv/sim/doctest.h"

namespace sv::sim {
namespace {

SCENARIO("A tuple with w=1.0 is a point") {
  GIVEN("a tuple with last element 1.0") {
    const Tuple4 a{4.3, -4.2, 3.1, 1.0};

    REQUIRE(a.size() == 4);
    CHECK(a.head<3>() == Vector3(4.3, -4.2, 3.1));
    CHECK(IsPoint(a));
    CHECK(!IsVector(a));
  }
}

SCENARIO("A tuple with w=0.0 is a vector") {
  GIVEN("a tuple with last element 1.0") {
    const Tuple4 a{4.3, -4.2, 3.1, 0.0};

    REQUIRE(a.size() == 4);
    CHECK(a.head<3>() == Vector3(4.3, -4.2, 3.1));
    CHECK(!IsPoint(a));
    CHECK(IsVector(a));
  }
}

SCENARIO("Point() creates tuples with w=1") {
  const Point3h p(4, -4, 3);
  CHECK(IsPoint(p));
  CHECK(p == Vector4(4, -4, 3, 1));
}

SCENARIO("Vector() creates tuples with w=0") {
  const Vector3h v(4, -4, 3);
  CHECK(IsVector(v));
  CHECK(v == Vector4(4, -4, 3, 0));
}

SCENARIO("Adding two tuples") {
  const Tuple4 a1(3, -2, 5, 1);
  const Tuple4 a2(-2, 3, 1, 0);
  CHECK(a1 + a2 == Tuple4(1, 1, 6, 1));
}

SCENARIO("Subtracing two points") {
  const auto p1 = Point3h(3, 2, 1);
  const auto p2 = Point3h(5, 6, 7);
  CHECK(p1 - p2 == Vector3h(-2, -4, -6));
}

SCENARIO("Subtracting a vector from a point") {
  const auto p = Point3h(3, 2, 1);
  const auto v = Vector3h(5, 6, 7);
  CHECK(p - v == Point3h(-2, -4, -6));
}

SCENARIO("Negating a tuple") {
  const Tuple4 a(1, -2, 3, -4);
  CHECK(-a == Tuple4(-1, 2, -3, 4));
}

SCENARIO("Multiplying a tuple by a scalar") {
  const Tuple4 a(1, -2, 3, -4);
  CHECK(a * 3.5 == Tuple4(3.5, -7, 10.5, -14));
}

SCENARIO("Dividing a tuple by a scalar") {
  const Tuple4 a(1, -2, 3, -4);
  CHECK(a / 2 == Tuple4(0.5, -1, 1.5, -2));
}

SCENARIO("Computing the magnitude of vector") {
  const auto v = Vector3h(1, 0, 0);
  CHECK(Magnitude(v) == 1.0);
}

SCENARIO("Normalizing vector") {
  const auto v = Vector3h(4, 0, 0);
  const auto n = Normalized(v);
  CHECK(n == Vector3h(1, 0, 0));
  CHECK(Magnitude(n) == 1.0);
}

SCENARIO("Dot product of two tuples") {
  const auto a = Vector3h(1, 2, 3);
  const auto b = Vector3h(2, 3, 4);
  CHECK(Dot(a, b) == 20);
}

SCENARIO("Cross product of two tuples") {
  const auto a = Vector3h(1, 2, 3);
  const auto b = Vector3h(2, 3, 4);
  CHECK(Cross(a, b) == Vector3h(-1, 2, -1));
  CHECK(Cross(b, a) == -Cross(a, b));
}

SCENARIO("Reflecting a vector apporaching at 45 deg") {
  const Vector3h v(1, -1, 0);
  const Vector3h n(0, 1, 0);
  CHECK(Reflect(v, n) == Vector3h(1, 1, 0));
}

SCENARIO("Reflecting a vector off a slanted surface") {
  const Vector3h v(0, -1, 0);
  const Vector3h n(std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0, 0);
  CHECK(Reflect(v, n).isApprox(Vector3h(1, 0, 0)));
}

}  // namespace
}  // namespace sv::sim