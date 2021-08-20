#include "sv/sim/transform.h"

#include "sv/sim/doctest.h"

namespace sv::sim {
namespace {

SCENARIO("Multiplying by a translation matrix") {
  Transform3h t;
  t = Translation3(5, -3, 2);
  const Point3h p(-3, 4, 5);
  CHECK(t * p == Point3h(2, 1, 7));
}

SCENARIO("Multiplying by the inverse of a translation matrix") {
  const auto t = Translation3(5, -3, 2);
  Transform3h inv;
  inv = t.inverse();
  const Point3h p(-3, 4, 5);
  CHECK(inv * p == Point3h(-8, 7, 3));
}

SCENARIO("Translation does not affect vectors") {
  Transform3h t;
  t = Translation3(5, -3, 2);
  const Vector3h v(-3, 4, 5);
  CHECK(t * v == v);
}

SCENARIO("A scaling matrix applied to a point") {
  Transform3h t;
  t = Scaling3(2, 3, 4);
  const Point3h p(-4, 6, 8);
  CHECK(t * p == Point3h(-8, 18, 32));
}

SCENARIO("Multiplying by the inverse of a scaling matrix") {
  Transform3h inv;
  inv = Scaling3(2, 3, 4).inverse();
  const Vector3h v(-4, 6, 8);
  CHECK(inv * v == Vector3h(-2, 2, 2));
}

SCENARIO("Reflection is scaling by a negative value") {
  Transform3h t;
  t = Scaling3(-1, 1, 1);
  const Point3h p(2, 3, 4);
  CHECK(t * p == Point3h(-2, 3, 4));
}

SCENARIO("Rotating a point around the x axis") {
  const Point3h p(0, 1, 0);
  Transform3h t;
  t = RotX(kPi / 2);
  CHECK((t * p).isApprox(Point3h(0, 0, 1)));
}

SCENARIO("The inverse of an x-rotation rotates in the opposite direction") {
  const Point3h p(0, 1, 0);
  Transform3h t;
  t = RotX(kPi / 2).inverse();
  CHECK((t * p).isApprox(Point3h(0, 0, -1)));
}

SCENARIO("A shearing transformation moves x in proportion to y") {
  Transform3h t;
  t = Shearing3(1, 0, 0, 0, 0, 0);
  const Point3h p(2, 3, 4);
  CHECK(t * p == Point3h(5, 3, 4));
}

SCENARIO("Individual transformations are applied in sequence") {
  const Point3h p(1, 0, 1);
  Transform3h A;
  A = RotX(kPi / 2);
  Transform3h B;
  B = Scaling3(5, 5, 5);
  Transform3h C;
  C = Translation3(10, 5, 7);

  const Point3h p2 = A * p;
  CHECK(p2.isApprox(Point3h(1, -1, 0)));

  const Point3h p3 = B * p2;
  CHECK(p3.isApprox(Point3h(5, -5, 0)));

  const Point3h p4 = C * p3;
  CHECK(p4.isApprox(Point3h(15, 0, 7)));
}

SCENARIO("Chained transformations must be applied in reverse order") {
  const Point3h p(1, 0, 1);
  Transform3h t;
  t = Translation3(10, 5, 7) * Scaling3(5, 5, 5) * RotX(kPi / 2);
  CHECK(t * p == Point3h(15, 0, 7));
}

SCENARIO("Chained transformations must be applied in reverse order, API2") {
  const Point3h p(1, 0, 1);
  Transform3h t = Transform3h::Identity();
  t.translate(Vector3(10, 5, 7)).scale(Vector3(5, 5, 5)).rotate(RotX(kPi / 2));
  CHECK(t * p == Point3h(15, 0, 7));
}

SCENARIO("The transformation matrix for the default orientation") {
  const Point3h from(0, 0, 0);
  const Point3h to(0, 0, -1);
  const Vector3h up(0, 1, 0);
  CHECK(ViewTransform(from, to, up).matrix() ==
        Transform3h::Identity().matrix());
}

SCENARIO("A view transformation matrix looking in positive z direction") {
  const Point3h from(0, 0, 0);
  const Point3h to(0, 0, 1);
  const Vector3h up(0, 1, 0);
  CHECK(ViewTransform(from, to, up).matrix() ==
        MakeTransform3h(Scaling3(-1, 1, -1)).matrix());
}

SCENARIO("The view transformation moves the world") {
  const Point3h from(0, 0, 8);
  const Point3h to(0, 0, 0);
  const Vector3h up(0, 1, 0);
  CHECK(ViewTransform(from, to, up).matrix() ==
        MakeTransform3h(Translation3(0, 0, -8)).matrix());
}

SCENARIO("An arbitrary view transformation") {
  const Point3h from(1, 3, 2);
  const Point3h to(4, -2, 8);
  const Vector3h up(1, 1, 0);
  Matrix4 m;
  m << -0.507092552837, 0.507092552837, 0.676123403783, -2.366431913240,
      0.767715933860, 0.606091526731, 0.121218305346, -2.828427124746,
      -0.358568582800, 0.597614304667, -0.717137165601, 0.000000000000, 0, 0, 0,
      1;
  CHECK(ViewTransform(from, to, up).matrix().isApprox(m));
}

}  // namespace
}  // namespace sv::sim