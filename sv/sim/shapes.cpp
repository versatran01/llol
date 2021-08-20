#include "sv/sim/shapes.h"

namespace sv::sim {

static constexpr scalar kBig = 100.0;
static constexpr scalar kSmall = 0.01;

/// Sphere
Intersection Sphere::LocalIntersect(const Ray& ray) const noexcept {
  const Vector3h sphere_to_ray = ray.origin - Point3h::Zero();
  const scalar a = SquaredNorm(ray.direction);  // > 0
  const scalar b = Dot(ray.direction, sphere_to_ray) * 2;
  const scalar c = SquaredNorm(sphere_to_ray) - 1;
  const auto discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return Miss();

  const auto sqrt_d = std::sqrt(discriminant);  // >= 0
  const auto two_a = 2 * a;
  const auto t1 = (-b - sqrt_d) / two_a;
  const auto t2 = (-b + sqrt_d) / two_a;

  if (t1 >= 0) return Hit(t1);
  if (t2 >= 0) return Hit(t2);
  return Miss();
  // return {{t1, this}, {t2, this}};
}

/// Cube
Intersection Cube::LocalIntersect(const Ray& ray) const noexcept {
  const auto xt = CheckAxis(ray.origin.x(), ray.direction.x());
  const auto yt = CheckAxis(ray.origin.y(), ray.direction.y());
  const auto zt = CheckAxis(ray.origin.z(), ray.direction.z());

  const auto tmin = std::max(xt(0), std::max(yt(0), zt(0)));
  const auto tmax = std::min(xt(1), std::min(yt(1), zt(1)));

  if (tmin > tmax) return Miss();
  if (tmin > 0) return Hit(tmin);
  if (tmax > 0) return Hit(tmax);
  return Miss();
}

Vector3h Cube::LocalNormalAt(const Point3h& point) const noexcept {
  const Point3h abs = point.cwiseAbs();
  const auto maxc = std::max(abs.x(), std::max(abs.y(), abs.z()));

  if (maxc == abs.x()) {
    return {point.x(), 0, 0};
  } else if (maxc == abs.y()) {
    return {0, point.y(), 0};
  } else {
    return {0, 0, point.z()};
  }
}

Vector2 Cube::CheckAxis(scalar origin, scalar direction) noexcept {
  const auto tmin_num = -1.0 - origin;
  const auto tmax_num = 1.0 - origin;
  auto tmin = tmin_num / direction;
  auto tmax = tmax_num / direction;
  if (tmin > tmax) std::swap(tmin, tmax);
  return {tmin, tmax};
}

/// Square
Intersection Square::LocalIntersect(const Ray& ray) const noexcept {
  const auto i = Plane::LocalIntersect(ray);
  if (!i.missed()) {
    const auto point = ray.Position(i.t);
    if (std::abs(point.x()) <= 1 && std::abs(point.y()) <= 1) {
      return i;
    }
  }
  return Miss();
}

/// Disk
Intersection Disk::LocalIntersect(const Ray& ray) const noexcept {
  const auto i = Plane::LocalIntersect(ray);
  if (!i.missed()) {
    const auto point = ray.Position(i.t);
    const scalar r2 = point.head<2>().squaredNorm();
    if (r2 <= 1) return i;
  }
  return Miss();
}

/// Cylinder
Intersection Cylinder::LocalIntersect(const Ray& ray) const noexcept {
  const auto a = ray.direction.head<2>().squaredNorm();  // >=0

  // ray is parallel to the z axis, just need to check cap
  if (a < kEps) {
    const auto t = IntersectCap(ray);
    if (t >= 0) return Hit(t);
    return Miss();
  }

  const auto b = 2 * ray.origin.x() * ray.direction.x() +
                 2 * ray.origin.y() * ray.direction.y();
  const auto c = ray.origin.head<2>().squaredNorm() - 1;

  const auto disc = b * b - 4 * a * c;
  if (disc < 0) return Miss();

  const auto sqrt_disc = std::sqrt(disc);  // >=0

  auto t0 = (-b - sqrt_disc) / (2 * a);
  auto t1 = (-b + sqrt_disc) / (2 * a);
  // t0 <= t1

  if (t0 > t1) std::swap(t0, t1);

  // intersect side of cylinder
  scalar t_a = Intersection::kMiss;
  if (t0 >= 0) {
    const auto z0 = ray.origin.z() + t0 * ray.direction.z();
    if (std::abs(z0) < 1) t_a = t0;
  } else if (t1 >= 0) {
    const auto z1 = ray.origin.z() + t1 * ray.direction.z();
    if (std::abs(z1) < 1) t_a = t1;
  }

  auto t_b = IntersectCap(ray);
  if (t_a > t_b) std::swap(t_a, t_b);

  if (t_a >= 0) return Hit(t_a);
  if (t_b >= 0) return Hit(t_b);
  return Miss();
}

scalar Cylinder::IntersectCap(const Ray& ray) const noexcept {
  // ray parallel to cap
  if (std::abs(ray.direction.z()) < kEps) return Intersection::kMiss;

  // Check for an intersection with the lower end cap by intersecting the
  // ray with the plane at z = -1
  auto t0 = (-1 - ray.origin.z()) / ray.direction.z();
  auto t1 = (1 - ray.origin.z()) / ray.direction.z();
  if (t0 > t1) std::swap(t0, t1);

  if (t0 >= 0 && CheckCap(ray.Position(t0))) return t0;
  if (t1 >= 0 && CheckCap(ray.Position(t1))) return t1;
  return Intersection::kMiss;
}

Vector3h Cylinder::LocalNormalAt(const Point3h& point) const noexcept {
  // Compute the square of the distance from the y axis
  const auto r2 = point.head<2>().squaredNorm();

  if (r2 < 1 && point.z() >= (1 - kEps)) {
    return {0, 0, 1};
  } else if (r2 < 1 && point.z() <= (-1 + kEps)) {
    return {0, 0, -1};
  } else {
    return {point.x(), point.y(), 0};
  }
}

}  // namespace sv::sim
