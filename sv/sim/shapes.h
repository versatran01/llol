#pragma once

#include "sv/sim/shape.h"

namespace sv::sim {

/// Unit sphere at origin
struct Sphere final : public Shape {
  using Shape::Shape;

  Sphere() noexcept : Shape(Type::SPHERE) {}

  Sphere(double radius, const Vector3& origin) noexcept
      : Shape(Type::SPHERE, {SO3{}, origin}, Vector3::Constant(radius)) {}

  /// Derived
  Intersection LocalIntersect(const Ray& ray) const override;
  Vector3h LocalNormalAt(const Point3h& point) const override {
    return point - Point3h::Zero();
  }
};

/// Plane in XY-plane
struct Plane : public Shape {
  using Shape::Shape;

  Plane(Type type = Type::PLANE) noexcept : Shape(type) {}

  Intersection LocalIntersect(const Ray& ray) const noexcept override {
    if (std::abs(ray.direction.z()) < kEps) return Intersection::Miss();
    return Hit(-ray.origin.z() / ray.direction.z());
  }

  Vector3h LocalNormalAt(const Point3h&) const noexcept override {
    return {0, 0, 1};
  }
};

/// Square in XY
struct Square final : public Plane {
  using Plane::Plane;

  Square() noexcept : Plane(Type::SQUARE) {}

  Intersection LocalIntersect(const Ray& ray) const noexcept override;
};

/// Disk in XY
struct Disk final : public Plane {
  using Plane::Plane;

  Disk() noexcept : Plane(Type::DISK) {}

  Intersection LocalIntersect(const Ray& ray) const noexcept override;
};

/// Cube
struct Cube final : public Shape {
  using Shape::Shape;

  Cube() noexcept : Shape(Type::CUBE) {}

  Intersection LocalIntersect(const Ray& ray) const noexcept override;
  Vector3h LocalNormalAt(const Point3h& point) const noexcept override;

  static Vector2 CheckAxis(double origin, double direction) noexcept;
};

/// Cylinder
struct Cylinder final : public Shape {
  using Shape::Shape;

  Cylinder() noexcept : Shape(Type::CYLINDER) {}

  Intersection LocalIntersect(const Ray& ray) const noexcept override;
  Vector3h LocalNormalAt(const Point3h& point) const noexcept override;

  double IntersectCap(const Ray& ray) const noexcept;
  static bool CheckCap(const Point3h& point) noexcept {
    return (point.head<2>().squaredNorm()) <= 1;
  }
};

}  // namespace sv::sim
