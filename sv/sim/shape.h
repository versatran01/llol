#pragma once

#include <memory>

#include "sv/sim/intersect.h"
#include "sv/sim/material.h"
#include "sv/sim/transform.h"

namespace sv::sim {

struct Shape {
  inline static int counter = 0;
  enum struct Type { BASE, SPHERE, PLANE, CUBE, CYLINDER, CONE, GROUP };

  Shape() = default;
  virtual ~Shape() noexcept = default;

  explicit Shape(Type type) : type(type) {}
  Shape(Type type, const SE3& pose, const Vector3& scale)
      : type(type), transform(pose, scale) {}

  friend bool operator==(const Shape& lhs, const Shape& rhs) noexcept {
    return lhs.id == rhs.id;
  }

  Intersection Intersect(const Ray& ray) const {
    return LocalIntersect(ray.Transformed(transform.inverse()));
  }

  Vector3h NormalAt(const Point3h& point) const;

  Point3h World2Object(const Point3h& point) const;
  Vector3h Normal2World(const Vector3h& normal) const;

  virtual Intersection LocalIntersect(const Ray&) const { return {}; }
  virtual Vector3h LocalNormalAt(const Point3h&) const { return {}; }

  Intersection MakeIntersection(scalar t) const { return {t, this}; }

  int id{counter++};
  Type type{Type::BASE};
  Transform3h transform{Transform3h::Identity()};
  Material material;
};

class PolyShape {
 public:
  PolyShape() = default;

  template <typename T>
  PolyShape(T x) : self_(std::make_shared<T>(std::move(x))) {}

  Intersection Intersect(const Ray& ray) const { return self_->Intersect(ray); }

  Intersection MakeIntersection(scalar t) const {
    return self_->MakeIntersection(t);
  }

  Vector3h NormalAt(const Point3h& point) const {
    return self_->NormalAt(point);
  }

  Shape::Type type() const { return self_->type; }
  int id() const { return self_->id; }
  const Material& material() const { return self_->material; }

 protected:
  std::shared_ptr<const Shape> self_{nullptr};
};

}  // namespace sv::sim
