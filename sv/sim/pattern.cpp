#include "sv/sim/pattern.h"

#include "sv/sim/shape.h"

namespace sv::sim {

scalar Pattern::ColorAtShape(const Shape& shape, const Point3h& point) const {
  const Point3h object_point = shape.transform.inverse() * point;
  const Point3h pattern_point = transform.inverse() * object_point;
  return ColorAt(pattern_point);
}

scalar Pattern::ColorAt(const Point3h& point) const {
  if (type == Type::STRIPE) {
    return ColorAtStripe(point);
  } else if (type == Type::GRADIENT) {
    return ColorAtGradient(point);
  } else if (type == Type::RING) {
    return ColorAtRing(point);
  } else if (type == Type::CHECKER) {
    return ColorAtChecker(point);
  } else {
    // BASE
    return (a + b) / 2.0;
  }
}

scalar Pattern::ColorAtStripe(const Point3h& point) const {
  if (static_cast<int>(std::floor(point.x())) % 2 == 0) return a;
  return b;
}

scalar Pattern::ColorAtGradient(const Point3h& point) const {
  const auto distance = b - a;
  const auto fraction = point.x() - std::floor(point.x());
  return a + distance * fraction;
}

scalar Pattern::ColorAtRing(const Point3h& point) const {
  const int f = std::floor(std::hypot(point.x(), point.y()));
  return f % 2 == 0 ? a : b;
}

scalar Pattern::ColorAtChecker(const Point3h& point) const {
  const int f =
      std::floor(point.x()) + std::floor(point.y()) + std::floor(point.z());
  return f % 2 == 0 ? a : b;
}

}  // namespace sv::sim
