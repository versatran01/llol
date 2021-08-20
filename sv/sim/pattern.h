#pragma once

#include "sv/sim/transform.h"

namespace sv::sim {

// fwd
struct Shape;

struct Pattern {
  enum struct Type { FLAT, STRIPE, GRADIENT, RING, CHECKER };

  Pattern() = default;
  Pattern(Type type, scalar a = 1.0, scalar b = 0.0) noexcept
      : type(type), a(a), b(b) {}

  static Pattern Flat() noexcept { return {Type::FLAT}; }
  static Pattern Stripe() noexcept { return {Type::STRIPE}; }
  static Pattern Gradient() noexcept { return {Type::GRADIENT}; }
  static Pattern Ring() noexcept { return {Type::RING}; }
  static Pattern Checker() noexcept { return {Type::CHECKER}; }

  scalar ColorAtShape(const Shape& shape, const Point3h& point) const;
  scalar ColorAt(const Point3h& point) const;

  scalar ColorAtStripe(const Point3h& point) const;
  scalar ColorAtGradient(const Point3h& point) const;
  scalar ColorAtRing(const Point3h& point) const;
  scalar ColorAtChecker(const Point3h& point) const;

  Type type{Type::FLAT};
  scalar a{1.0};
  scalar b{0.0};
  Transform3h transform;
};

}  // namespace sv::sim
