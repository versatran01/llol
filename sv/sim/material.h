#pragma once

#include <optional>

#include "sv/sim/pattern.h"  // Pattern
#include "sv/sim/tuple.h"    // Point3h, Vector3h

namespace sv::sim {

// fwd
struct Shape;

/// This is specific for lidar only
struct Material {
  scalar reflectivity{1.0};        // reflectivity of the material
  scalar ambient{0.4};             // background lighting, constant
  scalar diffuse{0.6};             // reflected from a matte surface
  std::optional<Pattern> pattern;  // replace relectivity

  // NOTE
  Color4 color;  // color for rviz, not used by render

  /// Ignores shadow, reflection and assumes light at the same point as eye
  scalar Lighting(const Point3h& light,
                  const Shape& shape,
                  const Point3h& point,
                  const Vector3h& normal,
                  scalar intensity = 1.0) const noexcept;
};

}  // namespace sv::sim
