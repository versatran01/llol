#include "sv/sim/material.h"

#include "sv/sim/shape.h"

namespace sv::sim {

scalar Material::Lighting(const Point3h& light,
                          const Shape& shape,
                          const Point3h& point,
                          const Vector3h& normal,
                          scalar intensity) const {
  const auto material_color =
      pattern ? pattern->ColorAtShape(shape, point) : reflectivity;
  const auto effective_color = material_color * intensity;

  // find the direction of the light source, which is the opposite of the ray
  // direction
  const Vector3h lightv = Normalized(light - point);

  // compute the ambient contribution
  const auto ambient_color = effective_color * ambient;

  // light_dot_normal represents the cosine of the angle between the light
  // vector and the normal vector
  // A negative number means the light is on the other side of the surface
  const auto light_dot_normal = Dot(lightv, normal);
  if (light_dot_normal < 0) return ambient_color;

  // comput ethe diffuse contribution
  const auto diffuse_color = effective_color * diffuse * light_dot_normal;
  return diffuse_color + ambient_color;
}

}  // namespace sv::sim
