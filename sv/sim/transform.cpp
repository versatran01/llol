#include "sv/sim/transform.h"

namespace sv::sim {

Matrix3 Shearing3(
    scalar xy, scalar xz, scalar yx, scalar yz, scalar zx, scalar zy) {
  Matrix3 s = Matrix3::Identity();
  s(0, 1) = xy;
  s(0, 2) = xz;
  s(1, 0) = yx;
  s(1, 2) = yz;
  s(2, 0) = zx;
  s(2, 1) = zy;
  return s;
}

Affine3 ViewTransform(const Point3h& from,
                      const Point3h& to,
                      const Vector3h& up) {
  const Vector3h forward = Normalized(to - from);
  const Vector3h upn = Normalized(up);
  const Vector3h left = Cross(forward, upn);
  const Vector3h true_up = Cross(left, forward);

  Matrix3 orientation;
  orientation.row(0) = left.head<3>();
  orientation.row(1) = true_up.head<3>();
  orientation.row(2) = -forward.head<3>();

  Affine3 t;
  t = orientation * Translation3(-from.head<3>());
  return t;
}

}  // namespace sv::sim