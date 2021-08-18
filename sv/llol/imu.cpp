#include "sv/llol/imu.h"

#include <glog/logging.h>

namespace sv {

using Vector3d = Eigen::Vector3d;

static const double kEps = Sophus::Constants<double>::epsilon();

Sophus::SO3d IntegrateRot(const Sophus::SO3d& R0,
                          const Eigen::Vector3d& omg,
                          double dt) {
  return R0 * Sophus::SO3d::exp(omg * dt);
}

NavState IntegrateEuler(const NavState& s0,
                        const ImuData& imu,
                        const Vector3d& g_w,
                        double dt) {
  NavState s1 = s0;
  // t
  s1.t = s0.t + dt;

  // gyr
  s1.rot = IntegrateRot(s0.rot, imu.gyr, dt);

  // acc
  // transform to worl frame
  const Vector3d a = s0.rot * imu.acc + g_w;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt * dt;

  return s1;
}

NavState IntegrateMidpoint(const NavState& s0,
                           const ImuData& imu0,
                           const ImuData& imu1,
                           const Vector3d& g_w) {
  NavState s1 = s0;
  const auto dt = imu1.t - imu0.t;
  // t
  s1.t = s0.t + dt;

  // gyro
  const auto omg_b = (imu0.gyr + imu1.gyr) / 2.0;
  s1.rot = IntegrateRot(s0.rot, omg_b, dt);

  // acc
  const Vector3d a0 = s0.rot * imu0.acc;
  const Vector3d a1 = s1.rot * imu1.acc;
  const Vector3d a = (a0 + a1) / 2.0 + g_w;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt * dt;

  return s1;
}

}  // namespace sv
