#include "sv/llol/imu.h"

#include <glog/logging.h>

namespace sv {

using Vector3d = Eigen::Vector3d;

static const double kEps = Sophus::Constants<double>::epsilon();

Sophus::SO3d IntegrateRot(const Sophus::SO3d& R0,
                          const Vector3d& omg,
                          double dt) {
  CHECK_GT(dt, 0);
  return R0 * Sophus::SO3d::exp(dt * omg);
}

NavState IntegrateEuler(const NavState& s0,
                        const ImuData& imu,
                        const Vector3d& g_w,
                        double dt) {
  CHECK_GT(dt, 0);
  NavState s1 = s0;
  // t
  s1.time = s0.time + dt;

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
  const auto dt = imu1.time - imu0.time;
  CHECK_GT(dt, 0);

  // t
  s1.time = s0.time + dt;

  // gyro
  const auto omg_b = (imu0.gyr + imu1.gyr) * 0.5;
  s1.rot = IntegrateRot(s0.rot, omg_b, dt);

  // acc
  const Vector3d a0 = s0.rot * imu0.acc;
  const Vector3d a1 = s1.rot * imu1.acc;
  const Vector3d a = (a0 + a1) * 0.5 + g_w;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt * dt;

  return s1;
}

int ImuTraj::Predict(double t0, double dt) {
  int ibuf = FindNextImu(buf, t0);
  if (ibuf < 0) return 0;  // no valid imu found

  // Now try to fill in later poses by integrating gyro only
  const int ibuf0 = ibuf;
  for (int i = 1; i < traj.size(); ++i) {
    const auto ti = t0 + dt * i;
    // increment ibuf if it is ealier than current cell time
    if (ti > buf[ibuf].time) {
      ++ibuf;
    }
    // make sure it is always valid
    if (ibuf >= buf.size()) {
      ibuf = buf.size() - 1;
    }
    const auto& imu = buf.at(ibuf);
    // Transform gyr to lidar frame
    const auto gyr_l = T_imu_lidar.so3().inverse() * imu.gyr;
    const auto omg_l = (dt * gyr_l).cast<float>();
    // TODO (chao): for now assume translation stays the same
    traj.at(i).translation() = traj.at(0).translation();
    traj.at(i).so3() = traj.at(i - 1).so3() * Sophus::SO3f::exp(omg_l);
  }

  return ibuf - ibuf0 + 1;
}

int FindNextImu(const ImuBuffer& buf, double t) {
  int ibuf = -1;
  for (int i = 0; i < buf.size(); ++i) {
    if (buf[i].time > t) {
      ibuf = i;
      break;
    }
  }
  return ibuf;
}

}  // namespace sv
