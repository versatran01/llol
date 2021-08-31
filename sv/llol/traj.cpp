#include "sv/llol/traj.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;

std::string Trajectory::Repr() const {
  return fmt::format(
      "Trajectory(size={}, g_pano=[{}], \nT_imu_lidar=\n{}\n, "
      "T_odom_pano=\n{}\n)",
      size(),
      g_pano.transpose(),
      T_imu_lidar.matrix3x4(),
      T_odom_pano.matrix3x4());
}

void Trajectory::InitExtrinsic(const SE3d& T_i_l,
                               const Vector3d& acc,
                               double g_norm) {
  CHECK(!states.empty());

  T_imu_lidar = T_i_l;
  // set all states to T_l_i since we want first sweep frame to align with pano
  const auto T_l_i = T_i_l.inverse();
  for (auto& s : states) {
    s.rot = T_l_i.so3();
    s.pos = T_l_i.translation();
  }

  // We want to initialized gravity vector with first imu measurement but it
  // should be in pano frame
  const Vector3d g_i = acc.normalized() * g_norm;
  g_pano = T_imu_lidar.so3().inverse() * g_i;
  T_odom_pano.so3().setQuaternion(
      Quaterniond::FromTwoVectors(Vector3d::UnitZ(), g_i));
}

int Trajectory::Predict(const ImuQueue& imuq, double t0, double dt, int n) {
  // Find the first imu from buffer that is right after t0
  int ibuf = imuq.IndexAfter(t0);
  CHECK_GE(ibuf, 0);
  const int ibuf0 = ibuf;

  // Now try to fill in the last n poses
  const int ist0 = size() - n - 1;
  CHECK_GE(ist0, 0);
  At(ist0).time = t0;

  for (int i = ist0 + 1; i < size(); ++i) {
    const auto ti = t0 + dt * i;
    // increment ibuf if it is ealier than current cell time
    if (imuq.RawAt(ibuf).time < ti) {
      ++ibuf;
    }
    // make sure it is always valid
    if (ibuf >= imuq.size()) {
      ibuf = imuq.size() - 1;
    }

    const auto imu = imuq.DebiasedAt(ibuf);

    // TODO (chao): for now assume translation stays the same
    const auto& prev = At(i - 1);
    auto& curr = At(i);
    //    IntegrateEuler(prev, imu, gravity, dt, curr);
    curr.time = prev.time + dt;
    // For do not propagate translation
    curr.pos = At(ist0).pos;
    curr.rot = prev.rot * SO3d::exp(imu.gyr * dt);
  }

  return ibuf - ibuf0 + 1;
}

void Trajectory::Rotate(int n) {
  CHECK_LE(0, n);
  CHECK_LT(n, size());
  std::rotate(states.begin(), states.begin() + n, states.end());
}

}  // namespace sv
