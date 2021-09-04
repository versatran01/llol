#include "sv/llol/traj.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Quaterniond = Eigen::Quaterniond;

Trajectory::Trajectory(int size, bool use_acc) : use_acc_{use_acc} {
  states.resize(size);
}

std::string Trajectory::Repr() const {
  return fmt::format(
      "Trajectory(size={}, use_acc={}, g_pano=[{}], \nT_imu_lidar=\n{}\n, "
      "T_odom_pano=\n{}\n)",
      size(),
      use_acc_,
      g_pano.transpose(),
      T_imu_lidar.matrix3x4(),
      T_odom_pano.matrix3x4());
}

void Trajectory::Init(const SE3d& tf_i_l, const Vector3d& acc, double g_norm) {
  CHECK(!states.empty());

  T_imu_lidar = tf_i_l;
  // set all states to T_l_i since we want first sweep frame to align with pano
  const auto tf_l_i = tf_i_l.inverse();
  for (auto& s : states) {
    s.rot = tf_l_i.so3();
    s.pos = tf_l_i.translation();
  }

  // We want to initialized gravity vector with first imu measurement but it
  // should be in pano frame
  const Vector3d g_i = acc.normalized() * g_norm;
  g_pano = T_imu_lidar.so3().inverse() * g_i;
  T_odom_pano.so3().setQuaternion(
      Quaterniond::FromTwoVectors(Vector3d::UnitZ(), g_i));
}

int Trajectory::Predict(const ImuQueue& imuq, double t0, double dt, int n) {
  // At the beginning of predict, the starting state of the trajectory is the
  // starting col of the previous grid. Since we add a new scan to the grid, the
  // starting point is shifted by n. We need to adjust the states such that the
  // new starting point matches the current grid.
  // In the case of adding a full sweep, this will just make the last state the
  // new beginning state
  PopOldest(n);

  // Find the first imu from buffer that is right after t0
  int ibuf = imuq.IndexAfter(t0);
  CHECK_GT(ibuf, 0) << fmt::format(
      "No imu found right before {}. Imu buffer size is {}, and the first imu "
      "in buffer has time {}",
      t0,
      imuq.size(),
      imuq.RawAt(0).time);

  const int ibuf0 = ibuf;

  // Find the state to start prediction
  const int ist0 = size() - n - 1;
  CHECK_GE(ist0, 0);
  states.at(ist0).time = t0;  // update its time
  const auto& st0 = At(ist0);

  auto imu0 = imuq.DebiasedAt(ibuf - 1);
  auto imu1 = imuq.DebiasedAt(ibuf);

  for (int i = ist0 + 1; i < size(); ++i) {
    const auto ti = t0 + dt * i;
    // increment ibuf if it is ealier than current cell end time
    if (imuq.RawAt(ibuf).time < ti && ibuf < imuq.size() - 1) {
      ++ibuf;
      imu0 = imu1;
      imu1 = imuq.DebiasedAt(ibuf);
    }

    const auto& prev = At(i - 1);
    auto& curr = At(i);

    if (use_acc_) {
      IntegrateEuler(prev, imu1, g_pano, dt, curr);
    } else {
      curr.time = prev.time + dt;
      // do not propagate translation
      curr.pos = st0.pos;
      curr.vel = st0.vel;
      curr.rot = IntegrateRot(prev.rot, imu0, imu1, prev.time, dt);
    }
  }

  return ibuf - ibuf0 + 1;
}

void Trajectory::PopOldest(int n) {
  CHECK_LE(0, n);
  CHECK_LT(n, size());
  std::rotate(states.begin(), states.begin() + n, states.end());
}

void Trajectory::Update(const Sophus::SE3d& tf_p2_p1) {
  // 1. Update all states into the new pano frame (identity)
  // T_p2_i = T_p2_p1 * T_p1_i
  //  = [R_21, p_21] * [R_1i, p_1i] = [R_21 * R_1i, R_21 * p_1i + p_21]
  for (auto& st : states) {
    st.rot = tf_p2_p1.so3() * st.rot;
    st.pos = tf_p2_p1.so3() * st.pos + tf_p2_p1.translation();
  }
  // 2. Update T_odom_pano
  // T_o_p2 = T_o_p1 * T_p1_p2
  T_odom_pano = T_odom_pano * tf_p2_p1.inverse();
  // 3. Update gravity in pano frame (rotation only)
  // g_p2 = R_p2_p1 * g_p1
  g_pano = tf_p2_p1.so3() * g_pano;
}

SE3d Trajectory::TfPanoLidar() const {
  const Sophus::SE3d T_pano_imu{states.back().rot, states.back().pos};
  return T_pano_imu * T_imu_lidar;
}

SE3d Trajectory::TfOdomLidar() const { return T_odom_pano * TfPanoLidar(); }

}  // namespace sv
