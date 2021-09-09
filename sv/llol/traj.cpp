#include "sv/llol/traj.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Quaterniond = Eigen::Quaterniond;

Trajectory::Trajectory(int size, const TrajectoryParams& params)
    : gravity_norm{params.gravity_norm},
      integrate_acc{params.integrate_acc},
      update_acc_bias{params.update_acc_bias} {
  CHECK_GT(size, 0);
  states.resize(size);
}

std::string Trajectory::Repr() const {
  return fmt::format(
      "Trajectory(size={}, integrate_acc={}, update_acc_bias={}, g_pano=[{}], "
      "g_norm={:.4f}, "
      "\nT_imu_lidar=\n{}\n"
      "T_odom_pano=\n{}\n)",
      size(),
      integrate_acc,
      update_acc_bias,
      g_pano.transpose(),
      gravity_norm,
      T_imu_lidar.matrix3x4(),
      T_odom_pano.matrix3x4());
}

void Trajectory::Init(const SE3d& tf_i_l, const Vector3d& acc) {
  CHECK(!states.empty());

  T_imu_lidar = tf_i_l;
  // set all states to T_l_i since we want first sweep frame to align with pano
  const auto tf_l_i = tf_i_l.inverse();
  for (auto& st : states) {
    st.rot = tf_l_i.so3();
    st.pos = tf_l_i.translation();
  }

  // We want to initialized gravity vector with first imu measurement but it
  // should be in pano frame
  Vector3d g_i = acc;
  if (gravity_norm > 0) {
    g_i = g_i.normalized() * gravity_norm;
  } else {
    gravity_norm = g_i.norm();
  }
  g_pano = T_imu_lidar.so3().inverse() * g_i;
  T_odom_pano.so3().setQuaternion(
      Quaterniond::FromTwoVectors(Vector3d::UnitZ(), g_i));
}

int Trajectory::PredictNew(const ImuQueue& imuq, double t0, double dt, int n) {
  CHECK_GT(dt, 0);

  // At the beginning of predict, the starting state of the trajectory is the
  // starting col of the previous grid. Since we add a new scan to the grid, the
  // starting point is shifted by n. We need to adjust the states such that the
  // new starting point matches the current grid.
  // In the case of adding a full sweep, this will just make the last state the
  // new beginning state
  PopOldest(n);

  // Find the first imu from buffer that is right after t0
  int ibuf = imuq.IndexAfter(t0);

  const int ibuf0 = ibuf;

  // Find the state to start prediction
  const int ist0 = size() - n - 1;
  // update the time of the state where we will start the prediction
  states.at(ist0).time = t0;
  const auto& st0 = At(ist0);

  auto imu0 = imuq.DebiasedAt(ibuf - 1);
  auto imu1 = imuq.DebiasedAt(ibuf);

  for (int ist = ist0 + 1; ist < size(); ++ist) {
    // time of the ith state
    const auto ti = t0 + dt * (ist - ist0);

    // increment ibuf if it is ealier than current cell end time
    if (imu1.time < ti && ibuf < imuq.size() - 1) {
      ++ibuf;
      imu0 = imu1;
      imu1 = imuq.DebiasedAt(ibuf);
    }

    const auto& prev = At(ist - 1);
    auto& curr = At(ist);

    if (integrate_acc) {
      IntegrateState(prev, imu0, imu1, g_pano, dt, curr);
    } else {
      curr.time = prev.time + dt;
      curr.vel = prev.vel;
      curr.pos = prev.pos + prev.vel * dt;
      curr.rot = IntegrateRot(prev.rot, prev.time, imu0, imu1, dt);
    }
  }

  return ibuf - ibuf0 + 1;
}

int Trajectory::PredictFull(const ImuQueue& imuq) {
  int ibuf = imuq.IndexAfter(front().time);
  const int ibuf0 = ibuf;

  auto imu0 = imuq.DebiasedAt(ibuf - 1);
  auto imu1 = imuq.DebiasedAt(ibuf);

  for (int ist = 1; ist < size(); ++ist) {
    const auto& prev = At(ist - 1);
    auto& curr = At(ist);
    const auto dt = curr.time - prev.time;

    // time of the ith state
    // increment ibuf if it is ealier than current cell end time
    if (imu1.time < curr.time && ibuf < imuq.size() - 1) {
      ++ibuf;
      imu0 = imu1;
      imu1 = imuq.DebiasedAt(ibuf);
    }

    if (integrate_acc) {
      IntegrateState(prev, imu0, imu1, g_pano, dt, curr);
    } else {
      curr.time = prev.time + dt;
      curr.vel = prev.vel;
      curr.pos = prev.pos + prev.vel * dt;
      curr.rot = IntegrateRot(prev.rot, prev.time, imu0, imu1, dt);
    }
  }

  return ibuf - ibuf0 + 1;
}

void Trajectory::PopOldest(int n) {
  CHECK_LE(0, n);
  CHECK_LT(n, size());
  std::rotate(states.begin(), states.begin() + n, states.end());
}

void Trajectory::MoveFrame(const Sophus::SE3d& tf_p2_p1) {
  // 1. Update all states into the new pano frame (identity)
  // T_p2_i = T_p2_p1 * T_p1_i
  //  = [R_21, p_21] * [R_1i, p_1i] = [R_21 * R_1i, R_21 * p_1i + p_21]
  const auto& R_p2_p1 = tf_p2_p1.so3();
  for (auto& st : states) {
    st.rot = R_p2_p1 * st.rot;
    st.vel = R_p2_p1 * st.vel;  // ignores angular velocity
    st.pos = R_p2_p1 * st.pos + tf_p2_p1.translation();
  }
  // 2. Update T_odom_pano
  // T_o_p2 = T_o_p1 * T_p1_p2
  T_odom_pano = T_odom_pano * tf_p2_p1.inverse();
  // 3. Update gravity in pano frame (rotation only)
  // g_p2 = R_p2_p1 * g_p1
  g_pano = R_p2_p1 * g_pano;
}

int Trajectory::UpdateBias(ImuQueue& imuq) {
  const auto t0 = states.front().time;
  const auto t1 = states.back().time;
  const auto dt_state = (t1 - t0) / (states.size() - 1);
  CHECK_GT(dt_state, 0);

  // Find next imu
  int ibuf = imuq.IndexAfter(t0);
  if (ibuf == imuq.size()) return 0;

  MeanVar3d bw{};
  MeanVar3d ba{};

  while (ibuf < imuq.size()) {
    // Get an imu and try to find its left and right state
    const auto& imu = imuq.RawAt(ibuf);
    const int ist = (imu.time - t0) / dt_state;
    // +2 to ignore last state
    if (ist + 2 >= states.size()) break;

    const auto& st0 = states.at(ist);
    const auto& st1 = states.at(ist + 1);

    // Compute expected gyr measurement by finite difference
    const auto R0_t = st0.rot.inverse();
    const Vector3d w_b = (R0_t * st1.rot).log() / dt_state;
    // w_t = w_m - b_w -> b_w = w_m - w_t
    bw.Add(imu.gyr - w_b);

    // Compute expected acc_w by finite difference
    const Vector3d a_w = (st1.vel - st0.vel) / dt_state;
    // Transform to expected body acc
    const Vector3d a_b = R0_t * (a_w + g_pano);
    ba.Add(imu.acc - a_b);

    ++ibuf;
  }

  imuq.bias.UpdateGyr(bw.mean, bw.Var());
  if (update_acc_bias) {
    imuq.bias.UpdateAcc(ba.mean, ba.Var());
  }
  return bw.n;
}

SE3d Trajectory::TfPanoLidar() const {
  const Sophus::SE3d T_pano_imu{states.back().rot, states.back().pos};
  return T_pano_imu * T_imu_lidar;
}

SE3d Trajectory::TfOdomLidar() const { return T_odom_pano * TfPanoLidar(); }

}  // namespace sv
