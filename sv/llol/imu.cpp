#include "sv/llol/imu.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"  // Hat3

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;

std::string NavState::Repr() const {
  return fmt::format("NavState(t={}, rot=[{}], pos=[{}], vel=[{}]",
                     time,
                     rot.unit_quaternion().coeffs().transpose(),
                     pos.transpose(),
                     vel.transpose());
}

void IntegrateEuler(const NavState& s0,
                    const ImuData& imu,
                    const Vector3d& g,
                    double dt,
                    NavState& s1) {
  CHECK_GT(dt, 0);
  // t
  s1.time = s0.time + dt;

  // gyr
  s1.rot = s0.rot * SO3d::exp(dt * imu.gyr);

  // acc
  // transform to fixed frame acc and remove gravity
  const Vector3d a = s0.rot * imu.acc - g;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt * dt;
}

NavState IntegrateMidpoint(const NavState& s0,
                           const ImuData& imu0,
                           const ImuData& imu1,
                           const Vector3d& g_w) {
  NavState s1 = s0;
  const auto dt = imu1.time - imu0.time;
  CHECK_GT(dt, 0);
  const auto dt2 = dt * dt;

  // t
  s1.time = s0.time + dt;

  // gyro
  const auto omg_b = (imu0.gyr + imu1.gyr) * 0.5;
  s1.rot = s0.rot * SO3d::exp(omg_b * dt);

  // acc
  const Vector3d a0 = s0.rot * imu0.acc;
  const Vector3d a1 = s1.rot * imu1.acc;
  const Vector3d a = (a0 + a1) * 0.5 + g_w;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt2;

  return s1;
}

int FindNextImu(const ImuBuffer& buf, double t) {
  int ibuf = -1;
  for (int i = 0; i < buf.size(); ++i) {
    if (buf.at(i).time > t) {
      ibuf = i;
      break;
    }
  }
  return ibuf;
}

void ImuTrajectory::InitExtrinsic(const SE3d& T_i_l, double gravity_norm) {
  CHECK(!states.empty());
  CHECK(!buf.empty());

  T_imu_lidar = T_i_l;
  // set all states to T_l_i since we want first sweep frame to align with pano
  const auto T_l_i = T_i_l.inverse();
  for (auto& s : states) {
    s.rot = T_l_i.so3();
    s.pos = T_l_i.translation();
  }

  // We want to initialized gravity vector with first imu measurement but it
  // should be in pano frame
  const Vector3d a_i = buf.back().acc;  // in imu frame
  const Vector3d g_i = a_i.normalized() * gravity_norm;
  gravity = T_imu_lidar.so3().inverse() * g_i;
  T_odom_pano.so3().setQuaternion(
      Quaterniond::FromTwoVectors(Vector3d::UnitZ(), g_i));
}

int ImuTrajectory::Predict(double t0, double dt, int n) {
  // Find the first imu from buffer that is right after t0
  int ibuf = FindNextImu(buf, t0);
  CHECK_GE(ibuf, 0);
  const int ibuf0 = ibuf;

  // Now try to fill in the last n poses
  const int ist0 = size() - n - 1;
  CHECK_GE(ist0, 0);
  StateAt(ist0).time = t0;

  for (int i = ist0 + 1; i < size(); ++i) {
    const auto ti = t0 + dt * i;
    // increment ibuf if it is ealier than current cell time
    if (buf[ibuf].time < ti) {
      ++ibuf;
    }
    // make sure it is always valid
    if (ibuf >= buf.size()) {
      ibuf = buf.size() - 1;
    }

    const auto imu = ImuAt(ibuf).DeBiased(bias);

    // TODO (chao): for now assume translation stays the same
    const auto& prev = StateAt(i - 1);
    auto& curr = StateAt(i);
    //    IntegrateEuler(prev, imu, gravity, dt, curr);
    curr.time = prev.time + dt;
    // For do not propagate translation
    curr.pos = StateAt(ist0).pos;
    curr.rot = prev.rot * SO3d::exp(imu.gyr * dt);
  }

  return ibuf - ibuf0 + 1;
}

ImuNoise::ImuNoise(double dt,
                   double acc_noise,
                   double gyr_noise,
                   double acc_bias_noise,
                   double gyr_bias_noise) {
  CHECK_GT(dt, 0);

  // Follows kalibr imu noise model
  // https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
  sigma2.segment<3>(kNa).setConstant(Sq(acc_noise) / dt);
  sigma2.segment<3>(kNw).setConstant(Sq(gyr_noise) / dt);
  sigma2.segment<3>(kBa).setConstant(Sq(acc_bias_noise) * dt);
  sigma2.segment<3>(kBw).setConstant(Sq(gyr_bias_noise) * dt);
}

std::string ImuNoise::Repr() const {
  return fmt::format(
      "ImuNoise(acc_cov=[{}], gyr_cov=[{}], acc_bias_cov=[{}], "
      "gyr_bias_cov=[{}])",
      sigma2.segment<3>(kNa).transpose(),
      sigma2.segment<3>(kNw).transpose(),
      sigma2.segment<3>(kBa).transpose(),
      sigma2.segment<3>(kBw).transpose());
}

void ImuPreintegration::Integrate(double dt,
                                  const ImuData& imu,
                                  const ImuNoise& noise) {
  // Assumes imu is already debiased
  CHECK_GT(dt, 0);
  const auto dt2 = dt * dt;

  const auto& a = imu.acc;
  const auto& w = imu.gyr;
  const Vector3d ga = gamma * a;

  // vins-mono eq 7
  const auto dgamma = SO3d::exp(w * dt);
  const Vector3d dbeta = ga * dt;
  const Vector3d dalpha = beta * dt + ga * dt2 * 0.5;

  // vins-mono eq 9
  // Ft =
  // [0  I        0    0   0]
  // [0  0  -R*[a]x   -R   0]
  // [0  0    -[w]x    0  -I]
  // last two rows are all zeros
  // F = I + Ft * dt
  const auto Rmat = gamma.matrix();
  F.block<3, 3>(Index::kAlpha, Index::kBeta) = kIdent3 * dt;
  F.block<3, 3>(Index::kBeta, Index::kTheta) = -Rmat * Hat3(a) * dt;
  F.block<3, 3>(Index::kBeta, Index::kBa) = -Rmat * dt;
  F.block<3, 3>(Index::kTheta, Index::kTheta) = kIdent3 - Hat3(w) * dt;
  F.block<3, 3>(Index::kTheta, Index::kBw) = -kIdent3 * dt;

  // vins-mono eq 10
  // Update covariance
  // P = F * P * F' + G * Qd * G'
  P = F * P * F.transpose();
  P.diagonal().tail<ImuNoise::kDim>() += noise.sigma2;

  // Update measurement
  alpha += dalpha;
  beta += dbeta;
  gamma *= dgamma;
  duration += dt;
  ++n;
}

// void ImuPreintegration::Integrate(double dt,
//                                  const ImuData& imu0,
//                                  const ImuData& imu1,
//                                  const ImuNoise& noise) {
//  CHECK_GT(dt, 0);
//  const auto dt2 = Sq(dt);

//  const Vector3d w = (imu0.gyr + imu1.gyr) / 2.0;
//  const auto dgamma = Sophus::SO3d::exp(w * dt);
//  const auto gamma1 = gamma * dgamma;

//  const auto& a0 = imu0.acc;
//  const Vector3d ga = (gamma * imu0.acc + gamma1 * imu1.acc) / 2.0;

//  const auto dbeta = ga * dt;
//  const auto dalpha = beta * dt + ga * dt2 * 0.5;

//  // vins-mono eq 9
//  // [0  I        0    0   0]
//  // [0  0  -R*[a]x   -R   0]
//  // [0  0    -[w]x    0  -I]
//  // last two rows are all zeros
//  const auto Rmat = gamma.matrix();
//  F.block<3, 3>(Index::ALPHA, Index::BETA) = kIdent3;
//  F.block<3, 3>(Index::BETA, Index::THETA) = -Rmat * Hat3(a0);
//  F.block<3, 3>(Index::BETA, Index::BA) = -Rmat;
//  F.block<3, 3>(Index::THETA, Index::THETA) = -Hat3(w);
//  F.block<3, 3>(Index::THETA, Index::BW) = -kIdent3;

//  // vins-mono eq 10
//  // Update covariance
//  P = F * P * F.transpose() * dt2;
//  P.diagonal().tail<12>() += noise.sigma2;

//  // Update measurement
//  alpha += dalpha;
//  beta += dbeta;
//  gamma *= dgamma;
//  duration += dt;
//  ++n;
//}

int ImuPreintegration::Compute(const ImuTrajectory& traj) {
  const auto t0 = traj.states.front().time;
  const auto t1 = traj.states.back().time;
  CHECK_LT(t0, t1);

  const int ibuf0 = FindNextImu(traj.buf, t0);
  CHECK_LE(0, ibuf0);

  // Keep integrate till we reach either the last imu or one right before t1
  double t = t0;
  int ibuf = ibuf0;

  while (true) {
    // This imu must exist
    const auto imu = traj.ImuAt(ibuf).DeBiased(traj.bias);
    Integrate(imu.time - t, imu, traj.noise);
    t = imu.time;

    // stop if we are at the last imu
    if (ibuf + 1 == traj.buf.size()) break;
    // or if next imu time is later than t1
    if (traj.ImuAt(ibuf + 1).time >= t1) break;
    // above ensures that ibuf should be valid

    ++ibuf;
  }

  // Use the imu at ibuf to finish integrating to t1
  const auto imu = traj.ImuAt(ibuf).DeBiased(traj.bias);
  Integrate(t1 - imu.time, imu, traj.noise);

  // Compute sqrt info
  U = MatrixSqrtUtU(P.inverse().eval());

  return n;
}

void ImuPreintegration::Reset() {
  duration = 0;
  n = 0;
  alpha.setZero();
  beta.setZero();
  gamma = SO3d{};
  F.setIdentity();
  P.setZero();
  U.setZero();
}

}  // namespace sv
