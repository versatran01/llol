#include "sv/llol/imu.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"  // Hat3

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Quaterniond = Eigen::Quaterniond;

ImuBias::ImuBias(double acc_bias_std, double gyr_bias_std) {
  acc_var.setConstant(Sq(acc_bias_std));
  gyr_var.setConstant(Sq(gyr_bias_std));
}

std::string ImuBias::Repr() const {
  return fmt::format("ImuBias(acc=[{}], gyr=[{}], acc_var=[{}], gyr_var=[{}])",
                     acc.transpose(),
                     gyr.transpose(),
                     acc_var.transpose(),
                     gyr_var.transpose());
}

std::string NavState::Repr() const {
  return fmt::format("NavState(t={}, rot=[{}], pos=[{}], vel=[{}]",
                     time,
                     rot.unit_quaternion().coeffs().transpose(),
                     pos.transpose(),
                     vel.transpose());
}

void ImuData::Debias(const ImuBias& bias) {
  acc -= bias.acc;
  gyr -= bias.gyr;
}

ImuData ImuData::DeBiased(const ImuBias& bias) const {
  ImuData out = *this;
  out.Debias(bias);
  return out;
}

SO3d IntegrateRot(const SO3d& rot,
                  double time,
                  const ImuData& imu0,
                  const ImuData& imu1,
                  double dt) {
  const auto time_mid = time + dt / 2.0;
  // Find interpolation factor
  const auto s =
      std::clamp((time_mid - imu0.time) / (imu1.time - imu0.time), 0.0, 1.0);
  // Linearly interpolate between two gyro measurements
  const Vector3d omg = (1 - s) * imu0.gyr + s * imu1.gyr;
  return rot * SO3d::exp(omg * dt);
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
  // assuming g_w = [0,0,+g], such that when accelerometer is sitting flat on
  // the ground the measured acceleration is [0,0,+g].
  //
  // a_m = a_b + R_b_w * g_w
  // a_w = R_w_b * a_b = R_w_b * (a_m - R_b_w * g_w) = R_w_b * a_m - g_w
  const Vector3d a = s0.rot * imu.acc - g;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt * dt;
}

void IntegrateState(const NavState& s0,
                    const ImuData& imu0,
                    const ImuData& imu1,
                    const Eigen::Vector3d& g,
                    double dt,
                    NavState& s1) {
  CHECK_GT(dt, 0);
  const auto dt2 = dt * dt;
  s1.time = s0.time + dt;

  const auto time_mid = s0.time + dt / 2.0;
  const auto s =
      std::clamp((time_mid - imu0.time) / (imu1.time - imu0.time), 0.0, 1.0);
  // Integrate rotation first
  const Vector3d omg = (1 - s) * imu0.gyr + s * imu1.gyr;
  s1.rot = s0.rot * SO3d::exp(omg * dt);

  // Given the two rotations, rotate both acc measurements into world frame
  const Vector3d a0 = s0.rot * imu0.acc;
  const Vector3d a1 = s1.rot * imu1.acc;
  // a is world frame acceleration without gravity
  const Vector3d a = (1 - s) * a0 + s * a1 - g;
  s1.vel = s0.vel + a * dt;
  s1.pos = s0.pos + s0.vel * dt + 0.5 * a * dt2;
}

int GetImuIndexAfterTime(const ImuBuffer& buf, double t) {
  int ibuf = -1;
  for (int i = 0; i < buf.size(); ++i) {
    if (buf.at(i).time > t) {
      ibuf = i;
      break;
    }
  }
  return ibuf;
}

ImuNoise::ImuNoise(double rate,
                   double acc_noise,
                   double gyr_noise,
                   double acc_bias_noise,
                   double gyr_bias_noise) {
  CHECK_GT(rate, 0);

  // Follows kalibr imu noise model also VectorNav material
  // https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
  // https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-imuspecs
  sigma2.segment<3>(kNa).setConstant(Sq(acc_noise) * rate);
  sigma2.segment<3>(kNw).setConstant(Sq(gyr_noise) * rate);
  sigma2.segment<3>(kNba).setConstant(Sq(acc_bias_noise) / rate);
  sigma2.segment<3>(kNbw).setConstant(Sq(gyr_bias_noise) / rate);
}

std::string ImuNoise::Repr() const {
  return fmt::format(
      "ImuNoise(acc_cov=[{}], gyr_cov=[{}], acc_bias_cov=[{}], "
      "gyr_bias_cov=[{}])",
      sigma2.segment<3>(kNa).transpose(),
      sigma2.segment<3>(kNw).transpose(),
      sigma2.segment<3>(kNba).transpose(),
      sigma2.segment<3>(kNbw).transpose());
}

std::string ImuQueue::Repr() const {
  return fmt::format("ImuQueue(size={}/{}, bias={}, noise={}",
                     size(),
                     capacity(),
                     bias,
                     noise);
}

void sv::ImuQueue::Add(const ImuData& imu) {
  buf.push_back(imu);
  if (!buf.empty()) {
    const auto dt = imu.time - buf[size() - 2].time;
    const auto dt2 = dt * dt;
    // Also propagate covariance for bias model
    bias.acc_var += noise.nbad() * dt2;
    bias.gyr_var += noise.nbwd() * dt2;
  }
}

ImuData ImuQueue::CalcMean() const {
  ImuData mean;
  // Use last imu time
  mean.time = buf.back().time;
  for (const auto& imu : buf) {
    mean.acc += imu.acc;
    mean.gyr += imu.gyr;
  }
  mean.acc /= size();
  mean.gyr /= size();
  return mean;
}

/// ImuPreintegration ==========================================================
int ImuPreintegration::Compute(const ImuQueue& imuq, double t0, double t1) {
  CHECK_LT(t0, t1);
  const int ibuf0 = imuq.IndexAfter(t0);
  CHECK_LE(0, ibuf0);

  // Keep integrate till we reach either the last imu or one right before t1
  double t = t0;
  int ibuf = ibuf0;

  while (true) {
    // This imu must exist
    const auto imu = imuq.DebiasedAt(ibuf);
    Integrate(imu.time - t, imu, imuq.noise);
    t = imu.time;

    // stop if we are at the last imu
    if (ibuf + 1 == imuq.size()) break;
    // or if next imu time is later than t1
    if (imuq.RawAt(ibuf + 1).time >= t1) break;
    // above ensures that ibuf should be valid

    ++ibuf;
  }

  // Use the imu at ibuf to finish integrating to t1
  const auto imu = imuq.DebiasedAt(ibuf);
  Integrate(t1 - imu.time, imu, imuq.noise);

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
  const Vector3d dalpha = beta * dt + 0.5 * ga * dt2;

  // vins-mono eq 9
  // Ft =
  // [0  I        0    0   0]
  // [0  0  -R*[a]x   -R   0]
  // [0  0    -[w]x    0  -I]
  // last two rows are all zeros
  // Fd = I + Ft * dt
  const auto Rmat = gamma.matrix();
  F.block<3, 3>(Index::kAlpha, Index::kBeta) = kMatEye3d * dt;
  F.block<3, 3>(Index::kBeta, Index::kTheta) = -Rmat * Hat3(a) * dt;
  F.block<3, 3>(Index::kBeta, Index::kBa) = -Rmat * dt;
  F.block<3, 3>(Index::kTheta, Index::kTheta) = SO3d::exp(-w * dt).matrix();
  F.block<3, 3>(Index::kTheta, Index::kBw) = -kMatEye3d * dt;

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

int ImuPreintegration2::Compute(const ImuQueue& imuq, double t0, double t1) {
  CHECK_LT(t0, t1);
  const int ibuf0 = imuq.IndexAfter(t0);
  CHECK_GT(ibuf0, 0);

  // Keep integrate till we reach either the last imu or one right before t1
  double t = t0;
  int ibuf = ibuf0;

  while (true) {
    // This imu must exist
    const auto imu = imuq.DebiasedAt(ibuf);
    Integrate(imu.time - t, imu, imuq.noise);
    t = imu.time;

    // stop if we are at the last imu
    if (ibuf + 1 == imuq.size()) break;
    // or if next imu time is later than t1
    if (imuq.RawAt(ibuf + 1).time >= t1) break;
    // above ensures that ibuf should be valid

    ++ibuf;
  }

  // Use the imu at ibuf to finish integrating to t1
  const auto imu = imuq.DebiasedAt(ibuf);
  Integrate(t1 - imu.time, imu, imuq.noise);

  // Compute sqrt info
  U = MatrixSqrtUtU(P.inverse().eval());

  return n;
}

void ImuPreintegration2::Reset() {
  duration = 0;
  n = 0;
  alpha.setZero();
  beta.setZero();
  gamma = SO3d{};
  F.setIdentity();
  P.setZero();
  U.setZero();
}

void ImuPreintegration2::Integrate(double dt,
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
  const Vector3d dalpha = beta * dt + 0.5 * ga * dt2;

  const auto Rmat = gamma.matrix();
  F.block<3, 3>(Index::kAlpha, Index::kBeta) = kMatEye3d * dt;
  F.block<3, 3>(Index::kAlpha, Index::kTheta) = -0.5 * Rmat * Hat3(a) * dt2;
  F.block<3, 3>(Index::kBeta, Index::kTheta) = -Rmat * Hat3(a) * dt;
  F.block<3, 3>(Index::kTheta, Index::kTheta) = SO3d::exp(-w * dt).matrix();

  P = F * P * F.transpose();
  P.topLeftCorner<3, 3>().diagonal() += 0.25 * dt2 * noise.nad();
  P.block<3, 3>(3, 3).diagonal() += noise.nad();
  P.block<3, 3>(0, 3).diagonal() += noise.nad() * dt * 0.5;
  P.block<3, 3>(3, 0).diagonal() += noise.nad() * dt * 0.5;
  P.bottomRightCorner<3, 3>().diagonal() += noise.nwd();

  // Update measurement
  alpha += dalpha;
  beta += dbeta;
  gamma *= dgamma;
  duration += dt;
  ++n;
}

}  // namespace sv
