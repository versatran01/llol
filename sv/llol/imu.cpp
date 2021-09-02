#include "sv/llol/imu.h"

#include <fmt/ostream.h>
#include <glog/logging.h>

#include "sv/util/math.h"  // Hat3

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Vector3d = Eigen::Vector3d;
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
                           const Vector3d& g) {
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
  const Vector3d a = (a0 + a1) * 0.5 + g;
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
  sigma2.segment<3>(kNba).setConstant(Sq(acc_bias_noise) * dt);
  sigma2.segment<3>(kNbw).setConstant(Sq(gyr_bias_noise) * dt);
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
  return fmt::format("ImuQueue(size={}, capacity={}, bias={}, noise={}",
                     size(),
                     capacity(),
                     bias,
                     noise);
}

void sv::ImuQueue::Add(const ImuData& imu) {
  buf.push_back(imu);
  // Also propagate covariance for bias model
  bias.acc_var += noise.nba();
  bias.gyr_var += noise.nbw();
}

ImuData ImuQueue::DebiasedAt(int i) const { return buf.at(i).DeBiased(bias); }

int ImuQueue::IndexAfter(double t) const { return FindNextImu(buf, t); }

int ImuQueue::UpdateBias(const std::vector<NavState>& states) {
  // TODO (chao): for now only update gyro bias
  const auto t0 = states.front().time;
  const auto t1 = states.back().time;
  const auto dt_state = (t1 - t0) / (states.size() - 1);

  // Find next imu
  int ibuf = FindNextImu(buf, t0);
  CHECK_LE(0, ibuf);

  MeanVar3d bw;

  while (ibuf < size()) {
    // Get an imu and try to find its left and right state
    const auto& imu = RawAt(ibuf);
    const int ist = (imu.time - t0) / dt_state;
    if (ist + 1 >= states.size()) break;

    const auto& st0 = states.at(ist);
    const auto& st1 = states.at(ist + 1);
    const Eigen::Vector3d gyr_hat =
        (st0.rot.inverse() * st1.rot).log() / dt_state;
    // w = w_m - b_w -> b_w = w_m - w
    //    bw.Add(imu.gyr - gyr_hat);
    bw.Add(imu.gyr - gyr_hat);

    ++ibuf;
  }

  //  LOG(INFO) << "n: " << bw.n;
  //  LOG(INFO) << "mean: " << bw.mean.transpose();
  //  LOG(INFO) << "var: " << bw.Var();

  // Simple update
  const Vector3d y = bw.mean - bias.gyr;
  const Vector3d S = bias.gyr_var + bw.Var();
  const Vector3d K = bias.gyr_var.cwiseProduct(S.cwiseInverse());
  bias.gyr += K.cwiseProduct(y);
  bias.gyr_var -= K.cwiseProduct(bias.gyr_var);
  //  LOG(INFO) << "bw: " << bias.gyr.transpose();
  //  LOG(INFO) << "bw_var: " << bias.gyr_var.transpose();

  return bw.n;
}

ImuData ImuQueue::CalcMean() const {
  ImuData mean;
  for (const auto& imu : buf) {
    mean.acc += imu.acc;
    mean.gyr += imu.gyr;
  }
  mean.acc /= size();
  mean.gyr /= size();
  return mean;
}

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
  const Vector3d dalpha = beta * dt + ga * dt2 * 0.5;

  // vins-mono eq 9
  // Ft =
  // [0  I        0    0   0]
  // [0  0  -R*[a]x   -R   0]
  // [0  0    -[w]x    0  -I]
  // last two rows are all zeros
  // F = I + Ft * dt
  const auto Rmat = gamma.matrix();
  F.block<3, 3>(Index::kAlpha, Index::kBeta) = kMatEye3d * dt;
  F.block<3, 3>(Index::kBeta, Index::kTheta) = -Rmat * Hat3(a) * dt;
  F.block<3, 3>(Index::kBeta, Index::kBa) = -Rmat * dt;
  F.block<3, 3>(Index::kTheta, Index::kTheta) = kMatEye3d - Hat3(w) * dt;
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

}  // namespace sv
