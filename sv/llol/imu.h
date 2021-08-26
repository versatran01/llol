#pragma once

#include <boost/circular_buffer.hpp>
#include <sophus/se3.hpp>

namespace sv {

static const Eigen::Vector3d kZero3d = Eigen::Vector3d::Zero();

struct NavState {
  double time{};
  Sophus::SO3d rot{};
  Eigen::Vector3d pos{kZero3d};
  Eigen::Vector3d vel{kZero3d};
};

struct ImuBias {
  Eigen::Vector3d acc{kZero3d};
  Eigen::Vector3d gyr{kZero3d};
};

/// @brief Time-stamped Imu data
struct ImuData {
  double time{};
  Eigen::Vector3d acc{kZero3d};
  Eigen::Vector3d gyr{kZero3d};

  void Debias(const ImuBias& bias) {
    acc -= bias.acc;
    gyr -= bias.gyr;
  }

  ImuData DeBiased(const ImuBias& bias) const {
    ImuData out = *this;
    out.Debias(bias);
    return out;
  }
};

/// Integrate rotation one step, assumes de-biased gyro data
Sophus::SO3d IntegrateRot(const Sophus::SO3d& R0,
                          const Eigen::Vector3d& omg,
                          double dt);

/// Integrate nav state one step, assume de-biased imu data
/// @param s0 is current state, imu is imu data, g_w is gravity in world frame
NavState IntegrateEuler(const NavState& s0,
                        const ImuData& imu,
                        const Eigen::Vector3d& g_w,
                        double dt);

NavState IntegrateMidpoint(const NavState& s0,
                           const ImuData& imu0,
                           const ImuData& imu1,
                           const Eigen::Vector3d& g_w);

using ImuBuffer = boost::circular_buffer<ImuData>;

int FindNextImu(const ImuBuffer& buf, double t);

/// @brief Accumulates imu data and integrate
/// @todo for now only integrate gyro for rotation
struct ImuModel {
  ImuBuffer buf{16};
  ImuBias bias{};
  Sophus::SE3d T_imu_lidar{};

  /// @brief Add imu data into buffer
  void Add(const ImuData& imu) { buf.push_back(imu.DeBiased(bias)); }

  /// @brief Given the first pose in poses, predict using imu
  /// @return Number of imus used
  int Predict(double t0, double dt, std::vector<Sophus::SE3f>& poses) const;
};

}  // namespace sv
