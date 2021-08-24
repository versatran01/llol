#pragma once

#include <absl/types/span.h>

#include <Eigen/Geometry>
#include <boost/circular_buffer.hpp>
#include <opencv2/core/types.hpp>
#include <sophus/se3.hpp>

namespace sv {

struct NavState {
  double time{};
  Sophus::SO3d rot{};
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
};

struct ImuBias {
  Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyr{Eigen::Vector3d::Zero()};
};

/// @brief Time-stamped Imu data
struct ImuData {
  double time{};
  Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyr{Eigen::Vector3d::Zero()};

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
struct ImuIntegrator {
  ImuBuffer buf{32};
  ImuBias bias;
  Sophus::SE3d T_imu_lidar;

  ImuIntegrator(int buffer_size = 32) : buf{32} {}

  int size() const { return buf.size(); }
  bool empty() const { return buf.empty(); }

  /// @brief Add imu data into buffer
  void Add(const ImuData& imu) { buf.push_back(imu.DeBiased(bias)); }

  /// @brief Given the first pose in poses, predict using imu
  int Integrate(double t0, double dt, absl::Span<Sophus::SE3f> poses) const;
};

}  // namespace sv
