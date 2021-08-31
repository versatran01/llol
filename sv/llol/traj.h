#pragma once

#include "sv/llol/imu.h"

namespace sv {

/// @brief Accumulates imu data and integrate
/// @todo for now only integrate gyro for rotation
struct Trajectory {
  Trajectory() = default;
  explicit Trajectory(int size) { states.resize(size); }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Trajectory& rhs) {
    return os << rhs.Repr();
  }

  int size() const { return states.size(); }
  NavState& At(int i) { return states.at(i); }
  const NavState& At(int i) const { return states.at(i); }

  /// @return the acc vector used to initialize gravity
  void InitExtrinsic(const Sophus::SE3d& T_i_l,
                     const Eigen::Vector3d& acc,
                     double g_norm);

  /// @brief Given the first pose in poses, predict using imu
  /// @return Number of imus used
  /// @todo Need to handle partial sweep
  int Predict(const ImuQueue& imuq, double t0, double dt, int n);

  /// @brief Rotate the states so that the traj starts at curr end
  void Rotate(int n);

  Eigen::Vector3d g_pano;        // gravity vector in pano frame
  Sophus::SE3d T_odom_pano{};    // tf from pano to odom frame
  Sophus::SE3d T_imu_lidar{};    // extrinsics lidar to imu
  std::vector<NavState> states;  // imu state wrt current pano
};

}  // namespace sv
