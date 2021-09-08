#pragma once

#include "sv/llol/imu.h"

namespace sv {

struct TrajectoryParams {
  bool integrate_acc{false};
  bool update_acc_bias{false};
};

/// @brief Accumulates imu data and integrate
/// @todo for now only integrate gyro for rotation
struct Trajectory {
  Trajectory() = default;
  explicit Trajectory(int size, const TrajectoryParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Trajectory& rhs) {
    return os << rhs.Repr();
  }

  int size() const { return states.size(); }
  NavState& At(int i) { return states.at(i); }
  const NavState& At(int i) const { return states.at(i); }
  double duration() const { return back().time - front().time; }

  /// @return the acc vector used to initialize gravity
  void Init(const Sophus::SE3d& tf_i_l,
            const Eigen::Vector3d& acc,
            double g_norm);

  /// @brief Given the first pose in poses, predict using imu
  /// @return Number of imus used
  /// @todo Need to handle partial sweep
  int Predict(const ImuQueue& imuq, double t0, double dt, int n);

  /// @brief Pop oldest states so that the traj starts at curr end
  void PopOldest(int n);

  /// @brief Update internal state given new transform
  void MoveFrame(const Sophus::SE3d& tf_p2_p1);

  /// @brief Update bias given optimized trajectory
  int UpdateBias(ImuQueue& imuq);

  Sophus::SE3d TfOdomLidar() const;
  Sophus::SE3d TfPanoLidar() const;

  const NavState& front() const { return states.front(); }
  const NavState& back() const { return states.back(); }

  /// Params
  bool integrate_acc{};
  bool update_acc_bias{};

  /// Data
  Eigen::Vector3d g_pano{};      // gravity vector in pano frame
  Sophus::SE3d T_odom_pano{};    // tf from pano to odom frame
  Sophus::SE3d T_imu_lidar{};    // extrinsics lidar to imu
  std::vector<NavState> states;  // imu state wrt current pano
};

}  // namespace sv
