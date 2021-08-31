#pragma once

#include "sv/llol/scan.h"
#include "sv/llol/traj.h"

namespace sv {

/// @struct Lidar Sweep is a Lidar Scan that covers 360 degree hfov
struct LidarSweep final : public LidarScan {
  LidarSweep() = default;
  explicit LidarSweep(const cv::Size& size) : LidarScan{size} {}

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Add a scan to this sweep
  /// @return Number of points added
  int Add(const LidarScan& scan);

  /// @brief Interpolate pose of each column
  void Interp(const Trajectory& traj, int gsize = 0);

  /// @brief Draw
  cv::Mat DrawRange() const;
};

LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
