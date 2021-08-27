#pragma once

#include <sophus/se3.hpp>

#include "sv/llol/scan.h"

namespace sv {

/// @struct Lidar Sweep is a Lidar Scan that covers 360 degree hfov
struct LidarSweep final : public LidarScan {
  /// Data
  std::vector<Sophus::SE3f> tfs;  // transforms of each columns to some frame

  LidarSweep() = default;
  explicit LidarSweep(const cv::Size& size);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarSweep& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Add a scan to this sweep
  /// @return Number of points added
  int Add(const LidarScan& scan);
  void Check(const LidarScan& scan) const;

  /// @brief Interpolate pose of each column
  void Interp(const std::vector<Sophus::SE3f>& traj, int gsize = 0);

  /// @brief Draw
  cv::Mat DrawRange() const;
};

LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
