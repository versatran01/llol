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

  /// @brief At
  const Sophus::SE3f& TfAt(int c) const { return tfs.at(c); }

  /// @brief Add a scan to this sweep
  /// @return Number of points added
  int Add(const LidarScan& scan);
  void Check(const LidarScan& scan) const;

  /// @brief Info
  bool full() const noexcept { return col_rg.end == xyzr.cols; }

  /// @brief Draw
  cv::Mat DispRange() const;
};

LidarSweep MakeTestSweep(const cv::Size& size);

}  // namespace sv
