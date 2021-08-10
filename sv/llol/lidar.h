
#pragma once

#include <opencv2/core/types.hpp>

#include "sv/util/math.h"  // SinCosF

namespace sv {

/// @struct LidarModel
struct LidarModel {
  LidarModel() = default;
  LidarModel(cv::Size size, float hfov);

  cv::Size size() const noexcept { return size_; }

  /// @brief xyzr to pixel, bad result is {-1, -1}
  cv::Point2i Forward(float x, float y, float z, float r) const;
  /// @brief pixel to xyz
  cv::Point3f Backward(int r, int c, float rg = 1.0) const;

  /// @brief compute row and col given xyzr
  int ToRow(float z, float r) const;
  int ToCol(float x, float y) const;

  /// @brief Check if r/c inside image
  bool RowInside(int r) const noexcept { return 0 <= r && r < size_.height; }
  bool ColInside(int c) const noexcept { return 0 <= c && c < size_.width; }

  /// @brief width / height
  double WidthHeightRatio() const { return size_.aspectRatio(); }
  float ElevAzimRatio() const { return elev_delta_ / azim_delta_; }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarModel& rhs);

  cv::Size size_{};
  float elev_max_{};
  float elev_delta_{};
  float azim_delta_{};
  std::vector<SinCosF> elevs_{};
  std::vector<SinCosF> azims_{};
};

}  // namespace sv
