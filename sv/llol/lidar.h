#pragma once

#include <opencv2/core/types.hpp>

#include "sv/util/math.h"  // SinCosF

namespace sv {

/// @struct LidarModel
struct LidarModel {
  LidarModel() = default;
  explicit LidarModel(const cv::Size& size_in, float vfov = 0.0F);

  /// @brief Repr / <<
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const LidarModel& rhs) {
    return os << rhs.Repr();
  }

  /// @brief xyzr to pixel, bad result is {-1, -1}
  cv::Point Forward(float x, float y, float z, float r) const;
  /// @brief pixel to xyz
  cv::Point3f Backward(int r, int c, float rg = 1.0) const;

  /// @brief compute row and col given xyzr
  int ToRow(float z, float r) const;
  int ToCol(float x, float y) const;

  /// @brief Check if r/c inside image
  bool RowInside(int r) const noexcept { return 0 <= r && r < size.height; }
  bool ColInside(int c) const noexcept { return 0 <= c && c < size.width; }

  /// Data
  cv::Size size{};
  float elev_max{};
  float elev_delta{};
  float azim_delta{};
  std::vector<SinCosF> elevs{};
  std::vector<SinCosF> azims{};
};

}  // namespace sv
