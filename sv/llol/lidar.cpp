#include "sv/llol/lidar.h"

#include <fmt/core.h>

#include "sv/util/ocv.h"

namespace sv {

/// LidarModel =================================================================
LidarModel::LidarModel(const cv::Size& size, float hfov) : size_{size} {
  if (hfov <= 0) {
    hfov = kTauF / size.aspectRatio();
  }

  elev_max_ = hfov / 2.0F;
  elev_delta_ = hfov / (size.height - 1);
  azim_delta_ = kTauF / size.width;
  elevs_.resize(size_.height);
  for (int i = 0; i < size_.height; ++i) {
    elevs_[i] = SinCosF{elev_max_ - i * elev_delta_};
  }
  azims_.resize(size_.width);
  for (int i = 0; i < size_.width; ++i) {
    azims_[i] = SinCosF{kTauF - i * azim_delta_};
  }
}

cv::Point2i LidarModel::Forward(float x, float y, float z, float r) const {
  cv::Point2i px{-1, -1};

  const int row = ToRow(z, r);
  if (!RowInside(row)) return px;

  const int col = ToCol(x, y);
  if (!ColInside(col)) return px;

  px.x = col;
  px.y = row;

  return px;
}

cv::Point3f LidarModel::Backward(int r, int c, float rg) const {
  const auto& elev = elevs_.at(r);
  const auto& azim = azims_.at(c);
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

int LidarModel::ToRow(float z, float r) const {
  const float elev = std::asin(z / r);
  return (elev_max_ - elev) / elev_delta_ + 0.5F;
}

int LidarModel::ToCol(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta_ + 0.5F;
}

std::string LidarModel::Repr() const {
  return fmt::format(
      "LidarModel(size={}, elev_max={:.4f}[deg], elev_delta={:.4f}[deg], "
      "azim_delta={:.4f}[deg])",
      sv::Repr(size_),
      Rad2Deg(elev_max_),
      Rad2Deg(elev_delta_),
      Rad2Deg(azim_delta_));
}

std::ostream& operator<<(std::ostream& os, const LidarModel& rhs) {
  return os << rhs;
}

}  // namespace sv
