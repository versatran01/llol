#include "sv/llol/lidar.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include "sv/util/ocv.h"

namespace sv {

/// LidarModel =================================================================
LidarModel::LidarModel(const cv::Size& size_in, float vfov) : size{size_in} {
  if (vfov <= 0) {
    vfov = kTauF / size.aspectRatio();
  }

  CHECK_GT(size.width, 0);
  CHECK_GT(size.height, 0);
  CHECK_LE(vfov, Deg2Rad(120.0));

  elev_max = vfov / 2.0F;
  elev_delta = vfov / (size.height - 1);
  azim_delta = kTauF / size.width;

  elevs.resize(size.height);
  for (int i = 0; i < size.height; ++i) {
    elevs[i] = SinCosF{elev_max - i * elev_delta};
  }

  azims.resize(size.width);
  for (int i = 0; i < size.width; ++i) {
    azims[i] = SinCosF{kTauF - i * azim_delta};
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
  const auto& elev = elevs.at(r);
  const auto& azim = azims.at(c);
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

int LidarModel::ToRow(float z, float r) const {
  const float elev = std::asin(z / r);
  return (elev_max - elev) / elev_delta + 0.5F;
}

int LidarModel::ToCol(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta + 0.5F;
}

std::string LidarModel::Repr() const {
  return fmt::format(
      "LidarModel(size={}, elev_max={:.4f}[deg], elev_delta={:.4f}[deg], "
      "azim_delta={:.4f}[deg])",
      sv::Repr(size),
      Rad2Deg(elev_max),
      Rad2Deg(elev_delta),
      Rad2Deg(azim_delta));
}

}  // namespace sv
