#include "sv/llol/lidar.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include "sv/util/math.h"
#include "sv/util/ocv.h"

namespace sv {

/// LidarModel =================================================================
LidarModel::LidarModel(const cv::Size& size_in, float vfov) : size{size_in} {
  if (vfov <= 0) {
    vfov = kTauF / size.aspectRatio();
  }

  CHECK_GT(size.width, 0);
  CHECK_GT(size.height, 0);
  CHECK_LE(Rad2Deg(vfov), 128.0) << "vertial fov too big";

  elev_max = vfov / 2.0F;
  elev_delta = vfov / (size.height - 1);
  azim_delta = kTauF / size.width;

  elevs.resize(size.height);
  for (int i = 0; i < size.height; ++i) {
    elevs[i] = SinCosF{elev_max - i * elev_delta};
  }

  azims.resize(size.width);
  for (int i = 0; i < size.width; ++i) {
    azims[i] = SinCosF{kTauF - (i + 0.5F) * azim_delta};
  }
}

cv::Point LidarModel::Forward(float x, float y, float z, float r) const {
  cv::Point bad{-1, -1};

  const auto row = ToRow(z, r);
  if (!RowInside(row)) return bad;
  const auto col = ToCol(x, y);
  if (!ColInside(col)) return bad;
  return {col, row};
}

cv::Point2f LidarModel::ForwardF(float x, float y, float z, float r) const {
  cv::Point2f bad{-1.0F, -1.0F};
  const auto row = ToRowF(z, r);
  if (!RowInside(row)) return bad;
  const auto col = ToColF(x, y);
  if (!ColInside(col)) return bad;
  return {col, row};
}

cv::Point3f LidarModel::Backward(int r, int c, float rg) const {
  //  CHECK_GT(rg, 0);
  const auto& elev = elevs.at(r);
  const auto& azim = azims.at(c);
  return {elev.cos * azim.cos * rg, elev.cos * azim.sin * rg, elev.sin * rg};
}

int LidarModel::ToRow(float z, float r) const {
  //  CHECK_GT(r, 0);
  const float elev = std::asin(z / r);
  return std::round((elev_max - elev) / elev_delta);
}

int LidarModel::ToCol(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return static_cast<int>(azim / azim_delta);
}

float LidarModel::ToRowF(float z, float r) const {
  const float elev = std::asin(z / r);
  return (elev_max - elev) / elev_delta;
}

float LidarModel::ToColF(float x, float y) const {
  const float azim = std::atan2(y, -x) + kPiF;
  return azim / azim_delta;
}

std::string LidarModel::Repr() const {
  return fmt::format(
      "LidarModel(size={}, elev_max={:.2f}[deg], elev_delta={:.4f}[deg], "
      "azim_delta={:.4f}[deg])",
      sv::Repr(size),
      Rad2Deg(elev_max),
      Rad2Deg(elev_delta),
      Rad2Deg(azim_delta));
}

}  // namespace sv
