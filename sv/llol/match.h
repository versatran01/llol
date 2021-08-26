#pragma once

#include <opencv2/core/types.hpp>

#include "sv/util/math.h"  // MeanCovar

namespace sv {

/// @struct Match
struct GicpMatch {
  static constexpr int kBadPx = -100;

  cv::Point px_g{kBadPx, kBadPx};  // 8 grid pixel coord
  MeanCovar3f mc_g{};              // 52 grid mean covar
  cv::Point px_p{kBadPx, kBadPx};  // 8 pano pixel coord
  MeanCovar3f mc_p{};              // 52 pano mean covar
  Eigen::Matrix3f U{};             // 36 sqrt of info

  /// @brief Whether this match is good
  bool Ok() const noexcept { return GridOk() && PanoOk(); }
  bool GridOk() const { return px_g.x >= 0 && mc_g.ok(); }
  bool PanoOk() const { return px_p.x >= 0 && mc_p.ok(); }

  void ResetGrid();
  void ResetPano();
  void Reset();

  void SqrtInfo(float lambda = 0.0F);
};

/// @brief Computes matrix square root using Cholesky
Eigen::Matrix3f MatrixSqrtUtU(const Eigen::Matrix3f& A);

}  // namespace sv
