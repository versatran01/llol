#pragma once

#include <opencv2/core/types.hpp>

#include "sv/util/math.h"

namespace sv {

/// @struct Match
struct GicpMatch {
  static constexpr int kBad = -100;

  cv::Point px_s{kBad, kBad};  // 8 sweep pixel coord
  MeanCovar3f mc_s{};          // 52 sweep mean covar
  cv::Point px_p{kBad, kBad};  // 8 pano pixel coord
  MeanCovar3f mc_p{};          // 52 pano mean covar
  Eigen::Matrix3f U{};         // 36 sqrt of info

  /// @brief Whether this match is good
  bool Ok() const noexcept { return SweepOk() && PanoOk(); }
  bool SweepOk() const { return px_s.x >= 0 && mc_s.ok(); }
  bool PanoOk() const { return px_p.x >= 0 && mc_p.ok(); }

  void ResetSweep();
  void ResetPano();
  void Reset();

  void SqrtInfo(float lambda = 0.0F);
};

/// @brief Computes matrix square root using Cholesky
Eigen::Matrix3f MatrixSqrtUtU(const Eigen::Matrix3f& A);

}  // namespace sv
