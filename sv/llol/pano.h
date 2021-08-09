#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/llol/lidar.h"

namespace sv {

struct PanoPixel {
  static constexpr float kScale = 256.0F;
  static constexpr uint16_t kMaxRaw = std::numeric_limits<uint16_t>::max();
  static constexpr float kMaxRange = static_cast<float>(kMaxRaw) / kScale;

  uint16_t raw{0};
  uint8_t age{0};
};

/// @class Depth Panorama
class DepthPano {
 public:
  static constexpr float kScale = 256.0F;
  static constexpr float kMaxRange = 65536.0F / kScale;

  DepthPano() = default;
  DepthPano(cv::Size size, float hfov = 0);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs);

  bool empty() const { return dbuf_.empty(); }
  size_t total() const { return dbuf_.total(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return model_.size_; }

  float GetRange(cv::Point pt) const { return dbuf_.at<ushort>(pt) / kScale; }
  void SetRange(cv::Point pt, float rg, cv::Mat& mat) {
    mat.at<ushort>(pt) = rg * kScale;
  }

  cv::Rect WinCenterAt(cv::Point pt, cv::Size size) const;
  cv::Rect BoundWinCenterAt(cv::Point pt, cv::Size size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const cv::Mat& sweep, bool tbb);
  int AddSweepRow(const cv::Mat& sweep, int row);

  /// @brief Render pano at a new location
  int Render(bool tbb);
  int RenderRow(int row1);

  /// @brief Computes mean and covar of points in window at
  void CalcMeanCovar(cv::Rect win, MeanCovar3f& mc) const;

  /// @brief Convert depth buffer to point buffer
  //  int DbufToPBuf(bool tbb);
  //  int DbufToPBufRow(int r);

  int num_sweeps_{0};
  LidarModel model_;
  cv::Mat dbuf_;   // depth buffer
  cv::Mat pbuf_;   // point buffer
  cv::Mat dbuf2_;  // depth buffer 2
};

}  // namespace sv
