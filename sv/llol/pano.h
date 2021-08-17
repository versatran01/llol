#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/llol/lidar.h"
#include "sv/llol/scan.h"

namespace sv {

inline cv::Rect WinCenterAt(const cv::Point& pt, const cv::Size& size) {
  return {{pt.x - size.width / 2, pt.y - size.height / 2}, size};
}

struct Pixel {
  static constexpr float kScale = 512.0F;
  static constexpr uint16_t kMaxRaw = std::numeric_limits<uint16_t>::max();
  static constexpr float kMaxRange = static_cast<float>(kMaxRaw) / kScale;

  uint16_t raw{0};

  float Metric() const noexcept { return raw / kScale; }
  void SetMetric(float rg) { raw = static_cast<uint16_t>(rg * kScale); }
};

enum struct BufferUpdate {
  ADD_NEW,  // depth is added to buffer
  OCCLUDE,  // depth is not added due to occlusion
  UPDATED   // depth is updated
};

/// @class Depth Panorama
struct DepthPano {
  /// Data
  int num_sweeps_{0};
  LidarModel model_;
  cv::Mat dbuf_;   // depth buffer
  cv::Mat dbuf2_;  // depth buffer 2

  /// @brief Ctors
  DepthPano() = default;
  explicit DepthPano(const cv::Size& size, float hfov = 0);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs) {
    return os << rhs.Repr();
  }

  bool empty() const { return dbuf_.empty(); }
  size_t total() const { return dbuf_.total(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return model_.size; }

  float RangeAt(const cv::Point& pt) const {
    return dbuf_.at<Pixel>(pt).Metric();
  }

  /// @brief Get a bounded window centered at pt with given size
  cv::Rect BoundWinCenterAt(const cv::Point& pt, const cv::Size& size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const LidarSweep& sweep, bool tbb = false);
  int AddSweepRow(const LidarSweep& sweep, int row);
  bool UpdateBuffer(const cv::Point& px, float rg);

  /// @brief Render pano at a new location
  /// @todo Currently disabled
  int Render(bool tbb);
  int RenderRow(int row);
};

}  // namespace sv
