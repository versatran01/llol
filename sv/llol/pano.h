#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "sv/llol/lidar.h"
#include "sv/llol/scan.h"

namespace sv {

inline cv::Rect WinCenterAt(const cv::Point& pt, const cv::Size& size) {
  return {{pt.x - size.width / 2, pt.y - size.height / 2}, size};
}

struct DepthPixel {
  static constexpr float kScale = 512.0F;
  static constexpr uint16_t kMaxRaw = std::numeric_limits<uint16_t>::max();
  static constexpr float kMaxRange = static_cast<float>(kMaxRaw) / kScale;

  uint16_t raw{0};
  uint16_t cnt{0};

  float GetMeter() const noexcept { return raw / kScale; }
  void SetMeter(float rg) { raw = static_cast<uint16_t>(rg * kScale); }
} __attribute__((packed));
static_assert(sizeof(DepthPixel) == 4, "Size of DepthPixel is not 4");

struct PanoParams {
  float hfov{0.0};
  int max_cnt{10};
  float range_ratio{0.1};
};

/// @class Depth Panorama
struct DepthPano {
  /// Params
  int max_cnt{10};
  float range_ratio{0.1};
  // TODO (chao): also add a min range?

  /// Data
  LidarModel model;
  cv::Mat dbuf;   // depth buffer
  cv::Mat dbuf2;  // depth buffer 2

  /// @brief Ctors
  DepthPano() = default;
  explicit DepthPano(const cv::Size& size, const PanoParams& params = {});

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs) {
    return os << rhs.Repr();
  }

  /// @brief At
  const auto& PixelAt(const cv::Point& pt) const {
    return dbuf.at<DepthPixel>(pt);
  }
  float RangeAt(const cv::Point& pt) const { return PixelAt(pt).GetMeter(); }
  /// @brief Get a bounded window centered at pt with given size
  cv::Rect BoundWinCenterAt(const cv::Point& pt, const cv::Size& size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const LidarSweep& sweep, int tbb_rows = 0);
  int AddSweepRow(const LidarSweep& sweep, int row);
  bool FuseDepth(const cv::Point& px, float rg);

  /// @brief Render pano at a new location
  /// @todo Currently disabled
  int Render(const Sophus::SE3f& tf_2_1, int tbb_rows = 0);
  int RenderRow(const Sophus::SE3f& tf_2_1, int row);
  bool UpdateBuffer(const cv::Point& px, float rg);

  /// @brief info
  bool empty() const { return dbuf.empty(); }
  size_t total() const { return dbuf.total(); }
  cv::Size size() const noexcept { return model.size; }
};

}  // namespace sv
