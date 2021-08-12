#pragma once

#include <opencv2/core/mat.hpp>

#include "sv/llol/lidar.h"

namespace sv {

struct Pixel {
  uint16_t raw{0};
  uint8_t num{0};

  static constexpr float kScale = 256.0F;
  static constexpr uint16_t kMaxRaw = std::numeric_limits<uint16_t>::max();
  static constexpr float kMaxRange = static_cast<float>(kMaxRaw) / kScale;

  float metric() const noexcept { return raw / kScale; }
};

// TODO (chao): make this a member function
int PanoRender(const LidarModel& model,
               const cv::Mat& dbuf1,
               cv::Mat& dbuf2,
               bool tbb = false);

/// @class Depth Panorama
class DepthPano {
 public:
  static constexpr float kScale = 256.0F;
  static constexpr float kMaxRange = 65536.0F / kScale;

  DepthPano() = default;
  explicit DepthPano(const cv::Size& size, float hfov = 0);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DepthPano& rhs);

  bool empty() const { return dbuf_.empty(); }
  size_t total() const { return dbuf_.total(); }
  bool num_sweeps() const noexcept { return num_sweeps_; }
  cv::Size size() const noexcept { return model_.size(); }

  float RangeAt(const cv::Point& pt) const {
    return dbuf_.at<ushort>(pt) / kScale;
  }

  cv::Rect WinCenterAt(const cv::Point& pt, const cv::Size& size) const;
  cv::Rect BoundWinCenterAt(const cv::Point& pt, const cv::Size& size) const;

  /// @brief Add a sweep to the pano
  int AddSweep(const cv::Mat& sweep, bool tbb = false);
  int AddSweepRow(const cv::Mat& sweep, int row);

  /// @brief Render pano at a new location
  int Render(bool tbb);

  int num_sweeps_{0};
  LidarModel model_;
  cv::Mat dbuf_;   // depth buffer
  cv::Mat dbuf2_;  // depth buffer 2
};

}  // namespace sv
