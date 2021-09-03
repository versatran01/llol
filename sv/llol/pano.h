#pragma once

#include "sv/llol/lidar.h"
#include "sv/llol/sweep.h"

namespace sv {

inline cv::Rect WinCenterAt(const cv::Point& pt, const cv::Size& size) {
  return {{pt.x - size.width / 2, pt.y - size.height / 2}, size};
}

/// @brief Pixel stored in DepthPano
struct DepthPixel {
  static constexpr float kScale = 512.0F;
  static constexpr uint16_t kMaxRaw = std::numeric_limits<uint16_t>::max();
  static constexpr float kMaxRange = static_cast<float>(kMaxRaw) / kScale;

  uint16_t raw{0};
  uint16_t cnt{0};

  float GetRange() const noexcept { return raw / kScale; }
  void SetRange(float rg) { raw = static_cast<uint16_t>(rg * kScale); }
  void SetRangeCount(float rg, int n) {
    SetRange(rg);
    cnt = n;
  }
} __attribute__((packed));
static_assert(sizeof(DepthPixel) == 4, "Size of DepthPixel is not 4");

struct PanoParams {
  float vfov{0.0F};
  int max_cnt{10};
  float min_range{0.5F};
  float range_ratio{0.1F};
  bool gravity_align{false};
};

/// @class Depth Panorama
struct DepthPano {
  /// Params
  int max_cnt{};
  float min_range{};
  float range_ratio{};
  bool gravity_align{};
  float num_added{};  // number of pano added, float since we could add partial

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
  auto& PixelAt(const cv::Point& pt) { return dbuf.at<DepthPixel>(pt); }
  const auto& PixelAt(const cv::Point& pt) const {
    return dbuf.at<DepthPixel>(pt);
  }
  float RangeAt(const cv::Point& pt) const { return PixelAt(pt).GetRange(); }
  /// @brief Get a bounded window centered at pt with given size
  cv::Rect BoundWinCenterAt(const cv::Point& pt, const cv::Size& size) const;

  /// @brief Add a sweep to the pano
  int Add(const LidarSweep& sweep, const cv::Range& curr, int gsize = 0);
  int AddRow(const LidarSweep& sweep, const cv::Range& curr, int row);
  bool FuseDepth(const cv::Point& px, float rg);

  /// @brief Render pano at a new location
  /// @note frame difference, ones is T_p1_p2, the other is T_p2_p1
  bool ShouldRender(const Sophus::SE3d& tf_p2_p1);
  int Render(Sophus::SE3f tf_p2_p1, int gsize = 0);
  int RenderRow(const Sophus::SE3f& tf_p2_p1, int row);
  bool UpdateBuffer(const cv::Point& px, float rg, int cnt);

  /// @brief info
  int rows() const { return dbuf.rows; }
  int cols() const { return dbuf.cols; }
  bool empty() const { return dbuf.empty(); }
  size_t total() const { return dbuf.total(); }
  cv::Size size() const noexcept { return model.size; }

  /// @brief Compute mean and covar on a window centered at px given range
  /// @return sum(cnt_i) / max_cnt
  float MeanCovarAt(const cv::Point& px,
                    const cv::Size& size,
                    float rg,
                    MeanCovar3f& mc) const;

  /// @brief Viz
  const std::vector<cv::Mat>& DrawRangeCount() const;
  const std::vector<cv::Mat>& DrawRangeCount2() const;
};

}  // namespace sv
