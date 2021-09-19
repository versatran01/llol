#pragma once

#include "sv/llol/lidar.h"
#include "sv/llol/sweep.h"

namespace sv {

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
  int min_sweeps{8};
  float min_range{0.5F};
  float max_range{0.0F};
  float win_ratio{0.1F};
  float fuse_ratio{0.05F};
  bool align_gravity{false};
  double min_match_ratio{0.9};
  double max_translation{1.5};
};

/// @class Depth Panorama
struct DepthPano {
  /// Params
  int max_cnt{};
  int min_sweeps{};
  float min_range{};
  float max_range{};
  float win_ratio{};
  float fuse_ratio{};
  bool align_gravity{};
  double min_match_ratio{};
  double max_translation{};

  /// Data
  LidarModel model;
  cv::Mat dbuf;
  cv::Mat dbuf2;
  float num_sweeps{-1};  // number of sweeps added

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

  /// @brief Add a partial sweep to the pano
  int Add(const LidarSweep& sweep, const cv::Range& curr, int gsize = 0);
  int AddRow(const LidarSweep& sweep, const cv::Range& curr, int row);
  bool FuseDepth(const cv::Point& px, float rg);

  /// @brief Render pano at a new location
  /// @note frame difference, ones is T_p1_p2, the other is T_p2_p1
  bool ShouldRender(const Sophus::SE3d& tf_p2_p1, double match_ratio) const;
  int Render(Sophus::SE3f tf_p2_p1, int gsize = 0);
  int RenderRow(const Sophus::SE3f& tf_p2_p1, int row);
  bool UpdateBuffer(const cv::Point& px, float rg, int cnt);

  /// @brief info
  int rows() const { return dbuf.rows; }
  int cols() const { return dbuf.cols; }
  bool empty() const { return dbuf.empty(); }
  size_t total() const { return dbuf.total(); }
  cv::Size size() const noexcept { return model.size; }
  bool ready() const { return num_sweeps >= 1; }

  /// @brief Compute mean and covar on a window centered at px given range
  /// @return sum(cnt_i) / max_cnt
  float CalcMeanCovar(cv::Rect win, float rg, MeanCovar3f& mc) const;

  /// @brief Viz
  const std::vector<cv::Mat>& DrawRangeCount() const;
  const std::vector<cv::Mat>& DrawRangeCount2() const;
};

}  // namespace sv
