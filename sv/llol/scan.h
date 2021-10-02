#pragma once

#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "sv/util/math.h"  // MeanCovar

namespace sv {

struct ScanBase {
  // start and delta time
  double time{};  // time of the last column
  double dt{};    // delta time between two columns

  // Scan related data
  cv::Mat mat;                    // storage
  cv::Range curr;                 // current range
  std::vector<Sophus::SE3f> tfs;  // tfs of each col to some frame

  ScanBase() = default;
  ScanBase(const cv::Size& size, int dtype);
  ScanBase(double time, double dt, const cv::Mat& scan, const cv::Range& curr);
  virtual ~ScanBase() noexcept = default;

  /// @brief Info
  int rows() const { return mat.rows; }
  int cols() const { return mat.cols; }
  int type() const { return mat.type(); }
  int total() const { return mat.total(); }
  bool empty() const { return mat.empty(); }
  int channels() const { return mat.channels(); }
  cv::Size size() const { return {mat.cols, mat.rows}; }

  double TimeAt(int col) const { return time - dt * (cols() - col); }
  const Sophus::SE3f& TfAt(int c) const { return tfs.at(c); }

  /// @brief Update view (curr and span) given new curr
  void UpdateView(const cv::Range& new_curr);
  /// @brief Update time (time and dt) given new time
  void UpdateTime(double new_time, double new_dt);
  /// @brief Extract the range channel (16UC1)
  cv::Mat ExtractRange() const;
};

/// @struct This should match the ouster scan
struct ScanPixel {
  float x{};
  float y{};
  float z{};
  uint16_t range_raw{};
  uint16_t intensity{};

  bool Ok() const noexcept { return !std::isnan(x); }
  auto Vec3fMap() const { return Eigen::Map<const Eigen::Vector3f>(&x); }
};
static_assert(sizeof(ScanPixel) == sizeof(float) * 4,
              "Size of ScanPixel must be 16");

/// @struct Lidar Scan like an image, with pixel (x,y,z,r)
struct LidarScan : public ScanBase {
  using PixelT = ScanPixel;
  static constexpr int kDtype = CV_32FC4;

  /// data
  double scale{};  // scale for converting range to float

  LidarScan() = default;
  /// @brief Ctor for allocating storage
  explicit LidarScan(const cv::Size& size);
  /// @brief Ctor for incoming lidar scan
  LidarScan(double time,
            double dt,
            double scale,
            const cv::Mat& scan,
            const cv::Range& curr);

  /// @brief At
  const auto& PixelAt(const cv::Point& px) const { return mat.at<PixelT>(px); }
  float RangeAt(const cv::Point& px) const {
    return PixelAt(px).range_raw / scale;
  }

  /// @brief Calculate smoothness and variance score of a cell starting at px
  cv::Vec2f CalcScore(const cv::Point& px, int width) const;
  /// @brief Calculate mean and covar of a cell in rect
  void CalcMeanCovar(const cv::Rect& rect, MeanCovar3f& mc) const;
};

LidarScan MakeTestScan(const cv::Size& size);

}  // namespace sv
