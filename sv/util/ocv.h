#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace sv {

/// @brief Get the corresponding cv enum value given type
/// @example cv_type<cv::Vec3f>::value == CV_32FC3
///          cv_type_v<cv::Vec3f> == CV_32FC3
template <typename T>
struct cv_type;

template <>
struct cv_type<uchar> {
  static constexpr int value = CV_8U;
};

template <>
struct cv_type<schar> {
  static constexpr int value = CV_8S;
};

template <>
struct cv_type<ushort> {
  static constexpr int value = CV_16U;
};

template <>
struct cv_type<short> {
  static constexpr int value = CV_16S;
};

template <>
struct cv_type<int> {
  static constexpr int value = CV_32S;
};

template <>
struct cv_type<float> {
  static constexpr int value = CV_32F;
};

template <>
struct cv_type<double> {
  static constexpr int value = CV_64F;
};

template <typename T, int N>
struct cv_type<cv::Vec<T, N>> {
  static constexpr int value = (CV_MAKETYPE(cv_type<T>::value, N));
};

template <typename T>
inline constexpr int cv_type_v = cv_type<T>::value;

/// @brief Convert cv::Mat::type() to string
/// @example CvTypeStr(CV_8UC1) == "8UC1"
std::string CvTypeStr(int type);

/// @brief Repr for various cv types
std::string Repr(const cv::Mat& mat);
std::string Repr(const cv::Size& size);
std::string Repr(const cv::Range& range);

/// Range * d
inline cv::Range& operator*=(cv::Range& lhs, int d) noexcept {
  lhs.start *= d;
  lhs.end *= d;
  return lhs;
}
inline cv::Range operator*(cv::Range lhs, int d) noexcept { return lhs *= d; }

/// Range / d
inline cv::Range& operator/=(cv::Range& lhs, int d) noexcept {
  lhs.start /= d;
  lhs.end /= d;
  return lhs;
}
inline cv::Range operator/(cv::Range lhs, int d) noexcept { return lhs /= d; }

/// Size / Size
inline cv::Size& operator/=(cv::Size& lhs, const cv::Size& rhs) noexcept {
  lhs.width /= rhs.width;
  lhs.height /= rhs.height;
  return lhs;
}

inline cv::Size operator/(cv::Size lhs, const cv::Size& rhs) noexcept {
  return lhs /= rhs;
}

}  // namespace sv
