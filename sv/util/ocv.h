#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

/// @brief Apply color map to mat
/// @details input must be 1-channel, assume after scale the max will be 1
///          default cmap is 10 = PINK. For float image it will set nan to
///          bad_color
cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale = 1.0,
                  int cmap = cv::COLORMAP_PINK,
                  uint8_t bad_color = 255);

/// @brief Create a window with name and show mat
void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

std::string Repr(const cv::Mat& mat);
std::string Repr(const cv::Size& size);
std::string Repr(const cv::Range& range);

}  // namespace sv
