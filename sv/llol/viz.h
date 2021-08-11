#pragma once

#include <visualization_msgs/MarkerArray.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "sv/llol/lidar.h"
#include "sv/llol/match.h"

namespace sv {

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

void MeanCovar2Marker(visualization_msgs::Marker& marker,
                      const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      double scale = 1.0);

// std::vector<visualization_msgs::Marker> Sweep2Markers(
//    const std_msgs::Header& header, const LidarSweep& sweep, float max_curve);

void Match2Markers(std::vector<visualization_msgs::Marker>& markers,
                   const std_msgs::Header& header,
                   const std::vector<PointMatch>& matches,
                   double scale = 1.0);

}  // namespace sv
