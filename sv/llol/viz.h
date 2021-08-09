#pragma once

#include <visualization_msgs/MarkerArray.h>

#include "sv/llol/lidar.h"
#include "sv/llol/match.h"

namespace sv {

MeanCovar3f CalcMeanCovar(const cv::Mat& mat);

void MeanCovar2Marker(visualization_msgs::Marker& marker,
                      const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      double scale = 1.0);

std::vector<visualization_msgs::Marker> Sweep2Markers(
    const std_msgs::Header& header, const LidarSweep& sweep, float max_curve);

void Match2Markers(std::vector<visualization_msgs::Marker>& markers,
                   const std_msgs::Header& header,
                   const std::vector<PointMatch>& matches,
                   double scale = 1.0);

}  // namespace sv
