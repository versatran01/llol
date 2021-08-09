#pragma once

#include <visualization_msgs/MarkerArray.h>

#include <opencv2/core/mat.hpp>

#include "sv/util/math.h"

namespace sv {

MeanCovar3f CalcMeanCovar(const cv::Mat& mat);

void MeanCovar2Marker(const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      visualization_msgs::Marker& marker);

auto Sweep2Gaussians(const cv::Mat& sweep, const cv::Mat& grid, float max_curve)
    -> visualization_msgs::MarkerArray;
}  // namespace sv
