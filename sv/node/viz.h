#pragma once

#include <geometry_msgs/PoseArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "sv/llol/grid.h"
#include "sv/llol/pano.h"

namespace sv {

using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using CloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

/// @brief Apply color map to mat
/// @details input must be 1-channel, assume after scale the max will be 1
/// default cmap is 10 = PINK. For float image it will set nan to bad_color
cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale = 1.0,
                  int cmap = cv::COLORMAP_PINK,
                  uint8_t bad_color = 255);

/// @brief Create a window with name and show mat
void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

void MeanCovar2Marker(const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      visualization_msgs::Marker& marker);

void Grid2Markers(const SweepGrid& grid,
                  const std_msgs::Header& header,
                  std::vector<visualization_msgs::Marker>& markers);

void Traj2PoseArray(const Trajectory& traj, geometry_msgs::PoseArray& parray);

void Pano2Cloud(const DepthPano& pano,
                const std_msgs::Header header,
                CloudXYZ& cloud);

void Sweep2Cloud(const LidarSweep& sweep,
                 const std_msgs::Header header,
                 CloudXYZI& cloud);

}  // namespace sv
