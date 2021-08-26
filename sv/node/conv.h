#pragma once

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Transform.h>
#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_eigen/tf2_eigen.h>

#include "sv/llol/cost.h"
#include "sv/llol/gicp.h"
#include "sv/llol/grid.h"
#include "sv/llol/imu.h"
#include "sv/llol/pano.h"
#include "sv/llol/scan.h"

namespace sv {

/// @brief Ros Conversion
void SE3d2Ros(const Sophus::SE3d& pose, geometry_msgs::Transform& tf);
void SO3d2Ros(const Sophus::SO3d& rot, geometry_msgs::Quaternion& q);
void SE3fVec2Ros(const std::vector<Sophus::SE3f>& poses,
                 geometry_msgs::PoseArray& parray);

/// @brief Factory methods
ImuData MakeImu(const sensor_msgs::Imu& imu_msg);
LidarScan MakeScan(const sensor_msgs::Image& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg);
LidarSweep MakeSweep(const sensor_msgs::CameraInfo& cinfo_msg);
SweepGrid MakeGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size);
GicpSolver MakeGicp(const ros::NodeHandle& pnh);
DepthPano MakePano(const ros::NodeHandle& pnh);

}  // namespace sv
