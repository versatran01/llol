#pragma once

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Transform.h>
#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_eigen/tf2_eigen.h>

#include "sv/llol/gicp.h"
#include "sv/llol/grid.h"
#include "sv/llol/imu.h"
#include "sv/llol/pano.h"
#include "sv/llol/scan.h"
#include "sv/llol/traj.h"

namespace sv {

/// @brief Ros Conversion
void SE3dToMsg(const Sophus::SE3d& se3, geometry_msgs::Pose& pose);
void SE3dToMsg(const Sophus::SE3d& se3, geometry_msgs::Transform& tf);
void SO3dToMsg(const Sophus::SO3d& so3, geometry_msgs::Quaternion& q);
void SE3fVecToMsg(const std::vector<Sophus::SE3f>& se3s,
                  geometry_msgs::PoseArray& parray);

/// @brief Factory methods
ImuData MakeImu(const sensor_msgs::Imu& imu_msg);
LidarScan MakeScan(const sensor_msgs::Image& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg);
LidarSweep MakeSweep(const sensor_msgs::CameraInfo& cinfo_msg);

ImuQueue InitImuq(const ros::NodeHandle& pnh);
Trajectory InitTraj(const ros::NodeHandle& pnh, int grid_cols);
SweepGrid InitGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size);
DepthPano InitPano(const ros::NodeHandle& pnh);
GicpSolver InitGicp(const ros::NodeHandle& pnh);

}  // namespace sv
