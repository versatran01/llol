#pragma once

#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include "sv/llol/grid.h"
#include "sv/llol/match.h"
#include "sv/llol/pano.h"
#include "sv/llol/scan.h"

namespace sv {

LidarScan MakeScan(const sensor_msgs::Image& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg);

LidarSweep MakeSweep(const sensor_msgs::CameraInfo& cinfo_msg);

SweepGrid MakeGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size);

ProjMatcher MakeMatcher(const ros::NodeHandle& pnh);

DepthPano MakePano(const ros::NodeHandle& pnh);

}  // namespace sv
