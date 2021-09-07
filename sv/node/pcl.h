#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <std_msgs/Header.h>

#include "sv/llol/grid.h"
#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"

namespace sv {

using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;
using CloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

void Pano2Cloud(const DepthPano& pano,
                const std_msgs::Header header,
                CloudXYZ& cloud);

void Sweep2Cloud(const LidarSweep& sweep,
                 const std_msgs::Header header,
                 CloudXYZI& cloud);

void Grid2Cloud(const SweepGrid& grid,
                const std_msgs::Header& header,
                CloudXYZ& cloud);
}  // namespace sv
