#pragma once

#include <geometry_msgs/Transform.h>
#include <tf2_eigen/tf2_eigen.h>

#include <sophus/se3.hpp>

namespace sv {

void SE3d2Transform(const Sophus::SE3d& pose, geometry_msgs::Transform& tf);

}  // namespace sv
