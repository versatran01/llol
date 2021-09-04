#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/point_cloud.h>
#include <ros/publisher.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include "sv/node/odom.h"
#include "sv/node/pcl.h"
#include "sv/node/viz.h"

namespace sv {

using geometry_msgs::PoseArray;
using geometry_msgs::PoseStamped;
using geometry_msgs::TransformStamped;
using nav_msgs::Odometry;
using nav_msgs::Path;
using visualization_msgs::MarkerArray;

void OdomNode::Publish(const std_msgs::Header& header) {
  static auto pub_path = pnh_.advertise<Path>("path", 1);
  static auto pub_odom = pnh_.advertise<Odometry>("odom", 1);
  static auto pub_traj = pnh_.advertise<PoseArray>("traj", 1);
  static auto pub_pose = pnh_.advertise<PoseStamped>("pose", 1);
  static auto pub_bias = pnh_.advertise<sensor_msgs::Imu>("imu_bias", 1);

  static auto pub_pano = pnh_.advertise<CloudXYZ>("pano", 1);
  static auto pub_sweep = pnh_.advertise<CloudXYZ>("sweep", 1);
  static auto pub_grid = pnh_.advertise<MarkerArray>("grid", 1);

  static tf2_ros::TransformBroadcaster tf_broadcaster;

  // Transform from pano to odom
  TransformStamped tf_o_p;
  tf_o_p.header.frame_id = odom_frame_;
  tf_o_p.header.stamp = header.stamp;
  tf_o_p.child_frame_id = pano_frame_;
  SE3dToMsg(traj_.T_odom_pano, tf_o_p.transform);
  tf_broadcaster.sendTransform(tf_o_p);

  static MarkerArray grid_marray;
  std_msgs::Header grid_header;
  grid_header.frame_id = pano_frame_;
  grid_header.stamp = header.stamp;
  Grid2Markers(grid_, grid_header, grid_marray.markers);
  pub_grid.publish(grid_marray);

  // Publish as pose array
  static PoseArray traj_parray;
  traj_parray.header.frame_id = pano_frame_;
  traj_parray.header.stamp = header.stamp;
  Traj2PoseArray(traj_, traj_parray);
  pub_traj.publish(traj_parray);

  // publish undistorted sweep
  static CloudXYZI sweep_cloud;
  std_msgs::Header sweep_header;
  sweep_header.frame_id = pano_frame_;
  sweep_header.stamp = header.stamp;
  Sweep2Cloud(sweep_, sweep_header, sweep_cloud);
  pub_sweep.publish(sweep_cloud);

  // Publish pano
  static CloudXYZ pano_cloud;
  std_msgs::Header pano_header;
  pano_header.frame_id = pano_frame_;
  pano_header.stamp = header.stamp;
  Pano2Cloud(pano_, pano_header, pano_cloud);
  pub_pano.publish(pano_cloud);

  // publish imu bias
  sensor_msgs::Imu imu_bias;
  imu_bias.header.stamp = header.stamp;
  imu_bias.header.frame_id = imu_frame_;
  tf2::toMsg(imuq_.bias.acc, imu_bias.linear_acceleration);
  tf2::toMsg(imuq_.bias.gyr, imu_bias.angular_velocity);
  pub_bias.publish(imu_bias);

  // publish latest traj as path
  static Path path;
  path.header.stamp = header.stamp;
  path.header.frame_id = odom_frame_;
  PoseStamped pose;
  pose.header.stamp = path.header.stamp;
  pose.header.frame_id = path.header.frame_id;
  SE3dToMsg(traj_.TfOdomLidar(), pose.pose);
  path.poses.push_back(pose);
  pub_path.publish(path);

  // publish odom
  Odometry odom;
  odom.header = path.header;
  odom.pose.pose = pose.pose;
  pub_odom.publish(odom);
  pub_pose.publish(pose);
}

}  // namespace sv
