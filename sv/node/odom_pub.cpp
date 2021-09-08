#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/point_cloud.h>
#include <ros/publisher.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include "sv/node/odom_node.h"
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
  //  static auto pub_odom = pnh_.advertise<Odometry>("odom", 1);
  static auto pub_traj = pnh_.advertise<PoseArray>("traj", 1);
  static auto pub_pose = pnh_.advertise<PoseStamped>("pose", 1);
  static auto pub_bias = pnh_.advertise<sensor_msgs::Imu>("imu_bias", 1);
  static auto pub_bias_std =
      pnh_.advertise<sensor_msgs::Imu>("imu_bias_std", 1);

  static auto pub_feat = pnh_.advertise<CloudXYZ>("feat", 1);
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

  std_msgs::Header pano_header;
  pano_header.frame_id = pano_frame_;
  pano_header.stamp = header.stamp;

  static MarkerArray grid_marray;
  if (pub_grid.getNumSubscribers() > 0) {
    Grid2Markers(grid_, pano_header, grid_marray.markers);
    pub_grid.publish(grid_marray);
  }

  // Publish as pose array
  static PoseArray traj_parray;
  if (pub_traj.getNumSubscribers() > 0) {
    traj_parray.header = pano_header;
    Traj2PoseArray(traj_, traj_parray);
    pub_traj.publish(traj_parray);
  }

  // publish undistorted sweep
  static CloudXYZI sweep_cloud;
  if (pub_sweep.getNumSubscribers() > 0) {
    Sweep2Cloud(sweep_, pano_header, sweep_cloud);
    pub_sweep.publish(sweep_cloud);
  }

  // Publish pano
  static CloudXYZ pano_cloud;
  if (pub_pano.getNumSubscribers() > 0) {
    Pano2Cloud(pano_, pano_header, pano_cloud);
    pub_pano.publish(pano_cloud);
  }

  // Publish match
  static CloudXYZ feat_cloud;
  if (pub_feat.getNumSubscribers() > 0) {
    Grid2Cloud(grid_, pano_header, feat_cloud);
    pub_feat.publish(feat_cloud);
  }

  // publish imu bias
  static sensor_msgs::Imu imu_bias;
  if (pub_bias.getNumSubscribers() > 0) {
    imu_bias.header.stamp = header.stamp;
    tf2::toMsg(imuq_.bias.acc, imu_bias.linear_acceleration);
    tf2::toMsg(imuq_.bias.gyr, imu_bias.angular_velocity);
    pub_bias.publish(imu_bias);
  }

  static sensor_msgs::Imu imu_bias_std;
  if (pub_bias_std.getNumSubscribers() > 0) {
    imu_bias_std.header = imu_bias.header;
    tf2::toMsg(imuq_.bias.acc_var.cwiseSqrt(),
               imu_bias_std.linear_acceleration);
    tf2::toMsg(imuq_.bias.gyr_var.cwiseSqrt(), imu_bias_std.angular_velocity);
    pub_bias_std.publish(imu_bias_std);
  }

  // publish latest traj as path
  PoseStamped pose;
  pose.header.stamp = header.stamp;
  pose.header.frame_id = odom_frame_;
  SE3dToMsg(traj_.TfOdomLidar(), pose.pose);
  pub_pose.publish(pose);

  static Path path;
  path.header = pose.header;

  if (path.poses.empty()) {
    path.poses.push_back(pose);
  } else {
    Eigen::Map<const Vector3d> prev_p(&path.poses.back().pose.position.x);
    Eigen::Map<const Vector3d> curr_p(&pose.pose.position.x);
    if ((prev_p - curr_p).norm() > path_dist_) {
      path.poses.push_back(pose);
    }
  }

  pub_path.publish(path);
}

}  // namespace sv
