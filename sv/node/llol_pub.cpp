#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/point_cloud.h>
#include <ros/publisher.h>
#include <sensor_msgs/Range.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include "sv/node/llol_node.h"
#include "sv/node/pcl.h"
#include "sv/node/viz.h"

namespace sv {

using Vector3d = Eigen::Vector3d;
using RowMat6d = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;
using RowMat34d = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
using geometry_msgs::PoseArray;
using geometry_msgs::PoseStamped;
using geometry_msgs::PoseWithCovarianceStamped;
using geometry_msgs::TransformStamped;
using nav_msgs::Odometry;
using nav_msgs::Path;
using visualization_msgs::MarkerArray;

void OdomNode::Publish(const std_msgs::Header& header) {
  static auto pub_path = pnh_.advertise<Path>("path", 1);
  //  static auto pub_odom = pnh_.advertise<Odometry>("odom", 1);
  static auto pub_traj = pnh_.advertise<PoseArray>("traj", 1);
  static auto pub_pose = pnh_.advertise<PoseStamped>("pose", 1);
  static auto pub_pose_cov =
      pnh_.advertise<PoseWithCovarianceStamped>("pose_cov", 1);
  static auto pub_bias = pnh_.advertise<sensor_msgs::Imu>("imu_bias", 1);
  //  static auto pub_bias_std =
  //      pnh_.advertise<sensor_msgs::Imu>("imu_bias_std", 1);

  static auto pub_grid = pnh_.advertise<MarkerArray>("grid", 1);
  static auto pub_feat = pnh_.advertise<CloudXYZ>("feat", 1);
  static auto pub_sweep = pnh_.advertise<CloudXYZ>("sweep", 1);
  // Need to have two levels here for ImageTransport to namespace the
  // camera_info topic properly
  static auto pub_pano_image = it_.advertiseCamera("pano/img", 1);
  static auto pub_pano_viz_image = it_.advertiseCamera("pano_viz/img", 1);
  static auto pub_pano_cloud = pnh_.advertise<CloudXYZ>("pano_cloud", 1);

  static auto pub_runtime = pnh_.advertise<sensor_msgs::Range>("runtime", 1);

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
  if (pub_pano_cloud.getNumSubscribers() > 0) {
    Pano2Cloud(pano_, pano_header, pano_cloud);
    pub_pano_cloud.publish(pano_cloud);
  }

  // Publish match
  static CloudXYZI feat_cloud;
  if (pub_feat.getNumSubscribers() > 0) {
    Grid2Cloud(grid_, pano_header, feat_cloud);
    pub_feat.publish(feat_cloud);
  }

  // publish imu bias
  if (pub_bias.getNumSubscribers() > 0) {
    sensor_msgs::Imu imu_bias;
    imu_bias.header.stamp = header.stamp;
    tf2::toMsg(imuq_.bias.acc, imu_bias.linear_acceleration);
    tf2::toMsg(imuq_.bias.gyr, imu_bias.angular_velocity);
    pub_bias.publish(imu_bias);
  }

  //  static sensor_msgs::Imu imu_bias_std;
  //  if (pub_bias_std.getNumSubscribers() > 0) {
  //    imu_bias_std.header = imu_bias.header;
  //    tf2::toMsg(imuq_.bias.acc_var.cwiseSqrt(),
  //               imu_bias_std.linear_acceleration);
  //    tf2::toMsg(imuq_.bias.gyr_var.cwiseSqrt(),
  //    imu_bias_std.angular_velocity); pub_bias_std.publish(imu_bias_std);
  //  }

  // publish time info
  if (pub_runtime.getNumSubscribers() > 0) {
    sensor_msgs::Range runtime;
    runtime.header = header;
    const auto stat = tm_.GetStats("Total");
    runtime.field_of_view = absl::ToDoubleSeconds(stat.mean());
    runtime.min_range = absl::ToDoubleSeconds(stat.min());
    runtime.max_range = absl::ToDoubleSeconds(stat.max());
    runtime.range = absl::ToDoubleSeconds(stat.last());
    pub_runtime.publish(runtime);
  }

  // publish pano image/cinfo
  static cv_bridge::CvImagePtr cv_ptr;
  static sensor_msgs::CameraInfoPtr cinfo_msg = 
    boost::make_shared<sensor_msgs::CameraInfo>();
  static sensor_msgs::ImagePtr image_msg;
  if (pub_pano_image.getNumSubscribers() > 0 || 
      pub_pano_viz_image.getNumSubscribers() > 0) {
    if (T_odom_pano_.has_value()) {
      cinfo_msg->header.stamp = header.stamp;
      cinfo_msg->header.frame_id = completed_pano_frame_;
      cinfo_msg->width = pano_.size().width;
      cinfo_msg->height = pano_.size().height;
      Eigen::Map<RowMat34d> P_map(&cinfo_msg->P[0]);
      P_map = T_odom_pano_->matrix3x4();
      cinfo_msg->R[0] = 1/DepthPixel::kScale;

      // Publish transform
      // This is one pano stamp older, so different frame from pano_frame_
      TransformStamped tf_o_pi;
      tf_o_pi.header.frame_id = odom_frame_;
      tf_o_pi.header.stamp = header.stamp;
      tf_o_pi.child_frame_id = completed_pano_frame_;
      SE3dToMsg(*T_odom_pano_, tf_o_pi.transform);
      tf_broadcaster.sendTransform(tf_o_pi);

      if (pub_pano_image.getNumSubscribers() > 0) {
        image_msg =
            cv_bridge::CvImage(cinfo_msg->header, "16UC2", pano_.dbuf2).toImageMsg();
        pub_pano_image.publish(image_msg, cinfo_msg);
      }
      if (pub_pano_viz_image.getNumSubscribers() > 0) {
        // extract depth channel for rqt
        cv::Mat channel[2];
        cv::split(pano_.dbuf2, channel);
        
        image_msg =
            cv_bridge::CvImage(cinfo_msg->header, "bgr8", 
                ApplyCmap(channel[0], 1.0 / 65536, cv::COLORMAP_JET)).toImageMsg();
        pub_pano_viz_image.publish(image_msg, cinfo_msg);
      }

      // clear so only publish once pano is done
      T_odom_pano_.reset();
    }
  }

  // publish latest traj as path
  PoseStamped pose;
  pose.header.stamp = header.stamp;
  pose.header.frame_id = odom_frame_;
  SE3dToMsg(traj_.TfOdomLidar(), pose.pose);
  pub_pose.publish(pose);

  if (pub_pose_cov.getNumSubscribers() > 0) {
    PoseWithCovarianceStamped pose_cov;
    pose_cov.header = pose.header;
    pose_cov.pose.pose = pose.pose;
    Eigen::Map<RowMat6d> cov(&pose_cov.pose.covariance[0]);
    // transform covariance from local frame to odom frame
    const auto R_odom_lidar = traj_.TfOdomLidar().so3().matrix();
    cov.topLeftCorner<3, 3>().noalias() = R_odom_lidar *
                                          traj_.cov.bottomRightCorner<3, 3>() *
                                          R_odom_lidar.transpose();
    cov.bottomRightCorner<3, 3>().noalias() = R_odom_lidar *
                                              traj_.cov.topLeftCorner<3, 3>() *
                                              R_odom_lidar.transpose();
    pub_pose_cov.publish(pose_cov);
  }

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
