#include "sv/node/conv.h"

#include <cv_bridge/cv_bridge.h>
#include <tf2_eigen/tf2_eigen.h>

namespace sv {

void SE3dToMsg(const Sophus::SE3d& se3, geometry_msgs::Transform& tf) {
  tf2::toMsg(se3.translation(), tf.translation);
  tf.rotation = tf2::toMsg(se3.unit_quaternion());
}

void SO3dToMsg(const Sophus::SO3d& so3, geometry_msgs::Quaternion& q) {
  q = tf2::toMsg(so3.unit_quaternion());
}

void SE3dToMsg(const Sophus::SE3d& se3, geometry_msgs::Pose& pose) {
  auto& p = pose.position;
  const auto& t = se3.translation();
  p.x = t.x();
  p.y = t.y();
  p.z = t.z();
  SO3dToMsg(se3.so3(), pose.orientation);
}

/// Make
ImuData MakeImu(const sensor_msgs::Imu& imu_msg) {
  ImuData imu;
  imu.time = imu_msg.header.stamp.toSec();
  const auto& a = imu_msg.linear_acceleration;
  const auto& w = imu_msg.angular_velocity;
  imu.acc = {a.x, a.y, a.z};
  imu.gyr = {w.x, w.y, w.z};
  return imu;
}

LidarScan MakeScan(const sensor_msgs::Image& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(image_msg, "32FC4");

  return {image_msg.header.stamp.toSec(),    // t
          cinfo_msg.K[0],                    // dt
          cinfo_msg.R[0],                    // scale
          cv_ptr->image,                     // xyzr
          cv::Range(cinfo_msg.roi.x_offset,  // col_rg
                    cinfo_msg.roi.x_offset + cinfo_msg.roi.width)};
}

LidarSweep MakeSweep(const sensor_msgs::CameraInfo& cinfo_msg) {
  return LidarSweep{cv::Size(cinfo_msg.width, cinfo_msg.height)};
}

SweepGrid InitGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size) {
  GridParams gp;
  gp.cell_rows = pnh.param<int>("cell_rows", gp.cell_rows);
  gp.cell_cols = pnh.param<int>("cell_cols", gp.cell_cols);
  gp.max_curve = pnh.param<double>("max_curve", gp.max_curve);
  gp.max_var = pnh.param<double>("max_var", gp.max_var);
  gp.nms = pnh.param<bool>("nms", gp.nms);
  return SweepGrid{sweep_size, gp};
}

DepthPano InitPano(const ros::NodeHandle& pnh) {
  PanoParams pp;
  const auto pano_rows = pnh.param<int>("rows", 256);
  const auto pano_cols = pnh.param<int>("cols", 1024);
  pp.vfov = Deg2Rad(pnh.param<double>("vfov", pp.vfov));
  pp.max_cnt = pnh.param<int>("max_cnt", pp.max_cnt);
  pp.min_sweeps = pnh.param<int>("min_sweeps", pp.min_sweeps);
  pp.min_range = pnh.param<double>("min_range", pp.min_range);
  pp.max_range = pnh.param<double>("max_range", pp.max_range);
  pp.win_ratio = pnh.param<double>("win_ratio", pp.win_ratio);
  pp.fuse_ratio = pnh.param<double>("fuse_ratio", pp.fuse_ratio);
  pp.align_gravity = pnh.param<bool>("align_gravity", pp.align_gravity);
  pp.min_match_ratio = pnh.param<double>("min_match_ratio", pp.min_match_ratio);
  pp.max_translation = pnh.param<double>("max_translation", pp.max_translation);
  return DepthPano({pano_cols, pano_rows}, pp);
}

GicpSolver InitGicp(const ros::NodeHandle& pnh) {
  GicpParams gp;
  gp.outer = pnh.param<int>("outer", gp.outer);
  gp.inner = pnh.param<int>("inner", gp.inner);
  gp.half_rows = pnh.param<int>("half_rows", gp.half_rows);
  gp.half_cols = pnh.param<int>("half_cols", gp.half_cols);
  gp.cov_lambda = pnh.param<double>("cov_lambda", gp.cov_lambda);
  gp.imu_weight = pnh.param<double>("imu_weight", gp.imu_weight);
  gp.min_eigval = pnh.param<double>("min_eigval", gp.min_eigval);
  return GicpSolver{gp};
}

Trajectory InitTraj(const ros::NodeHandle& pnh, int grid_cols) {
  TrajectoryParams tp;
  tp.use_acc = pnh.param<bool>("use_acc", tp.use_acc);
  tp.update_bias = pnh.param<bool>("update_bias", tp.update_bias);
  tp.gravity_norm = pnh.param<double>("gravity_norm", tp.gravity_norm);
  return Trajectory{grid_cols + 1, tp};
}

ImuQueue InitImuq(const ros::NodeHandle& pnh) {
  const auto buf_size = pnh.param<int>("buffer_size", 20);
  ImuQueue imuq(buf_size);

  const auto rate = pnh.param<double>("imu_rate", 100.0);
  const auto acc_noise = pnh.param<double>("acc_noise", 1e-2);
  const auto gyr_noise = pnh.param<double>("gyr_noise", 1e-3);
  const auto acc_bias_noise = pnh.param<double>("acc_bias_noise", 1e-3);
  const auto gyr_bias_noise = pnh.param<double>("gyr_bias_noise", 1e-4);
  imuq.noise = {rate, acc_noise, gyr_noise, acc_bias_noise, gyr_bias_noise};

  const auto acc_bias_std = pnh.param<double>("acc_bias_std", 1e-2);
  const auto gyr_bias_std = pnh.param<double>("gry_bias_std", 1e-3);
  imuq.bias = {acc_bias_std, gyr_bias_std};

  return imuq;
}

}  // namespace sv
