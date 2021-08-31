#include "sv/node/conv.h"

#include <cv_bridge/cv_bridge.h>

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

void SE3fVecToMsg(const std::vector<Sophus::SE3f>& poses,
                  geometry_msgs::PoseArray& parray) {
  parray.poses.resize(poses.size());

  for (int i = 0; i < poses.size(); ++i) {
    auto& pose = parray.poses.at(i);
    const auto& t = poses.at(i).translation();
    pose.position.x = t.x();
    pose.position.y = t.y();
    pose.position.z = t.z();
    SO3dToMsg(poses.at(i).so3(), pose.orientation);
  }
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
  gp.max_score = pnh.param<double>("max_score", gp.max_score);
  gp.nms = pnh.param<bool>("nms", gp.nms);
  return SweepGrid{sweep_size, gp};
}

DepthPano InitPano(const ros::NodeHandle& pnh) {
  PanoParams pp;
  const auto pano_rows = pnh.param<int>("rows", 256);
  const auto pano_cols = pnh.param<int>("cols", 1024);
  pp.vfov = Deg2Rad(pnh.param<double>("vfov", pp.vfov));
  pp.max_cnt = pnh.param<int>("max_cnt", pp.max_cnt);
  pp.min_range = pnh.param<double>("min_range", pp.min_range);
  pp.range_ratio = pnh.param<double>("range_ratio", pp.range_ratio);
  return DepthPano({pano_cols, pano_rows}, pp);
}

GicpSolver InitGicp(const ros::NodeHandle& pnh) {
  GicpParams gp;
  gp.outer = pnh.param<int>("outer", gp.outer);
  gp.inner = pnh.param<int>("inner", gp.inner);
  gp.half_rows = pnh.param<int>("half_rows", gp.half_rows);
  gp.cov_lambda = pnh.param<double>("cov_lambda", gp.cov_lambda);
  return GicpSolver{gp};
}

Trajectory InitTraj(const ros::NodeHandle& pnh, int grid_cols) {
  Trajectory traj(grid_cols + 1);
  return traj;
}

ImuQueue InitImuq(const ros::NodeHandle& pnh) {
  ImuQueue imuq;
  const auto dt = 1.0 / pnh.param<double>("imu_rate", 100.0);
  const auto acc_noise = pnh.param<double>("acc_noise", 1e-2);
  const auto gyr_noise = pnh.param<double>("gyr_noise", 1e-3);
  const auto acc_bias_noise = pnh.param<double>("acc_bias_noise", 1e-3);
  const auto gyr_bias_noise = pnh.param<double>("gyr_bias_noise", 1e-4);
  imuq.noise = {dt, acc_noise, gyr_noise, acc_bias_noise, gyr_bias_noise};
  const auto acc_bias_std = pnh.param<double>("acc_bias_std", 1e-2);
  const auto gyr_bias_std = pnh.param<double>("gry_bias_std", 1e-3);
  imuq.bias = {acc_bias_std, gyr_bias_std};
  return imuq;
}

}  // namespace sv
