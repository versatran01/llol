#include "sv/node/conv.h"

#include <cv_bridge/cv_bridge.h>

namespace sv {

void SE3d2Ros(const Sophus::SE3d& pose, geometry_msgs::Transform& tf) {
  tf2::toMsg(pose.translation(), tf.translation);
  tf.rotation = tf2::toMsg(pose.unit_quaternion());
}

void SO3d2Ros(const Sophus::SO3d& rot, geometry_msgs::Quaternion& q) {
  q = tf2::toMsg(rot.unit_quaternion());
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

SweepGrid MakeGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size) {
  GridParams gp;
  gp.cell_rows = pnh.param<int>("cell_rows", gp.cell_rows);
  gp.cell_cols = pnh.param<int>("cell_cols", gp.cell_cols);
  gp.max_score = pnh.param<double>("max_score", gp.max_score);
  gp.nms = pnh.param<bool>("nms", gp.nms);
  return SweepGrid{sweep_size, gp};
}

DepthPano MakePano(const ros::NodeHandle& pnh) {
  PanoParams pp;
  const auto pano_rows = pnh.param<int>("rows", 256);
  const auto pano_cols = pnh.param<int>("cols", 1024);
  pp.hfov = Deg2Rad(pnh.param<double>("hfov", pp.hfov));
  pp.max_cnt = pnh.param<int>("max_cnt", pp.max_cnt);
  pp.min_range = pnh.param<double>("min_range", pp.min_range);
  pp.range_ratio = pnh.param<double>("range_ratio", pp.range_ratio);
  return DepthPano({pano_cols, pano_rows}, pp);
}

GicpSolver MakeGicp(const ros::NodeHandle& pnh) {
  GicpParams gp;
  gp.outer = pnh.param<int>("outer", gp.outer);
  gp.inner = pnh.param<int>("inner", gp.inner);
  gp.half_rows = pnh.param<int>("half_rows", gp.half_rows);
  gp.cov_lambda = pnh.param<double>("cov_lambda", gp.cov_lambda);
  return GicpSolver{gp};
}

ImuData MakeImu(const sensor_msgs::Imu& imu_msg) {
  ImuData imu;
  imu.time = imu_msg.header.stamp.toSec();
  const auto& a = imu_msg.linear_acceleration;
  const auto& w = imu_msg.angular_velocity;
  imu.acc = {a.x, a.y, a.z};
  imu.gyr = {w.x, w.y, w.z};
  return imu;
}

void SE3fVec2Ros(const std::vector<Sophus::SE3f>& poses,
                 geometry_msgs::PoseArray& parray) {
  parray.poses.resize(poses.size());

  for (int i = 0; i < poses.size(); ++i) {
    auto& pose = parray.poses.at(i);
    const auto& t = poses.at(i).translation();
    pose.position.x = t.x();
    pose.position.y = t.y();
    pose.position.z = t.z();
    SO3d2Ros(poses.at(i).so3(), pose.orientation);
  }
}

}  // namespace sv
