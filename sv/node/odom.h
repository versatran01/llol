#pragma once

#include <image_transport/image_transport.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "sv/node/conv.h"
#include "sv/util/manager.h"

namespace sv {

struct OdomNode {
  /// ros
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_traj_;
  ros::Publisher pub_path_;
  ros::Publisher pub_sweep_;
  ros::Publisher pub_pano_;
  ros::Publisher pub_match_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  /// params
  bool vis_{true};
  int tbb_{0};

  bool tf_init_{false};
  bool lidar_init_{false};

  std::string lidar_frame_{};
  std::string pano_frame_{"pano"};
  std::string odom_frame_{"odom"};

  /// odom
  ImuTrajectory traj_;
  LidarSweep sweep_;
  SweepGrid grid_;
  DepthPano pano_;
  GicpSolver gicp_;

  TimerManager tm_{"llol"};

  /// Methods
  OdomNode(const ros::NodeHandle& pnh);
  void ImuCb(const sensor_msgs::Imu& imu_msg);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void Preprocess(const LidarScan& scan);
  void Init(const sensor_msgs::CameraInfo& cinfo_msg);
  void Register();
  void Register2();
  void PostProcess(const LidarScan& scan);
};
}  // namespace sv