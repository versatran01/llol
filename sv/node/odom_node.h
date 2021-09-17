#pragma once

#include <image_transport/image_transport.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
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
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  /// params
  int tbb_{0};
  int log_{0};
  bool vis_{true};

  bool rigid_{false};
  bool tf_init_{false};
  bool scan_init_{false};
  bool traj_updated_{false};
  double path_dist_{0.0};

  /// frames
  std::string imu_frame_{};
  std::string lidar_frame_{};
  std::string pano_frame_{"pano"};
  std::string odom_frame_{"odom"};

  /// odom
  ImuQueue imuq_;
  Trajectory traj_;
  LidarSweep sweep_;
  SweepGrid grid_;
  DepthPano pano_;
  GicpSolver gicp_;

  /// stats
  TimerManager tm_{"llol"};
  StatsManager sm_{"llol"};

  /// Methods
  OdomNode(const ros::NodeHandle& pnh);
  void ImuCb(const sensor_msgs::Imu& imu_msg);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void Publish(const std_msgs::Header& header);
  void Logging();

  void Initialize(const sensor_msgs::CameraInfo& cinfo_msg);
  void Preprocess(const LidarScan& scan);
  void Register();
  bool IcpRigid();
  //  bool IcpLinear();
  void PostProcess(const LidarScan& scan);
};

}  // namespace sv
