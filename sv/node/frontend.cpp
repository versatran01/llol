// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// sv
#include "sv/llol/imu.h"
#include "sv/node/factory.h"
#include "sv/node/viz.h"
#include "sv/util/manager.h"

// others
#include <fmt/ostream.h>

namespace sv {

using geometry_msgs::TransformStamped;
using visualization_msgs::MarkerArray;

struct OdomFrontend {
  /// ros
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Subscriber sub_imu_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  /// params
  bool vis_{true};
  int tbb_{0};

  bool imu_init_{false};
  bool lidar_init_{false};
  std::string lidar_frame_{};
  std::optional<TransformStamped> tf_imu_lidar_;

  /// odom
  ImuBuffer imu_buf_{32};
  LidarSweep sweep_;
  SweepGrid grid_;
  ProjMatcher matcher_;

  TimerManager tm_{"llol"};

  /// Methods
  OdomFrontend(const ros::NodeHandle& pnh);
  void ImuCb(const sensor_msgs::Imu& imu_msg);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void Preprocess(const LidarScan& scan);
  void InitLidar(const sensor_msgs::CameraInfo& cinfo_msg);
};

OdomFrontend::OdomFrontend(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
  sub_camera_ = it_.subscribeCamera("image", 10, &OdomFrontend::CameraCb, this);
  sub_imu_ = pnh_.subscribe("imu", 100, &OdomFrontend::ImuCb, this);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);
}

void OdomFrontend::ImuCb(const sensor_msgs::Imu& imu_msg) {
  // Add debiased data to buffer
  ImuData imu;
  imu.time = imu_msg.header.stamp.toSec();
  const auto& a = imu_msg.linear_acceleration;
  const auto& w = imu_msg.angular_velocity;
  imu.acc = {a.x, a.y, a.z};
  imu.gyr = {w.x, w.y, w.z};
  imu_buf_.push_back(imu);

  // tf stuff
  if (lidar_frame_.empty()) {
    ROS_WARN_STREAM("Lidar frame is empty");
    return;
  }

  if (!tf_imu_lidar_.has_value()) {
    try {
      tf_imu_lidar_ = tf_buffer_.lookupTransform(
          imu_msg.header.frame_id, lidar_frame_, ros::Time(0));

      const auto& t = tf_imu_lidar_->transform.translation;
      const auto& q = tf_imu_lidar_->transform.rotation;
      const Eigen::Vector3d t_imu_lidar{t.x, t.y, t.z};
      const Eigen::Quaterniond q_imu_lidar{q.w, q.x, q.y, q.z};
      const Sophus::SE3d T_imu_lidar{q_imu_lidar, t_imu_lidar};

      ROS_INFO_STREAM("Transform from lidar to imu\n" << T_imu_lidar.matrix());
    } catch (tf2::TransformException& ex) {
      ROS_WARN_STREAM(ex.what());
      return;
    }
  }
}

void OdomFrontend::CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                            const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
  if (lidar_frame_.empty()) {
    lidar_frame_ = image_msg->header.frame_id;
    ROS_INFO_STREAM("Lidar frame: " << lidar_frame_);
  }

  if (!lidar_init_) {
    InitLidar(*cinfo_msg);
  }

  const auto scan = MakeScan(*image_msg, *cinfo_msg);
  Preprocess(scan);

  ROS_DEBUG_STREAM_THROTTLE(1, tm_.ReportAll(true));
}

void OdomFrontend::Preprocess(const LidarScan& scan) {
  int npoints = 0;
  {  /// Add scan to sweep
    auto t = tm_.Scoped("Sweep.Add");
    npoints = sweep_.Add(scan);
  }

  std::pair<int, int> ncells = {0, 0};
  {  /// Reduce scan to grid and Filter
    auto _ = tm_.Scoped("Grid.Add");
    ncells = grid_.Add(scan, tbb_);
  }

  ROS_INFO_STREAM(fmt::format(
      "Scan points: {}, valid cells: {} / {} / {:02.2f}%, filtered cells: "
      "{} / {} / {:02.2f}%",
      npoints,
      ncells.first,
      grid_.total(),
      100.0 * ncells.first / grid_.total(),
      ncells.second,
      grid_.total(),
      100.0 * ncells.second / grid_.total()));

  if (vis_) {
    Imshow("sweep",
           ApplyCmap(sweep_.ExtractRange(), 1 / 32.0, cv::COLORMAP_PINK, 0));
    Imshow("score", ApplyCmap(grid_.score, 1 / 0.2, cv::COLORMAP_VIRIDIS));
    Imshow("filter",
           ApplyCmap(
               grid_.FilterMask(), 1 / grid_.max_score, cv::COLORMAP_VIRIDIS));
  }
}

void OdomFrontend::InitLidar(const sensor_msgs::CameraInfo& cinfo_msg) {
  sweep_ = MakeSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = MakeGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  matcher_ = MakeMatcher({pnh_, "match"});
  ROS_INFO_STREAM(matcher_);

  lidar_init_ = true;
  ROS_INFO_STREAM("Lidar initialized!");
}

}  // namespace sv

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_front");
  sv::OdomFrontend node(ros::NodeHandle("~"));
  ros::spin();
  return 0;
}
