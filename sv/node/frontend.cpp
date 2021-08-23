// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

// sv
#include "sv/node/factory.h"
#include "sv/node/viz.h"
#include "sv/util/manager.h"

namespace sv {

struct OdomFrontend {
  /// ros
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;

  /// params
  bool vis_{true};
  int tbb_{0};
  bool lidar_init_{false};
  std::string lidar_frame_{};

  /// odom
  LidarSweep sweep_;
  SweepGrid grid_;
  ProjMatcher matcher_;

  TimerManager tm_{"llol"};
  StatsManager sm_{"llol"};

  /// Methods
  OdomFrontend(const ros::NodeHandle& pnh);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void InitLidar(const sensor_msgs::CameraInfo& cinfo_msg);
};

OdomFrontend::OdomFrontend(const ros::NodeHandle& pnh) : pnh_{pnh}, it_{pnh} {
  sub_camera_ = it_.subscribeCamera("image", 10, &OdomFrontend::CameraCb, this);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);
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

  int npoints = 0;
  {  /// Add scan to sweep
    auto t = tm_.Scoped("Sweep.Add");
    sweep_.Add(scan);
  }

  std::pair<int, int> ncells;
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

  ROS_DEBUG_STREAM_THROTTLE(1, tm_.ReportAll(true));
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
