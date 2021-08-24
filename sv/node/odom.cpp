// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// sv
#include "sv/llol/imu.h"
#include "sv/node/conv.h"
#include "sv/node/viz.h"
#include "sv/util/manager.h"

// others
#include <fmt/ostream.h>
#include <glog/logging.h>

namespace sv {

using geometry_msgs::TransformStamped;
using visualization_msgs::MarkerArray;

struct OdomNode {
  /// ros
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_nom_;
  ros::Publisher pub_opt_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  /// params
  bool vis_{true};
  int tbb_{0};

  bool tf_init_{false};
  bool imu_init_{false};
  bool lidar_init_{false};
  std::string lidar_frame_{};
  std::string pano_frame_{"pano"};
  std::string odom_frame_{"odom"};

  /// odom
  ImuIntegrator imu_int_;
  //  ImuBuffer imu_buf_{32};
  LidarSweep sweep_;
  SweepGrid grid_;
  ProjMatcher matcher_;

  TimerManager tm_{"llol"};

  /// Methods
  OdomNode(const ros::NodeHandle& pnh);
  void ImuCb(const sensor_msgs::Imu& imu_msg);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void Preprocess(const LidarScan& scan);
  void InitLidar(const sensor_msgs::CameraInfo& cinfo_msg);
  void Integrate();
  void PostProcess();
};

OdomNode::OdomNode(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
  sub_camera_ = it_.subscribeCamera("image", 10, &OdomNode::CameraCb, this);
  sub_imu_ = pnh_.subscribe("imu", 100, &OdomNode::ImuCb, this);

  pub_nom_ = pnh_.advertise<geometry_msgs::PoseArray>("parray_nom", 1);
  pub_opt_ = pnh_.advertise<nav_msgs::Path>("path_opt", 1);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);
}

void OdomNode::ImuCb(const sensor_msgs::Imu& imu_msg) {
  // Add imu data to buffer
  const auto imu = MakeImu(imu_msg);
  imu_int_.Add(imu);
  //  imu_buf_.push_back(imu);

  if (tf_init_) return;

  // tf stuff
  if (lidar_frame_.empty()) {
    ROS_WARN_STREAM("Lidar frame is empty");
    return;
  }

  try {
    const auto tf_imu_lidar = tf_buffer_.lookupTransform(
        imu_msg.header.frame_id, lidar_frame_, ros::Time(0));

    const auto& t = tf_imu_lidar.transform.translation;
    const auto& q = tf_imu_lidar.transform.rotation;
    const Eigen::Vector3d t_imu_lidar{t.x, t.y, t.z};
    const Eigen::Quaterniond q_imu_lidar{q.w, q.x, q.y, q.z};
    imu_int_.T_imu_lidar = Sophus::SE3d{q_imu_lidar, t_imu_lidar};

    ROS_INFO_STREAM("Transform from lidar to imu\n"
                    << imu_int_.T_imu_lidar.matrix());
    tf_init_ = true;
  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM(ex.what());
    return;
  }
}

void OdomNode::CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                        const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
  if (lidar_frame_.empty()) {
    lidar_frame_ = image_msg->header.frame_id;
    ROS_INFO_STREAM("Lidar frame: " << lidar_frame_);
  }

  if (!lidar_init_) {
    InitLidar(*cinfo_msg);
  }

  // We can always process incoming scan no matter what
  const auto scan = MakeScan(*image_msg, *cinfo_msg);
  Preprocess(scan);

  if (tf_init_) {
    Integrate();
  }

  // TODO (chao): hack, need to remove
  grid_.tf_p_s.front() = grid_.tf_p_s.back();

  /// Transform from pano to odom
  TransformStamped tf_o_p;
  tf_o_p.header.frame_id = odom_frame_;
  tf_o_p.header.stamp = cinfo_msg->header.stamp;
  tf_o_p.child_frame_id = pano_frame_;
  tf_o_p.transform.rotation.w = 1.0;
  tf_broadcaster_.sendTransform(tf_o_p);

  ROS_DEBUG_STREAM_THROTTLE(1, tm_.ReportAll(true));
}

void OdomNode::Preprocess(const LidarScan& scan) {
  int npoints = 0;
  {  // Add scan to sweep
    auto t = tm_.Scoped("Sweep.Add");
    npoints = sweep_.Add(scan);
  }

  std::pair<int, int> ncells = {0, 0};
  {  // Reduce scan to grid and Filter
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

void OdomNode::InitLidar(const sensor_msgs::CameraInfo& cinfo_msg) {
  ROS_INFO_STREAM("+++ Initializing lidar");
  sweep_ = MakeSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = MakeGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  matcher_ = MakeMatcher({pnh_, "match"});
  ROS_INFO_STREAM(matcher_);

  lidar_init_ = true;
  ROS_INFO_STREAM("--- Lidar initialized!");
}

void OdomNode::Integrate() {
  auto timer = tm_.Manual("Imu.Integrate");

  const auto t0 = sweep_.time_begin();
  const auto dt = sweep_.dt * grid_.cell_size.width;
  auto& nominal = grid_.tf_p_s;

  int nimus = 0;
  {  // Integarte imu to fill nominal traj
    auto _ = tm_.Scoped("Imu.Integrate");
    nimus = imu_int_.Integrate(t0, dt, absl::MakeSpan(nominal));
  }
  ROS_INFO_STREAM("Integrate pose using " << nimus << " imus");

  // Publish as pose array
  static geometry_msgs::PoseArray parray;
  parray.header.frame_id = pano_frame_;
  parray.header.stamp = ros::Time(t0);
  SE3fSpan2Ros(absl::MakeConstSpan(nominal), parray);
  pub_nom_.publish(parray);
}

}  // namespace sv

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_node");
  sv::OdomNode node(ros::NodeHandle("~"));
  ros::spin();
  return 0;
}
