// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/camera_subscriber.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

// sv
#include "sv/node/odom.h"
#include "sv/node/viz.h"

namespace sv {

using geometry_msgs::PoseArray;
using geometry_msgs::TransformStamped;
using visualization_msgs::MarkerArray;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;

OdomNode::OdomNode(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
  sub_camera_ = it_.subscribeCamera("image", 10, &OdomNode::CameraCb, this);
  sub_imu_ = pnh_.subscribe("imu", 100, &OdomNode::ImuCb, this);

  pub_traj_ = pnh_.advertise<PoseArray>("traj", 1);
  pub_path_ = pnh_.advertise<nav_msgs::Path>("path", 1);
  pub_sweep_ = pnh_.advertise<CloudXYZ>("sweep", 1);
  pub_pano_ = pnh_.advertise<CloudXYZ>("pano", 1);
  pub_match_ = pnh_.advertise<MarkerArray>("match", 1);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);

  pano_ = InitPano({pnh_, "pano"});
  ROS_INFO_STREAM(pano_);
}

void OdomNode::ImuCb(const sensor_msgs::Imu& imu_msg) {
  // Add imu data to buffer
  const auto imu = MakeImuData(imu_msg);
  imu_.Add(imu);

  if (tf_init_) return;

  // tf stuff
  if (!lidar_init_) {
    ROS_WARN_STREAM("Lidar not initialized");
    return;
  }

  try {
    const auto tf_i_l = tf_buffer_.lookupTransform(
        imu_msg.header.frame_id, lidar_frame_, ros::Time(0));

    const auto& t = tf_i_l.transform.translation;
    const auto& q = tf_i_l.transform.rotation;
    const Eigen::Vector3d t_i_l{t.x, t.y, t.z};
    const Eigen::Quaterniond q_i_l{q.w, q.x, q.y, q.z};
    imu_.InitExtrinsic({q_i_l, t_i_l}, 9.80184);
    ROS_INFO_STREAM("Transform from lidar to imu:\n"
                    << imu_.T_imu_lidar.matrix());
    ROS_INFO_STREAM("Gravity: " << imu_.gravity.transpose());
    ROS_INFO_STREAM("Transform from pano to odom:\n"
                    << imu_.T_odom_pano.matrix());
    tf_init_ = true;
  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM(ex.what());
    return;
  }
}

void OdomNode::Init(const sensor_msgs::CameraInfo& cinfo_msg) {
  ROS_INFO_STREAM("+++ Initializing");
  sweep_ = InitSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = InitGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  imu_ = InitImu({pnh_, "imu"}, grid_.cols());
  ROS_INFO_STREAM(imu_.noise);

  gicp_ = InitGicp({pnh_, "gicp"});
  ROS_INFO_STREAM(gicp_);
}

void OdomNode::CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                        const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
  if (lidar_frame_.empty()) {
    lidar_frame_ = image_msg->header.frame_id;
    ROS_INFO_STREAM("Lidar frame: " << lidar_frame_);
  }

  // Allocate storage for sweep, grid and matcher
  if (!lidar_init_) {
    Init(*cinfo_msg);
    lidar_init_ = true;
    ROS_INFO_STREAM("Lidar initialized!");
  }

  if (!imu_.buf.full()) {
    ROS_WARN_STREAM(fmt::format(
        "Imu buffer not full: {}/{}", imu_.buf.size(), imu_.buf.capacity()));
    return;
  }

  if (!tf_init_) {
    ROS_WARN_STREAM("Transform not initialized");
    return;
  }

  // Transform from pano to odom
  TransformStamped tf_o_p;
  tf_o_p.header.frame_id = odom_frame_;
  tf_o_p.header.stamp = cinfo_msg->header.stamp;
  tf_o_p.child_frame_id = pano_frame_;
  SE3d2Ros(imu_.T_odom_pano, tf_o_p.transform);
  tf_broadcaster_.sendTransform(tf_o_p);

  // We can always process incoming scan no matter what
  const auto scan = MakeScan(*image_msg, *cinfo_msg);
  // Add scan to sweep, compute score and filter
  Preprocess(scan);

  Register();

  static MarkerArray match_marray;
  std_msgs::Header match_header;
  match_header.frame_id = pano_frame_;
  match_header.stamp = cinfo_msg->header.stamp;
  Grid2Markers(grid_, match_header, match_marray.markers);
  pub_match_.publish(match_marray);

  // Publish as pose array
  static PoseArray parray_traj;
  parray_traj.header.frame_id = pano_frame_;
  parray_traj.header.stamp = cinfo_msg->header.stamp;
  SE3fVec2Ros(grid_.tfs, parray_traj);
  pub_traj_.publish(parray_traj);

  PostProcess(scan);

  // publish undistorted sweep
  static CloudXYZ sweep_cloud;
  std_msgs::Header sweep_header;
  sweep_header.frame_id = pano_frame_;
  sweep_header.stamp = cinfo_msg->header.stamp;
  Sweep2Cloud(sweep_, sweep_header, sweep_cloud);
  pub_sweep_.publish(sweep_cloud);

  // Publish pano
  static CloudXYZ pano_cloud;
  std_msgs::Header pano_header;
  pano_header.frame_id = pano_frame_;
  pano_header.stamp = cinfo_msg->header.stamp;
  Pano2Cloud(pano_, pano_header, pano_cloud);
  pub_pano_.publish(pano_cloud);

  ROS_DEBUG_STREAM_THROTTLE(1, tm_.ReportAll(true));
}

void OdomNode::Preprocess(const LidarScan& scan) {
  cv::Vec2i n_cells{};
  {  // Reduce scan to grid and Filter
    auto _ = tm_.Scoped("Grid.Add");
    n_cells = grid_.Add(scan, tbb_);
  }

  const int grid_total = grid_.total();

  ROS_INFO_STREAM(
      fmt::format("[Grid.Add]: valid cells: {} / {} / {:02.2f}%, "
                  "filtered cells: {} / {} / {:02.2f}%",
                  n_cells[0],
                  grid_total,
                  100.0 * n_cells[0] / grid_total,
                  n_cells[1],
                  grid_total,
                  100.0 * n_cells[1] / grid_total));

  int n_imus{};
  {  // Integarte imu to fill nominal traj
    auto _ = tm_.Scoped("Imu.Integrate");
    const auto t0 = scan.t0;
    const auto dt = scan.dt * grid_.cell_size.width;
    n_imus = imu_.Predict(t0, dt);
  }
  ROS_INFO_STREAM("[Imu.Predict] using imus: " << n_imus);

  if (vis_) {
    Imshow("score", ApplyCmap(grid_.mat, 1 / 0.2, cv::COLORMAP_VIRIDIS));
    Imshow("filter",
           ApplyCmap(
               grid_.DrawFilter(), 1 / grid_.max_score, cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::PostProcess(const LidarScan& scan) {
  int n_added = 0;
  {  // Add sweep to pano
    auto _ = tm_.Scoped("Pano.Add");
    n_added = pano_.Add(sweep_, tbb_);
  }
  ROS_INFO_STREAM(fmt::format("[Pano.Add] Num added: {} / {} / {:02.2f}%",
                              n_added,
                              sweep_.total(),
                              100.0 * n_added / sweep_.total()));

  int n_points = 0;
  {  // Add scan to sweep
    auto _ = tm_.Scoped("Sweep.Add");
    n_points = sweep_.Add(scan);
  }
  ROS_INFO_STREAM(fmt::format("[Sweep.Add] Num added: {} / {} / {:02.2f}%",
                              n_points,
                              sweep_.total(),
                              100.0 * n_points / sweep_.total()));

  {  // Update sweep tfs for undistortion
    auto _ = tm_.Scoped("Sweep.Interp");
    sweep_.Interp(imu_, tbb_);
  }

  // TODO (chao): update first pose of traj for next round of imu integration
  imu_.states.front() = imu_.states.back();

  if (vis_) {
    const double max_range = 32.0;
    Imshow("sweep",
           ApplyCmap(sweep_.DrawRange(), 1 / max_range, cv::COLORMAP_PINK, 0));
    const auto& disps = pano_.DrawRangeCount();
    Imshow(
        "pano",
        ApplyCmap(
            disps[0], 1.0 / DepthPixel::kScale / max_range, cv::COLORMAP_PINK));
    Imshow("count",
           ApplyCmap(disps[1], 1.0 / pano_.max_cnt, cv::COLORMAP_VIRIDIS));
  }
}

}  // namespace sv

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_node");
  sv::OdomNode node(ros::NodeHandle("~"));
  ros::spin();
  return 0;
}
