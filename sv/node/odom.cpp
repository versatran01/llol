// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// sv
#include "sv/llol/gicp.h"
#include "sv/llol/imu.h"
#include "sv/node/conv.h"
#include "sv/node/viz.h"
#include "sv/util/manager.h"
#include "sv/util/solver.h"

// others
#include <fmt/ostream.h>
#include <glog/logging.h>

namespace sv {

using geometry_msgs::TransformStamped;
using visualization_msgs::MarkerArray;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;

struct OdomNode {
  /// ros
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_int_;
  ros::Publisher pub_opt_;
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
  bool imu_init_{false};
  bool pano_init_{false};
  bool sweep_init_{false};

  std::string lidar_frame_{};
  std::string pano_frame_{"pano"};
  std::string odom_frame_{"odom"};

  /// odom
  ImuIntegrator imu_int_;
  LidarSweep sweep_;
  SweepGrid grid_;
  DepthPano pano_;

  std::pair<int, int> icp_iters_;

  TimerManager tm_{"llol"};

  /// Methods
  OdomNode(const ros::NodeHandle& pnh);
  void ImuCb(const sensor_msgs::Imu& imu_msg);
  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void Preprocess(const LidarScan& scan);
  void InitSweep(const sensor_msgs::CameraInfo& cinfo_msg);
  void Integrate();
  void Register();
  void PostProcess();
};

OdomNode::OdomNode(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
  sub_camera_ = it_.subscribeCamera("image", 10, &OdomNode::CameraCb, this);
  sub_imu_ = pnh_.subscribe("imu", 100, &OdomNode::ImuCb, this);

  pub_int_ = pnh_.advertise<geometry_msgs::PoseArray>("parray_int", 1);
  pub_opt_ = pnh_.advertise<geometry_msgs::PoseArray>("parray_opt", 1);
  pub_sweep_ = pnh_.advertise<CloudXYZ>("sweep", 1);
  pub_pano_ = pnh_.advertise<CloudXYZ>("pano", 1);
  pub_match_ = pnh_.advertise<visualization_msgs::MarkerArray>("match", 1);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);

  ros::NodeHandle icp_nh{pnh_, "icp"};
  icp_iters_.first = icp_nh.param<int>("outer", 2);
  icp_iters_.second = icp_nh.param<int>("inner", 2);
  ROS_INFO_STREAM("Icp outer: " << icp_iters_.first
                                << " inner: " << icp_iters_.second);

  pano_ = MakePano({pnh_, "pano"});
  ROS_INFO_STREAM(pano_);
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

  // Transform from pano to odom
  TransformStamped tf_o_p;
  tf_o_p.header.frame_id = odom_frame_;
  tf_o_p.header.stamp = cinfo_msg->header.stamp;
  tf_o_p.child_frame_id = pano_frame_;
  tf_o_p.transform.rotation.w = 1.0;
  tf_broadcaster_.sendTransform(tf_o_p);

  // Allocate storage for sweep, grid and matcher
  if (!sweep_init_) {
    InitSweep(*cinfo_msg);
  }

  // We can always process incoming scan no matter what
  const auto scan = MakeScan(*image_msg, *cinfo_msg);
  // Add scan to sweep, compute score and filter
  Preprocess(scan);

  if (tf_init_) {
    // Predict poses using imu
    Integrate();

    // Publish as pose array
    static geometry_msgs::PoseArray parray_int;
    parray_int.header.frame_id = pano_frame_;
    parray_int.header.stamp = ros::Time(sweep_.time_begin());
    SE3fVec2Ros(grid_.tfs, parray_int);
    pub_int_.publish(parray_int);
  }

  if (pano_init_) {
    Register();

    static visualization_msgs::MarkerArray match_marray;
    std_msgs::Header match_header;
    match_header.frame_id = pano_frame_;
    match_header.stamp = cinfo_msg->header.stamp;
    Grid2Markers(grid_, match_header, match_marray.markers);
    pub_match_.publish(match_marray);

    // Publish as pose array
    static geometry_msgs::PoseArray parray_opt;
    parray_opt.header.frame_id = pano_frame_;
    parray_opt.header.stamp = ros::Time(sweep_.time_begin());
    SE3fVec2Ros(grid_.tfs, parray_opt);
    pub_opt_.publish(parray_opt);
  } else {
    ROS_WARN_STREAM("Pano not initialized");
  }

  PostProcess();

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
  int n_points{};
  {  // Add scan to sweep
    auto t = tm_.Scoped("Sweep.Add");
    n_points = sweep_.Add(scan);
  }

  std::pair<int, int> n_cells{};
  {  // Reduce scan to grid and Filter
    auto _ = tm_.Scoped("Grid.Add");
    n_cells = grid_.Add(scan, tbb_);
  }
  ROS_INFO_STREAM(fmt::format(
      "Scan points: {}, valid cells: {} / {} / {:02.2f}%, filtered cells: "
      "{} / {} / {:02.2f}%",
      n_points,
      n_cells.first,
      grid_.total(),
      100.0 * n_cells.first / grid_.total(),
      n_cells.second,
      grid_.total(),
      100.0 * n_cells.second / grid_.total()));

  if (vis_) {
    Imshow("sweep",
           ApplyCmap(sweep_.DispRange(), 1 / 32.0, cv::COLORMAP_PINK, 0));
    Imshow("score", ApplyCmap(grid_.score, 1 / 0.2, cv::COLORMAP_VIRIDIS));
    Imshow("filter",
           ApplyCmap(
               grid_.DispFilter(), 1 / grid_.max_score, cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::InitSweep(const sensor_msgs::CameraInfo& cinfo_msg) {
  ROS_INFO_STREAM("+++ Initializing lidar");
  sweep_ = MakeSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = MakeGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  sweep_init_ = true;
  ROS_INFO_STREAM("--- Lidar initialized!");
}

void OdomNode::Integrate() {
  auto timer = tm_.Manual("Imu.Integrate");

  const auto t0 = sweep_.time_begin();
  const auto dt = sweep_.dt * grid_.cell_size.width;

  int n_imus{};
  {  // Integarte imu to fill nominal traj
    auto _ = tm_.Scoped("Imu.Integrate");
    n_imus = imu_int_.Predict(t0, dt, grid_.tfs);
  }
  ROS_INFO_STREAM("Integrate imus: " << n_imus);
}

void OdomNode::Register() {
  // Outer icp iters
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  Eigen::Matrix<double, 12, 1> errors;
  TinySolver<AdGicpCostLinear> solver;
  solver.options.max_num_iterations = icp_iters_.second;

  for (int i = 0; i < icp_iters_.first; ++i) {
    errors.setZero();
    ROS_INFO_STREAM("Icp iteration: " << i);

    t_match.Resume();
    const auto n_matches = grid_.Match(pano_, tbb_);
    t_match.Stop(false);

    ROS_INFO_STREAM(fmt::format("Num matches: {} / {} / {:02.2f}% ",
                                n_matches,
                                grid_.total(),
                                100.0 * n_matches / grid_.total()));

    // Build
    t_build.Resume();
    GicpCostLinear cost(grid_, n_matches);
    AdGicpCostLinear adcost(cost);
    t_build.Stop(false);

    // Solve
    t_solve.Resume();
    solver.Solve(adcost, &errors);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // TODO: maybe try interp in SE3?
    ROS_INFO_STREAM("errors: " << errors.transpose());
    // Update
    auto& tfs_g = grid_.tfs;
    ROS_INFO_STREAM(
        "diff nominal before: "
        << (tfs_g.back().translation() - tfs_g.front().translation()).norm());
    const Vector6d dT = errors.tail<6>() - errors.head<6>();
    for (int i = 0; i < tfs_g.size(); ++i) {
      const double s = 1.0 * i / (tfs_g.size() - 1.0);
      CHECK_LE(0, s);
      CHECK_LE(s, 1);
      const Vector6d dTs = errors.head<6>() + s * dT;
      auto& T_p_s = tfs_g.at(i);
      T_p_s = T_p_s * Sophus::SE3f::exp(dTs.cast<float>());
      //      T_p_s.so3() *= Sophus::SO3f::exp(dTs.head<3>().cast<float>());
      //      T_p_s.translation() += dTs.tail<3>().cast<float>();
    }
    ROS_INFO_STREAM("diff dist: " << dT.tail<3>().norm());
    ROS_INFO_STREAM(
        "diff nominal after: "
        << (tfs_g.back().translation() - tfs_g.front().translation()).norm());

    if (vis_) {
      // display good match
      Imshow("match",
             ApplyCmap(grid_.DispMatch(),
                       1.0 / grid_.pano_win_size.area(),
                       cv::COLORMAP_VIRIDIS));
    }
  }
}

void OdomNode::PostProcess() {
  {  // Update sweep poses
    auto _ = tm_.Scoped("Grid.Interp");
    grid_.InterpSweep(sweep_, tbb_);
  }

  int n_added = 0;
  {  // Add sweep to pano
    auto _ = tm_.Scoped("Pano.Add");
    n_added = pano_.Add(sweep_, tbb_);
  }
  ROS_INFO_STREAM(fmt::format("Num added: {} / {} / {:02.2f}%",
                              n_added,
                              sweep_.total(),
                              100.0 * n_added / sweep_.total()));

  if (!pano_init_) {
    pano_init_ = true;
    ROS_INFO_STREAM("Pano initialized");
  }

  // TODO (chao): update first pose of grid for next round of imu integration
  grid_.tfs.front() = grid_.tfs.back();

  if (vis_) {
    const auto& disps = pano_.DispRangeCount();
    Imshow("pano", ApplyCmap(disps[0], 1.0 / DepthPixel::kScale / 30.0));
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
