#include "sv/node/odom.h"

#include <tf2_ros/transform_listener.h>

#include "sv/node/viz.h"

namespace sv {

OdomNode::OdomNode(const ros::NodeHandle& pnh) : pnh_{pnh}, it_{pnh} {
  sub_camera_ = it_.subscribeCamera("image", 20, &OdomNode::CameraCb, this);
  sub_imu_ = pnh_.subscribe("imu", 200, &OdomNode::ImuCb, this);

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);

  rigid_ = pnh_.param<bool>("rigid", true);
  ROS_WARN_STREAM("GICP: " << (rigid_ ? "Rigid" : "Linear"));

  imuq_ = InitImuq({pnh_, "imuq"});
  ROS_INFO_STREAM(imuq_);

  pano_ = InitPano({pnh_, "pano"});
  ROS_INFO_STREAM(pano_);
}

void OdomNode::ImuCb(const sensor_msgs::Imu& imu_msg) {
  static tf2_ros::Buffer tf_buffer;
  static tf2_ros::TransformListener tf_listener{tf_buffer};

  if (imu_frame_.empty()) {
    imu_frame_ = imu_msg.header.frame_id;
    ROS_INFO_STREAM("Imu frame: " << imu_frame_);
  }

  // Add imu data to buffer
  const auto imu = MakeImu(imu_msg);
  imuq_.Add(imu);

  if (tf_init_) return;

  if (!lidar_init_) {
    ROS_WARN_STREAM("Lidar not initialized");
    return;
  }

  // tf stuff
  try {
    const auto tf_i_l = tf_buffer.lookupTransform(
        imu_msg.header.frame_id, lidar_frame_, ros::Time(0));

    const auto& t = tf_i_l.transform.translation;
    const auto& q = tf_i_l.transform.rotation;
    const Vector3d t_i_l{t.x, t.y, t.z};
    const Eigen::Quaterniond q_i_l{q.w, q.x, q.y, q.z};

    // Just use current acc
    ROS_INFO_STREAM("buffer size: " << imuq_.size());
    const auto imu_mean = imuq_.CalcMean();
    traj_.Init({q_i_l, t_i_l}, imu.acc, 9.80184);
    //    imuq_.bias.gyr = imu_mean.gyr;
    //    imuq_.bias.gyr_var = imu_mean.gyr.array().square();
    ROS_INFO_STREAM(traj_);
    tf_init_ = true;
  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM(ex.what());
    return;
  }
}

void OdomNode::Init(const sensor_msgs::CameraInfo& cinfo_msg) {
  ROS_INFO_STREAM("+++ Initializing");
  sweep_ = MakeSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = InitGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  traj_ = InitTraj({pnh_, "traj"}, grid_.cols());

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

  if (!imuq_.full()) {
    ROS_WARN_STREAM(fmt::format(
        "Imu buffer not full: {}/{}", imuq_.size(), imuq_.capacity()));
    return;
  }

  if (!tf_init_) {
    ROS_WARN_STREAM("Transform not initialized");
    return;
  }

  if (!scan_init_) {
    if (cinfo_msg->binning_x == 0) {
      scan_init_ = true;
    } else {
      ROS_WARN_STREAM("Scan not initialized");
      return;
    }
  }

  // We can always process incoming scan no matter what
  const auto scan = MakeScan(*image_msg, *cinfo_msg);
  ROS_WARN_STREAM(
      fmt::format("Processing scan: [{},{})", scan.curr.start, scan.curr.end));
  // Add scan to sweep, compute score and filter
  Preprocess(scan);

  if (rigid_) {
    IcpRigid();
  } else {
    IcpLinear();
  }

  PostProcess(scan);

  Publish(cinfo_msg->header);

  ROS_DEBUG_STREAM_THROTTLE(0.5, tm_.ReportAll(true));
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
    const auto t0 = grid_.time_begin();
    const auto dt = grid_.dt;

    // Predict the segment of traj corresponding to current grid
    n_imus = traj_.Predict(imuq_, t0, dt, grid_.curr.size());
  }
  ROS_INFO_STREAM("[Imu.Predict] using imus: " << n_imus);

  if (vis_) {
    const auto& disps = grid_.DrawCurveVar();
    Imshow("curve", ApplyCmap(disps[0], 1 / 0.2, cv::COLORMAP_VIRIDIS));
    Imshow("var", ApplyCmap(disps[1], 1 / 0.2, cv::COLORMAP_VIRIDIS));
    Imshow("filter",
           ApplyCmap(
               grid_.DrawFilter(), 1 / grid_.max_curve, cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::PostProcess(const LidarScan& scan) {
  int n_added = 0;
  {  // Add sweep to pano
    // Note that at this point the new scan is not yet added to the sweep
    auto _ = tm_.Scoped("Pano.Add");
    n_added = pano_.Add(sweep_, scan.curr, tbb_);
  }
  ROS_INFO_STREAM(
      fmt::format("[Pano.Add] Num added: {} / {} / {:02.2f}%, pano: {}",
                  n_added,
                  sweep_.total(),
                  100.0 * n_added / sweep_.total(),
                  pano_.num_added));

  auto T_p1_p2 = traj_.TfPanoLidar();
  if (pano_.align_gravity) T_p1_p2.so3() = Sophus::SO3d{};
  const auto T_p2_p1 = T_p1_p2.inverse();
  if (pano_.ShouldRender(T_p2_p1)) {
    ROS_ERROR_STREAM("Render pano at new location n: "
                     << pano_.num_added << ", max: " << pano_.max_cnt);

    // TODO (chao): need to think about how to run this in background without
    // interfering with odom
    auto _ = tm_.Scoped("Pano.Render");
    // Render pano at the latest lidar pose wrt pano (T_p1_p2 = T_p1_lidar)
    pano_.Render(T_p2_p1.cast<float>(), tbb_);
    // Once rendering is done we need to update traj accordingly
    traj_.Update(T_p2_p1);
  }

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
    sweep_.Interp(traj_, tbb_);
  }

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
