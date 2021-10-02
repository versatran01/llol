#include "sv/node/llol_node.h"

#include <absl/strings/match.h>

#include "sv/node/viz.h"

namespace sv {

static constexpr double kMaxRange = 32.0;

OdomNode::OdomNode(const ros::NodeHandle& pnh)
    : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
  sub_camera_ = it_.subscribeCamera("image", 8, &OdomNode::CameraCb, this);
  sub_imu_ = pnh_.subscribe(
      "imu", 100, &OdomNode::ImuCb, this, ros::TransportHints().tcpNoDelay());

  vis_ = pnh_.param<bool>("vis", true);
  ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

  tbb_ = pnh_.param<int>("tbb", 0);
  ROS_INFO_STREAM("Tbb grainsize: " << tbb_);

  log_ = pnh_.param<int>("log", 0);
  ROS_INFO_STREAM("Log interval: " << log_);

  rigid_ = pnh_.param<bool>("rigid", true);
  ROS_WARN_STREAM("GICP: " << (rigid_ ? "Rigid" : "Linear"));

  path_dist_ = pnh_.param<double>("path_dist", 0.01);

  imuq_ = InitImuq({pnh_, "imuq"});
  ROS_INFO_STREAM(imuq_);

  pano_ = InitPano({pnh_, "pano"});
  ROS_INFO_STREAM(pano_);
}

void OdomNode::ImuCb(const sensor_msgs::Imu& imu_msg) {
  if (imu_frame_.empty()) {
    imu_frame_ = imu_msg.header.frame_id;
    ROS_INFO_STREAM("Imu frame: " << imu_frame_);
  }

  // Add imu data to buffer
  static std_msgs::Header prev_header;
  if (prev_header.seq > 0 && prev_header.seq + 1 != imu_msg.header.seq) {
    ROS_ERROR_STREAM("Missing imu data, prev: " << prev_header.seq << ", curr: "
                                                << imu_msg.header.seq);
  }
  prev_header = imu_msg.header;

  const auto imu = MakeImu(imu_msg);
  imuq_.Add(imu);

  if (tf_init_) return;

  if (lidar_frame_.empty()) {
    ROS_WARN_STREAM("Lidar not initialized");
    return;
  }

  if (!imuq_.full()) {
    ROS_WARN_STREAM(fmt::format(
        "Imu queue not full: {}/{}", imuq_.size(), imuq_.capacity()));
    return;
  }

  // Get tf between imu and lidar and initialize gravity direction
  try {
    const auto tf_i_l = tf_buffer_.lookupTransform(
        imu_msg.header.frame_id, lidar_frame_, ros::Time(0));

    const auto& t = tf_i_l.transform.translation;
    const auto& q = tf_i_l.transform.rotation;
    const Eigen::Vector3d t_i_l{t.x, t.y, t.z};
    const Eigen::Quaterniond q_i_l{q.w, q.x, q.y, q.z};

    // Just use current acc
    ROS_INFO_STREAM("buffer size: " << imuq_.size());
    const auto imu_mean = imuq_.CalcMean(10);
    const auto& imu0 = imuq_.buf.front();

    ROS_INFO_STREAM("acc_curr: " << imu.acc.transpose()
                                 << ", norm: " << imu.acc.norm());
    ROS_INFO_STREAM("acc_mean: " << imu_mean.acc.transpose()
                                 << ", norm: " << imu_mean.acc.norm());

    // Use the one that is closer to 9.8
    const Sophus::SE3d T_i_l{q_i_l, t_i_l};
    const auto curr_acc_norm = imu.acc.norm();
    const auto mean_acc_norm = imu_mean.acc.norm();
    const auto gnorm = traj_.gravity_norm;

    if (gnorm > 0 &&
        std::abs(curr_acc_norm - gnorm) < std::abs(mean_acc_norm - gnorm)) {
      ROS_INFO_STREAM("Use curr acc as gravity");
      traj_.Init(T_i_l, imu.acc);
    } else {
      ROS_INFO_STREAM("Use mean acc as gravity");
      traj_.Init(T_i_l, imu_mean.acc);
    }

    ROS_INFO_STREAM(traj_);
    tf_init_ = true;
  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM(ex.what());
    return;
  }
}

void OdomNode::Initialize(const sensor_msgs::CameraInfo& cinfo_msg) {
  ROS_INFO_STREAM("+++ Initializing");
  sweep_ = MakeSweep(cinfo_msg);
  ROS_INFO_STREAM(sweep_);

  grid_ = InitGrid({pnh_, "grid"}, sweep_.size());
  ROS_INFO_STREAM(grid_);

  traj_ = InitTraj({pnh_, "traj"}, grid_.cols());
  ROS_INFO_STREAM(traj_);

  gicp_ = InitGicp({pnh_, "gicp"});
  ROS_INFO_STREAM(gicp_);
}

void OdomNode::CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                        const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
  static std_msgs::Header prev_header;
  if (prev_header.seq > 0 && prev_header.seq + 1 != cinfo_msg->header.seq) {
    ROS_ERROR_STREAM("Missing lidar data, prev: "
                     << prev_header.seq << ", curr: " << cinfo_msg->header.seq);
  }
  prev_header = cinfo_msg->header;

  if (lidar_frame_.empty()) {
    lidar_frame_ = image_msg->header.frame_id;
    // Allocate storage for sweep, grid and matcher
    Initialize(*cinfo_msg);
    ROS_INFO_STREAM("Lidar frame: " << lidar_frame_);
    ROS_INFO_STREAM("Lidar initialized!");
  }

  if (!imuq_.full()) {
    ROS_WARN_STREAM(fmt::format(
        "Imu queue not full: {}/{}", imuq_.size(), imuq_.capacity()));
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
  ROS_DEBUG("Processing scan %d: [%d,%d)",
            static_cast<int>(cinfo_msg->header.seq),
            scan.curr.start,
            scan.curr.end);
  // Add scan to sweep, compute score and filter
  Preprocess(scan);

  Register();

  PostProcess();

  Logging();

  Publish(cinfo_msg->header);
}

void OdomNode::Preprocess(const LidarScan& scan) {
  // 1. Eject scan to pano, assuming traj is optimized
  int n_added = 0;
  {  // Note that at this point the new scan is not yet added to the sweep
    auto _ = tm_.Scoped("1.Pano.Add");
    n_added = pano_.Add(sweep_, scan.curr, tbb_);
  }
  sm_.GetRef("pano.add_points").Add(n_added);
  ROS_DEBUG_STREAM("[pano.Add] num added: " << n_added);

  // 2. Add current scan to sweep
  int n_points = 0;
  {  // Add scan to sweep
    auto _ = tm_.Scoped("2.Sweep.Add");
    n_points = sweep_.Add(scan);
  }
  sm_.GetRef("sweep.add").Add(n_points);
  ROS_DEBUG_STREAM("[sweep.Add] num added: " << n_points);

  // 3. Add current scan to grid
  cv::Vec2i n_cells{};
  {  // Reduce scan to grid and Filter
    auto _ = tm_.Scoped("3.Grid.Add");
    n_cells = grid_.Add(scan, tbb_);
  }
  ROS_DEBUG_STREAM("[grid.Add] Num valid cells: "
                   << n_cells[0] << ", num good cells: " << n_cells[1]);

  sm_.GetRef("grid.valid_cells").Add(n_cells[0]);
  sm_.GetRef("grid.good_cells").Add(n_cells[1]);

  // 4. Predict new scetion in trajectory
  int n_imus{};
  {  // Integarte imu to fill nominal traj
    auto _ = tm_.Scoped("4.Imu.Integrate");
    const int pred_cols = grid_.curr.size();
    const auto t0 = grid_.TimeAt(grid_.cols() - pred_cols);
    const auto dt = grid_.dt;

    // Predict the segment of traj corresponding to current grid
    n_imus = traj_.PredictNew(imuq_, t0, dt, pred_cols);
  }
  sm_.GetRef("traj.pred_imus").Add(n_imus);
  ROS_DEBUG_STREAM("[traj.Predict] num imus: " << n_imus);

  if (vis_) {
    const auto& disps = grid_.DrawCurveVar();

    Imshow("scan",
           ApplyCmap(scan.ExtractRange(),
                     1.0 / scan.scale / kMaxRange,
                     cv::COLORMAP_PINK,
                     0));
    Imshow("curve", ApplyCmap(disps[0], 1 / 0.25, cv::COLORMAP_VIRIDIS));
    Imshow("var", ApplyCmap(disps[1], 1 / 0.25, cv::COLORMAP_VIRIDIS));
    Imshow("filter",
           ApplyCmap(
               grid_.DrawFilter(), 1 / grid_.max_curve, cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::PostProcess() {
  const auto num_good_cells = grid_.NumCandidates();
  const auto num_matches = sm_.GetRef("grid.matches").last();
  const double match_ratio = num_matches / num_good_cells;
  ROS_DEBUG_STREAM("[pano.RenderCheck] match ratio: " << match_ratio);

  auto T_p1_p2 = traj_.TfPanoLidar();
  // Algin gravity means we will just set rotation to identity
  if (pano_.align_gravity) T_p1_p2.so3() = Sophus::SO3d{};

  // We use the inverse from now on
  const auto T_p2_p1 = T_p1_p2.inverse();

  int n_render = 0;
  if (pano_.ShouldRender(T_p2_p1, match_ratio)) {
    ROS_WARN_STREAM(
        "=Render= " << fmt::format(
            "sweeps: {:2.3f}, trans: {:.3f}, match: {:.2f}% = {}/{}",
            pano_.num_sweeps,
            T_p1_p2.translation().norm(),
            match_ratio * 100,
            num_matches,
            num_good_cells));

    // TODO (chao): need to think about how to run this in background without
    // interfering with odom
    auto _ = tm_.Scoped("Render");
    // Render pano at the latest lidar pose wrt pano (T_p1_p2 = T_p1_lidar)
    n_render = pano_.Render(T_p2_p1.cast<float>(), tbb_);
    // Once rendering is done we need to update traj accordingly
    traj_.MoveFrame(T_p2_p1);
  }
  if (n_render > 0) {
    sm_.GetRef("pano.render_points").Add(n_render);
    ROS_DEBUG_STREAM("[pano.Render] num render: " << n_render);
  }

  // 7. Update sweep transforms for undistortion
  {
    auto _ = tm_.Scoped("7.Sweep.Interp");
    sweep_.Interp(traj_, tbb_);
  }

  grid_.Interp(traj_);

  if (vis_) {
    Imshow("sweep",
           ApplyCmap(sweep_.ExtractRange(),
                     1.0 / sweep_.scale / kMaxRange,
                     cv::COLORMAP_PINK,
                     0));
    const auto& disps = pano_.DrawRangeCount();
    Imshow(
        "pano",
        ApplyCmap(
            disps[0], 1.0 / DepthPixel::kScale / kMaxRange, cv::COLORMAP_PINK));
    Imshow("count",
           ApplyCmap(disps[1], 1.0 / pano_.max_cnt, cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::Logging() {
  // Record total time
  TimerManager::StatsT stats;
  absl::Duration time;
  for (const auto& kv : tm_.dict()) {
    if (absl::StartsWith(kv.first, "Total")) continue;
    if (absl::StartsWith(kv.first, "Render")) continue;
    time += kv.second.last();
  }
  stats.Add(time);
  tm_.Update("Total", stats);

  if (log_ > 0) {
    ROS_INFO_STREAM_THROTTLE(log_, tm_.ReportAll(true));
  }
}

}  // namespace sv
