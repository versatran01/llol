// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_listener.h>

#include "sv/llol/match.h"
#include "sv/llol/optim.h"
#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"
#include "sv/llol/viz.h"
#include "sv/util/manager.h"

// ceres
#include <ceres/ceres.h>

namespace sv {

namespace cs = ceres;
class LlolNode {
 private:
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Subscriber sub_imu_;
  ros::Publisher pub_marray_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string lidar_frame_;
  std::optional<geometry_msgs::TransformStamped> tf_imu_lidar_;

  bool vis_{};
  bool tbb_{};
  bool init_{false};

  LidarSweep sweep_;
  DepthPano pano_;
  PointMatcher matcher_;
  TimerManager tm_{"llol"};

  Sophus::SE3d T_p_s_;

 public:
  explicit LlolNode(const ros::NodeHandle& pnh)
      : pnh_{pnh}, it_{pnh}, tf_listener_{tf_buffer_} {
    sub_imu_ = pnh_.subscribe("imu", 100, &LlolNode::ImuCb, this);
    sub_camera_ = it_.subscribeCamera("image", 10, &LlolNode::CameraCb, this);
    pub_marray_ = pnh_.advertise<visualization_msgs::MarkerArray>("marray", 1);

    vis_ = pnh_.param<bool>("vis", true);
    ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));

    tbb_ = pnh_.param<bool>("tbb", false);
    ROS_INFO_STREAM("Use tbb: " << (tbb_ ? "True" : "False"));

    auto pano_nh = ros::NodeHandle{pnh_, "pano"};
    const auto pano_rows = pano_nh.param<int>("rows", 256);
    const auto pano_cols = pano_nh.param<int>("cols", 1024);
    const auto pano_hfov = pano_nh.param<double>("hfov", -1.0);
    pano_ = DepthPano({pano_cols, pano_rows}, Deg2Rad(pano_hfov));
    ROS_INFO_STREAM(pano_);
  }

  void ImuCb(const sensor_msgs::Imu& imu_msg) {
    // tf stuff
    if (lidar_frame_.empty()) {
      ROS_WARN_STREAM_THROTTLE(1, "Lidar frame is not set, waiting");
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
        const SE3d T_imu_lidar{q_imu_lidar, t_imu_lidar};

        ROS_INFO_STREAM("Transform from lidar to imu\n"
                        << T_imu_lidar.matrix());
      } catch (tf2::TransformException& ex) {
        ROS_WARN_STREAM(ex.what());
        return;
      }
    }
  }

  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
    if (lidar_frame_.empty()) {
      lidar_frame_ = image_msg->header.frame_id;
      ROS_INFO_STREAM("Lidar frame: " << lidar_frame_);
    }

    // Currently disabled
    return;

    if (!init_) {
      // Initialized sweep
      {
        auto odom_nh = ros::NodeHandle{pnh_, "sweep"};
        int cell_rows = odom_nh.param<int>("cell_rows", 2);
        int cell_cols = odom_nh.param<int>("cell_cols", 16);

        sweep_ = LidarSweep{cv::Size(cinfo_msg->width, cinfo_msg->height),
                            {cell_cols, cell_rows}};
        ROS_INFO_STREAM(sweep_);
      }

      // Initialize matcher
      {
        auto match_nh = ros::NodeHandle{pnh_, "match"};
        MatcherParams params;
        params.nms = match_nh.param<bool>("nms", false);
        params.half_rows = match_nh.param<int>("half_rows", 2);
        params.max_curve = match_nh.param<double>("max_curve", 0.01);
        matcher_ = PointMatcher(sweep_.grid_size(), params);
        ROS_INFO_STREAM(matcher_);
      }

      init_ = true;
    }

    static bool wait_for_scan0{true};
    // Wait for the start of the sweep
    if (wait_for_scan0) {
      if (cinfo_msg->binning_x == 0) {
        ROS_INFO_STREAM("Start of sweep");
        wait_for_scan0 = false;
      } else {
        ROS_WARN_STREAM("Waiting for the first scan, current "
                        << cinfo_msg->binning_x);
        return;
      }
    }

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(image_msg, "32FC4");
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
      return;
    }

    const auto& scan = cv_ptr->image;
    const int col_beg = cinfo_msg->roi.x_offset;
    const int col_end = col_beg + cinfo_msg->roi.width;
    const cv::Range range(col_beg, col_end);

    // Predict poses, wrt pano
    //    times_[0] = cinfo_msg->header.stamp.toSec();
    //    times_[1] = times_[0] = cinfo_msg->width * cinfo_msg->K[0];
    //    poses_[0] = poses_[1];  // initialize guess of delta pose is identity

    int num_valid_cells = 0;
    {  /// Add scan to sweep
      auto _ = tm_.Scoped("Sweep/AddScan");
      num_valid_cells = sweep_.AddScan(scan, range, tbb_);
    }

    ROS_INFO_STREAM("Num valid cells: " << num_valid_cells);

    if (vis_) {
      cv::Mat sweep_disp;
      cv::extractChannel(sweep_.xyzr(), sweep_disp, 3);
      Imshow("sweep", ApplyCmap(sweep_disp, 1 / 30.0, cv::COLORMAP_PINK, 0));
      Imshow("grid", ApplyCmap(sweep_.grid(), 5, cv::COLORMAP_VIRIDIS, 255));
    }

    visualization_msgs::MarkerArray marray;

    /// Check if pano has data, if true then perform match
    if (pano_.num_sweeps() == 0) {
      ROS_INFO_STREAM("Pano is not initialized");
    } else {
      {  /// Match Features
        auto _ = tm_.Scoped("Matcher/Match");
        matcher_.Match(sweep_, pano_, tbb_);
      }

      ROS_INFO_STREAM("Num matches: " << matcher_.matches().size());
      Match2Markers(marray.markers, image_msg->header, matcher_.matches());

      // display good match
      Imshow("match",
             ApplyCmap(matcher_.Draw(sweep_),
                       1.0 / matcher_.pano_win_size_.area(),
                       cv::COLORMAP_VIRIDIS));

      cs::Solver::Summary summary;
      {  /// Optimization

        auto _ = tm_.Scoped("ICP/Solve");
        std::unique_ptr<ceres::LocalParameterization> local_params =
            std::make_unique<LocalParamSE3>();
        std::unique_ptr<ceres::LossFunction> loss =
            std::make_unique<cs::HuberLoss>(3);
        cs::Problem::Options problem_opt;
        problem_opt.loss_function_ownership = cs::DO_NOT_TAKE_OWNERSHIP;
        problem_opt.local_parameterization_ownership =
            cs::DO_NOT_TAKE_OWNERSHIP;
        cs::Problem problem{problem_opt};

        problem.AddParameterBlock(
            T_p_s_.data(), SE3d::num_parameters, local_params.get());

        // Build problem
        for (const auto& match : matcher_.matches()) {
          cs::CostFunction* cost =
              new cs::AutoDiffCostFunction<GicpFactor,
                                           GicpFactor::kNumResiduals,
                                           GicpFactor::kNumParams>(
                  new GicpFactor(match));
          problem.AddResidualBlock(cost, loss.get(), T_p_s_.data());
        }

        cs::Solver::Options solver_opt;
        solver_opt.linear_solver_type = ceres::DENSE_QR;
        solver_opt.max_num_iterations = 5;
        solver_opt.num_threads = tbb_ ? 4 : 1;
        solver_opt.minimizer_progress_to_stdout = true;
        cs::Solve(solver_opt, &problem, &summary);
      }
      ROS_INFO_STREAM("Pose: \n" << T_p_s_.matrix3x4());
      ROS_INFO_STREAM(summary.BriefReport());
    }

    /// Got a full sweep
    if (cinfo_msg->binning_x + 1 == cinfo_msg->binning_y) {
      ROS_INFO_STREAM("End of sweep");
      int num_added = 0;
      {
        auto _ = tm_.Scoped("Pano/AddSweep");
        num_added = pano_.AddSweep(sweep_.xyzr(), tbb_);
      }
      ROS_INFO_STREAM("Num added: " << num_added << ", sweep total: "
                                    << sweep_.xyzr().total());

      int num_rendered = 0;
      {
        auto _ = tm_.Scoped("Pano/Render");
        num_rendered = pano_.Render(tbb_);
      }
      ROS_INFO_STREAM("Num rendered: " << num_rendered
                                       << ", pano total: " << pano_.total());

      if (vis_) {
        Imshow("pano", ApplyCmap(pano_.dbuf_, 1 / DepthPano::kScale / 30.0));
        Imshow("pano2", ApplyCmap(pano_.dbuf2_, 1 / DepthPano::kScale / 30.0));
      }
    }

    pub_marray_.publish(marray);
    ROS_DEBUG_STREAM_THROTTLE(2, tm_.ReportAll());
  }
};

}  // namespace sv

int main(int argc, char** argv) {
  ros::init(argc, argv, "llol_node");
  sv::LlolNode node(ros::NodeHandle("~"));
  ros::spin();
}
