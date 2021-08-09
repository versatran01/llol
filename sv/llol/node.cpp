#include <ceres/ceres.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Eigenvalues>
#include <sophus/se3.hpp>

#include "sv/llol/match.h"
#include "sv/llol/pano.h"
#include "sv/llol/sweep.h"
#include "sv/util/manager.h"
#include "sv/util/ocv.h"

namespace sv {

MeanCovar3f CalcMeanCovar(const cv::Mat& mat) {
  CHECK_EQ(mat.type(), CV_32FC4);

  MeanCovar3f mc;
  for (int r = 0; r < mat.rows; ++r) {
    for (int c = 0; c < mat.cols; ++c) {
      const auto& xyzr = mat.at<cv::Vec4f>(r, c);
      if (std::isnan(xyzr[0])) continue;
      mc.Add({xyzr[0], xyzr[1], xyzr[2]});
    }
  }
  return mc;
}

void MeanCovar2Marker(const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      visualization_msgs::Marker& marker) {
  MakeRightHanded(eigvals, eigvecs);
  Eigen::Quaterniond quat(eigvecs);
  eigvals = eigvals.cwiseSqrt() * 2;

  marker.pose.position.x = mean.x();
  marker.pose.position.y = mean.y();
  marker.pose.position.z = mean.z();
  marker.pose.orientation.w = quat.w();
  marker.pose.orientation.x = quat.x();
  marker.pose.orientation.y = quat.y();
  marker.pose.orientation.z = quat.z();
  marker.scale.x = eigvals.x();
  marker.scale.y = eigvals.y();
  marker.scale.z = eigvals.z();
}

auto Sweep2Gaussians(const LidarSweep& sweep, float max_curve)
    -> visualization_msgs::MarkerArray {
  visualization_msgs::MarkerArray marray;
  marray.markers.reserve(sweep.grid_total());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

  visualization_msgs::Marker marker;
  marker.ns = "sweep";
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  for (int gr = 0; gr < sweep.grid().rows; ++gr) {
    for (int gc = 0; gc < sweep.grid_width(); ++gc) {
      const auto& curve = sweep.CurveAt(gr, gc);
      if (!(curve < max_curve)) continue;
      //  Get cell
      const auto cell = sweep.CellAt(gr, gc);
      const auto mc = CalcMeanCovar(cell);

      marker.id = gr * sweep.grid().cols + gc;
      es.compute(mc.covar());

      MeanCovar2Marker(mc.mean.cast<double>(),
                       es.eigenvalues().cast<double>(),
                       es.eigenvectors().cast<double>(),
                       marker);

      marray.markers.push_back(marker);
    }
  }
  return marray;
}

class LlolNode {
 private:
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  ros::Publisher pub_marray_;

  bool vis_{};
  bool tbb_{};

  LidarSweep sweep_;
  DepthPano pano_;
  PointMatcher matcher_;
  TimerManager tm_{"llol"};

  std::vector<double> times_;
  std::vector<Sophus::SE3d> poses_;

 public:
  explicit LlolNode(const ros::NodeHandle& pnh) : pnh_{pnh}, it_{pnh} {
    sub_camera_ = it_.subscribeCamera("image", 10, &LlolNode::CameraCb, this);
    pub_marray_ =
        pnh_.advertise<visualization_msgs::MarkerArray>("sweep_gaussian", 1);

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

    times_.resize(2);
    poses_.resize(2);
  }

  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
    if (sweep_.empty()) {
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
        MatcherParams mp;
        mp.nms = match_nh.param<bool>("nms", false);
        mp.half_rows = match_nh.param<int>("half_rows", 2);
        mp.max_curve = match_nh.param<double>("max_curve", 0.01);
        matcher_ = PointMatcher(sweep_.grid_total(), mp);
        ROS_INFO_STREAM(matcher_);
      }
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
    times_[0] = cinfo_msg->header.stamp.toSec();
    times_[1] = times_[0] = cinfo_msg->width * cinfo_msg->K[0];
    poses_[0] = poses_[1];  // initialize guess of delta pose is identity

    int num_valid_cells = 0;
    {  /// Add scan to sweep
      auto _ = tm_.Scoped("Sweep/AddScan");
      num_valid_cells = sweep_.AddScan(scan, range, tbb_);
    }

    ROS_INFO_STREAM("Num valid cells: " << num_valid_cells);

    if (vis_) {
      cv::Mat sweep_disp;
      cv::extractChannel(sweep_.sweep(), sweep_disp, 3);
      Imshow("sweep", ApplyCmap(sweep_disp, 1 / 30.0, cv::COLORMAP_PINK, 0));
      Imshow("grid", ApplyCmap(sweep_.grid(), 10, cv::COLORMAP_VIRIDIS, 255));
    }

    auto marray = Sweep2Gaussians(sweep_, 0.01);
    for (auto& m : marray.markers) m.header = image_msg->header;
    pub_marray_.publish(marray);

    /// Check if pano has data, if true then perform match
    if (pano_.num_sweeps() == 0) {
      ROS_INFO_STREAM("Pano is not initialized");
    } else {
      {  /// Match Features
        auto _ = tm_.Scoped("Matcher/Match");
        matcher_.Match(sweep_, pano_);
      }

      ROS_INFO_STREAM("Num matches: " << matcher_.matches().size());

      // display good match
      cv::Mat match_disp(sweep_.grid_size(), CV_32FC1);
      match_disp.setTo(std::numeric_limits<float>::quiet_NaN());

      float max_pts = matcher_.win_size().area();
      for (const auto& match : matcher_.matches()) {
        match_disp.at<float>(sweep_.PixelToCell(match.pt)) =
            match.dst.n / max_pts;
      }
      Imshow("match",
             ApplyCmap(matcher_.Draw(sweep_), 1.0, cv::COLORMAP_VIRIDIS));
    }

    /// Got a full sweep
    if (cinfo_msg->binning_x + 1 == cinfo_msg->binning_y) {
      ROS_INFO_STREAM("End of sweep");
      int num_added;
      {
        auto _ = tm_.Scoped("Pano/AddSweep");
        num_added = pano_.AddSweep(sweep_.sweep(), tbb_);
      }
      ROS_INFO_STREAM("Num added: " << num_added
                                    << ", sweep total: " << sweep_.total());

      int num_rendered = 0;
      {
        auto _ = tm_.Scoped("Pano/Render");
        num_rendered = pano_.Render(tbb_);
      }
      ROS_INFO_STREAM("Num rendered: " << num_rendered
                                       << ", pano total: " << pano_.total());

      if (vis_) {
        Imshow("pano", ApplyCmap(pano_.buf_, 1 / DepthPano::kScale / 30.0));
        Imshow("pano2", ApplyCmap(pano_.buf2_, 1 / DepthPano::kScale / 30.0));
      }
    }

    ROS_DEBUG_STREAM(tm_.ReportAll());
  }
};

}  // namespace sv

int main(int argc, char** argv) {
  ros::init(argc, argv, "llol_node");
  sv::LlolNode node(ros::NodeHandle("~"));
  ros::spin();
}
