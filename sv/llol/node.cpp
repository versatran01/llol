#include <absl/flags/flag.h>
#include <cv_bridge/cv_bridge.h>
#include <fmt/core.h>
#include <glog/logging.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tbb/parallel_for.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/highgui.hpp>

#include "sv/llol/odom.h"
#include "sv/util/manager.h"

namespace sv {

cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale = 1.0,
                  int cmap = cv::COLORMAP_PINK,
                  uint8_t bad_value = 255) {
  CHECK_EQ(input.channels(), 1);

  cv::Mat disp;
  input.convertTo(disp, CV_8UC1, scale * 255.0);
  cv::applyColorMap(disp, disp, cmap);

  if (input.depth() >= CV_32F) {
    disp.setTo(bad_value, cv::Mat(~(input > 0)));
  }

  return disp;
}

class LlolNode {
 private:
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;

  bool vis_{true};
  bool tbb_{false};
  bool wait_for_scan0_{true};
  int cv_win_flag_{};

  LidarSweep sweep_;
  DepthPano pano_;
  PointMatcher matcher_;
  TimerManager tm_{"llol"};

 public:
  explicit LlolNode(const ros::NodeHandle& pnh) : pnh_{pnh}, it_{pnh} {
    sub_camera_ = it_.subscribeCamera("image", 10, &LlolNode::CameraCb, this);

    vis_ = pnh_.param<bool>("vis", true);
    ROS_INFO_STREAM("Visualize: " << (vis_ ? "True" : "False"));
    cv_win_flag_ =
        cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED;

    tbb_ = pnh_.param<bool>("tbb", false);
    ROS_INFO_STREAM("Use tbb: " << (tbb_ ? "True" : "False"));

    auto pano_nh = ros::NodeHandle{pnh_, "pano"};
    const auto pano_rows = pano_nh.param<int>("rows", 256);
    const auto pano_cols = pano_nh.param<int>("cols", 1024);
    const auto pano_hfov = pano_nh.param<double>("hfov", -1.0);
    pano_ = DepthPano({pano_cols, pano_rows}, Deg2Rad(pano_hfov));
    ROS_INFO_STREAM(pano_);
  }

  void Imshow(const std::string& name, const cv::Mat& mat) {
    cv::namedWindow(name, cv_win_flag_);
    cv::imshow(name, mat);
    cv::waitKey(1);
  }

  void ProcessScan(const cv::Mat& scan, const cv::Range& range) {
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

    // Wait for the start of the sweep
    if (wait_for_scan0_) {
      if (cinfo_msg->binning_x == 0) {
        ROS_INFO_STREAM("Start of sweep");
        wait_for_scan0_ = false;
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

    ProcessScan(scan, range);

    /// Got a full sweep
    if (cinfo_msg->binning_x + 1 == cinfo_msg->binning_y) {
      ROS_INFO_STREAM("End of sweep");
      int num_added;
      {
        auto _ = tm_.Scoped("Pano/AddSweep");
        num_added = pano_.AddSweep(sweep_, tbb_);
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
