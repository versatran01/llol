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
  FeatureGrid feat_;
  DepthPano pano_;
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
    int pano_rows = pano_nh.param<int>("rows", 256);
    int pano_cols = pano_nh.param<int>("cols", 1024);
    pano_ = DepthPano({pano_cols, pano_rows});
    ROS_INFO_STREAM(pano_);
  }

  void Imshow(const std::string& name, const cv::Mat& mat) {
    cv::namedWindow(name, cv_win_flag_);
    cv::imshow(name, mat);
    cv::waitKey(1);
  }

  void ProcessScan(const cv::Mat& scan, const cv::Range& range) {
    /// Add scan to sweep
    {
      auto _ = tm_.Scoped("Sweep/AddScan");
      sweep_.AddScan(scan, range);
    }

    if (vis_) {
      cv::Mat sweep_range;
      cv::extractChannel(sweep_.mat_, sweep_range, 3);
      Imshow("sweep", ApplyCmap(sweep_range, 1 / 30.0, cv::COLORMAP_PINK, 0));
    }

    /// Check if pano has weep
    if (pano_.num_sweeps() == 0) {
      ROS_INFO_STREAM("Pano is not initialized");
    } else {
      {
        /// Detect Feature
        auto _ = tm_.Scoped("Feat/Detect");
        feat_.Detect(sweep_, tbb_);
      }

      ROS_INFO_STREAM("Num cells: " << feat_.NumCells());

      if (vis_) {
        Imshow("score", ApplyCmap(feat_.mat(), 10, cv::COLORMAP_VIRIDIS, 255));
      }
    }
  }

  void CameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
    // Initialized sweep
    if (sweep_.empty()) {
      sweep_ = LidarSweep(cv::Size(cinfo_msg->width, cinfo_msg->height));
      ROS_INFO_STREAM(sweep_);
      auto odom_nh = ros::NodeHandle{pnh_, "odom"};
      int feat_win_rows = odom_nh.param<int>("feat_win_rows", 2);
      int feat_win_cols = odom_nh.param<int>("feat_win_cols", 16);
      feat_ = FeatureGrid(sweep_.size(), {feat_win_cols, feat_win_rows});
      ROS_INFO_STREAM(feat_);
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
      {
        auto _ = tm_.Scoped("Pano/AddSweep");
        pano_.AddSweep(sweep_, tbb_);
      }

      {
        auto _ = tm_.Scoped("Pano/Render");
        pano_.Render(tbb_);
      }

      if (vis_) {
        Imshow("pano", ApplyCmap(pano_.mat_, 1 / DepthPano::kScale / 30.0));
        Imshow("pano2", ApplyCmap(pano_.mat2_, 1 / DepthPano::kScale / 30.0));
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
