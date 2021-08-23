#include "sv/node/factory.h"

#include <cv_bridge/cv_bridge.h>

namespace sv {

LidarScan MakeScan(const sensor_msgs::Image& image_msg,
                   const sensor_msgs::CameraInfo& cinfo_msg) {
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(image_msg, "32FC4");

  return {image_msg.header.stamp.toSec(),    // t
          cinfo_msg.K[0],                    // dt
          cv_ptr->image,                     // xyzr
          cv::Range(cinfo_msg.roi.x_offset,  // col_rg
                    cinfo_msg.roi.x_offset + cinfo_msg.roi.width)};
}

SweepGrid MakeGrid(const ros::NodeHandle& pnh, const cv::Size& sweep_size) {
  GridParams gp;
  gp.cell_rows = pnh.param<int>("cell_rows", gp.cell_rows);
  gp.cell_cols = pnh.param<int>("cell_cols", gp.cell_cols);
  gp.max_score = pnh.param<double>("max_score", gp.max_score);
  gp.nms = pnh.param<bool>("nms", gp.nms);
  return SweepGrid{sweep_size, gp};
}

LidarSweep MakeSweep(const sensor_msgs::CameraInfo& cinfo_msg) {
  return LidarSweep{cv::Size(cinfo_msg.width, cinfo_msg.height)};
}

ProjMatcher MakeMatcher(const ros::NodeHandle& pnh) {
  MatcherParams mp;
  mp.half_rows = pnh.param<int>("half_rows", mp.half_rows);
  mp.cov_lambda = pnh.param<double>("cov_lambda", mp.cov_lambda);
  return ProjMatcher{mp};
}

DepthPano MakePano(const ros::NodeHandle& pnh) {
  PanoParams pp;
  const auto pano_rows = pnh.param<int>("rows", 256);
  const auto pano_cols = pnh.param<int>("cols", 1024);
  pp.hfov = Deg2Rad(pnh.param<double>("hfov", pp.hfov));
  pp.max_cnt = pnh.param<int>("max_cnt", pp.max_cnt);
  pp.min_range = pnh.param<double>("min_range", pp.min_range);
  pp.range_ratio = pnh.param<double>("range_ratio", pp.range_ratio);
  return DepthPano({pano_cols, pano_rows}, pp);
}

}  // namespace sv
