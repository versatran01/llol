#include "sv/llol/viz.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

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

visualization_msgs::MarkerArray Sweep2Gaussians(const cv::Mat& sweep,
                                                const cv::Mat& grid,
                                                float max_curve) {
  visualization_msgs::MarkerArray marray;
  marray.markers.reserve(grid.total());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

  const cv::Size cell_size{sweep.cols / grid.cols, sweep.rows / grid.rows};
  const int min_pts = cell_size.area() * 0.75;

  visualization_msgs::Marker marker;
  marker.ns = "sweep";
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 0.5;
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;

  int k = 0;
  for (int gr = 0; gr < grid.rows; ++gr) {
    for (int gc = 0; gc < grid.cols; ++gc) {
      const auto& curve = grid.at<float>(gr, gc);
      if (!(curve < max_curve)) continue;
      //  Get cell
      const cv::Rect rect{
          cv::Point{gc * cell_size.width, gr * cell_size.height}, cell_size};
      const cv::Mat cell{sweep, rect};
      const auto mc = CalcMeanCovar(cell);

      if (mc.n < min_pts) continue;
      es.compute(mc.covar());
      MeanCovar2Marker(mc.mean.cast<double>(),
                       es.eigenvalues().cast<double>(),
                       es.eigenvectors().cast<double>(),
                       marker);

      marker.id = ++k;
      marray.markers.push_back(marker);
    }
  }
  return marray;
}

}  // namespace sv
