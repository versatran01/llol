#include "sv/llol/viz.h"

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace sv {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void MeanCovar2Marker(Marker& marker,
                      const Eigen::Vector3d& mean,
                      Eigen::Vector3d eigvals,
                      Eigen::Matrix3d eigvecs,
                      double scale) {
  MakeRightHanded(eigvals, eigvecs);
  const Eigen::Quaterniond quat(eigvecs);
  eigvals = eigvals.cwiseSqrt() * 2 * scale;

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

// std::vector<Marker> Sweep2Markers(const std_msgs::Header& header,
//                                  const LidarSweep& sweep,
//                                  float max_curve) {
//  std::vector<Marker> markers;
//  markers.reserve(sweep.grid().total());

//  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

//  const auto& grid = sweep.grid();
//  const auto& cell_size = sweep.cell_size();
//  const int min_pts = cell_size.area() * 0.75;

//  Marker marker;
//  marker.header = header;
//  marker.ns = "sweep";
//  marker.type = Marker::SPHERE;
//  marker.action = Marker::ADD;
//  marker.color.a = 0.5;
//  marker.color.r = 0.0;
//  marker.color.g = 1.0;
//  marker.color.b = 0.0;

//  MeanCovar3f mc;

//  int k = 0;
//  for (int gr = 0; gr < grid.rows; ++gr) {
//    for (int gc = 0; gc < grid.cols; ++gc) {
//      const auto& curve = grid.at<float>(gr, gc);
//      if (!(curve < max_curve)) continue;
//      //  Get cell
//      const cv::Rect rect{
//          cv::Point{gc * cell_size.width, gr * cell_size.height}, cell_size};
//      const cv::Mat cell{sweep.xyzr(), rect};

//      mc.Reset();
//      MatXyzr2MeanCovar(cell, mc);

//      if (mc.n < min_pts) continue;
//      es.compute(mc.covar());
//      MeanCovar2Marker(marker,
//                       mc.mean.cast<double>(),
//                       es.eigenvalues().cast<double>(),
//                       es.eigenvectors().cast<double>());

//      marker.id = ++k;
//      markers.push_back(marker);
//    }
//  }
//  return markers;
//}

void Match2Markers(std::vector<Marker>& markers,
                   const std_msgs::Header& header,
                   const std::vector<PointMatch>& matches,
                   double scale) {
  markers.reserve(matches.size() * 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

  Marker pano_mk;
  pano_mk.header = header;
  pano_mk.ns = "match_pano";
  pano_mk.type = Marker::SPHERE;
  pano_mk.action = Marker::ADD;
  pano_mk.color.a = 0.8;
  pano_mk.color.r = 0.0;
  pano_mk.color.g = 1.0;
  pano_mk.color.b = 0.0;

  Marker sweep_mk;
  sweep_mk.header = header;
  sweep_mk.ns = "match_sweep";
  sweep_mk.type = Marker::SPHERE;
  sweep_mk.action = Marker::ADD;
  sweep_mk.color.a = 0.8;
  sweep_mk.color.r = 1.0;
  sweep_mk.color.g = 0.0;
  sweep_mk.color.b = 0.0;

  for (int i = 0; i < matches.size(); ++i) {
    const auto& match = matches[i];
    es.compute(match.dst.Covar());
    MeanCovar2Marker(pano_mk,
                     match.dst.mean.cast<double>(),
                     es.eigenvalues().cast<double>(),
                     es.eigenvectors().cast<double>(),
                     scale);

    pano_mk.id = i;
    markers.push_back(pano_mk);

    es.compute(match.src.Covar());
    MeanCovar2Marker(sweep_mk,
                     match.src.mean.cast<double>(),
                     es.eigenvalues().cast<double>(),
                     es.eigenvectors().cast<double>(),
                     scale);

    sweep_mk.id = i;
    markers.push_back(sweep_mk);
  }
}

cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale,
                  int cmap,
                  uint8_t bad_color) {
  CHECK_EQ(input.channels(), 1);

  cv::Mat disp;
  input.convertTo(disp, CV_8UC1, scale * 255.0);
  cv::applyColorMap(disp, disp, cmap);

  if (input.depth() >= CV_32F) {
    disp.setTo(bad_color, cv::Mat(~(input > 0)));
  }

  return disp;
}

void Imshow(const std::string& name, const cv::Mat& mat, int flag) {
  cv::namedWindow(name, flag);
  cv::imshow(name, mat);
  cv::waitKey(1);
}

}  // namespace sv
