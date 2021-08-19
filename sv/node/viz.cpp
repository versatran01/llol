#include "sv/node/viz.h"

#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tbb/parallel_for.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace sv {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void MeanCovar2Marker(Marker& marker,
                      const Eigen::Vector3f& mean,
                      Eigen::Vector3f eigvals,
                      Eigen::Matrix3f eigvecs) {
  MakeRightHanded(eigvals, eigvecs);
  const Eigen::Quaternionf quat(eigvecs);
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

void Match2Markers(const std::vector<NormalMatch>& matches,
                   const std_msgs::Header& header,
                   std::vector<Marker>& markers) {
  markers.reserve(matches.size() * 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

  Marker pano_mk;
  pano_mk.header = header;
  pano_mk.ns = "match_pano";
  pano_mk.type = Marker::SPHERE;
  pano_mk.action = Marker::ADD;
  pano_mk.color.a = 0.5;
  pano_mk.color.r = 0.0;
  pano_mk.color.g = 1.0;
  pano_mk.color.b = 0.0;

  Marker sweep_mk;
  sweep_mk.header = header;
  sweep_mk.ns = "match_sweep";
  sweep_mk.type = Marker::SPHERE;
  sweep_mk.action = Marker::ADD;
  sweep_mk.color.a = 0.5;
  sweep_mk.color.r = 1.0;
  sweep_mk.color.g = 0.0;
  sweep_mk.color.b = 0.0;

  Eigen::Matrix3f covar;

  // TODO (chao): only draw matches up to width
  for (int i = 0; i < matches.size(); ++i) {
    const auto& match = matches[i];
    if (!match.ok()) continue;

    covar = match.mc_p.Covar();
    covar.diagonal().array() += 1e-6;
    es.compute(covar);
    MeanCovar2Marker(
        pano_mk, match.mc_p.mean, es.eigenvalues(), es.eigenvectors());

    pano_mk.id = i;
    markers.push_back(pano_mk);

    //    covar = match.mc_s.Covar();
    //    covar.diagonal().array() += 1e-6;
    //    es.compute(covar);
    //    MeanCovar2Marker(
    //        sweep_mk, match.mc_s.mean, es.eigenvalues(), es.eigenvectors());

    //    sweep_mk.id = i;
    //    markers.push_back(sweep_mk);
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

void Pano2Cloud(const DepthPano& pano,
                const std_msgs::Header header,
                Cloud& cloud) {
  const auto size = pano.size();
  if (cloud.empty()) {
    cloud.resize(pano.total());
    cloud.width = size.width;
    cloud.height = size.height;
  }

  pcl_conversions::toPCL(header, cloud.header);
  tbb::parallel_for(tbb::blocked_range<int>(0, size.height),
                    [&](const auto& blk) {
                      for (int r = blk.begin(); r < blk.end(); ++r) {
                        for (int c = 0; c < size.width; ++c) {
                          const auto rg = pano.RangeAt({c, r});
                          auto& pc = cloud.at(c, r);
                          if (rg == 0) {
                            pc.x = pc.y = pc.z = kNaNF;
                          } else {
                            const auto pp = pano.model.Backward(r, c, rg);
                            pc.x = pp.x;
                            pc.y = pp.y;
                            pc.z = pp.z;
                          }
                        }
                      }
                    });
}

}  // namespace sv
