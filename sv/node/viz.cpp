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

void Grid2Markers(const SweepGrid& grid,
                  const std_msgs::Header& header,
                  std::vector<Marker>& markers) {
  markers.resize(grid.total());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;

  Marker pano_mk;
  pano_mk.header = header;
  pano_mk.ns = "match";
  pano_mk.type = Marker::SPHERE;
  pano_mk.color.a = 0.5;
  pano_mk.color.r = 0.0;
  pano_mk.color.g = 1.0;
  pano_mk.color.b = 0.0;

  Eigen::Matrix3f covar;

  // TODO (chao): only draw matches up to width?
  // TODO (chao): also need to remove bad match
  for (int r = 0; r < grid.rows(); ++r) {
    for (int c = 0; c < grid.cols(); ++c) {
      const auto i = grid.Px2Ind({c, r});
      const auto& match = grid.MatchAt({c, r});

      auto& marker = markers.at(grid.Px2Ind({c, r}));
      marker = pano_mk;
      marker.id = i;

      if (match.Ok()) {
        marker.action = Marker::ADD;
        covar = match.mc_p.Covar();
        covar.diagonal().array() += 1e-10;
        es.compute(covar);
        MeanCovar2Marker(
            marker, match.mc_p.mean, es.eigenvalues(), es.eigenvectors());
      } else {
        marker.action = Marker::DELETE;
      }
    }
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
                CloudXYZ& cloud) {
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

void Sweep2Cloud(const LidarSweep& sweep,
                 const std_msgs::Header header,
                 CloudXYZ& cloud) {
  const auto size = sweep.size();
  if (cloud.empty()) {
    cloud.resize(sweep.total());
    cloud.width = size.width;
    cloud.height = size.height;
  }

  pcl_conversions::toPCL(header, cloud.header);
  tbb::parallel_for(tbb::blocked_range<int>(0, size.height),
                    [&](const auto& blk) {
                      for (int r = blk.begin(); r < blk.end(); ++r) {
                        for (int c = 0; c < size.width; ++c) {
                          const auto& tf = sweep.tfs.at(c);
                          const auto& xyzr = sweep.XyzrAt({c, r});
                          auto& pc = cloud.at(c, r);
                          if (std::isnan(xyzr[0])) {
                            pc.x = pc.y = pc.z = kNaNF;
                          } else {
                            Eigen::Map<const Eigen::Vector3f> xyz(&xyzr[0]);
                            pc.getArray3fMap() = tf * xyz;
                          }
                        }
                      }
                    });
}

}  // namespace sv
