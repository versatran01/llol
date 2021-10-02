#include "sv/node/pcl.h"

#include <pcl_conversions/pcl_conversions.h>
#include <tbb/parallel_for.h>

namespace sv {

using Vector3f = Eigen::Vector3f;

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
                          auto& pc = cloud.at(c, r);

                          const auto rg = pano.RangeAt({c, r});
                          if (rg == 0) {
                            pc.x = pc.y = pc.z = kNaNF;
                            continue;
                          }

                          const auto pp = pano.model.Backward(r, c, rg);
                          pc.x = pp.x;
                          pc.y = pp.y;
                          pc.z = pp.z;
                        }
                      }
                    });
}

void Sweep2Cloud(const LidarSweep& sweep,
                 const std_msgs::Header header,
                 CloudXYZI& cloud) {
  const auto size = sweep.size();
  if (cloud.empty()) {
    cloud.resize(sweep.total());
    cloud.width = size.width;
    cloud.height = size.height;
  }

  pcl_conversions::toPCL(header, cloud.header);
  tbb::parallel_for(
      tbb::blocked_range<int>(0, size.height), [&](const auto& blk) {
        for (int r = blk.begin(); r < blk.end(); ++r) {
          for (int c = 0; c < size.width; ++c) {
            auto& pt = cloud.at(c, r);

            const auto& pixel = sweep.PixelAt({c, r});
            if (!pixel.Ok()) {
              pt.x = pt.y = pt.z = pt.intensity = kNaNF;
              continue;
            }

            pt.getVector3fMap() = sweep.TfAt(c) * pixel.Vec3fMap();
            pt.intensity = pixel.intensity;
          }
        }
      });
}

void Grid2Cloud(const SweepGrid& grid,
                const std_msgs::Header& header,
                CloudXYZI& cloud) {
  const auto size = grid.size();
  if (cloud.empty()) {
    cloud.resize(grid.total());
    cloud.width = size.width;
    cloud.height = size.height;
  }

  pcl_conversions::toPCL(header, cloud.header);
  tbb::parallel_for(tbb::blocked_range<int>(0, size.height),
                    [&](const auto& blk) {
                      for (int r = blk.begin(); r < blk.end(); ++r) {
                        for (int c = 0; c < size.width; ++c) {
                          auto& pt = cloud.at(c, r);
                          const auto& match = grid.MatchAt({c, r});

                          if (!match.GridOk()) {
                            pt.x = pt.y = pt.z = kNaNF;
                            continue;
                          }

                          pt.getVector3fMap() = grid.TfAt(c) * match.mc_g.mean;
                          pt.intensity = match.PanoOk() ? 1.0 : 0.5;
                        }
                      }
                    });
}

}  // namespace sv
