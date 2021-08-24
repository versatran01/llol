#include "sv/llol/icp.h"

#include <glog/logging.h>

namespace sv {

using SE3d = Sophus::SE3d;

Icp::Icp(const IcpParams& params)
    : n_outer{params.n_outer}, n_inner{params.n_inner} {}

void Icp::Register(const SweepGrid& grid) {}

}  // namespace sv
