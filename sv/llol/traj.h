#pragma once

#include <sophus/se3.hpp>

namespace sv {

struct SinglePose {
  double t;  // time
  Sophus::SE3d T_p_s{};
};

}  // namespace sv
