#include "sv/util/math.h"

#include <Eigen/Geometry>

namespace sv {

Eigen::Matrix3d CalCovar3d(const Eigen::Matrix3Xd& X) {
  const Eigen::Vector3d m = X.rowwise().mean();   // mean
  const Eigen::Matrix3Xd Xm = (X.colwise() - m);  // centered
  return ((Xm * Xm.transpose()) / (X.cols() - 1));
}

}  // namespace sv
