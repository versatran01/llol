#include "sv/util/math.h"

#include <Eigen/Geometry>

namespace sv {

Eigen::Matrix3d CalCovar3d(const Eigen::Matrix3Xd& X) {
  const Eigen::Vector3d m = X.rowwise().mean();   // mean
  const Eigen::Matrix3Xd Xm = (X.colwise() - m);  // centered
  return ((Xm * Xm.transpose()) / (X.cols() - 1));
}

void MakeRightHanded(Eigen::Vector3f& eigvals, Eigen::Matrix3f& eigvecs) {
  auto hand = eigvecs.col(0).cross(eigvecs.col(1)).dot(eigvecs.col(2));
  if (hand < 0) {
    eigvecs.col(0).swap(eigvecs.col(1));
    eigvals.row(0).swap(eigvals.row(1));
  }
}

}  // namespace sv
