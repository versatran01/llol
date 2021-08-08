#include "sv/util/math.h"

#include <Eigen/Dense>

namespace sv {

Eigen::Matrix3d CalCovar3d(const Eigen::Matrix3Xd& X) {
  const Eigen::Vector3d m = X.rowwise().mean();   // mean
  const Eigen::Matrix3Xd Xm = (X.colwise() - m);  // centered
  return ((Xm * Xm.transpose()) / (X.cols() - 1));
}

void MakeRightHanded(Eigen::Vector3d& eigvals, Eigen::Matrix3d& eigvecs) {
  // Note that sorting of eigenvalues may end up with left-hand coordinate
  // system. So here we correctly sort it so that it does end up being
  // right-handed and normalised.
  auto hand = eigvecs.col(0).cross(eigvecs.col(1)).dot(eigvecs.col(2));
  if (hand < 0) {
    eigvecs.col(0).swap(eigvecs.col(1));
    eigvals.row(0).swap(eigvals.row(1));
  }
}

}  // namespace sv
