#include "sv/llol/match.h"

#include <Eigen/Cholesky>  // llt
#include <Eigen/Geometry>  // inverse

namespace sv {

void GicpMatch::ResetGrid() {
  px_g = {kBad, kBad};
  mc_g.Reset();
}

void GicpMatch::ResetPano() {
  px_p = {kBad, kBad};
  mc_p.Reset();
}

void GicpMatch::Reset() {
  ResetGrid();
  ResetPano();
  U.setZero();
}

void GicpMatch::SqrtInfo(float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  cov.diagonal().array() += lambda;
  U = MatrixSqrtUtU(cov.inverse().eval());
}

Eigen::Matrix3f MatrixSqrtUtU(const Eigen::Matrix3f& A) {
  return A.selfadjointView<Eigen::Upper>().llt().matrixU();
}

}  // namespace sv
