#include "sv/llol/match.h"

#include <Eigen/Cholesky>  // llt
#include <Eigen/Geometry>  // inverse

namespace sv {

void GicpMatch::ResetSweep() {
  px_s = {kBad, kBad};
  mc_s.Reset();
}

void GicpMatch::ResetPano() {
  px_p = {kBad, kBad};
  mc_p.Reset();
}

void GicpMatch::Reset() {
  ResetSweep();
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
