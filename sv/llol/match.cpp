#include "sv/llol/match.h"

#include <Eigen/Cholesky>  // llt
#include <Eigen/Geometry>  // inverse

namespace sv {

void PointMatch::ResetGrid() {
  px_g = {kBadPx, kBadPx};
  mc_g.Reset();
}

void PointMatch::ResetPano() {
  px_p = {kBadPx, kBadPx};
  mc_p.Reset();
}

void PointMatch::Reset() {
  ResetGrid();
  ResetPano();
  U.setZero();
}

void PointMatch::SqrtInfo(float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  if (lambda > 0) {
    cov.diagonal().array() += lambda;
  }
  U = MatrixSqrtUtU(cov.inverse().eval());
}

Eigen::Matrix3f MatrixSqrtUtU(const Eigen::Matrix3f& A) {
  return A.selfadjointView<Eigen::Upper>().llt().matrixU();
}

}  // namespace sv
