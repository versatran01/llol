#include "sv/llol/match.h"

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

void PointMatch::CalcSqrtInfo(float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  if (lambda > 0) {
    cov.diagonal().array() += lambda;
  }
  U = MatrixSqrtUtU(cov.inverse().eval());
}

}  // namespace sv
