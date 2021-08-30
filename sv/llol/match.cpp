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

void PointMatch::CalcSqrtInfo(const Eigen::Matrix3f& R_p_g, float lambda) {
  Eigen::Matrix3f cov = mc_p.Covar();
  cov += R_p_g * mc_g.Covar() * R_p_g.transpose();
  if (lambda > 0) {
    cov.diagonal().array() += lambda;
  }
  U = MatrixSqrtUtU(cov.inverse().eval());
}

}  // namespace sv
