#include "sv/llol/match.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(LinalgTest, TestMatrixSqrtUpper) {
  Eigen::Matrix3Xf X = Eigen::Matrix3Xf::Random(3, 100);
  const Eigen::Matrix3f A = X * X.transpose();
  const Eigen::Matrix3f U = MatrixSqrtUtU(A);
  const Eigen::Matrix3f UtU = U.transpose() * U;
  EXPECT_TRUE(A.isApprox(UtU));
}

}  // namespace
}  // namespace sv
