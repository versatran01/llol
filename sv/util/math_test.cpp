#include "sv/util/math.h"

#include <gtest/gtest.h>

namespace sv {

TEST(MathTest, TestAngleConversion) {
  EXPECT_DOUBLE_EQ(Deg2Rad(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(90.0), M_PI / 2.0);
  EXPECT_DOUBLE_EQ(Deg2Rad(180.0), M_PI);
  EXPECT_DOUBLE_EQ(Deg2Rad(360.0), M_PI * 2);
  EXPECT_DOUBLE_EQ(Deg2Rad(-180.0), -M_PI);

  EXPECT_DOUBLE_EQ(Rad2Deg(0.0), 0.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI / 2), 90.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI), 180.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(M_PI * 2), 360.0);
  EXPECT_DOUBLE_EQ(Rad2Deg(-M_PI), -180.0);
}

TEST(MathTest, TestMeanCovar) {
  for (int i = 3; i < 50; i += 10) {
    const auto X = Eigen::Matrix3Xd::Random(3, i).eval();
    const auto cov0 = Covariance(X);

    MeanCovar3d mc;
    for (int i = 0; i < X.cols(); ++i) mc.Add(X.col(i));
    const auto cov1 = mc.covar();

    EXPECT_TRUE(cov0.isApprox(cov1));
  }
}

}  // namespace sv
