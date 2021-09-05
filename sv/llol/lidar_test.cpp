#include "sv/llol/lidar.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(LidarTest, TestCtor) {
  const LidarModel lm{{1024, 64}};

  EXPECT_EQ(lm.size.height, 64);
  EXPECT_EQ(lm.size.width, 1024);
  EXPECT_EQ(lm.elevs.size(), 64);
  EXPECT_EQ(lm.azims.size(), 1024);
}

TEST(LidarTest, TestInside) {
  const LidarModel lm{{1024, 64}};

  EXPECT_EQ(lm.RowInside(0), true);
  EXPECT_EQ(lm.RowInside(-1), false);
  EXPECT_EQ(lm.RowInside(63), true);
  EXPECT_EQ(lm.RowInside(64), false);

  EXPECT_EQ(lm.ColInside(0), true);
  EXPECT_EQ(lm.ColInside(-1), false);
  EXPECT_EQ(lm.ColInside(1023), true);
  EXPECT_EQ(lm.ColInside(1024), false);
}

TEST(LidarTest, TestToRow) {
  const LidarModel lm{{32, 8}, Deg2Rad(70.0F)};
  EXPECT_EQ(lm.elev_max, Deg2Rad(35.0F));
  EXPECT_EQ(lm.elev_delta, Deg2Rad(10.0F));
  EXPECT_EQ(lm.elevs.front().sin, std::sin(Deg2Rad(35.0F)));
  EXPECT_EQ(lm.elevs.front().cos, std::cos(Deg2Rad(35.0F)));
  EXPECT_EQ(lm.elevs.back().sin, std::sin(Deg2Rad(-35.0F)));
  EXPECT_EQ(lm.elevs.back().cos, std::cos(Deg2Rad(-35.0F)));

  //     0       1       2       3      4         5         6        7
  // |   *   |   *    |  *   |   *  |   *    |    *    |    *    |   *    |
  // 40  35  30  25  20  15  10  5  0   -5  -10  -15  -20  -25  -30  -35  -40
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(45.0F)), 1), -1);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(40.01F)), 1), -1);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(39.99F)), 1), 0);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(35.0F)), 1), 0);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(30.01F)), 1), 0);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(29.99F)), 1), 1);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(25.0F)), 1), 1);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(20.01F)), 1), 1);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(19.99F)), 1), 2);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(15.00)), 1), 2);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(10.01)), 1), 2);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(0.01F)), 1), 3);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-0.01F)), 1), 4);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-30.1F)), 1), 7);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-35.0F)), 1), 7);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-39.9F)), 1), 7);

  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-40.01F)), 1), 8);
  EXPECT_EQ(lm.ToRow(std::sin(Deg2Rad(-45.0F)), 1), 8);
}

TEST(LidarTest, TestToCol) {
  const LidarModel lm{{8, 2}};
  EXPECT_FLOAT_EQ(lm.azim_delta, Deg2Rad(360.0F / 8));

  EXPECT_FLOAT_EQ(lm.azims.front().sin, std::sin(Deg2Rad(-22.5F)));
  EXPECT_FLOAT_EQ(lm.azims.front().cos, std::cos(Deg2Rad(-22.5F)));
  EXPECT_FLOAT_EQ(lm.azims.back().sin, std::sin(Deg2Rad(22.5F)));
  EXPECT_FLOAT_EQ(lm.azims.back().cos, std::cos(Deg2Rad(22.5F)));

  EXPECT_EQ(lm.ToCol(1.0, 0.0), 8);
  EXPECT_EQ(lm.ToCol(1.0, 0.01), 7);
  EXPECT_EQ(lm.ToCol(1.0, -0.01), 0);
  EXPECT_EQ(lm.ToCol(1.0, -0.99), 0);
  EXPECT_EQ(lm.ToCol(1.0, -1.0), 1);
  EXPECT_EQ(lm.ToCol(0.01, -1.0), 1);
  EXPECT_EQ(lm.ToCol(0.0, -1.0), 2);
  EXPECT_EQ(lm.ToCol(-1.0, -1.0), 3);
  EXPECT_EQ(lm.ToCol(-1.0, 0.0), 4);
  EXPECT_EQ(lm.ToCol(-1.0, 1.0), 5);
  EXPECT_EQ(lm.ToCol(0.0, 1.0), 6);
  EXPECT_EQ(lm.ToCol(1.0, 1.0), 7);
}

}  // namespace
}  // namespace sv
