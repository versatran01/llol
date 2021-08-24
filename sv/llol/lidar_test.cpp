#include "sv/llol/lidar.h"

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

TEST(LidarTest, TestToRowToCol) {
  const LidarModel lm{{32, 2}};

  EXPECT_EQ(lm.ToRow(0.0001F, 1), 0);
  EXPECT_EQ(lm.ToRow(0, 1), 1);
}

}  // namespace
}  // namespace sv
