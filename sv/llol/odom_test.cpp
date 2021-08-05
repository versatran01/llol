#include "sv/llol/odom.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.empty(), true);
  EXPECT_EQ(ls.full(), false);
}

TEST(LidarSweepTest, TestAddScan) {
  LidarSweep ls({8, 4});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.empty(), false);
  EXPECT_EQ(ls.full(), false);

  cv::Mat scan(4, 4, CV_32FC4);
  ls.AddScan(scan, {0, 4});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 4);
  EXPECT_EQ(ls.empty(), false);
  EXPECT_EQ(ls.full(), false);

  ls.AddScan(scan, {4, 8});
  std::cout << ls << "\n";
  EXPECT_EQ(ls.width(), 8);
  EXPECT_EQ(ls.full(), true);
}

TEST(DepthPanoTest, TestCtor) {
  DepthPano dp({256, 64});
  EXPECT_EQ(dp.empty(), false);
  std::cout << dp << "\n";
}

TEST(DepthPanoTest, TestWinAt) {
  DepthPano dp({256, 64});
  const auto win = dp.WinAt({0, 0}, {5, 7});
  EXPECT_EQ(win.x, -5);
  EXPECT_EQ(win.y, -7);
  EXPECT_EQ(win.width, 11);
  EXPECT_EQ(win.height, 15);
}

TEST(DepthPanoTest, TestBoundedWinAt) {
  DepthPano dp({256, 64});
  const auto win = dp.BoundedWinAt({0, 0}, {5, 7});
  EXPECT_EQ(win.x, 0);
  EXPECT_EQ(win.y, 0);
  EXPECT_EQ(win.width, 6);
  EXPECT_EQ(win.height, 8);
}

}  // namespace
}  // namespace sv
