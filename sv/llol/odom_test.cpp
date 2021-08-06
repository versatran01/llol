#include "sv/llol/odom.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(LidarSweepTest, TestDefault) {
  LidarSweep ls;
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.empty(), true);
  EXPECT_EQ(ls.full(), false);
}

TEST(LidarSweepTest, TestCtor) {
  LidarSweep ls({8, 4}, {2, 1});
  std::cout << ls << "\n";

  EXPECT_EQ(ls.width(), 0);
  EXPECT_EQ(ls.empty(), false);
  EXPECT_EQ(ls.full(), false);

  EXPECT_EQ(ls.sweep().rows, 4);
  EXPECT_EQ(ls.sweep().cols, 8);
  EXPECT_EQ(ls.sweep().channels(), 4);

  EXPECT_EQ(ls.grid().rows, 4);
  EXPECT_EQ(ls.grid().cols, 4);
  EXPECT_EQ(ls.grid().channels(), 1);
}

TEST(LidarSweepTest, TestAddScan) {
  LidarSweep ls({8, 4}, {2, 1});
  cv::Mat scan;
  scan.create(4, 4, CV_32FC4);
  scan.setTo(1);

  const int n = ls.AddScan(scan, {0, 4}, false);

  EXPECT_EQ(ls.range().start, 0);
  EXPECT_EQ(ls.range().end, 4);
  EXPECT_EQ(n, 8);

  const int n2 = ls.AddScan(scan, {4, 8}, false);
  EXPECT_EQ(ls.range().start, 4);
  EXPECT_EQ(ls.range().end, 8);
  EXPECT_EQ(n2, 8);
  EXPECT_EQ(ls.full(), true);
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
