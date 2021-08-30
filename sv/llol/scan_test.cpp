#include "sv/llol/scan.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(ScanTest, TestCtor) {
  ScanBase s{cv::Size{20, 10}, CV_8UC3};
  EXPECT_EQ(s.rows(), 10);
  EXPECT_EQ(s.cols(), 20);
  EXPECT_EQ(s.type(), CV_8UC3);
  EXPECT_EQ(s.total(), 200);
  EXPECT_EQ(s.channels(), 3);
}

TEST(ScanTest, TestUpdate) {
  ScanBase s{cv::Size{20, 10}, CV_8UC1};

  s.UpdateView({0, 10});
  EXPECT_EQ(s.curr.start, 0);
  EXPECT_EQ(s.curr.end, 10);

  s.UpdateView({10, 20});
  EXPECT_EQ(s.curr.start, 10);
  EXPECT_EQ(s.curr.end, 20);

  s.UpdateView({0, 10});
  EXPECT_EQ(s.curr.start, 0);
  EXPECT_EQ(s.curr.end, 10);

  s.UpdateView({10, 20});
  EXPECT_EQ(s.curr.start, 10);
  EXPECT_EQ(s.curr.end, 20);
}

TEST(ScanTest, TestColMod) {
  EXPECT_EQ(ColMod(0 - 64, 64), 0);
  EXPECT_EQ(ColMod(1 - 64, 64), 1);
  EXPECT_EQ(ColMod(63 - 64, 64), 63);

  EXPECT_EQ(ColMod(0 - 2, 64), 62);
  EXPECT_EQ(ColMod(1 - 2, 64), 63);
  EXPECT_EQ(ColMod(2 - 2, 64), 0);
  EXPECT_EQ(ColMod(63 - 2, 64), 61);
}

}  // namespace
}  // namespace sv
