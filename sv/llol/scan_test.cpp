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

}  // namespace
}  // namespace sv
