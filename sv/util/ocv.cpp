#include "sv/util/ocv.h"

#include <fmt/color.h>
#include <glog/logging.h>

namespace sv {

std::string CvTypeStr(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
std::string Repr(const cv::Mat& mat) {
  return fmt::format("(hwc=({},{},{}), depth={})",
                     mat.rows,
                     mat.cols,
                     mat.channels(),
                     mat.depth());
}

std::string Repr(const cv::Size& size) {
  return fmt::format("(h={}, w={})", size.height, size.width);
}

std::string Repr(const cv::Range& range) {
  return fmt::format("[{},{})", range.start, range.end);
}

}  // namespace sv
