#include "sv/util/ocv.h"

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

cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale,
                  int cmap,
                  uint8_t bad_color) {
  CHECK_EQ(input.channels(), 1);

  cv::Mat disp;
  input.convertTo(disp, CV_8UC1, scale * 255.0);
  cv::applyColorMap(disp, disp, cmap);

  if (input.depth() >= CV_32F) {
    disp.setTo(bad_color, cv::Mat(~(input > 0)));
  }

  return disp;
}

void Imshow(const std::string& name, const cv::Mat& mat, int flag) {
  cv::namedWindow(name, flag);
  cv::imshow(name, mat);
  cv::waitKey(1);
}

}  // namespace sv
