#include <ros/ros.h>

#include "sv/node/llol_node.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "llol_node");
  sv::OdomNode node(ros::NodeHandle("~"));
  ros::spin();
  return 0;
}
