#include <ros/ros.h>

#include "sv/node/odom.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_node");
  sv::OdomNode node(ros::NodeHandle("~"));
  ros::spin();
  return 0;
}
