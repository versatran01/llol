#include "sv/node/conv.h"

namespace sv {

void SE3d2Transform(const Sophus::SE3d& pose, geometry_msgs::Transform& tf) {
  tf2::toMsg(pose.translation(), tf.translation);
  tf.rotation = tf2::toMsg(pose.unit_quaternion());
}

}  // namespace sv
