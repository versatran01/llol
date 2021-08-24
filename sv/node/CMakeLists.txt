cc_library(
  NAME ros1
  DEPS ${catkin_LIBRARIES}
  INCS ${catkin_INCLUDE_DIRS}
  INTERFACE)

cc_library(
  NAME node_conv
  SRCS "conv.cpp"
  DEPS sv_ros1 sv_llol_scan sv_llol_grid sv_llol_match sv_llol_pano sv_llol_imu)

cc_library(
  NAME node_viz
  SRCS "viz.cpp"
  DEPS sv_llol_grid sv_llol_pano sv_ros1 opencv_highgui opencv_imgproc)

cc_binary(
  NAME node_odom
  SRCS "odom.cpp"
  DEPS sv_node_conv sv_node_viz sv_llol_imu sv_llol_gicp sv_util_manager)
