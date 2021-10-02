cc_library(
  NAME ros1
  DEPS ${catkin_LIBRARIES}
  INCS ${catkin_INCLUDE_DIRS}
  INTERFACE)

cc_library(
  NAME node_viz
  SRCS "viz.cpp"
  DEPS sv_llol_grid sv_llol_pano sv_ros1 opencv_highgui opencv_imgproc)

cc_library(
  NAME node_pcl
  SRCS "pcl.cpp"
  DEPS sv_llol_sweep sv_llol_pano sv_ros1)

cc_library(
  NAME node_conv
  SRCS "conv.cpp"
  DEPS sv_ros1
       sv_llol_sweep
       sv_llol_grid
       sv_llol_pano
       sv_llol_imu
       sv_llol_gicp
       sv_llol_cost)

cc_binary(
  NAME node_llol
  SRCS "llol_main.cpp" "llol_node.cpp" "llol_pub.cpp" "llol_icp.cpp"
  DEPS sv_node_conv sv_node_viz sv_node_pcl sv_util_manager)
