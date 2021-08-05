cc_library(
  NAME llol_odom
  SRCS "odom.cpp"
  DEPS sv_base sv_log opencv_core Eigen3::Eigen TBB::tbb)
cc_test(
  NAME llol_odom_test
  SRCS "odom_test.cpp"
  DEPS sv_llol_odom)
cc_bench(
  NAME llol_odom_bench
  SRCS "odom_bench.cpp"
  DEPS sv_llol_odom)

cc_binary(
  NAME llol_node
  SRCS "node.cpp"
  DEPS sv_llol_odom sv_util_manager sv_ros1)
