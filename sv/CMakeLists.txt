cc_library(
  NAME base
  DEPS sv_options
  INCLUDES ${CMAKE_SOURCE_DIR}
  INTERFACE)

cc_library(
  NAME log
  DEPS fmt::fmt glog::glog
  INTERFACE)

# util
cc_library(
  NAME util_manager
  SRCS "util/manager.cpp"
  DEPS sv_base sv_log absl::time absl::flat_hash_map)

cc_library(
  NAME util_math
  SRCS "util/math.cpp"
  DEPS sv_base Eigen3::Eigen)
cc_test(
  NAME util_math_test
  SRCS "util/math_test.cpp"
  DEPS sv_util_math)
cc_bench(
  NAME util_math_bench
  SRCS "util/math_bench.cpp"
  DEPS sv_util_math)

cc_library(
  NAME util_ocv
  SRCS "util/ocv.cpp"
  DEPS sv_base sv_log opencv_core opencv_imgproc opencv_highgui)

# llol
cc_library(
  NAME llol_lidar
  SRCS "llol/lidar.cpp"
  DEPS sv_base sv_log sv_util_math sv_util_ocv TBB::tbb)
cc_test(
  NAME llol_lidar_test
  SRCS "llol/lidar_test.cpp"
  DEPS sv_llol_lidar)
cc_bench(
  NAME llol_lidar_bench
  SRCS "llol/lidar_bench.cpp"
  DEPS sv_llol_lidar)

cc_library(
  NAME llol_match
  SRCS "llol/match.cpp"
  DEPS sv_llol_lidar)

if(catkin_FOUND)
  cc_binary(
    NAME llol_node
    SRCS "llol/node.cpp"
    DEPS sv_llol_lidar
         sv_llol_match
         sv_util_ocv
         sv_util_manager
         sv_ros1
         Ceres::ceres
         Sophus::Sophus)
endif()
