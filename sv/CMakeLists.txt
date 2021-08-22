# base =========================================================================
cc_library(
  NAME base
  DEPS sv_options
  INCS ${CMAKE_SOURCE_DIR}
  INTERFACE)

if(BUILD_MARCH_NATIVE)
  target_compile_options(sv_base INTERFACE "-march=native")
endif()

cc_library(
  NAME tbb
  DEPS ${TBB_LIBRARIES}
  INCS ${TBB_INCLUDE_DIRS}
  INTERFACE)

cc_library(
  NAME log
  DEPS fmt::fmt glog::glog
  INTERFACE)

# util =========================================================================
cc_library(
  NAME util_timer
  HDRS "util/timer.h"
  DEPS sv_base absl::time
  INTERFACE)

cc_library(
  NAME util_manager
  SRCS "util/manager.cpp"
  DEPS sv_base sv_log sv_util_timer absl::flat_hash_map)

cc_library(
  NAME util_math
  SRCS "util/math.cpp"
  DEPS sv_base Eigen3::Eigen)
cc_test(
  NAME util_math_test
  SRCS "util/math_test_bench.cpp"
  DEPS sv_util_math benchmark::benchmark)
cc_bench(
  NAME util_math_bench
  SRCS "util/math_test_bench.cpp"
  DEPS sv_util_math GTest::GTest)

cc_library(
  NAME util_ocv
  SRCS "util/ocv.cpp"
  DEPS sv_base sv_log opencv_core opencv_imgproc opencv_highgui)

# llol =========================================================================
cc_library(
  NAME llol_lidar
  SRCS "llol/lidar.cpp"
  DEPS sv_util_math sv_util_ocv)
cc_test(
  NAME llol_lidar_test
  SRCS "llol/lidar_test.cpp"
  DEPS sv_llol_lidar)

cc_library(
  NAME llol_scan
  SRCS "llol/scan.cpp"
  DEPS sv_util_ocv Sophus::Sophus)
cc_test(
  NAME llol_scan_test
  SRCS "llol/scan_test.cpp"
  DEPS sv_llol_scan benchmark::benchmark)
cc_bench(
  NAME llol_scan_bench
  SRCS "llol/scan_test.cpp"
  DEPS sv_llol_scan GTest::GTest)

cc_library(
  NAME llol_grid
  SRCS "llol/grid.cpp"
  DEPS sv_util_math sv_llol_scan sv_tbb)
cc_test(
  NAME llol_grid_test
  SRCS "llol/grid_test.cpp"
  DEPS sv_llol_grid benchmark::benchmark)
cc_bench(
  NAME llol_grid_bench
  SRCS "llol/grid_test.cpp"
  DEPS sv_llol_grid GTest::GTest)

cc_library(
  NAME llol_pano
  SRCS "llol/pano.cpp"
  DEPS sv_llol_lidar sv_llol_scan sv_tbb)
cc_test(
  NAME llol_pano_test
  SRCS "llol/pano_test.cpp"
  DEPS sv_llol_pano benchmark::benchmark)
cc_bench(
  NAME llol_pano_bench
  SRCS "llol/pano_test.cpp"
  DEPS sv_llol_pano GTest::GTest)

cc_library(
  NAME llol_match
  SRCS "llol/match.cpp"
  DEPS sv_llol_scan sv_llol_grid sv_llol_pano)
cc_test(
  NAME llol_match_test
  SRCS "llol/match_test.cpp"
  DEPS sv_llol_match benchmark::benchmark)
cc_bench(
  NAME llol_match_bench
  SRCS "llol/match_test.cpp"
  DEPS sv_llol_match GTest::GTest)

cc_library(
  NAME llol_factor
  SRCS "llol/factor.cpp"
  DEPS sv_llol_match sv_tbb Sophus::Sophus)

cc_library(
  NAME llol_imu
  SRCS "llol/imu.cpp"
  DEPS sv_base Sophus::Sophus Eigen3::Eigen)
# cc_test(
#   NAME llol_imu_test
#   SRCS "llol/imu_test.cpp"
#   DEPS sv_llol_imu benchmark::benchmark)
cc_bench(
  NAME llol_imu_bench
  SRCS "llol/imu_test.cpp"
  DEPS sv_llol_imu GTest::GTest)

# node =========================================================================
if(catkin_FOUND AND BUILD_NODE)
  cc_library(
    NAME ros1
    DEPS ${catkin_LIBRARIES}
    INCS ${catkin_INCLUDE_DIRS}
    INTERFACE)

  cc_library(
    NAME node_conv
    SRCS "node/conv.cpp"
    DEPS sv_base sv_ros1)

  cc_library(
    NAME node_viz
    SRCS "node/viz.cpp"
    DEPS sv_llol_grid sv_llol_pano sv_ros1)

  cc_binary(
    NAME node_odom
    SRCS "node/odom.cpp"
    DEPS sv_llol_match sv_llol_factor sv_node_viz sv_node_conv sv_util_manager
         Ceres::ceres)
endif()

# temp =========================================================================
cc_binary(
  NAME curve_fitting
  SRCS "temp/curve_fitting.cpp"
  DEPS sv_util_timer Ceres::ceres)
cc_bench(
  NAME ceres_bench
  SRCS "temp/ceres_bench.cpp"
  DEPS Ceres::ceres sv_tbb)

# sim ==========================================================================
cc_library(
  NAME sim_rt
  SRCS "sim/transform.cpp"
  DEPS sv_base Eigen3::Eigen Sophus::Sophus)

cc_library(
  NAME doctest_main
  SRCS "sim/doctest_main.cpp"
  DEPS sv_base)

cc_binary(
  NAME sim_test
  SRCS "sim/tuple_test.cpp" "sim/transform_test.cpp" "sim/ray_test.cpp"
  DEPS sv_sim_rt sv_doctest_main)
