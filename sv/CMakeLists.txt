cc_library(
  NAME base
  DEPS sv_options
  INCLUDES ${CMAKE_SOURCE_DIR}
  INTERFACE)

cc_library(
  NAME log
  DEPS fmt::fmt glog::glog
  INTERFACE)

# util =========================================================================
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
  DEPS sv_base sv_log sv_util_math sv_util_ocv)

cc_library(
  NAME llol_sweep
  SRCS "llol/sweep.cpp"
  DEPS sv_base sv_log sv_util_math sv_util_ocv)
cc_test(
  NAME llol_sweep_test
  SRCS "llol/sweep_test_bench.cpp"
  DEPS sv_llol_sweep benchmark::benchmark)
cc_bench(
  NAME llol_sweep_bench
  SRCS "llol/sweep_test_bench.cpp"
  DEPS sv_llol_sweep GTest::GTest)

cc_library(
  NAME llol_pano
  SRCS "llol/pano.cpp"
  DEPS sv_llol_lidar TBB::tbb)
cc_test(
  NAME llol_pano_test
  SRCS "llol/pano_test_bench.cpp"
  DEPS sv_llol_pano benchmark::benchmark)
cc_bench(
  NAME llol_pano_bench
  SRCS "llol/pano_test_bench.cpp"
  DEPS sv_llol_pano GTest::GTest)

cc_library(
  NAME llol_match
  SRCS "llol/match.cpp"
  DEPS sv_llol_sweep sv_llol_pano)

cc_library(
  NAME llol_optim
  SRCS "llol/optim.cpp"
  DEPS sv_llol_match Ceres::ceres Sophus::Sophus)


# node =========================================================================
if(catkin_FOUND)
  cc_library(
    NAME llol_viz
    SRCS "llol/viz.cpp"
    DEPS sv_llol_sweep sv_llol_match sv_ros1)

  cc_binary(
    NAME llol_node
    SRCS "llol/node.cpp"
    DEPS sv_llol_match sv_llol_optim sv_llol_viz sv_util_manager sv_ros1)
endif()
