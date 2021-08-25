cc_library(
  NAME llol_lidar
  SRCS "lidar.cpp"
  DEPS sv_util_math sv_util_ocv)
cc_test(
  NAME llol_lidar_test
  SRCS "lidar_test.cpp"
  DEPS sv_llol_lidar)

cc_library(
  NAME llol_scan
  SRCS "scan.cpp"
  DEPS sv_util_math sv_util_ocv Sophus::Sophus)
cc_test(
  NAME llol_scan_test
  SRCS "scan_test.cpp"
  DEPS sv_llol_scan benchmark::benchmark)
cc_bench(
  NAME llol_scan_bench
  SRCS "scan_test.cpp"
  DEPS sv_llol_scan GTest::GTest)

cc_library(
  NAME llol_match
  SRCS "match.cpp"
  DEPS sv_base sv_util_math opencv_core)
cc_test(
  NAME llol_match_test
  SRCS "match_test.cpp"
  DEPS sv_llol_match)

cc_library(
  NAME llol_pano
  SRCS "pano.cpp"
  DEPS sv_llol_lidar sv_llol_scan sv_tbb)
cc_test(
  NAME llol_pano_test
  SRCS "pano_test.cpp"
  DEPS sv_llol_pano benchmark::benchmark)
cc_bench(
  NAME llol_pano_bench
  SRCS "pano_test.cpp"
  DEPS sv_llol_pano GTest::GTest)

cc_library(
  NAME llol_grid
  SRCS "grid.cpp"
  DEPS sv_llol_match sv_llol_scan sv_llol_pano)
cc_test(
  NAME llol_grid_test
  SRCS "grid_test.cpp"
  DEPS sv_llol_grid benchmark::benchmark)
cc_bench(
  NAME llol_grid_bench
  SRCS "grid_test.cpp"
  DEPS sv_llol_grid GTest::GTest)

cc_library(
  NAME llol_gicp
  SRCS "gicp.cpp"
  DEPS sv_llol_grid sv_util_solver sv_tbb Ceres::ceres)

cc_library(
  NAME llol_imu
  SRCS "imu.cpp"
  DEPS sv_base sv_log sv_util_ocv Sophus::Sophus Eigen3::Eigen Boost::boost)
cc_test(
  NAME llol_imu_test
  SRCS "imu_test.cpp"
  DEPS sv_llol_imu benchmark::benchmark)
cc_bench(
  NAME llol_imu_bench
  SRCS "imu_test.cpp"
  DEPS sv_llol_imu GTest::GTest)
