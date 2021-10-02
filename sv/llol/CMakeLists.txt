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
  DEPS sv_llol_scan)

cc_library(
  NAME llol_imu
  SRCS "imu.cpp"
  DEPS sv_util_math Sophus::Sophus Boost::boost)
cc_test(
  NAME llol_imu_test
  SRCS "imu_test.cpp"
  DEPS sv_llol_imu benchmark::benchmark)
cc_bench(
  NAME llol_imu_bench
  SRCS "imu_test.cpp"
  DEPS sv_llol_imu GTest::GTest)

cc_library(
  NAME llol_traj
  SRCS "traj.cpp"
  DEPS sv_llol_imu)
cc_test(
  NAME llol_traj_test
  SRCS "traj_test.cpp"
  DEPS sv_llol_traj)

cc_library(
  NAME llol_sweep
  SRCS "sweep.cpp"
  DEPS sv_llol_scan sv_llol_traj sv_tbb)
cc_test(
  NAME llol_sweep_test
  SRCS "sweep_test.cpp"
  DEPS sv_llol_sweep benchmark::benchmark)
cc_bench(
  NAME llol_sweep_bench
  SRCS "sweep_test.cpp"
  DEPS sv_llol_sweep GTest::GTest)

cc_library(
  NAME llol_match
  SRCS "match.cpp"
  DEPS sv_util_math opencv_core)

cc_library(
  NAME llol_pano
  SRCS "pano.cpp"
  DEPS sv_llol_lidar sv_llol_sweep sv_tbb)
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
  DEPS sv_llol_traj sv_llol_match sv_llol_scan sv_tbb)
cc_test(
  NAME llol_grid_test
  SRCS "grid_test.cpp"
  DEPS sv_llol_grid benchmark::benchmark)
cc_bench(
  NAME llol_grid_bench
  SRCS "grid_test.cpp"
  DEPS sv_llol_grid GTest::GTest)

cc_library(
  NAME llol_cost
  SRCS "cost.cpp"
  DEPS sv_llol_grid sv_util_nlls)
cc_test(
  NAME llol_cost_test
  SRCS "cost_test.cpp"
  DEPS sv_llol_cost benchmark::benchmark)
cc_bench(
  NAME llol_cost_bench
  SRCS "cost_test.cpp"
  DEPS sv_llol_cost GTest::GTest)

cc_library(
  NAME llol_gicp
  SRCS "gicp.cpp"
  DEPS sv_llol_cost sv_llol_pano sv_util_ocv sv_tbb)
cc_test(
  NAME llol_gicp_test
  SRCS "gicp_test.cpp"
  DEPS sv_llol_gicp benchmark::benchmark)
cc_bench(
  NAME llol_gicp_bench
  SRCS "gicp_test.cpp"
  DEPS sv_llol_gicp GTest::GTest)
