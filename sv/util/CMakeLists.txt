cc_library(
  NAME util_timer
  HDRS "timer.h"
  DEPS sv_base absl::time
  INTERFACE)

cc_library(
  NAME util_manager
  SRCS "manager.cpp"
  DEPS sv_base sv_log sv_util_timer absl::flat_hash_map)

cc_library(
  NAME util_math
  SRCS "math.cpp"
  DEPS sv_base sv_log Eigen3::Eigen)
cc_test(
  NAME util_math_test
  SRCS "math_test.cpp"
  DEPS sv_util_math benchmark::benchmark)
cc_bench(
  NAME util_math_bench
  SRCS "math_test.cpp"
  DEPS sv_util_math GTest::GTest)

cc_library(
  NAME util_ocv
  SRCS "ocv.cpp"
  DEPS sv_base sv_log opencv_core)
cc_test(
  NAME util_ocv_test
  SRCS "ocv_test.cpp"
  DEPS sv_util_ocv)

cc_library(
  NAME util_nlls
  SRCS "nlls.cpp"
  DEPS sv_base sv_log Eigen3::Eigen)
cc_test(
  NAME util_nlls_test
  SRCS "nlls_test.cpp"
  DEPS sv_util_nlls)

# cc_library(
#   NAME util_solver
#   SRCS "solver.cpp"
#   DEPS sv_base sv_log Eigen3::Eigen)
# cc_test(
#   NAME util_solver_test
#   SRCS "solver_test.cpp"
#   DEPS sv_util_solver)