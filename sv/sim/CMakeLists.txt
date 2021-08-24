cc_library(
  NAME sim_rt
  SRCS "transform.cpp"
  DEPS sv_base Eigen3::Eigen Sophus::Sophus)

cc_library(
  NAME doctest_main
  SRCS "doctest_main.cpp"
  DEPS sv_base)

cc_binary(
  NAME sim_test
  SRCS "tuple_test.cpp" "transform_test.cpp" "ray_test.cpp"
  DEPS sv_sim_rt sv_doctest_main)
