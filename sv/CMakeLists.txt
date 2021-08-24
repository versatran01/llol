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

add_subdirectory(util)
add_subdirectory(llol)

if(catkin_FOUND AND BUILD_NODE)
  add_subdirectory(node)
endif()

add_subdirectory(sim)

# temp =========================================================================
cc_binary(
  NAME curve_fitting
  SRCS "temp/curve_fitting.cpp"
  DEPS sv_util_timer Ceres::ceres)
cc_bench(
  NAME ceres_bench
  SRCS "temp/ceres_bench.cpp"
  DEPS Ceres::ceres sv_tbb)
