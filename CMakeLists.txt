cmake_minimum_required(VERSION 3.15)
project(llol)

set(CMAKE_CXX_STANDARD 17)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_NODE "Build ros nodes" On)
option(BUILD_TESTING "Build tests" Off)
option(BUILD_BENCHMARK "Build benchmarks" Off)
option(BUILD_MARCH_NATIVE "Build with -march=native" OFF)

set(CC_TARGET_PREFIX sv)
include(CMakeHelpers)

add_library(sv_options INTERFACE)
include(CompilerWarnings)
enable_warnings(sv_options)
include(Sanitizers)
enable_sanitizers(sv_options)

find_package(
  catkin QUIET
  COMPONENTS roscpp
             pcl_ros
             tf2_ros
             tf2_eigen
             cv_bridge
             image_transport
             pcl_conversions
             nav_msgs
             sensor_msgs
             diagnostic_msgs
             visualization_msgs)

find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
endif()
if(BUILD_BENCHMARK)
  find_package(benchmark REQUIRED)
endif()

if(catkin_FOUND AND BUILD_NODE)
  catkin_package()

endif()

enable_testing()
add_subdirectory(sv)
