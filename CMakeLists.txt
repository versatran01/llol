cmake_minimum_required(VERSION 3.15)
project(llol)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_TESTING "Build tests" Off)
option(BUILD_BENCHMARK "Build benchmarks" Off)

set(CC_TARGET_PREFIX llol)
include(CMakeHelpers)

add_library(llol_options INTERFACE)
include(CompilerWarnings)
enable_warnings(llol_options)
include(Sanitizers)
enable_sanitizers(llol_options)

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             tf2_ros
             image_transport
             sensor_msgs
             nav_msgs
             cv_bridge
             image_geometry)

# base
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(TBB REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(GTest REQUIRED)
find_package(GMock REQUIRED)
find_package(benchmark REQUIRED)

if(catkin_FOUND)
  catkin_package(INCLUDE_DIRS ${CMAKE_SOURCE_DIR})

  cc_library(
    NAME ros1
    DEPS ${catkin_LIBRARIES}
    INCLUDES ${catkin_INCLUDE_DIRS}
    INTERFACE)
endif()

enable_testing()
add_subdirectory(llol)
