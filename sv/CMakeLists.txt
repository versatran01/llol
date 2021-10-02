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
