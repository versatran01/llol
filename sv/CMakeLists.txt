cc_library(
  NAME base
  DEPS sv_options
  INCLUDES ${CMAKE_SOURCE_DIR}
  INTERFACE)

add_subdirectory(util)
add_subdirectory(llol)
