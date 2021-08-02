cc_library(
  NAME base
  DEPS llol_options
  INCLUDES ${CMAKE_SOURCE_DIR}
  INTERFACE)

add_subdirectory(util)
