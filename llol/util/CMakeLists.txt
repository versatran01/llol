cc_library(
  NAME log
  DEPS fmt::fmt glog::glog
  INTERFACE)

cc_library(
  NAME manager
  SRCS "manager.cpp"
  DEPS llol_base llol_log)
