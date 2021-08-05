cc_library(
  NAME log
  DEPS fmt::fmt glog::glog
  INTERFACE)

cc_library(
  NAME util_manager
  SRCS "manager.cpp"
  DEPS sv_base sv_log absl::time absl::flat_hash_map)
