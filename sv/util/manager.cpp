#include "sv/util/manager.h"

#include <fmt/color.h>
#include <fmt/ostream.h>
#include <glog/logging.h>

namespace sv {

std::string StatsManager::ReportStats(const std::string& name,
                                      const StatsT& stats) const {
  std::string str = fmt::format(fmt::fg(fmt::color::orange), "[{:<16}]", name);

  str += fmt::format(
      " n: {:<8} | sum: {:<12f} | min: {:<12e} | max: {:<12e} | "
      "mean: {:<12} | last: {:<12f} |",
      stats.count(),
      stats.sum(),
      stats.min(),
      stats.max(),
      stats.mean(),
      stats.last());
  return str;
}

void StatsManager::Counter::Commit() {
  if (!stats_.ok()) return;  // Noop if there's no stats to commit
  manager_->Update(name_, stats_);
  stats_ = StatsT{};  // reset stats
}

StatsManager& GlobalStatsManager() {
  static StatsManager sm{};
  return sm;
}

TimerManager::ManualTimer::ManualTimer(std::string name,
                                       TimerManager* manager,
                                       bool start)
    : name_{std::move(name)}, manager_{manager} {
  CHECK_NOTNULL(manager_);
  if (start) {
    timer_.Start();
  } else {
    timer_.Reset();
  }
}

void TimerManager::ManualTimer::Stop(bool record) {
  timer_.Stop();
  if (record) {
    stats_.Add(absl::Nanoseconds(timer_.Elapsed()));
  }
}

void TimerManager::ManualTimer::Commit() {
  Stop(true);
  if (!stats_.ok()) return;  // Noop if there's no stats to commit

  // Already checked in ctor
  // CHECK_NOTNULL(manager_);
  manager_->Update(name_, stats_);
  stats_ = StatsT{};  // reset stats
}

std::string TimerManager::ReportStats(const std::string& name,
                                      const StatsT& stats) const {
  std::string str =
      fmt::format(fmt::fg(fmt::color::light_sky_blue), "[{:<16}]", name);
  str += fmt::format(
      " n: {:<8} | sum: {:<12} | min: {:<12} | max: {:<12} | mean: {:<12} | "
      "last: {:<12} |",
      stats.count(),
      stats.sum(),
      stats.min(),
      stats.max(),
      stats.mean(),
      stats.last());
  return str;
}

TimerManager& GlobalTimerManager() {
  static TimerManager tm{};
  return tm;
}

}  // namespace sv
