#include "llol/util/manager.h"

#include <fmt/color.h>
#include <fmt/ostream.h>
#include <glog/logging.h>

namespace llol {

std::string StatsManager::ReportStats(const std::string& name,
                                      const StatsT& stats) const {
  std::string str = fmt::format(fmt::fg(fmt::color::orange), "[{:<16}]", name);

  str += fmt::format(
      " n: {:<8} | sum: {:<14f} | min: {:<14e} | max: {:<14e} | "
      "mean: {:<14} | last: {:<14f} |",
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

TimerManager::ManualTimer::ManualTimer(std::string name, TimerManager* manager)
    : name_{std::move(name)}, manager_{manager} {
  CHECK_NOTNULL(manager_);
  Start();
}

void TimerManager::ManualTimer::Stop() {
  if (timer_.IsStopped()) return;
  timer_.Stop();
  stats_.Add(absl::Nanoseconds(timer_.Elapsed()));
}

void TimerManager::ManualTimer::Commit() {
  if (timer_.IsRunning()) Stop();
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
      " n: {:<8} | sum: {:<14} | min: {:<14} | max: {:<14} | "
      "mean: {:<14} | last: {:<14} |",
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

}  // namespace llol
