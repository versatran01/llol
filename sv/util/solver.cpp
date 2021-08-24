#include "sv/util/solver.h"

#include <fmt/core.h>

namespace sv {

std::string Repr(SolverStatus status) {
  switch (status) {
    case SolverStatus::COST_TOO_SMALL:
      return "COST_TOO_SMALL";
    case SolverStatus::GRADIENT_TOO_SMALL:
      return "GRAD_TOO_SMALL";
    case SolverStatus::RELATIVE_STEP_SIZE_TOO_SMALL:
      return "REL_STEP_SIZE_TOO_SMALL";
    case SolverStatus::HIT_MAX_ITERATIONS:
      return "HIT_MAX_ITERS";
    default:
      return "UNKNOWN";
  }
}

std::string SolverSummary::Report() const {
  return fmt::format(
      "init_cost={:.6e}, final_cost={:.6e}, grad_max_norm={:.6e}, iters={}, "
      "status={}",
      initial_cost,
      final_cost,
      gradient_max_norm,
      iterations,
      Repr(status));
}

}  // namespace sv
