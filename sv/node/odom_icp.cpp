#include "sv/llol/cost.h"
#include "sv/node/odom_node.h"
#include "sv/node/viz.h"
#include "sv/util/solver.h"

namespace sv {

void OdomNode::Register() {
  bool icp_ok = false;

  // 2 is because the first sweep added to pano is junk, so we need to wait for
  // the second sweep to be added
  if (pano_.ready()) {
    icp_ok = IcpRigid();
  } else {
    ROS_WARN_STREAM("Pano is not ready, num sweeps: " << pano_.num_sweeps);
  }

  ROS_DEBUG_STREAM("velocity: " << traj_.back().vel.transpose()
                                << ", norm: " << traj_.back().vel.norm());

  // Do not update bias if icp was not running
  if (icp_ok && traj_.update_bias) {
    traj_.UpdateBias(imuq_);
    ROS_DEBUG_STREAM("gyr_bias: " << imuq_.bias.gyr.transpose());
    ROS_DEBUG_STREAM("acc_bias: " << imuq_.bias.acc.transpose());
  }

  if (vis_) {
    // display good match
    Imshow("match",
           ApplyCmap(grid_.DrawMatch(),
                     1.0 / gicp_.pano_win.area(),
                     cv::COLORMAP_VIRIDIS));
  }
}

bool OdomNode::IcpRigid() {
  auto t_match = tm_.Manual("5.Grid.Match", false);
  auto t_solve = tm_.Manual("6.Icp.Solve", false);

  using Cost = GicpRigidCost;
  static Cost cost(tbb_);
  cost.imu_weight = gicp_.imu_weight;
  const int n_preint = cost.UpdatePreint(traj_, imuq_);
  ROS_DEBUG_STREAM("[cost.Preint] num imus: " << n_preint);

  static TinySolver2<Cost> solver;
  auto& opts = solver.options;
  opts.max_num_iterations = gicp_.iters.second;
  opts.gradient_tolerance = 1e-8;
  opts.min_eigenvalue = gicp_.min_eigval;

  bool icp_ok = false;

  for (int i = 0; i < gicp_.iters.first; ++i) {
    cost.ResetError();

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

    ROS_DEBUG_STREAM("[grid.Match] num matched: " << n_matches);
    if (n_matches < 10) {
      ROS_WARN_STREAM("Not enough matches: " << n_matches);
      break;
    }

    // Build
    t_solve.Resume();
    // Update matches in cost
    cost.UpdateMatches(grid_);
    // Solve
    solver.Solve(cost, &cost.error);
    // Update trajectory
    cost.UpdateTraj(traj_);
    // Repropagate full trajectory from the starting point
    const int n_imus = traj_.PredictFull(imuq_);
    t_solve.Stop(false);
    ROS_DEBUG_STREAM("[Traj.PredictFull] using imus: " << n_imus);

    // early exit
    icp_ok = true;
    if (i >= 1 && solver.summary.IsConverged()) {
      ROS_DEBUG_STREAM(
          fmt::format("[Icp] converged at outer: {}/{}, inner: {}/{}",
                      i + 1,
                      gicp_.iters.first,
                      solver.summary.iterations,
                      gicp_.iters.second));
      break;
    }
  }

  t_match.Commit();
  t_solve.Commit();

  ROS_DEBUG_STREAM(solver.summary.Report());
  sm_.GetRef("grid.matches").Add(cost.matches.size());

  return icp_ok;
}

bool OdomNode::IcpLinear() {
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  using Cost = GicpLinearCost;
  static Cost cost(tbb_);
  cost.imu_weight = gicp_.imu_weight;
  cost.UpdatePreint(traj_, imuq_);

  static TinySolver2<Cost> solver;
  auto& opts = solver.options;
  opts.max_num_iterations = gicp_.iters.second;
  opts.gradient_tolerance = 1e-8;
  opts.min_eigenvalue = gicp_.min_eigval;

  bool icp_ok = false;

  for (int i = 0; i < gicp_.iters.first; ++i) {
    cost.ResetError();

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

    ROS_DEBUG_STREAM("[grid.Match] num matched: " << n_matches);
    if (n_matches < 10) {
      ROS_WARN_STREAM("Not enough matches: " << n_matches);
      break;
    }

    // Build
    t_build.Resume();
    cost.UpdateMatches(grid_);
    t_build.Stop(false);

    // Solve
    t_solve.Resume();
    solver.Solve(cost, &cost.error);
    t_solve.Stop(false);

    // Update state
    cost.UpdateTraj(traj_);
    icp_ok = true;

    // early exit
    if (i >= 1 && solver.summary.IsConverged()) break;
  }

  ROS_DEBUG_STREAM(solver.summary.Report());
  sm_.GetRef("grid.matches").Add(cost.matches.size());

  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
  return icp_ok;
}

}  // namespace sv
