#include "sv/llol/cost.h"
#include "sv/llol/nlls.h"
#include "sv/node/odom_node.h"
#include "sv/node/viz.h"

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
  if (icp_ok) {
    if (traj_.update_bias) {
      traj_.UpdateBias(imuq_);
      ROS_DEBUG_STREAM("gyr_bias: " << imuq_.bias.gyr.transpose());
      ROS_DEBUG_STREAM("acc_bias: " << imuq_.bias.acc.transpose());
    }
  }

  if (vis_) {
    // display good match
    Imshow("match",
           ApplyCmap(grid_.DrawMatch(),
                     1.0 / (gicp_.half_win.area() * 4.0),
                     cv::COLORMAP_VIRIDIS));
  }
}

bool OdomNode::IcpRigid() {
  auto t_match = tm_.Manual("5.Grid.Match", false);
  auto t_solve = tm_.Manual("6.Icp.Solve", false);

  static GicpCost cost(tbb_);
  cost.imu_weight = gicp_.imu_weight;
  cost.UpdatePreint(traj_, imuq_);
  ROS_DEBUG_STREAM("[cost.Preint] num imus: " << cost.preint.n);

  static NllsSolver solver;
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

    if (n_matches < 10) {
      ROS_WARN_STREAM("[grid.Match] Not enough matches: " << n_matches);
      break;
    } else {
      ROS_DEBUG_STREAM("[grid.Match] num matched: " << n_matches);
    }

    // Build
    t_solve.Resume();
    cost.UpdateMatches(grid_);
    solver.Solve(cost, cost.error.data());
    cost.UpdateTraj(traj_);
    // Repropagate full trajectory from the starting point
    const int n_imus = traj_.PredictFull(imuq_);
    t_solve.Stop(false);
    ROS_DEBUG_STREAM("[Traj.PredictFull] using imus: " << n_imus);

    icp_ok = true;
    if (i >= 2 && solver.summary.IsConverged()) {
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

  traj_.cov = solver.GetJtJ();
  ROS_DEBUG_STREAM(solver.summary.Report());
  sm_.GetRef("grid.matches").Add(cost.matches.size());

  return icp_ok;
}

// bool OdomNode::IcpLinear() {
//  auto t_match = tm_.Manual("5.Grid.Match", false);
//  auto t_solve = tm_.Manual("6.Icp.Solve", false);

//  using Cost = GicpLinearCost;
//  static Cost cost(tbb_);
//  cost.imu_weight = gicp_.imu_weight;
//  const int n_preint = cost.UpdatePreint(traj_, imuq_);
//  ROS_DEBUG_STREAM("[cost.Preint] num imus: " << n_preint);

//  static TinySolver2<Cost> solver;
//  auto& opts = solver.options;
//  opts.max_num_iterations = gicp_.iters.second;
//  opts.gradient_tolerance = 1e-8;
//  opts.min_eigenvalue = gicp_.min_eigval;

//  bool icp_ok = false;

//  for (int i = 0; i < gicp_.iters.first; ++i) {
//    cost.ResetError();

//    t_match.Resume();
//    // Need to update cell tfs before match
//    grid_.Interp(traj_);
//    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
//    t_match.Stop(false);

//    if (n_matches < 10) {
//      ROS_WARN_STREAM("[grid.Match] Not enough matches: " << n_matches);
//      break;
//    } else {
//      ROS_DEBUG_STREAM("[grid.Match] num matched: " << n_matches);
//    }

//    t_solve.Resume();
//    cost.UpdateMatches(grid_);
//    solver.Solve(cost, &cost.error);
//    cost.UpdateTraj(traj_);
//    t_solve.Stop(false);

//    icp_ok = true;

//    if (solver.summary.initial_cost == solver.summary.final_cost) {
//      ROS_WARN_STREAM("[Icp] final cost did not improve, iter: " << i + 1);
//    }

//    // early exit
//    if (i >= 2 && solver.summary.IsConverged()) {
//      ROS_DEBUG_STREAM(
//          fmt::format("[Icp] converged at outer: {}/{}, inner: {}/{}",
//                      i + 1,
//                      gicp_.iters.first,
//                      solver.summary.iterations,
//                      gicp_.iters.second));
//      break;
//    }
//  }

//  t_match.Commit();
//  t_solve.Commit();

//  ROS_DEBUG_STREAM(solver.summary.Report());
//  sm_.GetRef("grid.matches").Add(cost.matches.size());

//  return icp_ok;
//}

}  // namespace sv
