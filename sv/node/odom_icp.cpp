#include "sv/llol/cost.h"
#include "sv/node/odom_node.h"
#include "sv/node/viz.h"
#include "sv/util/solver.h"

namespace sv {

void OdomNode::Register() {
  if (rigid_) {
    IcpRigid();
  } else {
    IcpLinear();
  }

  ROS_WARN_STREAM("velocity: " << traj_.back().vel.transpose()
                               << ", norm: " << traj_.back().vel.norm());
  traj_.UpdateBias(imuq_);
  ROS_WARN_STREAM("gyr_bias: " << imuq_.bias.gyr.transpose());
  ROS_WARN_STREAM("acc_bias: " << imuq_.bias.acc.transpose());

  if (vis_) {
    // display good match
    Imshow("match",
           ApplyCmap(grid_.DrawMatch(),
                     1.0 / gicp_.pano_win.area(),
                     cv::COLORMAP_VIRIDIS));
  }
}

void OdomNode::IcpRigid() {
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  using Cost = GicpRigidCost;
  static Cost cost(tbb_);

  static TinySolver2<Cost> solver;
  auto& opts = solver.options;
  opts.max_num_iterations = gicp_.iters.second;
  opts.gradient_tolerance = 1e-8;

  Cost::ErrorVector err;

  for (int i = 0; i < gicp_.iters.first; ++i) {
    err.setZero();

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

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
    solver.Solve(cost, &err);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // Update state
    cost.UpdateTraj(traj_, err.data());
  }

  sm_.Get("grid.matches").Add(cost.matches.size());

  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
}

void OdomNode::IcpLinear() {
  // Outer icp iters
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

  Cost::ErrorVector err;
  for (int i = 0; i < gicp_.iters.first; ++i) {
    err.setZero();

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

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
    solver.Solve(cost, &err);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // Update state
    cost.UpdateTraj(traj_, err.data());
  }

  sm_.Get("grid.matches").Add(cost.matches.size());
  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
}

}  // namespace sv
