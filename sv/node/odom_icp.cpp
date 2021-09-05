#include "sv/llol/cost.h"
#include "sv/node/odom_node.h"
#include "sv/node/viz.h"
#include "sv/util/solver.h"

namespace sv {

// TODO (chao): refactor this, too much duplicate!!!
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
  opts.min_eigenvalue = 0.01;

  Eigen::Matrix<double, Cost::kNumParams, 1> err_sum;
  err_sum.setZero();
  Eigen::Matrix<double, Cost::kNumParams, 1> err;
  for (int i = 0; i < gicp_.iters.first; ++i) {
    err.setZero();
    ROS_INFO_STREAM("Icp iteration: " << i);

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

    ROS_INFO_STREAM(fmt::format("Num matches: {} / {} / {:02.2f}% ",
                                n_matches,
                                grid_.total(),
                                100.0 * n_matches / grid_.total()));

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
    // accumulate e
    err_sum += err;

    const Cost::State<double> es(err.data());
    const auto eR = Sophus::SO3d::exp(es.r0());
    for (auto& st1 : traj_.states) {
      st1.rot = eR * st1.rot;
      st1.pos = eR * st1.pos + es.p0();

      if (i > 1) {
        const auto& st0 = traj_.At(i - 1);
        st1.vel = (st1.pos - st0.pos) / (st1.time - st0.time);
      }
    }
  }

  const Cost::State<double> ess(err_sum.data());
  ROS_WARN_STREAM("err_rot: " << ess.r0().transpose()
                              << ", norm: " << ess.r0().norm() << "\n");
  ROS_WARN_STREAM("err_pos: " << ess.p0().transpose()
                              << ", norm: " << ess.p0().norm());
  ROS_WARN_STREAM("velocity: " << traj_.states.back().vel.transpose()
                               << ", norm: " << traj_.states.back().vel.norm());

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
  opts.min_eigenvalue = 0.01;

  Eigen::Matrix<double, Cost::kNumParams, 1> err_sum;
  err_sum.setZero();
  Eigen::Matrix<double, Cost::kNumParams, 1> err;
  for (int i = 0; i < gicp_.iters.first; ++i) {
    err.setZero();
    ROS_INFO_STREAM("Icp iter: " << i);

    t_match.Resume();
    // Need to update cell tfs before match
    grid_.Interp(traj_);
    const auto n_matches = gicp_.Match(grid_, pano_, tbb_);
    t_match.Stop(false);

    ROS_INFO_STREAM(fmt::format("Num matches: {} / {} / {:02.2f}% ",
                                n_matches,
                                grid_.total(),
                                100.0 * n_matches / grid_.total()));

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
    // accumulate e
    err_sum += err;

    const Cost::State<double> es(err.data());
    const auto eR = Sophus::SO3d::exp(es.r0());

    for (int i = 0; i < traj_.size(); ++i) {
      auto& st1 = traj_.At(i);
      const double s = i / (traj_.size() - 1.0);
      st1.rot = eR * st1.rot;
      st1.pos = eR * st1.pos + s * es.p0();
      if (i > 1) {
        const auto& st0 = traj_.At(i - 1);
        st1.vel = (st1.pos - st0.pos) / (st1.time - st0.time);
      }
    }
  }

  const Cost::State<double> ess(err_sum.data());
  ROS_WARN_STREAM("err_rot: " << ess.r0().transpose()
                              << ", norm: " << ess.r0().norm() << "\n");
  ROS_WARN_STREAM("err_pos: " << ess.p0().transpose()
                              << ", norm: " << ess.p0().norm());
  ROS_WARN_STREAM("velocity: " << traj_.states.back().vel.transpose()
                               << ", norm: " << traj_.states.back().vel.norm());

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

  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
}

}  // namespace sv
