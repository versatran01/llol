#include "sv/llol/cost.h"
#include "sv/node/odom.h"
#include "sv/node/viz.h"
#include "sv/util/solver2.h"

namespace sv {

void OdomNode::Register() {
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  using Cost = GicpRigidCost;

  Eigen::Matrix<double, Cost::kNumParams, 1> err_sum;
  err_sum.setZero();
  Eigen::Matrix<double, Cost::kNumParams, 1> err;
  //  TinySolver<AdCost<Cost>> solver;
  TinySolver2<Cost> solver;
  solver.options.max_num_iterations = gicp_.iters.second;
  solver.options.min_eigenvalue = 0.05;

  traj_updated_ = false;

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
    Cost cost(grid_, tbb_);
    //    AdCost<Cost> adcost(cost);
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
    for (auto& st : traj_.states) {
      st.rot = eR * st.rot;
      st.pos = eR * st.pos + es.p0();
    }

    traj_updated_ = true;

    if (vis_) {
      // display good match
      Imshow("match",
             ApplyCmap(grid_.DrawMatch(),
                       1.0 / gicp_.pano_win.area(),
                       cv::COLORMAP_VIRIDIS));
    }
  }

  if (traj_updated_) {
    const Cost::State<double> ess(err_sum.data());
    ROS_WARN_STREAM(fmt::format(
        "err_rot: [{}], norm={:6e}", ess.r0().transpose(), ess.r0().norm()));

    //    imuq_.UpdateBias(traj_.states);

  } else {
    ROS_WARN_STREAM("Trajectory not updated");
  }

  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
}

void OdomNode::Register2() {
  // Outer icp iters
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  using Cost = GicpLinearCost;

  Eigen::Matrix<double, Cost::kNumParams, 1> err_sum;
  err_sum.setZero();

  Eigen::Matrix<double, Cost::kNumParams, 1> err;
  TinySolver<AdCost<Cost>> solver;
  solver.options.max_num_iterations = gicp_.iters.second;

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
    Cost cost(grid_, traj_, imuq_, tbb_);
    AdCost<Cost> adcost(cost);
    t_build.Stop(false);
    ROS_INFO_STREAM("Num residuals: " << cost.NumResiduals());

    // Solve
    t_solve.Resume();
    solver.Solve(adcost, &err);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // Update state
    // accumulate e
    err_sum += err;

    const Cost::State<double> es(err.data());
    const auto eR = Sophus::SO3d::exp(es.r0());
    const auto dep = (es.p1() - es.p0()).eval();
    for (int i = 0; i < traj_.size(); ++i) {
      auto& st1 = traj_.At(i);
      const double s = i / (traj_.size() - 1.0);
      const auto ep = es.p0() + s * dep;
      st1.rot = eR * st1.rot;
      st1.pos = eR * st1.pos + ep;
      if (i > 1) {
        const auto& st0 = traj_.At(i - 1);
        st1.vel =
            st1.rot.inverse() * (st1.pos - st0.pos) / (st1.time - st0.time);
      }
    }

    if (vis_) {
      // display good match
      Imshow("match",
             ApplyCmap(grid_.DrawMatch(),
                       1.0 / gicp_.pano_win.area(),
                       cv::COLORMAP_VIRIDIS));
    }
  }

  t_match.Commit();
  t_solve.Commit();
  t_build.Commit();
}

}  // namespace sv
