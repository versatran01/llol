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

  Eigen::Matrix<double, Cost::kNumParams, 1> x;
  //  TinySolver<AdCost<Cost>> solver;
  TinySolver<Cost> solver;
  solver.options.max_num_iterations = gicp_.iters.second;

  ImuPreintegration preint;
  {
    auto _ = tm_.Scoped("Icp.Preint");
    preint.Compute(traj_);
  }
  ROS_INFO_STREAM("n: " << preint.n << ", dura: " << preint.duration);

  for (int i = 0; i < gicp_.iters.first; ++i) {
    x.setZero();
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
    ROS_INFO_STREAM("Num residuals: " << cost.NumResiduals());

    // Solve
    t_solve.Resume();
    solver.Solve(cost, &x);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // Update state
    const ErrorState<double> es(x.data());
    const auto dR = Sophus::SO3d::exp(es.r0());
    for (auto& st : traj_.states) {
      st.rot = dR * st.rot;
      st.pos = dR * st.pos + es.p0();
    }

    if (vis_) {
      // display good match
      Imshow("match",
             ApplyCmap(grid_.DrawMatch(),
                       1.0 / gicp_.pano_win.area(),
                       cv::COLORMAP_VIRIDIS));
    }
  }
}

void OdomNode::Register2() {
  // Outer icp iters
  auto t_match = tm_.Manual("Grid.Match", false);
  auto t_build = tm_.Manual("Icp.Build", false);
  auto t_solve = tm_.Manual("Icp.Solve", false);

  using Cost = GicpAndImuCost;

  Eigen::Matrix<double, Cost::kNumParams, 1> x;
  static TinySolver<AdCost<Cost>> solver;
  solver.options.max_num_iterations = gicp_.iters.second;

  for (int i = 0; i < gicp_.iters.first; ++i) {
    x.setZero();
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
    Cost cost(grid_, traj_, tbb_);
    AdCost<Cost> adcost(cost);
    t_build.Stop(false);
    ROS_INFO_STREAM("Num residuals: " << cost.NumResiduals());

    // Solve
    t_solve.Resume();
    solver.Solve(adcost, &x);
    t_solve.Stop(false);
    ROS_INFO_STREAM(solver.summary.Report());

    // Update state
    const ErrorState<double> es(x.data());
    // Update bias
    //    traj_.bias.acc += es.ba();
    //    traj_.bias.gyr += es.bw();
    // Update velocity
    //    traj_.states.front().vel += es.v0();
    //    traj_.states.back().vel += es.v1();

    // Update pose
    //    const auto der = (es.r1() - es.r0()).eval();
    //    const auto eR = Sophus::SO3d::exp(es.r0());
    //    const auto dep = (es.p1() - es.p0()).eval();
    //    for (int i = 0; i < imu_.states.size(); ++i) {
    //      auto& st = imu_.StateAt(i);
    //      const double s = i / (imu_.states.size() - 1.0);
    //      //      const auto eR = Sophus::SO3d::exp(es.r0() + s * der);
    //      const auto ep = es.p0() + s * dep;
    //      st.rot = eR * st.rot;
    //      st.pos = eR * st.pos + ep;
    //    }

    const auto dR = Sophus::SO3d::exp(es.r0());
    for (auto& st : traj_.states) {
      st.rot = dR * st.rot;
      st.pos = dR * st.pos + es.p0();
    }

    if (vis_) {
      // display good match
      Imshow("match",
             ApplyCmap(grid_.DrawMatch(),
                       1.0 / gicp_.pano_win.area(),
                       cv::COLORMAP_VIRIDIS));
    }
  }
}

}  // namespace sv
