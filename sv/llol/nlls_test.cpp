#include "sv/llol/nlls.h"

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

namespace sv {
namespace {

template <typename T>
bool EvalResAndJac(const T* parameters, T* residuals, T* jacobian) {
  T x = parameters[0];
  T y = parameters[1];
  T z = parameters[2];

  residuals[0] = x + static_cast<T>(2) * y + static_cast<T>(4) * z;
  residuals[1] = y * z;

  if (jacobian) {
    jacobian[0 * 2 + 0] = static_cast<T>(1);
    jacobian[0 * 2 + 1] = static_cast<T>(0);

    jacobian[1 * 2 + 0] = static_cast<T>(2);
    jacobian[1 * 2 + 1] = z;

    jacobian[2 * 2 + 0] = static_cast<T>(4);
    jacobian[2 * 2 + 1] = y;
  }
  return true;
}

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::VectorXd VecX;

class ExampleCost final : public CostBase {
 public:
  int NumResiduals() const override { return 2; }
  int NumParameters() const override { return 3; }

  bool Compute(const double* parameters,
               double* residuals,
               double* jacobian) const override {
    return EvalResAndJac(parameters, residuals, jacobian);
  }
};

void TestSolver(const CostBase& f, double* x) {
  Vec2 residuals;
  f.Compute(x, residuals.data(), nullptr);
  EXPECT_GT(residuals.squaredNorm() / 2.0, 1e-10);

  NllsSolver solver;
  solver.Solve(f, x);
  EXPECT_NEAR(0.0, solver.summary.final_cost, 1e-10);
}

// A test case for when the number of parameters and residuals is
// dynamically sized.
TEST(NllsSolverTest, TestParametersAndResidualsDynamic) {
  VecX x0(3);
  x0 << 0.76026643, -30.01799744, 0.55192142;
  ExampleCost f;
  TestSolver(f, x0.data());
}

}  // namespace
}  // namespace sv
