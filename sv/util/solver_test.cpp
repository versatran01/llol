
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: mierle@gmail.com (Keir Mierle)

#include "sv/util/solver.h"

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"
#include "sv/util/solver.h"

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

class ExampleStatic {
 public:
  typedef double Scalar;
  enum {
    // Can also be Eigen::Dynamic.
    NUM_RESIDUALS = 2,
    NUM_PARAMETERS = 3,
  };
  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvalResAndJac(parameters, residuals, jacobian);
  }
};

class ExampleParametersDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = 2,
    NUM_PARAMETERS = Eigen::Dynamic,
  };

  int NumParameters() const { return 3; }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvalResAndJac(parameters, residuals, jacobian);
  }
};

class ExampleResidualsDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 3,
  };

  int NumResiduals() const { return 2; }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvalResAndJac(parameters, residuals, jacobian);
  }
};

class ExampleAllDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = Eigen::Dynamic,
  };

  int NumResiduals() const { return 2; }

  int NumParameters() const { return 3; }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvalResAndJac(parameters, residuals, jacobian);
  }
};

template <typename Function, typename Vector>
void TestSolver(const Function& f, const Vector& x0) {
  Vector x = x0;
  Vec2 residuals;
  f(x.data(), residuals.data(), NULL);
  EXPECT_GT(residuals.squaredNorm() / 2.0, 1e-10);

  TinySolver2<Function> solver;
  solver.Solve(f, &x);
  EXPECT_NEAR(0.0, solver.summary.final_cost, 1e-10);
}

// A test case for when the cost function is statically sized.
TEST(TinySolver, SimpleExample) {
  Vec3 x0(0.76026643, -30.01799744, 0.55192142);
  ExampleStatic f;
  TestSolver(f, x0);
}

// A test case for when the number of parameters is dynamically sized.
TEST(TinySolver, ParametersDynamic) {
  VecX x0(3);
  x0 << 0.76026643, -30.01799744, 0.55192142;
  ExampleParametersDynamic f;
  TestSolver(f, x0);
}

// A test case for when the number of residuals is dynamically sized.
TEST(TinySolver, ResidualsDynamic) {
  Vec3 x0(0.76026643, -30.01799744, 0.55192142);
  ExampleResidualsDynamic f;
  TestSolver(f, x0);
}

// A test case for when the number of parameters and residuals is
// dynamically sized.
TEST(TinySolver, ParametersAndResidualsDynamic) {
  VecX x0(3);
  x0 << 0.76026643, -30.01799744, 0.55192142;
  ExampleAllDynamic f;
  TestSolver(f, x0);
}

}  // namespace
}  // namespace sv
