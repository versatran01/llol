/*
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
//
// WARNING WARNING WARNING
// WARNING WARNING WARNING  Tiny solver is experimental and will change.
// WARNING WARNING WARNING
//
// A tiny least squares solver using Levenberg-Marquardt, intended for solving
// small dense problems with low latency and low overhead. The implementation
// takes care to do all allocation up front, so that no memory is allocated
// during solving. This is especially useful when solving many similar problems;
// for example, inverse pixel distortion for every pixel on a grid.
//
// Note: This code has no dependencies beyond Eigen, including on other parts of
// Ceres, so it is possible to take this file alone and put it in another
// project without the rest of Ceres.
//
// Algorithm based off of:
//
// [1] K. Madsen, H. Nielsen, O. Tingleoff.
//     Methods for Non-linear Least Squares Problems.
//     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
*/

#pragma once

#include <glog/logging.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <cmath>

namespace sv {

// To use tiny solver, create a class or struct that allows computing the cost
// function (described below). This is similar to a ceres::CostFunction, but is
// different to enable statically allocating all memory for the solver
// (specifically, enum sizes). Key parts are the Scalar typedef, the enums to
// describe problem sizes (needed to remove all heap allocations), and the
// operator() overload to evaluate the cost and (optionally) jacobians.
//
//   struct TinySolverCostFunctionTraits {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = <int> OR Eigen::Dynamic,
//       NUM_PARAMETERS = <int> OR Eigen::Dynamic,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* jacobian) const;
//
//     int NumResiduals() const;  -- Needed if NUM_RESIDUALS == Eigen::Dynamic.
//     int NumParameters() const; -- Needed if NUM_PARAMETERS == Eigen::Dynamic.
//   };
//
// For operator(), the size of the objects is:
//
//   double* parameters -- NUM_PARAMETERS or NumParameters()
//   double* residuals  -- NUM_RESIDUALS or NumResiduals()
//   double* jacobian   -- NUM_RESIDUALS * NUM_PARAMETERS in column-major format
//                         (Eigen's default); or NULL if no jacobian requested.
//
// An example (fully statically sized):
//
//   struct MyCostFunctionExample {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = 2,
//       NUM_PARAMETERS = 3,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* jacobian) const {
//       residuals[0] = x + 2*y + 4*z;
//       residuals[1] = y * z;
//       if (jacobian) {
//         jacobian[0 * 2 + 0] = 1;   // First column (x).
//         jacobian[0 * 2 + 1] = 0;
//
//         jacobian[1 * 2 + 0] = 2;   // Second column (y).
//         jacobian[1 * 2 + 1] = z;
//
//         jacobian[2 * 2 + 0] = 4;   // Third column (z).
//         jacobian[2 * 2 + 1] = y;
//       }
//       return true;
//     }
//   };
//
// The solver supports either statically or dynamically sized cost
// functions. If the number of residuals is dynamic then the Function
// must define:
//
//   int NumResiduals() const;
//
// If the number of parameters is dynamic then the Function must
// define:
//
//   int NumParameters() const;
//

enum class SolverStatus {
  GRADIENT_TOO_SMALL,            // eps > max(J'*f(x))
  RELATIVE_STEP_SIZE_TOO_SMALL,  // eps > ||dx|| / (||x|| + eps)
  COST_TOO_SMALL,                // eps > ||f(x)||^2 / 2
  HIT_MAX_ITERATIONS,

  // TODO(sameeragarwal): Deal with numerical failures.
};

std::string Repr(SolverStatus status);

struct SolverOptions {
  double gradient_tolerance = 1e-10;  // eps > max(J'*f(x))
  double parameter_tolerance = 1e-8;  // eps > ||dx|| / ||x||
  double cost_threshold =
      std::numeric_limits<double>::epsilon();  // eps > ||f(x)||
  double initial_trust_region_radius = 1e4;
  int max_num_iterations = 50;
  double min_eigenvalue = 0.0;
};

struct SolverSummary {
  double initial_cost = -1;       // 1/2 ||f(x)||^2
  double final_cost = -1;         // 1/2 ||f(x)||^2
  double gradient_max_norm = -1;  // max(J'f(x))
  int iterations = -1;
  int degenerate_directions = 0;
  SolverStatus status = SolverStatus::HIT_MAX_ITERATIONS;

  std::string Report() const;
  bool IsConverged() const;
};

/// @brief This version allocates once and use Eigen::Map
template <typename Function,
          typename LinearSolver =
              Eigen::LDLT<Eigen::Matrix<typename Function::Scalar,
                                        Function::NUM_PARAMETERS,
                                        Function::NUM_PARAMETERS>>>
class TinySolver2 {
 public:
  // This class needs to have an Eigen aligned operator new as it contains
  // fixed-size Eigen types.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum {
    NUM_RESIDUALS = Function::NUM_RESIDUALS,
    NUM_PARAMETERS = Function::NUM_PARAMETERS
  };
  using Scalar = typename Function::Scalar;
  using Parameters = Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>;
  using ParametersMap = Eigen::Map<Parameters>;
  using Residuals = Eigen::Matrix<Scalar, NUM_RESIDUALS, 1>;
  using ResidualsMap = Eigen::Map<Residuals>;
  using Jacobian = Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS>;
  using JacobianMap = Eigen::Map<Jacobian>;
  using Hessian = Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS>;
  using HessianMap = Eigen::Map<Hessian>;

  bool Update(const Function& function, const Parameters& x) {
    if (!function(x.data(), error_.data(), jacobian_.data())) {
      return false;
    }

    error_ = -error_;

    // On the first iteration, compute a diagonal (Jacobi) scaling
    // matrix, which we store as a vector.
    if (summary.iterations == 0) {
      // jacobi_scaling = 1 / (1 + diagonal(J'J))
      //
      // 1 is added to the denominator to regularize small diagonal
      // entries.
      jacobi_scaling_ = 1.0 / (1.0 + jacobian_.colwise().norm().array());
    }

    // This explicitly computes the normal equations, which is numerically
    // unstable. Nevertheless, it is often good enough and is fast.
    //
    // TODO(sameeragarwal): Refactor this to allow for DenseQR
    // factorization.
    jacobian_ = jacobian_ * jacobi_scaling_.asDiagonal();
    jtj_.noalias() = jacobian_.transpose() * jacobian_;
    g_.noalias() = jacobian_.transpose() * error_;
    summary.gradient_max_norm = g_.array().abs().maxCoeff();
    cost_ = error_.squaredNorm() / 2.0;
    return true;
  }

  const SolverSummary& Solve(const Function& function, Parameters* x_and_min) {
    Initialize<NUM_RESIDUALS, NUM_PARAMETERS>(function);
    CHECK_NOTNULL(x_and_min);
    Parameters& x = *x_and_min;
    summary = SolverSummary();
    summary.iterations = 0;

    bool need_remap = false;

    // TODO(sameeragarwal): Deal with failure here.
    Update(function, x);
    summary.initial_cost = cost_;
    summary.final_cost = cost_;

    if (summary.gradient_max_norm < options.gradient_tolerance) {
      summary.status = SolverStatus::GRADIENT_TOO_SMALL;
      return summary;
    }

    if (cost_ < options.cost_threshold) {
      summary.status = SolverStatus::COST_TOO_SMALL;
      return summary;
    }

    // Solution remapping
    // https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7487211
    if (options.min_eigenvalue > 0) {
      // Compute eigen values and eigen vectors of jtj
      eigen_solver_.compute(jtj_);
      const auto& eigvals = eigen_solver_.eigenvalues();

      // Determine a number m of eigvals smaller than a threshold
      const int n = dx_.rows();
      int m = 0;
      for (; m < n; ++m) {
        if (eigvals[m] >= options.min_eigenvalue) break;
      }
      // LOG(INFO) << "eigenvalues: " << eigvals.transpose();
      // Obiviously if all eigenvalues are good then no need for remapping
      need_remap = m > 0;
      summary.degenerate_directions = m;
      if (need_remap) {
        // Construct Vf^-1 * Vu
        const auto& Vf = eigen_solver_.eigenvectors();
        Vf_inv_Vu_.setZero();
        Vf_inv_Vu_.rightCols(n - m) = Vf.rightCols(n - m);
        Vf_inv_Vu_.applyOnTheLeft(Vf.inverse());
      }
    }

    Scalar u = 1.0 / options.initial_trust_region_radius;
    Scalar v = 2;

    for (summary.iterations = 1;
         summary.iterations < options.max_num_iterations;
         summary.iterations++) {
      jtj_regularized_ = jtj_;
      const Scalar min_diagonal = 1e-6;
      const Scalar max_diagonal = 1e32;
      for (int i = 0; i < lm_diagonal_.rows(); ++i) {
        lm_diagonal_[i] = std::sqrt(
            u * std::min(std::max(jtj_(i, i), min_diagonal), max_diagonal));
        jtj_regularized_(i, i) += lm_diagonal_[i] * lm_diagonal_[i];
      }

      // TODO(sameeragarwal): Check for failure and deal with it.
      linear_solver_.compute(jtj_regularized_);
      lm_step_ = linear_solver_.solve(g_);
      dx_.noalias() = jacobi_scaling_.asDiagonal() * lm_step_;

      // Adding parameter_tolerance to x.norm() ensures that this
      // works if x is near zero.
      const Scalar parameter_tolerance =
          options.parameter_tolerance *
          (x.norm() + options.parameter_tolerance);
      if (dx_.norm() < parameter_tolerance) {
        summary.status = SolverStatus::RELATIVE_STEP_SIZE_TOO_SMALL;
        break;
      }

      if (need_remap) {
        // dx = Vf^-1 * Vu * dx
        dx_ = Vf_inv_Vu_ * dx_;
      }
      x_new_ = x + dx_;

      // TODO(keir): Add proper handling of errors from user eval of cost
      // functions.
      function(&x_new_[0], &f_x_new_[0], NULL);

      const Scalar cost_change = (2 * cost_ - f_x_new_.squaredNorm());

      // TODO(sameeragarwal): Better more numerically stable evaluation.
      const Scalar model_cost_change = lm_step_.dot(2 * g_ - jtj_ * lm_step_);

      // rho is the ratio of the actual reduction in error to the reduction
      // in error that would be obtained if the problem was linear. See [1]
      // for details.
      Scalar rho(cost_change / model_cost_change);
      if (rho > 0) {
        // Accept the Levenberg-Marquardt step because the linear
        // model fits well.
        x = x_new_;

        // TODO(sameeragarwal): Deal with failure.
        Update(function, x);
        if (summary.gradient_max_norm < options.gradient_tolerance) {
          summary.status = SolverStatus::GRADIENT_TOO_SMALL;
          break;
        }

        if (cost_ < options.cost_threshold) {
          summary.status = SolverStatus::COST_TOO_SMALL;
          break;
        }

        Scalar tmp = Scalar(2 * rho - 1);
        u = u * std::max(1 / 3., 1 - tmp * tmp * tmp);
        v = 2;
        continue;
      }

      // Reject the update because either the normal equations failed to solve
      // or the local linear model was not good (rho < 0). Instead, increase u
      // to move closer to gradient descent.
      u *= v;
      v *= 2;
    }

    summary.final_cost = cost_;
    return summary;
  }

  SolverOptions options;
  SolverSummary summary;

 private:
  // Preallocate everything, including temporary storage needed for solving the
  // linear system. This allows reusing the intermediate storage across solves.
  LinearSolver linear_solver_;
  Scalar cost_;

  ParametersMap dx_{nullptr, 0}, x_new_{nullptr, 0};
  ParametersMap g_{nullptr, 0}, jacobi_scaling_{nullptr, 0};
  ParametersMap lm_diagonal_{nullptr, 0}, lm_step_{nullptr, 0};

  ResidualsMap error_{nullptr, 0}, f_x_new_{nullptr, 0};
  JacobianMap jacobian_{nullptr, 0, 0};
  HessianMap jtj_{nullptr, 0, 0}, jtj_regularized_{nullptr, 0, 0};
  HessianMap Vf_inv_Vu_{nullptr, 0, 0};

  std::vector<Scalar> storage_;

  // Remapping stuff
  using EigenSolver = Eigen::SelfAdjointEigenSolver<Hessian>;
  EigenSolver eigen_solver_;

  // The following definitions are needed for template metaprogramming.
  template <bool Condition, typename T>
  struct enable_if;

  template <typename T>
  struct enable_if<true, T> {
    typedef T type;
  };

  // The number of parameters and residuals are dynamically sized.
  template <int R, int P>
  typename enable_if<(R == Eigen::Dynamic && P == Eigen::Dynamic), void>::type
  Initialize(const Function& function) {
    Initialize(function.NumResiduals(), function.NumParameters());
  }

  // The number of parameters is dynamically sized and the number of
  // residuals is statically sized.
  template <int R, int P>
  typename enable_if<(R == Eigen::Dynamic && P != Eigen::Dynamic), void>::type
  Initialize(const Function& function) {
    Initialize(function.NumResiduals(), P);
  }

  // The number of parameters is statically sized and the number of
  // residuals is dynamically sized.
  template <int R, int P>
  typename enable_if<(R != Eigen::Dynamic && P == Eigen::Dynamic), void>::type
  Initialize(const Function& function) {
    Initialize(R, function.NumParameters());
  }

  // The number of parameters and residuals are statically sized.
  template <int R, int P>
  typename enable_if<(R != Eigen::Dynamic && P != Eigen::Dynamic), void>::type
  Initialize(const Function&) {
    Initialize(R, P);
  }

  void Initialize(int num_residuals, int num_parameters) {
    const int num_jacobian = num_residuals * num_parameters;
    const int num_hessian = num_parameters * num_parameters;
    const int total = num_parameters * 6 + num_residuals * 2 +
                      num_jacobian * 1 + num_hessian * 3;
    storage_.resize(total);
    auto* s = storage_.data();

    // https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html#TutorialMapPlacementNew
    // dx_.resize(num_parameters);
    // x_new_.resize(num_parameters);
    // g_.resize(num_parameters);
    // jacobi_scaling_.resize(num_parameters);
    // lm_diagonal_.resize(num_parameters);
    // lm_step_.resize(num_parameters);

    new (&dx_) ParametersMap(s, num_parameters);
    s += num_parameters;
    new (&x_new_) ParametersMap(s, num_parameters);
    s += num_parameters;
    new (&g_) ParametersMap(s, num_parameters);
    s += num_parameters;
    new (&jacobi_scaling_) ParametersMap(s, num_parameters);
    s += num_parameters;
    new (&lm_diagonal_) ParametersMap(s, num_parameters);
    s += num_parameters;
    new (&lm_step_) ParametersMap(s, num_parameters);
    s += num_parameters;

    // error_.resize(num_residuals);
    // f_x_new_.resize(num_residuals);
    new (&error_) ResidualsMap(s, num_residuals);
    s += num_residuals;
    new (&f_x_new_) ResidualsMap(s, num_residuals);
    s += num_residuals;

    // jacobian_.resize(num_residuals, num_parameters);
    new (&jacobian_) JacobianMap(s, num_residuals, num_parameters);
    s += num_jacobian;

    // jtj_.resize(num_parameters, num_parameters);
    // jtj_regularized_.resize(num_parameters, num_parameters);
    new (&jtj_) HessianMap(s, num_parameters, num_parameters);
    s += num_hessian;
    new (&jtj_regularized_) HessianMap(s, num_parameters, num_parameters);
    s += num_hessian;
    new (&Vf_inv_Vu_) HessianMap(s, num_parameters, num_parameters);
    s += num_hessian;

    CHECK_EQ(s - storage_.data(), total);
  }
};

}  // namespace sv
