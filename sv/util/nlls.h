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

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace sv {

struct CostBase {
  virtual ~CostBase() noexcept = default;

  bool operator()(const double* x, double* r, double* J) const {
    return Compute(x, r, J);
  }

  virtual bool Compute(const double* x, double* r, double* J) const = 0;
  virtual int NumResiduals() const = 0;
  virtual int NumParameters() const = 0;
};

enum class NllsStatus {
  GRADIENT_TOO_SMALL,            // eps > max(J'*f(x))
  RELATIVE_STEP_SIZE_TOO_SMALL,  // eps > ||dx|| / (||x|| + eps)
  COST_TOO_SMALL,                // eps > ||f(x)||^2 / 2
  HIT_MAX_ITERATIONS
};

std::string Repr(NllsStatus status);

struct NllsOptions {
  double gradient_tolerance = 1e-10;  // eps > max(J'*f(x))
  double parameter_tolerance = 1e-8;  // eps > ||dx|| / ||x||
  double cost_threshold =             // eps > ||f(x)||
      std::numeric_limits<double>::epsilon();
  double initial_trust_region_radius = 1e4;
  int max_num_iterations = 50;
  double min_eigenvalue = 0.0;
};

struct NllsSummary {
  double initial_cost = -1;       // 1/2 ||f(x)||^2
  double final_cost = -1;         // 1/2 ||f(x)||^2
  double gradient_max_norm = -1;  // max(J'f(x))
  int iterations = -1;
  int degenerate_directions = 0;
  NllsStatus status = NllsStatus::HIT_MAX_ITERATIONS;

  std::string Report() const;
  bool IsConverged() const;
};

/// @brief This version allocates once and use Eigen::Map
class NllsSolver {
 public:
  // This class needs to have an Eigen aligned operator new as it contains
  // fixed-size Eigen types.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Scalar = double;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMat =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorMap = Eigen::Map<Vector>;
  using MatrixMap = Eigen::Map<Matrix>;
  using RowMatMap = Eigen::Map<RowMat>;

  bool Update(const CostBase& function, const Scalar* x);
  const NllsSummary& Solve(const CostBase& function, double* x_and_min);
  Matrix GetJtJ() const { return jtj_; }

  NllsOptions options;
  NllsSummary summary;

 private:
  // Preallocate everything, including temporary storage needed for solving the
  // linear system. This allows reusing the intermediate storage across solves.
  using LinearSolver = Eigen::LDLT<Matrix>;
  LinearSolver linear_solver_;
  Scalar cost_;

  VectorMap dx_{nullptr, 0}, x_new_{nullptr, 0};
  VectorMap g_{nullptr, 0}, jacobi_scaling_{nullptr, 0};
  VectorMap lm_diag_{nullptr, 0}, lm_step_{nullptr, 0};

  VectorMap error_{nullptr, 0}, f_x_new_{nullptr, 0};
  RowMatMap jacobian_{nullptr, 0, 0};  // jacobian is row major
  MatrixMap jtj_{nullptr, 0, 0}, jtj_reg_{nullptr, 0, 0};
  MatrixMap Vf_inv_Vu_{nullptr, 0, 0};

  std::vector<Scalar> storage_;

  // Remapping stuff
  using EigenSolver = Eigen::SelfAdjointEigenSolver<Matrix>;
  EigenSolver eigen_solver_;

  void Initialize(int num_residuals, int num_parameters);
};

}  // namespace sv
