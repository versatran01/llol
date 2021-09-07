#include "sv/llol/nlls.h"

#include <fmt/core.h>
#include <glog/logging.h>

namespace sv {

std::string Repr(NllsStatus status) {
  switch (status) {
    case NllsStatus::COST_TOO_SMALL:
      return "COST_TOO_SMALL";
    case NllsStatus::GRADIENT_TOO_SMALL:
      return "GRAD_TOO_SMALL";
    case NllsStatus::RELATIVE_STEP_SIZE_TOO_SMALL:
      return "REL_STEP_SIZE_TOO_SMALL";
    case NllsStatus::HIT_MAX_ITERATIONS:
      return "HIT_MAX_ITERS";
    default:
      return "UNKNOWN";
  }
}

std::string NllsSummary::Report() const {
  return fmt::format(
      "init_cost={:.6e}, final_cost={:.6e}, grad_max_norm={:.6e}, iters={}, "
      "status={}",
      initial_cost,
      final_cost,
      gradient_max_norm,
      iterations,
      Repr(status));
}

bool NllsSolver::Update(const CostBase& function, const Scalar* x) {
  if (!function.Compute(x, error_.data(), jacobian_.data())) {
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

const NllsSummary& NllsSolver::Solve(const CostBase& function,
                                     double* x_and_min) {
  Initialize(function.NumResiduals(), function.NumParameters());
  CHECK_NOTNULL(x_and_min);
  ParametersMap x(x_and_min, function.NumParameters());
  summary = NllsSummary();
  summary.iterations = 0;

  bool need_remap = false;

  // TODO(sameeragarwal): Deal with failure here.
  Update(function, x.data());
  summary.initial_cost = cost_;
  summary.final_cost = cost_;

  if (summary.gradient_max_norm < options.gradient_tolerance) {
    summary.status = NllsStatus::GRADIENT_TOO_SMALL;
    return summary;
  }

  if (cost_ < options.cost_threshold) {
    summary.status = NllsStatus::COST_TOO_SMALL;
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

  for (summary.iterations = 1; summary.iterations < options.max_num_iterations;
       summary.iterations++) {
    jtj_reg_ = jtj_;
    const Scalar min_diagonal = 1e-6;
    const Scalar max_diagonal = 1e32;
    for (int i = 0; i < lm_diag_.rows(); ++i) {
      lm_diag_[i] = std::sqrt(
          u * std::min(std::max(jtj_(i, i), min_diagonal), max_diagonal));
      jtj_reg_(i, i) += lm_diag_[i] * lm_diag_[i];
    }

    // TODO(sameeragarwal): Check for failure and deal with it.
    linear_solver_.compute(jtj_reg_);
    lm_step_ = linear_solver_.solve(g_);
    dx_.noalias() = jacobi_scaling_.asDiagonal() * lm_step_;

    // Adding parameter_tolerance to x.norm() ensures that this
    // works if x is near zero.
    const Scalar parameter_tolerance =
        options.parameter_tolerance * (x.norm() + options.parameter_tolerance);
    if (dx_.norm() < parameter_tolerance) {
      summary.status = NllsStatus::RELATIVE_STEP_SIZE_TOO_SMALL;
      break;
    }

    if (need_remap) {
      // dx = Vf^-1 * Vu * dx
      dx_ = Vf_inv_Vu_ * dx_;
    }
    x_new_ = x + dx_;

    // TODO(keir): Add proper handling of errors from user eval of cost
    // functions.
    function.Compute(x_new_.data(), f_x_new_.data(), nullptr);

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
      Update(function, x.data());
      if (summary.gradient_max_norm < options.gradient_tolerance) {
        summary.status = NllsStatus::GRADIENT_TOO_SMALL;
        break;
      }

      if (cost_ < options.cost_threshold) {
        summary.status = NllsStatus::COST_TOO_SMALL;
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

void NllsSolver::Initialize(int num_residuals, int num_parameters) {
  const int num_jacobian = num_residuals * num_parameters;
  const int num_hessian = num_parameters * num_parameters;
  const int total = num_parameters * 6  // dx, xnew, g, jscale, lm_diag, lm_step
                    + num_residuals * 2  // error, f_x_new
                    + num_jacobian * 1   // jacobian
                    + num_hessian * 3;   // jtj, jtj_reg, Vf_inv_Vu
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
  new (&lm_diag_) ParametersMap(s, num_parameters);
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
  new (&jtj_reg_) HessianMap(s, num_parameters, num_parameters);
  s += num_hessian;
  new (&Vf_inv_Vu_) HessianMap(s, num_parameters, num_parameters);
  s += num_hessian;

  CHECK_EQ(s - storage_.data(), total);
}

}  // namespace sv
