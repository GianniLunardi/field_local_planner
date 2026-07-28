#include <ccbf/proxqp_solver.hpp>

#include <cmath>
#include <stdexcept>

#include <proxsuite/proxqp/dense/dense.hpp>

namespace ccbf {
namespace {

QpStatus normalizeStatus(proxsuite::proxqp::QPSolverOutput status) noexcept {
  using ProxqpStatus = proxsuite::proxqp::QPSolverOutput;
  switch (status) {
    case ProxqpStatus::PROXQP_SOLVED:
      return QpStatus::SOLVED;
    case ProxqpStatus::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE:
      return QpStatus::SOLVED_CLOSEST_PRIMAL_FEASIBLE;
    case ProxqpStatus::PROXQP_MAX_ITER_REACHED:
      return QpStatus::MAX_ITERATIONS;
    case ProxqpStatus::PROXQP_PRIMAL_INFEASIBLE:
      return QpStatus::PRIMAL_INFEASIBLE;
    case ProxqpStatus::PROXQP_DUAL_INFEASIBLE:
      return QpStatus::DUAL_INFEASIBLE;
    case ProxqpStatus::PROXQP_NOT_RUN:
      return QpStatus::NOT_RUN;
  }
  return QpStatus::UNKNOWN;
}

void validateProblem(const QpProblem& problem) {
  const Eigen::Index dimension = problem.H.rows();
  if (dimension <= 0 || problem.H.cols() != dimension) {
    throw std::invalid_argument("QP Hessian must be a non-empty square matrix");
  }
  if (problem.q.size() != dimension) {
    throw std::invalid_argument("QP linear cost has the wrong dimension");
  }
  if (problem.C.cols() != dimension) {
    throw std::invalid_argument("QP inequality matrix has the wrong column count");
  }
  if (problem.lower.size() != problem.C.rows() ||
      problem.upper.size() != problem.C.rows()) {
    throw std::invalid_argument("QP inequality bounds have the wrong dimension");
  }
  if (!problem.H.allFinite() || !problem.q.allFinite() ||
      !problem.C.allFinite()) {
    throw std::invalid_argument("QP cost and constraint matrices must be finite");
  }
  if (!problem.H.isApprox(problem.H.transpose(), 1e-12)) {
    throw std::invalid_argument("QP Hessian must be symmetric");
  }
  for (Eigen::Index i = 0; i < problem.lower.size(); ++i) {
    if (std::isnan(problem.lower(i)) || std::isnan(problem.upper(i)) ||
        problem.lower(i) > problem.upper(i)) {
      throw std::invalid_argument("QP inequality bounds are invalid");
    }
  }
}

}  // namespace

const char* toString(QpStatus status) noexcept {
  switch (status) {
    case QpStatus::SOLVED:
      return "solved";
    case QpStatus::SOLVED_CLOSEST_PRIMAL_FEASIBLE:
      return "solved_closest_primal_feasible";
    case QpStatus::MAX_ITERATIONS:
      return "max_iterations";
    case QpStatus::PRIMAL_INFEASIBLE:
      return "primal_infeasible";
    case QpStatus::DUAL_INFEASIBLE:
      return "dual_infeasible";
    case QpStatus::NOT_RUN:
      return "not_run";
    case QpStatus::UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

QpSolveResult ProxqpSolver::solve(const QpProblem& problem) const {
  validateProblem(problem);

  QpSolveResult result;
  try {
    const proxsuite::proxqp::isize dimension =
        static_cast<proxsuite::proxqp::isize>(problem.H.rows());
    const proxsuite::proxqp::isize inequality_count =
        static_cast<proxsuite::proxqp::isize>(problem.C.rows());

    Eigen::MatrixXd equality_matrix(0, problem.H.rows());
    Eigen::VectorXd equality_vector(0);

    proxsuite::proxqp::dense::QP<double> qp(
        dimension, 0, inequality_count);
    qp.settings.eps_abs = 1e-9;
    qp.settings.eps_rel = 0.0;
    qp.init(
        problem.H,
        problem.q,
        equality_matrix,
        equality_vector,
        problem.C,
        problem.lower,
        problem.upper);
    qp.solve();

    result.status = normalizeStatus(qp.results.info.status);
    result.success = result.status == QpStatus::SOLVED;
    result.iterations = static_cast<int>(qp.results.info.iter);
    result.primal_residual = qp.results.info.pri_res;
    result.dual_residual = qp.results.info.dua_res;
    if (result.success) {
      result.solution = qp.results.x;
    }
  } catch (const std::exception&) {
    result.status = QpStatus::UNKNOWN;
    result.success = false;
    result.solution.resize(0);
  }

  return result;
}

}  // namespace ccbf
