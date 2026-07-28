#pragma once

#include <limits>

#include <Eigen/Core>

#include <ccbf/qp_builder.hpp>

namespace ccbf {

enum class QpStatus {
  SOLVED,
  SOLVED_CLOSEST_PRIMAL_FEASIBLE,
  MAX_ITERATIONS,
  PRIMAL_INFEASIBLE,
  DUAL_INFEASIBLE,
  NOT_RUN,
  UNKNOWN,
};

const char* toString(QpStatus status) noexcept;

struct QpSolveResult {
  bool success{false};
  Eigen::VectorXd solution;
  QpStatus status{QpStatus::NOT_RUN};
  int iterations{0};
  double primal_residual{std::numeric_limits<double>::infinity()};
  double dual_residual{std::numeric_limits<double>::infinity()};
};

class ProxqpSolver {
 public:
  QpSolveResult solve(const QpProblem& problem) const;
};

}  // namespace ccbf
