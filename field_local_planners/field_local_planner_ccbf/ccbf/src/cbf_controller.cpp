#include <ccbf/cbf_controller.hpp>

#include <stdexcept>
#include <utility>

namespace ccbf {

CbfController::CbfController(std::shared_ptr<const Dynamics> dynamics)
    : dynamics_(std::move(dynamics)) {
  if (!dynamics_) {
    throw std::invalid_argument("CbfController dynamics must not be null");
  }
}

ControlResult CbfController::computeControl(
    const Eigen::Ref<const Eigen::VectorXd>& state,
    const Eigen::Ref<const Eigen::VectorXd>& nominal_control,
    const std::vector<BarrierPtr>& barriers) const {
  const QpProblem problem =
      QpBuilder::build(*dynamics_, state, nominal_control, barriers);
  const QpSolveResult solve_result = solver_.solve(problem);

  ControlResult result;
  result.success = solve_result.success;
  result.status = solve_result.status;
  result.iterations = solve_result.iterations;
  result.primal_residual = solve_result.primal_residual;
  result.dual_residual = solve_result.dual_residual;
  if (solve_result.success) {
    result.control = solve_result.solution;
  }
  return result;
}

}  // namespace ccbf
