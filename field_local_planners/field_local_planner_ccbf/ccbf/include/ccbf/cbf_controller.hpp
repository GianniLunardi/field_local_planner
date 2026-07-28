#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <ccbf/dynamics.hpp>
#include <ccbf/proxqp_solver.hpp>
#include <ccbf/qp_builder.hpp>

namespace ccbf {

struct ControlResult {
  bool success{false};
  Eigen::VectorXd control;
  QpStatus status{QpStatus::NOT_RUN};
  int iterations{0};
  double primal_residual{std::numeric_limits<double>::infinity()};
  double dual_residual{std::numeric_limits<double>::infinity()};
};

class CbfController {
 public:
  explicit CbfController(std::shared_ptr<const Dynamics> dynamics);

  ControlResult computeControl(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const Eigen::Ref<const Eigen::VectorXd>& nominal_control,
      const std::vector<BarrierPtr>& barriers) const;

 private:
  std::shared_ptr<const Dynamics> dynamics_;
  ProxqpSolver solver_;
};

}  // namespace ccbf
