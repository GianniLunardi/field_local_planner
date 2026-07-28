#include <ccbf/qp_builder.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ccbf {

QpProblem QpBuilder::build(
    const Dynamics& dynamics,
    const Eigen::Ref<const Eigen::VectorXd>& state,
    const Eigen::Ref<const Eigen::VectorXd>& nominal_control,
    const std::vector<BarrierPtr>& barriers) {
  const int state_dimension = dynamics.stateDimension();
  const int input_dimension = dynamics.inputDimension();

  if (state_dimension <= 0 || input_dimension <= 0) {
    throw std::invalid_argument("Dynamics dimensions must be positive");
  }
  if (state.size() != state_dimension) {
    throw std::invalid_argument("State dimension does not match the dynamics");
  }
  if (nominal_control.size() != input_dimension) {
    throw std::invalid_argument("Nominal control dimension does not match the dynamics");
  }
  if (!state.allFinite() || !nominal_control.allFinite()) {
    throw std::invalid_argument("State and nominal control must be finite");
  }

  Eigen::VectorXd drift(state_dimension);
  Eigen::MatrixXd input_matrix(state_dimension, input_dimension);
  dynamics.evaluate(state, drift, input_matrix);
  if (!drift.allFinite() || !input_matrix.allFinite()) {
    throw std::invalid_argument("Dynamics evaluation must be finite");
  }

  QpProblem problem;
  problem.H = Eigen::MatrixXd::Identity(input_dimension, input_dimension);
  problem.q = -nominal_control;
  problem.C.resize(static_cast<Eigen::Index>(barriers.size()), input_dimension);
  problem.lower.resize(static_cast<Eigen::Index>(barriers.size()));
  problem.upper = Eigen::VectorXd::Constant(
      static_cast<Eigen::Index>(barriers.size()),
      std::numeric_limits<double>::infinity());

  Eigen::VectorXd gradient(state_dimension);
  for (std::size_t i = 0; i < barriers.size(); ++i) {
    const BarrierPtr& barrier = barriers[i];
    if (!barrier) {
      throw std::invalid_argument("Barrier pointers must not be null");
    }
    if (barrier->stateDimension() != state_dimension) {
      throw std::invalid_argument("Barrier state dimension does not match the dynamics");
    }

    const double h = barrier->value(state);
    gradient.setZero();
    barrier->gradient(state, gradient);
    const double alpha = barrier->alpha(h);
    if (!std::isfinite(h) || !std::isfinite(alpha) || !gradient.allFinite()) {
      throw std::invalid_argument("Barrier evaluation must be finite");
    }

    const Eigen::Index row = static_cast<Eigen::Index>(i);
    problem.C.row(row).noalias() = gradient.transpose() * input_matrix;
    problem.lower(row) = -alpha - gradient.dot(drift);
  }

  return problem;
}

}  // namespace ccbf
