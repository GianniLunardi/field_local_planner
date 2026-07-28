#include <ccbf/dynamics.hpp>

#include <cmath>
#include <stdexcept>

namespace ccbf {

int Omnidirectional::stateDimension() const noexcept {
  return 3;
}

int Omnidirectional::inputDimension() const noexcept {
  return 3;
}

void Omnidirectional::evaluate(
    const Eigen::Ref<const Eigen::VectorXd>& state,
    Eigen::Ref<Eigen::VectorXd> drift,
    Eigen::Ref<Eigen::MatrixXd> input_matrix) const {
  if (state.size() != stateDimension()) {
    throw std::invalid_argument("Omnidirectional state must have dimension 3");
  }
  if (drift.size() != stateDimension()) {
    throw std::invalid_argument("Omnidirectional drift output must have dimension 3");
  }
  if (input_matrix.rows() != stateDimension() ||
      input_matrix.cols() != inputDimension()) {
    throw std::invalid_argument(
        "Omnidirectional input matrix output must have dimensions 3x3");
  }
  if (!state.allFinite()) {
    throw std::invalid_argument("Omnidirectional state must be finite");
  }

  drift.setZero();

  const double yaw = state(2);
  input_matrix <<
      std::cos(yaw), -std::sin(yaw), 0.0,
      std::sin(yaw),  std::cos(yaw), 0.0,
      0.0,            0.0,           1.0;
}

}  // namespace ccbf
