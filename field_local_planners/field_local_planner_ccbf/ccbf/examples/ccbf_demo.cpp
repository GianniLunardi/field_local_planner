#include <ccbf/barrier.hpp>
#include <ccbf/cbf_controller.hpp>
#include <ccbf/dynamics.hpp>
#include <ccbf/proxqp_solver.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

int main() {
  try {
    auto dynamics = std::make_shared<ccbf::Omnidirectional>();
    auto obstacle = std::make_shared<ccbf::SphericalObstacleBarrier>(
        Eigen::Vector2d(1.0, 0.0), 0.5, 2.0);
    const std::vector<ccbf::BarrierPtr> barriers{obstacle};

    Eigen::VectorXd state(3);
    state << 0.0, 0.0, 0.0;
    Eigen::VectorXd nominal_control(3);
    nominal_control << 1.0, 0.0, 0.0;

    ccbf::CbfController controller(dynamics);
    const ccbf::ControlResult result =
        controller.computeControl(state, nominal_control, barriers);

    std::cout << "state:          " << state.transpose() << '\n'
              << "nominal control:" << nominal_control.transpose() << '\n'
              << "barrier value:  " << obstacle->value(state) << '\n'
              << "solver status:  " << ccbf::toString(result.status) << '\n'
              << "iterations:     " << result.iterations << '\n'
              << "primal residual:" << result.primal_residual << '\n'
              << "dual residual:  " << result.dual_residual << '\n';

    if (!result.success) {
      std::cerr << "CCBF demo failed: ProxQP did not solve the safety QP\n";
      return 1;
    }

    Eigen::VectorXd drift(3);
    Eigen::MatrixXd input_matrix(3, 3);
    dynamics->evaluate(state, drift, input_matrix);
    Eigen::VectorXd gradient(3);
    obstacle->gradient(state, gradient);
    const double cbf_residual =
        gradient.dot(drift + input_matrix * result.control) +
        obstacle->alpha(obstacle->value(state));

    std::cout << "safe control:   " << result.control.transpose() << '\n'
              << "CBF residual:   " << cbf_residual << '\n';

    constexpr double kTolerance = 1e-5;
    if (std::abs(result.control(0) - 0.75) > kTolerance ||
        std::abs(result.control(1)) > kTolerance ||
        std::abs(result.control(2)) > kTolerance ||
        cbf_residual < -kTolerance) {
      std::cerr << "CCBF demo failed: unexpected filtered control or CBF residual\n";
      return 1;
    }

    std::cout << "CCBF demo passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CCBF demo failed with exception: " << error.what() << '\n';
    return 1;
  }
}
