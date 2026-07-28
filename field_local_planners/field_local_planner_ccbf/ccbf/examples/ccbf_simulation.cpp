#include <ccbf/barrier.hpp>
#include <ccbf/cbf_controller.hpp>
#include <ccbf/dynamics.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Obstacle {
  Eigen::Vector2d center;
  double radius;
};

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

Eigen::Vector3d pathlineControl(
    const Eigen::Vector3d& state,
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& goal) {
  constexpr double kMaximumForwardSpeed = 0.8;
  constexpr double kMaximumLateralSpeed = 0.6;
  constexpr double kGoalGain = 1.0;
  constexpr double kPathGain = 1.2;
  constexpr double kYawGain = 2.0;

  const Eigen::Vector2d path = goal - start;
  const Eigen::Vector2d tangent = path.normalized();
  const Eigen::Vector2d normal(-tangent.y(), tangent.x());
  const Eigen::Vector2d position = state.head<2>();

  const double remaining_distance = tangent.dot(goal - position);
  const double cross_track_error = normal.dot(position - start);
  const double forward_speed = clamp(
      kGoalGain * remaining_distance, 0.0, kMaximumForwardSpeed);
  const double lateral_speed = clamp(
      -kPathGain * cross_track_error,
      -kMaximumLateralSpeed,
      kMaximumLateralSpeed);
  const Eigen::Vector2d world_velocity =
      forward_speed * tangent + lateral_speed * normal;

  const double cosine = std::cos(state(2));
  const double sine = std::sin(state(2));

  Eigen::Vector3d body_control;
  body_control << cosine * world_velocity.x() + sine * world_velocity.y(),
                  -sine * world_velocity.x() + cosine * world_velocity.y(),
                  -kYawGain * state(2);
  return body_control;
}

void writeHeader(std::ofstream& output, std::size_t obstacle_count) {
  output << "time,x,y,yaw,"
         << "nominal_vx,nominal_vy,nominal_yaw_rate,"
         << "safe_vx,safe_vy,safe_yaw_rate,"
         << "min_barrier,goal_x,goal_y";
  for (std::size_t i = 0; i < obstacle_count; ++i) {
    output << ",obstacle_" << i + 1 << "_x"
           << ",obstacle_" << i + 1 << "_y"
           << ",obstacle_" << i + 1 << "_radius";
  }
  output << '\n';
}

void writeRow(
    std::ofstream& output,
    double time,
    const Eigen::Vector3d& state,
    const Eigen::Vector3d& nominal_control,
    const Eigen::Vector3d& safe_control,
    double minimum_barrier,
    const Eigen::Vector2d& goal,
    const std::vector<Obstacle>& obstacles) {
  output << time << ','
         << state(0) << ',' << state(1) << ',' << state(2) << ','
         << nominal_control(0) << ',' << nominal_control(1) << ','
         << nominal_control(2) << ','
         << safe_control(0) << ',' << safe_control(1) << ','
         << safe_control(2) << ','
         << minimum_barrier << ',' << goal.x() << ',' << goal.y();
  for (const Obstacle& obstacle : obstacles) {
    output << ',' << obstacle.center.x()
           << ',' << obstacle.center.y()
           << ',' << obstacle.radius;
  }
  output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string output_path =
        argc > 1 ? argv[1] : "ccbf_simulation.csv";
    std::ofstream output(output_path);
    if (!output) {
      std::cerr << "Unable to open simulation output: " << output_path << '\n';
      return 1;
    }
    output << std::setprecision(12);

    const Eigen::Vector2d start(0.0, 0.0);
    const Eigen::Vector2d goal(10.0, 0.0);
    const std::vector<Obstacle> obstacles{
        {{2.0, 0.30}, 0.55},
        {{3.6, -0.32}, 0.55},
        {{5.2, 0.28}, 0.55},
        {{6.8, -0.30}, 0.55},
        {{8.4, 0.25}, 0.55},
    };

    auto dynamics = std::make_shared<ccbf::Omnidirectional>();
    std::vector<ccbf::BarrierPtr> barriers;
    barriers.reserve(obstacles.size());
    for (const Obstacle& obstacle : obstacles) {
      barriers.push_back(
          std::make_shared<ccbf::SphericalObstacleBarrier>(
              obstacle.center, obstacle.radius, 2.0));
    }

    // With composite barriers
    auto composite = std::make_shared<ccbf::CompositeBarrier>(
    obstacles.size(), 100);

    composite->getBarriers(barriers);

    std::vector<ccbf::BarrierPtr> composite_barriers;
    composite_barriers.reserve(1);
    composite_barriers.push_back(composite);

    ccbf::CbfController controller(dynamics);

    constexpr double kTimeStep = 0.02;
    constexpr double kMaximumTime = 40.0;
    constexpr double kGoalTolerance = 0.08;
    constexpr double kSafetyTolerance = 1e-7;
    const int maximum_steps =
        static_cast<int>(std::ceil(kMaximumTime / kTimeStep));

    Eigen::Vector3d state(start.x(), start.y(), 0.0);
    Eigen::VectorXd drift(3);
    Eigen::MatrixXd input_matrix(3, 3);
    double global_minimum_barrier = std::numeric_limits<double>::infinity();
    bool reached_goal = false;

    writeHeader(output, obstacles.size());

    for (int step = 0; step < maximum_steps; ++step) {
      const double distance_to_goal = (state.head<2>() - goal).norm();
      if (distance_to_goal <= kGoalTolerance) {
        reached_goal = true;
        break;
      }

      const Eigen::Vector3d nominal_control =
          pathlineControl(state, start, goal);
      const ccbf::ControlResult result =
          controller.computeControl(state, nominal_control, composite_barriers);
      if (!result.success) {
        std::cerr << "Safety QP failed at step " << step
                  << " with status " << ccbf::toString(result.status) << '\n';
        return 1;
      }

      double minimum_barrier = std::numeric_limits<double>::infinity();
      for (const ccbf::BarrierPtr& barrier : barriers) {
        minimum_barrier = std::min(minimum_barrier, barrier->value(state));
      }
      global_minimum_barrier =
          std::min(global_minimum_barrier, minimum_barrier);
      if (minimum_barrier < -kSafetyTolerance) {
        std::cerr << "Safety boundary violated at step " << step
                  << ": minimum h = " << minimum_barrier << '\n';
        return 1;
      }

      writeRow(
          output,
          step * kTimeStep,
          state,
          nominal_control,
          result.control,
          minimum_barrier,
          goal,
          obstacles);

      dynamics->evaluate(state, drift, input_matrix);
      state += kTimeStep * (drift + input_matrix * result.control);
      state(2) = std::atan2(std::sin(state(2)), std::cos(state(2)));
    }

    output.close();

    const double final_distance = (state.head<2>() - goal).norm();
    std::cout << "Simulation CSV:       " << output_path << '\n'
              << "Final position:       " << state.head<2>().transpose() << '\n'
              << "Distance to goal:     " << final_distance << '\n'
              << "Minimum barrier h:    " << global_minimum_barrier << '\n'
              << "Goal reached safely:  " << (reached_goal ? "yes" : "no")
              << '\n';

    if (!reached_goal) {
      std::cerr << "Robot did not reach the goal within "
                << kMaximumTime << " seconds\n";
      return 1;
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CCBF simulation failed with exception: "
              << error.what() << '\n';
    return 1;
  }
}
