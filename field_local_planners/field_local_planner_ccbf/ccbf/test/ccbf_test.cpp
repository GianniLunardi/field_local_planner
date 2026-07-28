#include <ccbf/barrier.hpp>
#include <ccbf/cbf_controller.hpp>
#include <ccbf/dynamics.hpp>
#include <ccbf/qp_builder.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::shared_ptr<ccbf::Omnidirectional> makeDynamics() {
  return std::make_shared<ccbf::Omnidirectional>();
}

ccbf::BarrierPtr makeObstacle(
    double x = 1.0,
    double y = 0.0,
    double radius = 0.5,
    double alpha_gain = 2.0) {
  return std::make_shared<ccbf::SphericalObstacleBarrier>(
      Eigen::Vector2d(x, y), radius, alpha_gain);
}

Eigen::Vector3d zeroState() {
  return Eigen::Vector3d::Zero();
}

}  // namespace

TEST(DynamicsTest, RotatesBodyVelocityIntoWorldFrame) {
  ccbf::Omnidirectional dynamics;
  Eigen::Vector3d state(0.0, 0.0, std::acos(-1.0) / 2.0);
  Eigen::VectorXd drift(3);
  Eigen::MatrixXd input_matrix(3, 3);

  dynamics.evaluate(state, drift, input_matrix);

  EXPECT_TRUE(drift.isZero(1e-12));
  Eigen::Matrix3d expected;
  expected << 0.0, -1.0, 0.0,
              1.0,  0.0, 0.0,
              0.0,  0.0, 1.0;
  EXPECT_TRUE(input_matrix.isApprox(expected, 1e-12));
}

TEST(BarrierTest, ComputesValueGradientAndClassKFunction) {
  ccbf::SphericalObstacleBarrier barrier(
      Eigen::Vector2d(1.0, 0.0), 0.5, 2.0);
  const Eigen::Vector3d state = zeroState();
  Eigen::VectorXd gradient(3);

  barrier.gradient(state, gradient);

  EXPECT_DOUBLE_EQ(barrier.value(state), 0.75);
  EXPECT_TRUE(gradient.isApprox(Eigen::Vector3d(-2.0, 0.0, 0.0)));
  EXPECT_DOUBLE_EQ(barrier.alpha(barrier.value(state)), 1.5);
}

TEST(QpBuilderTest, BuildsExpectedSafetyConstraint) {
  ccbf::Omnidirectional dynamics;
  const Eigen::Vector3d state = zeroState();
  const Eigen::Vector3d nominal_control(1.0, 0.0, 0.0);
  const std::vector<ccbf::BarrierPtr> barriers{makeObstacle()};

  const ccbf::QpProblem problem =
      ccbf::QpBuilder::build(dynamics, state, nominal_control, barriers);

  EXPECT_TRUE(problem.H.isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(problem.q.isApprox(-nominal_control));
  ASSERT_EQ(problem.C.rows(), 1);
  EXPECT_TRUE(problem.C.row(0).isApprox(Eigen::RowVector3d(-2.0, 0.0, 0.0)));
  EXPECT_DOUBLE_EQ(problem.lower(0), -1.5);
  EXPECT_TRUE(std::isinf(problem.upper(0)));
}

TEST(QpBuilderTest, AddsOneConstraintPerObstacle) {
  ccbf::Omnidirectional dynamics;
  const Eigen::Vector3d state = zeroState();
  const Eigen::Vector3d nominal_control = Eigen::Vector3d::Zero();
  const std::vector<ccbf::BarrierPtr> barriers{
      makeObstacle(1.0, 0.0),
      makeObstacle(0.0, 1.0),
  };

  const ccbf::QpProblem problem =
      ccbf::QpBuilder::build(dynamics, state, nominal_control, barriers);

  ASSERT_EQ(problem.C.rows(), 2);
  EXPECT_TRUE(problem.C.row(0).isApprox(Eigen::RowVector3d(-2.0, 0.0, 0.0)));
  EXPECT_TRUE(problem.C.row(1).isApprox(Eigen::RowVector3d(0.0, -2.0, 0.0)));
}

TEST(ControllerTest, PreservesNominalControlWithoutObstacles) {
  ccbf::CbfController controller(makeDynamics());
  const Eigen::Vector3d state = zeroState();
  const Eigen::Vector3d nominal_control(0.4, -0.2, 0.3);

  const ccbf::ControlResult result =
      controller.computeControl(state, nominal_control, {});

  ASSERT_TRUE(result.success);
  EXPECT_TRUE(result.control.isApprox(nominal_control, 1e-7));
}

TEST(ControllerTest, LimitsControlApproachingObstacle) {
  ccbf::CbfController controller(makeDynamics());
  const Eigen::Vector3d state = zeroState();
  const Eigen::Vector3d nominal_control(1.0, 0.0, 0.2);

  const ccbf::ControlResult result =
      controller.computeControl(state, nominal_control, {makeObstacle()});

  ASSERT_TRUE(result.success);
  EXPECT_NEAR(result.control(0), 0.75, 1e-6);
  EXPECT_NEAR(result.control(1), 0.0, 1e-7);
  EXPECT_NEAR(result.control(2), 0.2, 1e-7);
}

TEST(ControllerTest, PreservesControlMovingAwayFromObstacle) {
  ccbf::CbfController controller(makeDynamics());
  const Eigen::Vector3d state = zeroState();
  const Eigen::Vector3d nominal_control(-1.0, 0.25, -0.1);

  const ccbf::ControlResult result =
      controller.computeControl(state, nominal_control, {makeObstacle()});

  ASSERT_TRUE(result.success);
  EXPECT_TRUE(result.control.isApprox(nominal_control, 1e-7));
}

TEST(ControllerTest, DoesNotReturnNominalControlWhenProblemIsInfeasible) {
  ccbf::CbfController controller(makeDynamics());
  const Eigen::Vector3d state(1.0, 0.0, 0.0);
  const Eigen::Vector3d nominal_control(0.5, 0.0, 0.0);

  const ccbf::ControlResult result =
      controller.computeControl(state, nominal_control, {makeObstacle()});

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.control.size(), 0);
  EXPECT_NE(result.status, ccbf::QpStatus::SOLVED);
}

TEST(ValidationTest, RejectsInvalidInputs) {
  ccbf::Omnidirectional dynamics;
  const Eigen::Vector2d short_state = Eigen::Vector2d::Zero();
  const Eigen::Vector3d nominal_control = Eigen::Vector3d::Zero();

  EXPECT_THROW(
      ccbf::QpBuilder::build(dynamics, short_state, nominal_control, {}),
      std::invalid_argument);
  EXPECT_THROW(
      ccbf::SphericalObstacleBarrier(Eigen::Vector2d::Zero(), 0.0),
      std::invalid_argument);
  EXPECT_THROW(
      ccbf::CbfController(std::shared_ptr<const ccbf::Dynamics>()),
      std::invalid_argument);

  Eigen::Vector3d non_finite_control = nominal_control;
  non_finite_control(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      ccbf::QpBuilder::build(dynamics, zeroState(), non_finite_control, {}),
      std::invalid_argument);
}
