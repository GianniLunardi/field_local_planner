#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <ccbf/barrier.hpp>
#include <ccbf/dynamics.hpp>

namespace ccbf {

using BarrierPtr = std::shared_ptr<const BarrierFunction>;

struct QpProblem {
  Eigen::MatrixXd H;
  Eigen::VectorXd q;
  Eigen::MatrixXd C;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

class QpBuilder {
 public:
  static QpProblem build(
      const Dynamics& dynamics,
      const Eigen::Ref<const Eigen::VectorXd>& state,
      const Eigen::Ref<const Eigen::VectorXd>& nominal_control,
      const std::vector<BarrierPtr>& barriers);
};

}  // namespace ccbf
