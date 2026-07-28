#pragma once

#include <Eigen/Core>

namespace ccbf {

class Dynamics {
 public:
  virtual ~Dynamics() = default;

  virtual int stateDimension() const noexcept = 0;
  virtual int inputDimension() const noexcept = 0;

  virtual void evaluate(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      Eigen::Ref<Eigen::VectorXd> drift,
      Eigen::Ref<Eigen::MatrixXd> input_matrix) const = 0;
};

class Omnidirectional : public Dynamics {
 public:
  int stateDimension() const noexcept override;
  int inputDimension() const noexcept override;

  void evaluate(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      Eigen::Ref<Eigen::VectorXd> drift,
      Eigen::Ref<Eigen::MatrixXd> input_matrix) const override;
};

}  // namespace ccbf
