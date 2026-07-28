#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>

namespace ccbf {

class BarrierFunction {
 public:
  virtual ~BarrierFunction() = default;

  virtual int stateDimension() const noexcept = 0;

  virtual double value(
      const Eigen::Ref<const Eigen::VectorXd>& state) const = 0;

  virtual void gradient(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      Eigen::Ref<Eigen::VectorXd> gradient) const = 0;

  virtual double alpha(double h) const = 0;
};

using BarrierPtr = std::shared_ptr<const BarrierFunction>;

class SphericalObstacleBarrier : public BarrierFunction {
 public:
  SphericalObstacleBarrier(
      const Eigen::Vector2d& center,
      double effective_radius,
      double alpha_gain = 2.0);

  int stateDimension() const noexcept override;

  double value(
      const Eigen::Ref<const Eigen::VectorXd>& state) const override;

  void gradient(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      Eigen::Ref<Eigen::VectorXd> gradient) const override;

  double alpha(double h) const override;

  const Eigen::Vector2d& center() const noexcept;
  double effectiveRadius() const noexcept;
  double alphaGain() const noexcept;

 private:
  Eigen::Vector2d center_;
  double effective_radius_;
  double alpha_gain_;
};

class CompositeBarrier : public BarrierFunction {
 public:
  CompositeBarrier(
    const int num_obs,
    double kappa
  );     
  
  int stateDimension() const noexcept override;

  double value(
      const Eigen::Ref<const Eigen::VectorXd>& state) const override;

  void gradient(
      const Eigen::Ref<const Eigen::VectorXd>& state,
      Eigen::Ref<Eigen::VectorXd> gradient) const override;

  double alpha(double h) const override;

  const Eigen::Vector2d& center() const noexcept;
  double effectiveRadius() const noexcept;
  double alphaGain() const noexcept;

  void getBarriers(const std::vector<BarrierPtr>& barriers) {
    barriers_ = barriers;
  }
  
 private:
  Eigen::Vector2d center_;
  double effective_radius_;
  double alpha_gain_;
  double exp_sum_;
  double h_;

  int num_obs_;
  double kappa_;
  std::vector<BarrierPtr> barriers_;
};    

}  // namespace ccbf
