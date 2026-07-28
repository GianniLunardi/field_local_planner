#include <ccbf/barrier.hpp>

#include <cmath>
#include <stdexcept>

namespace ccbf {

SphericalObstacleBarrier::SphericalObstacleBarrier(
    const Eigen::Vector2d& center,
    double effective_radius,
    double alpha_gain)
    : center_(center),
      effective_radius_(effective_radius),
      alpha_gain_(alpha_gain) {
  if (!center_.allFinite()) {
    throw std::invalid_argument("Obstacle center must be finite");
  }
  if (!std::isfinite(effective_radius_) || effective_radius_ <= 0.0) {
    throw std::invalid_argument("Obstacle effective radius must be finite and positive");
  }
  if (!std::isfinite(alpha_gain_) || alpha_gain_ <= 0.0) {
    throw std::invalid_argument("Barrier alpha gain must be finite and positive");
  }
}

int SphericalObstacleBarrier::stateDimension() const noexcept {
  return 3;
}

double SphericalObstacleBarrier::value(
    const Eigen::Ref<const Eigen::VectorXd>& state) const {
  if (state.size() != stateDimension()) {
    throw std::invalid_argument("Spherical obstacle state must have dimension 3");
  }
  if (!state.allFinite()) {
    throw std::invalid_argument("Spherical obstacle state must be finite");
  }

  const Eigen::Vector2d offset = state.head<2>() - center_;
  return offset.squaredNorm() - effective_radius_ * effective_radius_;
}

void SphericalObstacleBarrier::gradient(
    const Eigen::Ref<const Eigen::VectorXd>& state,
    Eigen::Ref<Eigen::VectorXd> gradient) const {
  if (state.size() != stateDimension()) {
    throw std::invalid_argument("Spherical obstacle state must have dimension 3");
  }
  if (gradient.size() != stateDimension()) {
    throw std::invalid_argument(
        "Spherical obstacle gradient output must have dimension 3");
  }
  if (!state.allFinite()) {
    throw std::invalid_argument("Spherical obstacle state must be finite");
  }

  gradient << 2.0 * (state(0) - center_(0)),
              2.0 * (state(1) - center_(1)),
              0.0;
}

double SphericalObstacleBarrier::alpha(double h) const {
  if (!std::isfinite(h)) {
    throw std::invalid_argument("Barrier value must be finite");
  }
  return alpha_gain_ * h;
}

const Eigen::Vector2d& SphericalObstacleBarrier::center() const noexcept {
  return center_;
}

double SphericalObstacleBarrier::effectiveRadius() const noexcept {
  return effective_radius_;
}

double SphericalObstacleBarrier::alphaGain() const noexcept {
  return alpha_gain_;
}

CompositeBarrier::CompositeBarrier(
    const int num_obs,
    double kappa)
    : alpha_gain_(2.0),
      num_obs_(num_obs),
      kappa_(kappa) {
  if (!std::isfinite(num_obs_) || num_obs_ < 0.0) {
    throw std::invalid_argument("Obstacles must be greater equal than zero");
  }
  // TODO: update the single barrier here??
}

int CompositeBarrier::stateDimension() const noexcept {
  return 3;
}

double CompositeBarrier::value(
    const Eigen::Ref<const Eigen::VectorXd>& state) const {
  if (state.size() != stateDimension()) {
    throw std::invalid_argument("Spherical obstacle state must have dimension 3");
  }
  if (!state.allFinite()) {
    throw std::invalid_argument("Spherical obstacle state must be finite");
  }

  double exp_sum = 0.0;
  for (size_t i = 0; i < num_obs_; i++) {
    exp_sum += std::exp(-kappa_ * barriers_[i]->value(state));
  }
  return -1.0 / kappa_ * std::log(exp_sum);

}

void CompositeBarrier::gradient(
    const Eigen::Ref<const Eigen::VectorXd>& state,
    Eigen::Ref<Eigen::VectorXd> gradient) const {
  if (state.size() != stateDimension()) {
    throw std::invalid_argument("Spherical obstacle state must have dimension 3");
  }
  if (gradient.size() != stateDimension()) {
    throw std::invalid_argument(
        "Spherical obstacle gradient output must have dimension 3");
  }
  if (!state.allFinite()) {
    throw std::invalid_argument("Spherical obstacle state must be finite");
  }

  Eigen::VectorXd single_gradient(state.size());
  // L_{g}
  Eigen::VectorXd Lg_h = Eigen::VectorXd::Zero(state.size());
  double exp_sum = 0.0;
  for (size_t i = 0; i < num_obs_; i++) {
    barriers_[i]->gradient(state, single_gradient);
    // TODO: avoid recompute barrier value
    const double weight =
        std::exp(-kappa_ * barriers_[i]->value(state));
    Lg_h += weight * single_gradient;
    exp_sum += weight;
  }
  gradient = Lg_h / exp_sum;
}

double CompositeBarrier::alpha(double h) const {
  if (!std::isfinite(h)) {
    throw std::invalid_argument("Barrier value must be finite");
  }
  return alpha_gain_ * h;
}

}  // namespace ccbf
