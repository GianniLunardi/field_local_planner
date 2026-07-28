#!/usr/bin/env python3
"""Composite-CBF simulation using disturbed obstacle-contour point clouds.

Each of the five real circular obstacles is represented to the controller by
50 disturbed contour points. Every point defines a small obstacle CBF, and all
250 point barriers are combined into one conservative smooth-min composite CBF.
The real circles are used only for visualization and clearance validation.

Run from the ``ccbf`` package directory with the existing pyenv environment:

    PYENV_VERSION=cbfpy pyenv exec python scripts/ccbf_pc_sim.py

For a headless run that saves the final frame:

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/cbfpy-matplotlib \
      PYENV_VERSION=cbfpy pyenv exec python \
      scripts/ccbf_pc_sim.py --no-show --save /tmp/ccbf_pc_simulation.png
"""

import os

# Set the JAX/runtime options before importing JAX, as in car_demo.py.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cbfpy import CBF, CBFConfig
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Circle


START = np.array([0.0, 0.0])
GOAL = np.array([10.0, 0.0])
OBSTACLES = np.array(
    [
        [2.0, 0.30, 0.55],
        [3.6, -0.32, 0.55],
        [5.2, 0.28, 0.55],
        [6.8, -0.30, 0.55],
        [8.4, 0.25, 0.55],
    ],
    dtype=float,
)

POINTS_PER_CIRCLE = 50
POINT_CLOUD_NOISE_STD = 0.02
POINT_CLOUD_SEED = 7
POINT_OBSTACLE_RADIUS = 0.06
COMPOSITE_BETA = 100.0


def generate_contour_point_cloud(
    obstacles: np.ndarray,
    points_per_circle: int,
    noise_standard_deviation: float,
    seed: int,
) -> np.ndarray:
    """Sample reproducible, radially disturbed points on circle contours."""

    obstacle_array = np.asarray(obstacles, dtype=float)
    if obstacle_array.ndim != 2 or obstacle_array.shape[1] != 3:
        raise ValueError("obstacles must have shape (N, 3): [x, y, radius]")
    if points_per_circle < 3:
        raise ValueError("points_per_circle must be at least 3")
    if noise_standard_deviation < 0.0:
        raise ValueError("point-cloud noise must be non-negative")

    random_generator = np.random.default_rng(seed)
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        points_per_circle,
        endpoint=False,
    )
    unit_directions = np.column_stack((np.cos(angles), np.sin(angles)))
    point_clouds = []

    for center_x, center_y, radius in obstacle_array:
        radial_noise = random_generator.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=points_per_circle,
        )
        noisy_radii = radius + radial_noise
        if np.any(noisy_radii <= 0.0):
            raise ValueError("point-cloud disturbance produced a non-positive radius")
        center = np.array([center_x, center_y])
        point_clouds.append(center + noisy_radii[:, None] * unit_directions)

    return np.vstack(point_clouds)


POINT_CLOUD = generate_contour_point_cloud(
    OBSTACLES,
    POINTS_PER_CIRCLE,
    POINT_CLOUD_NOISE_STD,
    POINT_CLOUD_SEED,
)


@dataclass(frozen=True)
class SimulationSettings:
    """Numerical and controller settings copied from the C++ example."""

    time_step: float = 0.02
    maximum_time: float = 40.0
    goal_tolerance: float = 0.08
    safety_tolerance: float = 1e-7
    qp_tolerance: float = 1e-6
    maximum_forward_speed: float = 0.8
    maximum_lateral_speed: float = 0.6
    goal_gain: float = 1.0
    path_gain: float = 1.2
    yaw_gain: float = 2.0


@dataclass(frozen=True)
class SimulationResult:
    """In-memory history produced by the fixed-step simulation."""

    states: np.ndarray
    nominal_controls: np.ndarray
    safe_controls: np.ndarray
    barrier_values: np.ndarray
    qp_residuals: np.ndarray
    time_step: float

    @property
    def state_times(self) -> np.ndarray:
        return np.arange(self.states.shape[0], dtype=float) * self.time_step

    @property
    def control_times(self) -> np.ndarray:
        return np.arange(self.safe_controls.shape[0], dtype=float) * self.time_step

    @property
    def minimum_barriers(self) -> np.ndarray:
        return np.min(self.barrier_values, axis=1)

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]


class PointCloudCompositeCBFConfig(CBFConfig):
    """Composite CBF configuration built from point-cloud obstacles."""

    def __init__(
        self,
        point_cloud: ArrayLike,
        point_obstacle_radius: float,
        composite_beta: float,
    ):
        point_array = np.asarray(point_cloud, dtype=float)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("point_cloud must have shape (N, 2)")
        if point_array.shape[0] == 0 or not np.all(np.isfinite(point_array)):
            raise ValueError("point_cloud must contain finite points")
        if not np.isfinite(point_obstacle_radius) or point_obstacle_radius <= 0.0:
            raise ValueError("point_obstacle_radius must be finite and positive")
        if not np.isfinite(composite_beta) or composite_beta <= 0.0:
            raise ValueError("composite_beta must be finite and positive")

        self.point_cloud = jnp.asarray(point_array)
        self.point_obstacle_radius = float(point_obstacle_radius)
        self.composite_beta = float(composite_beta)

        # ======================= MODEL DEFINITION =======================
        # State:   z = [x, y, yaw]
        # Control: u = [body_vx, body_vy, yaw_rate]
        #
        # When replacing the model, update n and m here together with f()
        # and g() below. A different state/control layout may also require
        # changes to pathline_control(), the integration step in
        # run_simulation(), and the x/y/yaw indices used by the plot.
        super().__init__(
            n=3,
            m=3,
            relax_qp=False,
            solver_tol=1e-7,
        )

    def f(self, z: ArrayLike) -> Array:
        """Uncontrolled dynamics in z_dot = f(z) + g(z) u."""

        # ======================= MODEL DEFINITION =======================
        # Replace this function to define a nonzero drift for a new model.
        del z
        return jnp.zeros(3)

    def g(self, z: ArrayLike) -> Array:
        """Rotate body-frame velocity commands into the world frame."""

        # ======================= MODEL DEFINITION =======================
        # Replace this matrix together with f() and the n/m values above
        # when changing the control-affine model.
        yaw = z[2]
        cosine = jnp.cos(yaw)
        sine = jnp.sin(yaw)
        return jnp.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def h_1(self, z: ArrayLike) -> Array:
        """Return one smooth composite of all point-obstacle barriers."""

        # ====================== BARRIER FUNCTIONS ======================
        # Each point acts as a disk obstacle. A positive point_barrier means
        # the robot is outside that point's safety disk.
        offsets = z[:2] - self.point_cloud
        point_barriers = (
            jnp.sum(offsets * offsets, axis=1)
            - self.point_obstacle_radius**2
        )

        # This is the active composite CBF. The unnormalized smooth minimum
        # is conservative: composite_barrier <= min(point_barriers). Enforcing
        # composite_barrier >= 0 therefore keeps every point barrier positive.
        composite_barrier = -logsumexp(
            -self.composite_beta * point_barriers
        ) / self.composite_beta
        return jnp.atleast_1d(composite_barrier)

    def point_barriers(self, z: ArrayLike) -> Array:
        """Return all 250 underlying point CBFs for diagnostics."""

        offsets = z[:2] - self.point_cloud
        return (
            jnp.sum(offsets * offsets, axis=1)
            - self.point_obstacle_radius**2
        )

    def alpha(self, h: ArrayLike) -> Array:
        """Class-K gain matching SphericalObstacleBarrier in the C++ demo."""

        return 2.0 * jnp.asarray(h)


def pathline_control(
    state: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    settings: SimulationSettings,
) -> np.ndarray:
    """Compute the nominal body-frame command used by the C++ simulation."""

    path = goal - start
    path_length = np.linalg.norm(path)
    if not np.isfinite(path_length) or path_length <= 0.0:
        raise ValueError("start and goal must define a nonzero finite path")

    tangent = path / path_length
    normal = np.array([-tangent[1], tangent[0]])
    position = state[:2]

    remaining_distance = float(tangent @ (goal - position))
    cross_track_error = float(normal @ (position - start))
    forward_speed = np.clip(
        settings.goal_gain * remaining_distance,
        0.0,
        settings.maximum_forward_speed,
    )
    lateral_speed = np.clip(
        -settings.path_gain * cross_track_error,
        -settings.maximum_lateral_speed,
        settings.maximum_lateral_speed,
    )
    world_velocity = forward_speed * tangent + lateral_speed * normal

    cosine = math.cos(float(state[2]))
    sine = math.sin(float(state[2]))
    return np.array(
        [
            cosine * world_velocity[0] + sine * world_velocity[1],
            -sine * world_velocity[0] + cosine * world_velocity[1],
            -settings.yaw_gain * state[2],
        ],
        dtype=float,
    )


def evaluate_barriers(
    config: PointCloudCompositeCBFConfig, state: ArrayLike
) -> np.ndarray:
    """Evaluate and validate the configured composite barrier."""

    barriers = np.asarray(config.h_1(state), dtype=float)
    if barriers.ndim != 1 or not np.all(np.isfinite(barriers)):
        raise RuntimeError("barrier evaluation returned invalid values")
    return barriers


def run_simulation(
    config: PointCloudCompositeCBFConfig,
    settings: SimulationSettings,
    start: np.ndarray = START,
    goal: np.ndarray = GOAL,
) -> SimulationResult:
    """Run the fixed-step CBF-filtered rollout and return its full history."""

    cbf = CBF.from_config(config)
    maximum_steps = int(math.ceil(settings.maximum_time / settings.time_step))

    state = jnp.array([start[0], start[1], 0.0])
    initial_barriers = evaluate_barriers(config, state)
    if np.min(initial_barriers) < -settings.safety_tolerance:
        raise RuntimeError(
            "initial state is unsafe: "
            f"minimum h = {np.min(initial_barriers):.6g}"
        )

    states = [np.asarray(state, dtype=float)]
    nominal_controls = []
    safe_controls = []
    barrier_values = [initial_barriers]
    qp_residuals = []
    reached_goal = False

    for step in range(maximum_steps):
        state_numpy = np.asarray(state, dtype=float)
        if np.linalg.norm(state_numpy[:2] - goal) <= settings.goal_tolerance:
            reached_goal = True
            break

        nominal_control = pathline_control(state_numpy, start, goal, settings)
        nominal_control_jax = jnp.asarray(nominal_control)
        safe_control = cbf.safety_filter(state, nominal_control_jax)
        safe_control_numpy = np.asarray(safe_control, dtype=float)

        if safe_control_numpy.shape != (config.m,) or not np.all(
            np.isfinite(safe_control_numpy)
        ):
            raise RuntimeError(
                f"CBF QP returned an invalid control at step {step}: "
                f"{safe_control_numpy}"
            )

        # CBFpy's safety_filter returns only the primal control and does not
        # expose qpax's convergence flag. Verify the actual G u <= h residual.
        inequality_matrix = np.asarray(
            cbf.G_qp(state, nominal_control_jax), dtype=float
        )
        inequality_bound = np.asarray(
            cbf.h_qp(state, nominal_control_jax), dtype=float
        )
        residual = inequality_matrix @ safe_control_numpy - inequality_bound
        maximum_residual = float(np.max(residual))
        if not np.isfinite(maximum_residual) or (
            maximum_residual > settings.qp_tolerance
        ):
            raise RuntimeError(
                "CBF QP constraint check failed at "
                f"step {step}: max(G u - h) = {maximum_residual:.6g}"
            )

        nominal_controls.append(nominal_control)
        safe_controls.append(safe_control_numpy)
        qp_residuals.append(maximum_residual)

        state_derivative = config.f(state) + config.g(state) @ safe_control
        state = state + settings.time_step * state_derivative
        state = state.at[2].set(
            jnp.arctan2(jnp.sin(state[2]), jnp.cos(state[2]))
        )

        next_state = np.asarray(state, dtype=float)
        next_barriers = evaluate_barriers(config, state)
        minimum_barrier = float(np.min(next_barriers))
        if minimum_barrier < -settings.safety_tolerance:
            raise RuntimeError(
                "safety boundary violated after "
                f"step {step}: minimum h = {minimum_barrier:.6g}"
            )

        states.append(next_state)
        barrier_values.append(next_barriers)

    if not reached_goal:
        final_position = states[-1][:2]
        reached_goal = (
            np.linalg.norm(final_position - goal) <= settings.goal_tolerance
        )

    if not reached_goal:
        final_distance = np.linalg.norm(states[-1][:2] - goal)
        raise RuntimeError(
            "robot did not reach the goal within "
            f"{settings.maximum_time:.1f} seconds; "
            f"final distance = {final_distance:.6g}"
        )

    return SimulationResult(
        states=np.asarray(states),
        nominal_controls=np.asarray(nominal_controls),
        safe_controls=np.asarray(safe_controls),
        barrier_values=np.asarray(barrier_values),
        qp_residuals=np.asarray(qp_residuals),
        time_step=settings.time_step,
    )


def minimum_true_circle_clearance(
    states: np.ndarray,
    obstacles: np.ndarray,
) -> float:
    """Return the closest trajectory clearance from any real circle."""

    positions = np.asarray(states, dtype=float)[:, :2]
    obstacle_array = np.asarray(obstacles, dtype=float)
    clearances = [
        np.linalg.norm(positions - obstacle[:2], axis=1) - obstacle[2]
        for obstacle in obstacle_array
    ]
    minimum_clearance = float(np.min(np.column_stack(clearances)))
    if not np.isfinite(minimum_clearance):
        raise RuntimeError("true-circle clearance evaluation is not finite")
    return minimum_clearance


def create_figure(
    result: SimulationResult,
    obstacles: np.ndarray,
    point_cloud: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> Tuple[Figure, Callable[[int], tuple]]:
    """Create the diagnostic figure and return its frame-update callback."""

    figure = plt.figure(figsize=(13, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.6, 1.0))
    trajectory_axis = figure.add_subplot(grid[:, 0])
    velocity_axis = figure.add_subplot(grid[0, 1])
    barrier_axis = figure.add_subplot(grid[1, 1])

    trajectory_axis.plot(
        [start[0], goal[0]],
        [start[1], goal[1]],
        "--",
        color="0.65",
        linewidth=1.5,
        label="nominal pathline",
    )
    trajectory_line, = trajectory_axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.2,
        label="CBF-filtered trajectory",
    )
    trajectory_axis.scatter(
        start[0],
        start[1],
        marker="o",
        s=70,
        color="tab:green",
        zorder=5,
        label="start",
    )
    trajectory_axis.scatter(
        goal[0],
        goal[1],
        marker="*",
        s=180,
        color="gold",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="goal",
    )

    for index, (center_x, center_y, radius) in enumerate(obstacles):
        trajectory_axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                facecolor="tab:red",
                edgecolor="darkred",
                alpha=0.28,
                linewidth=1.5,
                label="real circular obstacles" if index == 0 else None,
            )
        )
        trajectory_axis.scatter(
            center_x,
            center_y,
            marker="x",
            color="darkred",
            s=35,
            zorder=4,
        )
        trajectory_axis.text(
            center_x,
            center_y + radius + 0.08,
            str(index + 1),
            horizontalalignment="center",
            color="darkred",
        )

    trajectory_axis.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        marker="o",
        s=13,
        facecolors="black",
        edgecolors="none",
        alpha=0.85,
        zorder=6,
        label=f"disturbed contour points ({point_cloud.shape[0]})",
    )

    robot_marker, = trajectory_axis.plot(
        [],
        [],
        marker="o",
        markersize=8,
        color="tab:blue",
        markeredgecolor="navy",
        zorder=7,
        label="robot",
    )
    heading_line, = trajectory_axis.plot(
        [],
        [],
        color="navy",
        linewidth=2.0,
        zorder=8,
    )
    time_text = trajectory_axis.text(
        0.98,
        0.97,
        "",
        transform=trajectory_axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )

    nominal_speed = np.linalg.norm(result.nominal_controls[:, :2], axis=1)
    safe_speed = np.linalg.norm(result.safe_controls[:, :2], axis=1)
    velocity_axis.plot(
        result.control_times,
        nominal_speed,
        "--",
        color="0.45",
        label="nominal speed",
    )
    velocity_axis.plot(
        result.control_times,
        safe_speed,
        color="tab:blue",
        label="safe speed",
    )
    velocity_cursor = velocity_axis.axvline(
        0.0,
        color="tab:orange",
        linewidth=1.2,
        label="animation time",
    )
    velocity_axis.set_ylabel("linear speed [m/s]")
    velocity_axis.set_title("Nominal and filtered commands")
    velocity_axis.grid(True, alpha=0.3)
    velocity_axis.legend()

    barrier_axis.plot(
        result.state_times,
        result.minimum_barriers,
        color="tab:purple",
        label="composite point-cloud h",
    )
    barrier_axis.axhline(
        0.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="safety boundary",
    )
    barrier_cursor = barrier_axis.axvline(
        0.0,
        color="tab:orange",
        linewidth=1.2,
        label="animation time",
    )
    barrier_axis.set_xlabel("time [s]")
    barrier_axis.set_ylabel("barrier value h")
    barrier_axis.set_title("Composite point-cloud CBF")
    barrier_axis.grid(True, alpha=0.3)
    barrier_axis.legend()

    trajectory_axis.set_title(
        f"{point_cloud.shape[0]}-point composite CBFpy simulation"
    )
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )

    all_x = np.concatenate(
        [
            result.states[:, 0],
            obstacles[:, 0] - obstacles[:, 2],
            obstacles[:, 0] + obstacles[:, 2],
            point_cloud[:, 0],
        ]
    )
    all_y = np.concatenate(
        [
            result.states[:, 1],
            obstacles[:, 1] - obstacles[:, 2],
            obstacles[:, 1] + obstacles[:, 2],
            point_cloud[:, 1],
        ]
    )
    x_margin = max(0.5, 0.05 * np.ptp(all_x))
    y_margin = max(0.5, 0.10 * np.ptp(all_y))
    trajectory_axis.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
    trajectory_axis.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)

    figure.tight_layout()

    def update_frame(frame: int) -> tuple:
        state = result.states[frame]
        trajectory_line.set_data(
            result.states[: frame + 1, 0],
            result.states[: frame + 1, 1],
        )
        robot_marker.set_data([state[0]], [state[1]])

        heading_length = 0.22
        heading_line.set_data(
            [state[0], state[0] + heading_length * math.cos(state[2])],
            [state[1], state[1] + heading_length * math.sin(state[2])],
        )

        current_time = result.state_times[frame]
        time_text.set_text(f"time = {current_time:.2f} s")
        velocity_cursor_time = min(current_time, result.control_times[-1])
        velocity_cursor.set_xdata([velocity_cursor_time, velocity_cursor_time])
        barrier_cursor.set_xdata([current_time, current_time])
        return (
            trajectory_line,
            robot_marker,
            heading_line,
            time_text,
            velocity_cursor,
            barrier_cursor,
        )

    update_frame(0)
    return figure, update_frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate circular-obstacle avoidance using a disturbed point "
            "cloud and one composite CBF."
        )
    )
    parser.add_argument(
        "--save",
        metavar="PNG_PATH",
        type=Path,
        help="Save the final visualization frame to this path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive Matplotlib animation window.",
    )
    return parser.parse_args()


def print_summary(
    result: SimulationResult,
    goal: np.ndarray,
    point_count: int,
    true_circle_clearance: float,
) -> None:
    final_distance = np.linalg.norm(result.final_state[:2] - goal)
    print(f"Point-cloud points:   {point_count}")
    print(f"Final position:       {result.final_state[:2]}")
    print(f"Distance to goal:     {final_distance:.12g}")
    print(f"Simulation time:      {result.state_times[-1]:.2f} s")
    print(f"Simulation steps:     {result.safe_controls.shape[0]}")
    print(f"Minimum composite h:  {np.min(result.minimum_barriers):.12g}")
    print(f"True-circle clearance: {true_circle_clearance:.12g}")
    print(f"Maximum QP residual:  {np.max(result.qp_residuals):.12g}")
    print("Goal reached safely:  yes")


def main() -> int:
    arguments = parse_arguments()
    settings = SimulationSettings()

    try:
        config = PointCloudCompositeCBFConfig(
            POINT_CLOUD,
            POINT_OBSTACLE_RADIUS,
            COMPOSITE_BETA,
        )
        result = run_simulation(config, settings)
        true_circle_clearance = minimum_true_circle_clearance(
            result.states,
            OBSTACLES,
        )
        if true_circle_clearance < -settings.safety_tolerance:
            raise RuntimeError(
                "point-cloud CBF trajectory intersects a real circle: "
                f"minimum clearance = {true_circle_clearance:.6g}"
            )
        figure, update_frame = create_figure(
            result,
            OBSTACLES,
            POINT_CLOUD,
            START,
            GOAL,
        )
    except (RuntimeError, ValueError) as error:
        print(f"CBFpy simulation failed: {error}", file=sys.stderr)
        return 1

    print_summary(
        result,
        GOAL,
        POINT_CLOUD.shape[0],
        true_circle_clearance,
    )

    if arguments.save is not None:
        update_frame(result.states.shape[0] - 1)
        try:
            figure.savefig(arguments.save, dpi=160, bbox_inches="tight")
        except OSError as error:
            print(f"Unable to save plot to {arguments.save}: {error}", file=sys.stderr)
            plt.close(figure)
            return 1
        print(f"Saved plot:           {arguments.save}")

    if arguments.no_show:
        # Render the final state without constructing an animation that will
        # never be displayed (which would trigger a Matplotlib warning).
        update_frame(result.states.shape[0] - 1)
        figure.canvas.draw()
        plt.close(figure)
        return 0

    update_frame(0)
    animation = FuncAnimation(
        figure,
        update_frame,
        frames=result.states.shape[0],
        interval=result.time_step * 1000.0,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    # Keep the animation referenced until the interactive window is closed.
    del animation
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
