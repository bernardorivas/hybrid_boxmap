"""Cellwise fixed-time suspension grids for the three physical examples.

These builders are the CMGDB-style counterparts of the much cheaper
periodic-orbit smoke runners.  Every base cell is evaluated, every
guard-intersecting cell receives a globally identified unit-handle phase grid,
and exits are sent to an absorbing cemetery cell.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from ..src.fixed_time_suspension_grid import (
    FixedTimeSuspensionGridResult,
    ProgressCallback,
    compute_fixed_time_suspension_grid,
)
from ..src.grid import Grid
from .bouncing_ball import BouncingBall
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    PERIOD_TWO_POINT_A,
    GarciaPassiveWalker,
    post_impact_state,
)
from .rimless_wheel import RimlessWheel


def _three_points(lower: float, upper: float) -> tuple[float, ...]:
    if upper < lower:
        return ()
    values = np.asarray((lower, 0.5 * (lower + upper), upper), dtype=float)
    return tuple(float(value) for value in np.unique(np.round(values, 14)))


def _endpoints(lower: float, upper: float) -> tuple[float, ...]:
    if upper < lower:
        return ()
    values = np.asarray((lower, upper), dtype=float)
    return tuple(float(value) for value in np.unique(np.round(values, 14)))


def _garcia_orbit_tube_cells(
    grid: Grid,
    walker: GarciaPassiveWalker,
    *,
    radius_cells: int,
    sample_step: float,
) -> tuple[int, ...]:
    """Return a grid-cell tube around the stored two-stride physical orbit.

    The orbit selects a local active family only.  Every cell in the resulting
    tube is subsequently evaluated by the fixed-time suspension box map; this
    is not the old construction that inserted only the cells hit by orbit
    samples.
    """

    if radius_cells < 0:
        raise ValueError("radius_cells must be non-negative")
    if not np.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("sample_step must be finite and positive")
    trajectory = walker.system.simulate(
        post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 8.0),
        max_jumps=3,
        dense_output=True,
        max_step=min(0.01, sample_step),
    )
    core: set[int] = set()
    for segment in trajectory.segments:
        sample_count = max(
            2,
            int(np.ceil((segment.t_end - segment.t_start) / sample_step)) + 1,
        )
        times = np.linspace(segment.t_start, segment.t_end, sample_count)
        if segment.solution is None:
            continue
        raw_states = np.asarray(segment.solution(times), dtype=float)
        if raw_states.ndim == 1:
            raw_states = raw_states[:, np.newaxis]
        states = raw_states.T
        for state in states:
            core.update(grid.find_boxes_containing_point(state))
    if not core:
        raise RuntimeError("stored Garcia orbit does not meet the declared grid")

    active: set[int] = set()
    shape = tuple(int(value) for value in grid.subdivisions)
    for cell_index in core:
        coordinates = np.asarray(np.unravel_index(cell_index, shape), dtype=int)
        ranges = tuple(
            range(
                max(0, int(coordinate) - radius_cells),
                min(shape[axis] - 1, int(coordinate) + radius_cells) + 1,
            )
            for axis, coordinate in enumerate(coordinates)
        )
        active.update(
            int(np.ravel_multi_index(neighbor, shape))
            for neighbor in itertools.product(*ranges)
        )
    return tuple(sorted(active))


def build_bouncing_ball_suspension_grid(
    *,
    subdivisions: Sequence[int] = (64, 256),
    t_star: float = 0.5,
    handle_slabs: int = 16,
    padding_cells: float = 1.0,
    gravity: float = 9.81,
    restitution: float = 0.5,
    progress_callback: ProgressCallback | None = None,
) -> FixedTimeSuspensionGridResult:
    """Compute the bouncing-ball suspension relation on ``[0,2]x[-5,5]``."""

    ball = BouncingBall(
        domain_bounds=[(0.0, 2.0), (-5.0, 5.0)],
        g=gravity,
        c=restitution,
        max_jumps=64,
    )
    grid = Grid(
        bounds=[[0.0, 2.0], [-5.0, 5.0]],
        subdivisions=[int(value) for value in subdivisions],
    )

    def guard_sampler(lower, upper):
        if not lower[0] <= 0.0 <= upper[0] or lower[1] > 0.0:
            return ()
        velocity_upper = min(float(upper[1]), 0.0)
        return tuple(
            np.asarray((0.0, velocity), dtype=float)
            for velocity in _three_points(float(lower[1]), velocity_upper)
        )

    return compute_fixed_time_suspension_grid(
        model_name="bouncing_ball_zeno_suspension",
        system=ball.system,
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=t_star,
        handle_slabs=handle_slabs,
        padding_cells=padding_cells,
        max_jumps=64,
        max_step=0.02,
        cemetery_label="ball_exit",
        progress_callback=progress_callback,
    )


def build_rimless_wheel_suspension_grid(
    *,
    subdivisions: Sequence[int] = (201, 201),
    t_star: float = 0.5,
    handle_slabs: int = 16,
    padding_cells: float = 1.0,
    alpha: float = 0.4,
    gamma: float = 0.2,
    progress_callback: ProgressCallback | None = None,
) -> FixedTimeSuspensionGridResult:
    """Compute the wheel relation on the traditional global plotting domain."""

    bounds = [(-0.2, 0.6), (-0.5, 1.0)]
    wheel = RimlessWheel(
        domain_bounds=bounds,
        alpha=alpha,
        gamma=gamma,
        max_jumps=24,
    )
    grid = Grid(
        bounds=[list(value) for value in bounds],
        subdivisions=[int(value) for value in subdivisions],
    )
    # The traditional plotting domain ends exactly on the impact section.
    # Snap only floating-point roundoff to that boundary; a genuinely exterior
    # guard must remain exterior rather than being invented at the box edge.
    raw_guard_angle = float(alpha + gamma)
    angle_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0,
        abs(raw_guard_angle),
        abs(bounds[0][0]),
        abs(bounds[0][1]),
    )
    if abs(raw_guard_angle - bounds[0][0]) <= angle_tolerance:
        guard_angle = float(bounds[0][0])
    elif abs(raw_guard_angle - bounds[0][1]) <= angle_tolerance:
        guard_angle = float(bounds[0][1])
    else:
        guard_angle = raw_guard_angle

    def guard_sampler(lower, upper):
        if (
            guard_angle < float(lower[0]) - angle_tolerance
            or guard_angle > float(upper[0]) + angle_tolerance
            or upper[1] < 0.0
        ):
            return ()
        velocity_lower = max(float(lower[1]), 0.0)
        return tuple(
            np.asarray((guard_angle, velocity), dtype=float)
            for velocity in _three_points(velocity_lower, float(upper[1]))
        )

    return compute_fixed_time_suspension_grid(
        model_name="rimless_wheel_walking_suspension",
        system=wheel.system,
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=t_star,
        handle_slabs=handle_slabs,
        padding_cells=padding_cells,
        max_jumps=24,
        max_step=0.02,
        cemetery_label="wheel_exit",
        progress_callback=progress_callback,
    )


def build_garcia_passive_walker_suspension_grid(
    *,
    subdivisions: Sequence[int] = (12, 12, 12, 12),
    t_star: float = 0.5,
    handle_slabs: int = 16,
    padding_cells: float = 1.0,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    active_tube_radius: int | None = 1,
    tube_sample_step: float = 0.01,
    progress_callback: ProgressCallback | None = None,
) -> FixedTimeSuspensionGridResult:
    """Compute a local four-dimensional gait-plus-exit suspension relation."""

    # Local box around the complete stored period-two orbit, not merely its
    # post-impact section points.  A dense two-stride regression has ranges
    # theta +/-0.2551, theta_dot [-0.2824,-0.0664], phi +/-0.5101, and
    # phi_dot [-0.4297,0.0541]; the margins below keep that orbit in the
    # declared compact state space.
    bounds = [
        (-0.29, 0.29),
        (-0.305, -0.045),
        (-0.56, 0.56),
        (-0.47, 0.09),
    ]
    walker = GarciaPassiveWalker(
        gamma=gamma,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
        domain_bounds=bounds,
        max_jumps=20,
    )
    grid = Grid(
        bounds=[list(value) for value in bounds],
        subdivisions=[int(value) for value in subdivisions],
    )
    active_base_cells = (
        None
        if active_tube_radius is None
        else _garcia_orbit_tube_cells(
            grid,
            walker,
            radius_cells=active_tube_radius,
            sample_step=tube_sample_step,
        )
    )

    def guard_sampler(lower, upper):
        theta_lower = float(lower[0])
        theta_upper = min(
            float(upper[0]),
            0.5 * float(upper[2]),
            -guard_delta,
        )
        theta_lower = max(theta_lower, 0.5 * float(lower[2]))
        if theta_upper < theta_lower:
            return ()
        theta_dot_lower = float(lower[1])
        theta_dot_upper = min(
            float(upper[1]),
            0.5 * (float(upper[3]) - transversality_eta),
        )
        if theta_dot_upper < theta_dot_lower:
            return ()

        representatives = []
        for theta in _endpoints(theta_lower, theta_upper):
            for theta_dot in _endpoints(
                theta_dot_lower,
                theta_dot_upper,
            ):
                phi_dot_lower = max(
                    float(lower[3]),
                    2.0 * theta_dot + transversality_eta,
                )
                for phi_dot in _endpoints(
                    phi_dot_lower,
                    float(upper[3]),
                ):
                    representatives.append(
                        np.asarray(
                            (theta, theta_dot, 2.0 * theta, phi_dot),
                            dtype=float,
                        )
                    )
        return tuple(representatives)

    result = compute_fixed_time_suspension_grid(
        model_name="garcia_passive_walker_period_two_suspension",
        system=walker.system,
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=t_star,
        handle_slabs=handle_slabs,
        padding_cells=padding_cells,
        max_jumps=20,
        max_step=0.02,
        cemetery_label="walker_exit",
        progress_callback=progress_callback,
        active_base_cells=active_base_cells,
    )
    result.relation_graph.graph["domain_role"] = "local gait isolating-neighborhood candidate"
    result.relation_graph.graph["active_family"] = (
        "full local box" if active_tube_radius is None else "sampled-orbit cell tube"
    )
    result.relation_graph.graph["active_tube_radius_cells"] = active_tube_radius
    return result


__all__ = [
    "build_bouncing_ball_suspension_grid",
    "build_rimless_wheel_suspension_grid",
    "build_garcia_passive_walker_suspension_grid",
]
