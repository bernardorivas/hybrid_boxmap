"""Fixed-time unit-handle suspension pipeline for the bouncing ball.

The materialized orbit is the Zeno limiting state ``(h, v) = (0, 0)``.  Its
self-reset is represented by a handle of duration one, so it is a periodic
fiber in suspension time rather than an ordinary-time zero-duration loop.

The returned trajectory witnesses are genuine integrations.  The finite cell
relation is a deterministic point-sampling model and is not advertised as a
rigorous whole-cell enclosure or as a computed Conley index.
"""

from __future__ import annotations

import numpy as np

from ._fixed_time_suspension import (
    FixedTimeSuspensionPipelineResult,
    PhysicalEndpointWitness,
    materialize_periodic_orbit_relation,
)
from .bouncing_ball import BouncingBall
from ..src.grid import Grid
from ..src.sampled_suspension import simulate_suspension_endpoint


MODEL_NAME = "bouncing_ball_zeno_suspension"
DEFAULT_T_STAR = 0.5
DEFAULT_HANDLE_SLABS = 8


def analytic_next_impact_time(
    state: np.ndarray,
    *,
    gravity: float = 9.81,
) -> float:
    """Return the next non-negative ballistic impact time.

    For ``h >= 0`` this is the non-negative root of
    ``h + v*t - gravity*t**2/2 = 0``.  At the rest state the result is zero.
    """

    h, velocity = np.asarray(state, dtype=np.float64)
    if gravity <= 0 or not np.isfinite(gravity):
        raise ValueError("gravity must be finite and positive")
    if h < 0:
        raise ValueError("height must be non-negative")
    return float((velocity + np.sqrt(velocity * velocity + 2.0 * gravity * h)) / gravity)


def build_bouncing_ball_suspension_pipeline(
    *,
    t_star: float = DEFAULT_T_STAR,
    handle_slabs: int = DEFAULT_HANDLE_SLABS,
    sample_count: int = 160,
    gravity: float = 9.81,
    restitution: float = 0.5,
    compute_cmgdb: bool = False,
) -> FixedTimeSuspensionPipelineResult:
    """Compute the sampled fixed-time relation on the Zeno suspension fiber."""

    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not 0.0 <= restitution < 1.0:
        raise ValueError("the Zeno pipeline requires 0 <= restitution < 1")
    energy_cap = 12.5
    velocity_cap = float(np.sqrt(2.0 * energy_cap))
    ball = BouncingBall(
        domain_bounds=[
            (0.0, energy_cap / gravity),
            (-velocity_cap, velocity_cap),
        ],
        g=gravity,
        c=restitution,
        max_jumps=max(12, handle_slabs + 3),
    )
    grid = Grid(
        bounds=[
            [0.0, energy_cap / gravity],
            [-velocity_cap, velocity_cap],
        ],
        subdivisions=[8, 11],
    )

    falling_initial = np.array([1.0, 0.0])
    impact_time = analytic_next_impact_time(falling_initial, gravity=gravity)
    falling_trajectory = ball.system.simulate(
        falling_initial,
        (0.0, impact_time + 0.1),
        max_jumps=1,
        dense_output=True,
        max_step=0.01,
    )
    if not falling_trajectory.jump_times:
        raise RuntimeError("the numerical bouncing-ball trajectory missed its impact")
    numerical_impact_time = float(falling_trajectory.jump_times[0])
    falling_endpoint = simulate_suspension_endpoint(
        ball.system,
        falling_initial,
        t_star,
        max_jumps=3,
        max_step=0.01,
    )
    rest_endpoint = simulate_suspension_endpoint(
        ball.system,
        np.array([0.0, 0.0]),
        t_star,
        max_jumps=max(4, handle_slabs),
        max_step=0.01,
    )
    witnesses = (
        PhysicalEndpointWitness(
            name="falling_ball_fixed_time_endpoint",
            initial_state=(1.0, 0.0),
            endpoint=falling_endpoint,
        ),
        PhysicalEndpointWitness(
            name="zeno_rest_fiber_endpoint",
            initial_state=(0.0, 0.0),
            endpoint=rest_endpoint,
        ),
    )
    diagnostics = {
        "gravity": float(gravity),
        "coefficient_of_restitution": float(restitution),
        "energy_cap": energy_cap,
        "analytic_first_impact_time": impact_time,
        "numerical_first_impact_time": numerical_impact_time,
        "impact_time_absolute_error": abs(impact_time - numerical_impact_time),
        "active_region": "h >= 0 and v^2/2 + g*h <= 12.5 (rectangular cover used here)",
    }
    return materialize_periodic_orbit_relation(
        model_name=MODEL_NAME,
        system=ball.system,
        initial_state=(0.0, 0.0),
        grid=grid,
        t_star=t_star,
        orbit_period=1.0,
        handle_slabs=handle_slabs,
        sample_count=sample_count,
        max_jumps=max(12, handle_slabs + 3),
        max_step=0.01,
        witnesses=witnesses,
        diagnostics=diagnostics,
        compute_cmgdb=compute_cmgdb,
    )


if __name__ == "__main__":
    print(build_bouncing_ball_suspension_pipeline().summary())


__all__ = [
    "MODEL_NAME",
    "DEFAULT_T_STAR",
    "DEFAULT_HANDLE_SLABS",
    "analytic_next_impact_time",
    "build_bouncing_ball_suspension_pipeline",
]
