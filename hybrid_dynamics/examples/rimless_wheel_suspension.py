"""Fixed-time unit-handle suspension pipeline for the rimless wheel.

The runner initializes the analytically determined one-step walking gait,
checks conservation of stance energy, and samples the exact suspension clock
around one flow-plus-reset cycle.  SCCs are exact for the resulting finite
sample relation, but that relation is not a validated whole-cell enclosure.
"""

from __future__ import annotations

import numpy as np

from ._fixed_time_suspension import (
    FixedTimeSuspensionPipelineResult,
    PhysicalEndpointWitness,
    materialize_periodic_orbit_relation,
)
from .rimless_wheel import RimlessWheel
from ..src.grid import Grid
from ..src.sampled_suspension import simulate_suspension_endpoint


MODEL_NAME = "rimless_wheel_walking_suspension"
DEFAULT_T_STAR = 0.5
DEFAULT_HANDLE_SLABS = 8


def walking_fixed_point_speed(*, alpha: float, gamma: float) -> float:
    """Return the post-impact angular speed of the forward walking gait."""

    impact_factor = float(np.cos(2.0 * alpha))
    if not 0.0 < impact_factor < 1.0:
        raise ValueError("this forward-gait formula requires 0 < cos(2*alpha) < 1")
    reset_angle = gamma - alpha
    guard_angle = gamma + alpha
    potential_drop = float(np.cos(reset_angle) - np.cos(guard_angle))
    if potential_drop <= 0.0:
        raise ValueError("the selected slope/spoke angles do not drive a forward gait")
    speed_squared = (
        2.0
        * impact_factor
        * impact_factor
        * potential_drop
        / (1.0 - impact_factor * impact_factor)
    )
    return float(np.sqrt(speed_squared))


def stance_energy(state: np.ndarray) -> float:
    """Conserved swing-phase energy ``omega**2/2 + cos(theta)``."""

    theta, omega = np.asarray(state, dtype=np.float64)
    return float(0.5 * omega * omega + np.cos(theta))


def build_rimless_wheel_suspension_pipeline(
    *,
    t_star: float = DEFAULT_T_STAR,
    handle_slabs: int = DEFAULT_HANDLE_SLABS,
    sample_count: int = 240,
    alpha: float = 0.4,
    gamma: float = 0.2,
    compute_cmgdb: bool = False,
) -> FixedTimeSuspensionPipelineResult:
    """Compute a sampled fixed-time relation around the stable walking gait."""

    reset_angle = gamma - alpha
    guard_angle = gamma + alpha
    post_speed = walking_fixed_point_speed(alpha=alpha, gamma=gamma)
    initial = np.array([reset_angle, post_speed])
    wheel = RimlessWheel(
        domain_bounds=[
            (reset_angle - 0.01, guard_angle + 0.01),
            (0.2, 0.9),
        ],
        alpha=alpha,
        gamma=gamma,
        max_jumps=12,
    )
    grid = Grid(
        bounds=[
            [reset_angle - 0.01, guard_angle + 0.01],
            [0.2, 0.9],
        ],
        subdivisions=[16, 14],
    )

    one_step = wheel.system.simulate(
        initial,
        (0.0, 8.0),
        max_jumps=1,
        dense_output=True,
        max_step=0.01,
    )
    if not one_step.jump_times:
        raise RuntimeError("the rimless-wheel gait did not reach its spoke-impact guard")
    flow_time = float(one_step.jump_times[0])
    guard_state, reset_state = one_step.jump_states[0]
    orbit_period = flow_time + 1.0
    witness = PhysicalEndpointWitness(
        name="walking_gait_fixed_time_endpoint",
        initial_state=tuple(float(value) for value in initial),
        endpoint=simulate_suspension_endpoint(
            wheel.system,
            initial,
            t_star,
            max_jumps=3,
            max_step=0.01,
        ),
    )
    diagnostics = {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "impact_velocity_factor": float(np.cos(2.0 * alpha)),
        "post_impact_fixed_point_speed": post_speed,
        "flow_time": flow_time,
        "stance_energy_initial": stance_energy(initial),
        "stance_energy_at_guard": stance_energy(guard_state),
        "stance_energy_residual": abs(
            stance_energy(initial) - stance_energy(guard_state)
        ),
        "reset_fixed_point_residual": float(np.linalg.norm(reset_state - initial)),
        "guard_transverse_speed": float(guard_state[1]),
    }
    return materialize_periodic_orbit_relation(
        model_name=MODEL_NAME,
        system=wheel.system,
        initial_state=initial,
        grid=grid,
        t_star=t_star,
        orbit_period=orbit_period,
        handle_slabs=handle_slabs,
        sample_count=sample_count,
        max_jumps=8,
        max_step=0.01,
        skeleton_flow_arcs=1,
        witnesses=(witness,),
        diagnostics=diagnostics,
        compute_cmgdb=compute_cmgdb,
    )


if __name__ == "__main__":
    print(build_rimless_wheel_suspension_pipeline().summary())


__all__ = [
    "MODEL_NAME",
    "DEFAULT_T_STAR",
    "DEFAULT_HANDLE_SLABS",
    "walking_fixed_point_speed",
    "stance_energy",
    "build_rimless_wheel_suspension_pipeline",
]
