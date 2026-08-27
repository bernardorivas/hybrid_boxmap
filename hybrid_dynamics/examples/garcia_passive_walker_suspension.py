"""Garcia simplest-passive-walker fixed-time suspension pipeline.

This is the four-dimensional limiting model of Garcia, Chatterjee, Ruina, and
Coleman, *The Simplest Walking Model: Stability, Complexity, and Scaling*,
Journal of Biomechanical Engineering 120 (1998), DOI 10.1115/1.2798313.

State order is ``(theta, theta_dot, phi, phi_dot)``.  The physical event-driven
simulation stays four-dimensional; the period-two post-impact points below are
only regression seeds, not a replacement by a Poincare return-map model.

The legacy smoke pipeline below can add one absorbing cemetery graph cell
while preserving the three raw terminal labels in diagnostics.  The Atlas
local-relation workflow uses open exits instead and never inserts that cell.
"""

from __future__ import annotations

import numpy as np

from ._fixed_time_suspension import (
    CemeteryCell,
    FixedTimeSuspensionPipelineResult,
    PhysicalEndpointWitness,
    materialize_periodic_orbit_relation,
)
from ..src.grid import Grid
from ..src.hybrid_system import HybridSystem
from ..src.sampled_suspension import simulate_suspension_endpoint


MODEL_NAME = "garcia_passive_walker_period_two_suspension"
DEFAULT_T_STAR = 0.5
DEFAULT_HANDLE_SLABS = 8
DEFAULT_GAMMA = 0.0172
DEFAULT_GUARD_DELTA = 0.05
DEFAULT_TRANSVERSALITY_ETA = 0.10
PERIOD_TWO_POINT_A = (0.238312423452016, -0.233987869217989)
PERIOD_TWO_POINT_B = (0.255026005923842, -0.246392826580685)
RAW_FAILURE_LABELS = (
    "forward_fall",
    "backward_fall",
    "insufficient_energy",
)

PHYSICAL_DOMAIN_BOUNDS = (
    (-0.32, 0.32),
    (-0.35, 0.05),
    (-0.65, 0.65),
    (-0.55, 0.15),
)


def guard_aligned_domain_hull(
    physical_bounds: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the rectangular hull in ``(theta, omega, q, nu)`` coordinates.

    Here ``q = phi - 2 theta`` and ``nu = phi_dot - 2 omega``.  The image of a
    physical coordinate rectangle is a parallelepiped; the Atlas base chart
    deliberately uses this containing rectangular hull and later treats the
    local tube, rather than the full chart rectangle, as the active domain.
    """

    if len(physical_bounds) != 4:
        raise ValueError("Garcia physical bounds must have four axes")
    theta, omega, phi, phi_dot = (
        (float(lower), float(upper)) for lower, upper in physical_bounds
    )
    for lower, upper in (theta, omega, phi, phi_dot):
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("physical bounds must be finite increasing intervals")
    return [
        theta,
        omega,
        (phi[0] - 2.0 * theta[1], phi[1] - 2.0 * theta[0]),
        (phi_dot[0] - 2.0 * omega[1], phi_dot[1] - 2.0 * omega[0]),
    ]


GUARD_ALIGNED_DOMAIN_BOUNDS = tuple(
    guard_aligned_domain_hull(PHYSICAL_DOMAIN_BOUNDS)
)


class InadmissibleHeelstrikeError(ValueError):
    """Raised when the numerical event solver stops off the declared guard.

    The continuous scalar event is the minimum of the three closed guard
    inequalities.  Its zero set also contains boundary pieces on which contact
    itself has not occurred.  Every proposed collision is therefore checked
    against the geometric equality and transversality conditions before the
    Garcia collision map is applied.  A rejected boundary event becomes an
    explicit failed/open-exit sample in the Atlas callback instead of a
    spurious jump.
    """


class _GarciaHybridSystem(HybridSystem):
    """Reject initial states beyond contact before generic jump handling.

    ``HybridTrajectory`` correctly locates crossings during integration, but its
    generic initial jump-set convention treats every nonnegative event value as
    an immediate jump.  For this equality guard, a penetrated or otherwise
    ineligible initial state is instead an open exit.
    """

    def __init__(self, *args, guard_validator, **kwargs) -> None:
        self._guard_validator = guard_validator
        super().__init__(*args, **kwargs)

    def simulate(self, initial_state, time_span, **kwargs):
        point = np.asarray(initial_state, dtype=np.float64)
        value = float(self.evaluate_event_function(float(time_span[0]), point))
        direction = getattr(self.event_function, "direction", 0)
        initially_in_jump_set = bool(
            (direction > 0 and value >= 0.0)
            or (direction < 0 and value <= 0.0)
        )
        if initially_in_jump_set and not self._guard_validator(point):
            raise InadmissibleHeelstrikeError(
                "initial state lies beyond contact but not on the declared "
                "transverse heel-strike guard"
            )
        return super().simulate(point, time_span, **kwargs)


def ode_fun(_time: float, state: np.ndarray, walker: "GarciaPassiveWalker") -> np.ndarray:
    """Nondimensional swing-phase equations (Garcia et al., eqs. 1--2)."""

    theta, theta_dot, phi, phi_dot = state
    theta_ddot = np.sin(theta - walker.gamma)
    phi_ddot = (
        theta_ddot
        + theta_dot * theta_dot * np.sin(phi)
        - np.cos(theta - walker.gamma) * np.sin(phi)
    )
    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])


def heelstrike_gap(state: np.ndarray) -> float:
    """Signed geometric foot-contact gap ``phi - 2*theta``."""

    theta, _, phi, _ = np.asarray(state, dtype=np.float64)
    return float(phi - 2.0 * theta)


def heelstrike_transversality(state: np.ndarray) -> float:
    """Time derivative of :func:`heelstrike_gap`."""

    _, theta_dot, _, phi_dot = np.asarray(state, dtype=np.float64)
    return float(phi_dot - 2.0 * theta_dot)


def reset_map(state: np.ndarray) -> np.ndarray:
    """Plastic heel-strike collision and leg relabeling (Garcia et al., eq. 4)."""

    theta, theta_dot, _, _ = np.asarray(state, dtype=np.float64)
    collision_factor = np.cos(2.0 * theta)
    return np.array(
        [
            -theta,
            collision_factor * theta_dot,
            -2.0 * theta,
            collision_factor * (1.0 - collision_factor) * theta_dot,
        ],
    )


def encode_guard_aligned_state(state: np.ndarray) -> np.ndarray:
    """Encode ``(theta, omega, phi, phi_dot)`` as ``(theta, omega, q, nu)``."""

    theta, omega, phi, phi_dot = np.asarray(state, dtype=np.float64)
    return np.array(
        [theta, omega, phi - 2.0 * theta, phi_dot - 2.0 * omega],
        dtype=np.float64,
    )


def decode_guard_aligned_state(state: np.ndarray) -> np.ndarray:
    """Invert :func:`encode_guard_aligned_state` exactly up to roundoff."""

    theta, omega, q, nu = np.asarray(state, dtype=np.float64)
    return np.array(
        [theta, omega, q + 2.0 * theta, nu + 2.0 * omega],
        dtype=np.float64,
    )


def encode_guard_aligned_velocity(velocity: np.ndarray) -> np.ndarray:
    """Apply the tangent map of the linear guard-aligned coordinate change."""

    theta_dot, omega_dot, phi_dot, phi_ddot = np.asarray(
        velocity,
        dtype=np.float64,
    )
    return np.array(
        [
            theta_dot,
            omega_dot,
            phi_dot - 2.0 * theta_dot,
            phi_ddot - 2.0 * omega_dot,
        ],
        dtype=np.float64,
    )


def guard_aligned_ode_fun(
    _time: float,
    state: np.ndarray,
    walker: "GuardAlignedGarciaPassiveWalker",
) -> np.ndarray:
    """Garcia swing dynamics in the guard-aligned base chart.

    With ``z=(theta, omega, q, nu)`` and ``phi=q+2 theta``, the contact guard
    is the cubical hyperplane ``q=0`` and ``q_dot=nu``.
    """

    theta, omega, q, nu = np.asarray(state, dtype=np.float64)
    theta_ddot = np.sin(theta - walker.gamma)
    phi = q + 2.0 * theta
    nu_dot = (
        -theta_ddot
        + omega * omega * np.sin(phi)
        - np.cos(theta - walker.gamma) * np.sin(phi)
    )
    return np.array([omega, theta_ddot, nu, nu_dot], dtype=np.float64)


def guard_aligned_reset_map(state: np.ndarray) -> np.ndarray:
    """Garcia reset in guard-aligned coordinates.

    For every state (and in particular on the declared ``q=0`` guard), with
    ``c=cos(2 theta)``, physical reset followed by coordinate encoding is

    ``(-theta, c omega, 0, -c(1+c) omega)``.

    Thus both the guard and reset patches lie on the same internal cubical
    hyperplane ``q=0`` while remaining disjoint in the ``theta`` coordinate.
    """

    theta, omega, _, _ = np.asarray(state, dtype=np.float64)
    collision_factor = np.cos(2.0 * theta)
    return np.array(
        [
            -theta,
            collision_factor * omega,
            0.0,
            -collision_factor * (1.0 + collision_factor) * omega,
        ],
        dtype=np.float64,
    )


def guard_aligned_post_impact_state(theta: float, theta_dot: float) -> np.ndarray:
    """Lift a stored post-impact section point into guard-aligned coordinates."""

    return encode_guard_aligned_state(post_impact_state(theta, theta_dot))


def post_impact_state(theta: float, theta_dot: float) -> np.ndarray:
    """Lift a post-impact section point to the full four-dimensional state."""

    collision_factor = np.cos(2.0 * theta)
    return np.array(
        [
            theta,
            theta_dot,
            2.0 * theta,
            (1.0 - collision_factor) * theta_dot,
        ],
    )


class GarciaPassiveWalker:
    """The four-dimensional simplest passive dynamic walker."""

    def __init__(
        self,
        *,
        gamma: float = DEFAULT_GAMMA,
        guard_delta: float = DEFAULT_GUARD_DELTA,
        transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
        domain_bounds: list[tuple[float, float]] | None = None,
        max_jumps: int = 20,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if guard_delta <= 0.0:
            raise ValueError("guard_delta must be positive")
        if transversality_eta <= 0.0:
            raise ValueError("transversality_eta must be positive")
        self.gamma = float(gamma)
        self.guard_delta = float(guard_delta)
        self.transversality_eta = float(transversality_eta)
        self.domain_bounds = domain_bounds or list(PHYSICAL_DOMAIN_BOUNDS)

        def ode(time: float, state: np.ndarray) -> np.ndarray:
            return ode_fun(time, state, self)

        def event(_time: float, state: np.ndarray) -> float:
            # The minimum is a continuous scalar whose positive side requires
            # contact penetration, a stance leg past the scuffing margin, and
            # positive transverse gap speed.  The desired gait event occurs
            # when the geometric gap is the last active constraint.  If another
            # constraint becomes active first, ``validated_reset`` rejects that
            # non-contact stop and the Atlas callback records an open exit.
            theta = float(state[0])
            return min(
                heelstrike_gap(state),
                -theta - self.guard_delta,
                heelstrike_transversality(state) - self.transversality_eta,
            )

        def validated_reset(state: np.ndarray) -> np.ndarray:
            point = np.asarray(state, dtype=np.float64)
            if not self.is_admissible_heelstrike(point):
                raise InadmissibleHeelstrikeError(
                    "event solver stopped outside the declared transverse "
                    "heel-strike guard"
                )
            return reset_map(point)

        event.terminal = True
        event.direction = 1
        self.event_function = event
        self.system = _GarciaHybridSystem(
            ode=ode,
            event_function=event,
            reset_map=validated_reset,
            domain_bounds=self.domain_bounds,
            max_jumps=max_jumps,
            event_direction=1,
            rtol=rtol,
            atol=atol,
            guard_validator=self.is_admissible_heelstrike,
        )

    def is_admissible_heelstrike(self, state: np.ndarray, *, atol: float = 1e-8) -> bool:
        """Check the closed, transverse heel-strike conditions."""

        return bool(
            abs(heelstrike_gap(state)) <= atol
            and state[0] <= -self.guard_delta + atol
            and heelstrike_transversality(state)
            >= self.transversality_eta - atol
        )


class GuardAlignedGarciaPassiveWalker:
    """Coordinate-conjugate Garcia walker whose base guard is ``q=0``.

    The object exposes the same ``system`` and heel-strike validation interface
    as :class:`GarciaPassiveWalker`.  Its state is
    ``(theta, omega, q, nu)``.  ``physical_domain_bounds`` records the original
    chart and ``domain_bounds`` is its guard-aligned rectangular hull.
    """

    def __init__(
        self,
        *,
        gamma: float = DEFAULT_GAMMA,
        guard_delta: float = DEFAULT_GUARD_DELTA,
        transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
        physical_domain_bounds: list[tuple[float, float]] | None = None,
        domain_bounds: list[tuple[float, float]] | None = None,
        max_jumps: int = 20,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if guard_delta <= 0.0:
            raise ValueError("guard_delta must be positive")
        if transversality_eta <= 0.0:
            raise ValueError("transversality_eta must be positive")
        self.gamma = float(gamma)
        self.guard_delta = float(guard_delta)
        self.transversality_eta = float(transversality_eta)
        self.physical_domain_bounds = list(
            physical_domain_bounds or PHYSICAL_DOMAIN_BOUNDS
        )
        self.domain_bounds = domain_bounds or guard_aligned_domain_hull(
            self.physical_domain_bounds
        )

        def ode(time: float, state: np.ndarray) -> np.ndarray:
            return guard_aligned_ode_fun(time, state, self)

        def event(_time: float, state: np.ndarray) -> float:
            theta, _, q, nu = np.asarray(state, dtype=np.float64)
            return min(
                float(q),
                -float(theta) - self.guard_delta,
                float(nu) - self.transversality_eta,
            )

        def validated_reset(state: np.ndarray) -> np.ndarray:
            point = np.asarray(state, dtype=np.float64)
            if not self.is_admissible_heelstrike(point):
                raise InadmissibleHeelstrikeError(
                    "event solver stopped outside the declared transverse "
                    "guard-aligned heel-strike patch"
                )
            return guard_aligned_reset_map(point)

        event.terminal = True
        event.direction = 1
        self.event_function = event
        self.system = _GarciaHybridSystem(
            ode=ode,
            event_function=event,
            reset_map=validated_reset,
            domain_bounds=self.domain_bounds,
            max_jumps=max_jumps,
            event_direction=1,
            rtol=rtol,
            atol=atol,
            guard_validator=self.is_admissible_heelstrike,
        )

    @staticmethod
    def encode_physical_state(state: np.ndarray) -> np.ndarray:
        return encode_guard_aligned_state(state)

    @staticmethod
    def decode_physical_state(state: np.ndarray) -> np.ndarray:
        return decode_guard_aligned_state(state)

    def is_admissible_heelstrike(
        self,
        state: np.ndarray,
        *,
        atol: float = 1e-8,
    ) -> bool:
        theta, _, q, nu = np.asarray(state, dtype=np.float64)
        return bool(
            abs(float(q)) <= atol
            and float(theta) <= -self.guard_delta + atol
            and float(nu) >= self.transversality_eta - atol
        )


def build_garcia_passive_walker_suspension_pipeline(
    *,
    t_star: float = DEFAULT_T_STAR,
    handle_slabs: int = DEFAULT_HANDLE_SLABS,
    sample_count: int = 420,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    compute_cmgdb: bool = False,
) -> FixedTimeSuspensionPipelineResult:
    """Compute the sampled fixed-time relation around the period-two gait."""

    if not np.isclose(gamma, DEFAULT_GAMMA, rtol=0.0, atol=1e-14):
        raise ValueError(
            "the stored period-two regression seed is calibrated at gamma=0.0172; "
            "continue/re-solve the gait before running another slope",
        )
    walker = GarciaPassiveWalker(
        gamma=gamma,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
        max_jumps=12,
    )
    grid = Grid(
        bounds=[list(bounds) for bounds in walker.domain_bounds],
        subdivisions=[12, 12, 12, 12],
    )
    initial = post_impact_state(*PERIOD_TWO_POINT_A)
    two_steps = walker.system.simulate(
        initial,
        (0.0, 10.5),
        max_jumps=3,
        dense_output=True,
        max_step=0.01,
    )
    if len(two_steps.jump_times) < 2:
        raise RuntimeError("the Garcia regression seed did not complete two heel strikes")
    first_guard, first_reset = two_steps.jump_states[0]
    second_guard, second_reset = two_steps.jump_states[1]
    if not walker.is_admissible_heelstrike(first_guard):
        raise RuntimeError("the first detected event is not an admissible heel strike")
    if not walker.is_admissible_heelstrike(second_guard):
        raise RuntimeError("the second detected event is not an admissible heel strike")

    first_flow_time = float(two_steps.jump_times[0])
    second_flow_time = float(two_steps.jump_times[1] - two_steps.jump_times[0])
    orbit_period = float(two_steps.jump_times[1] + 2.0)
    fixed_time_witness = PhysicalEndpointWitness(
        name="period_two_gait_fixed_time_endpoint",
        initial_state=tuple(float(value) for value in initial),
        endpoint=simulate_suspension_endpoint(
            walker.system,
            initial,
            t_star,
            max_jumps=3,
            max_step=0.01,
        ),
    )
    handle_witness = PhysicalEndpointWitness(
        name="first_heelstrike_handle_midpoint",
        initial_state=tuple(float(value) for value in initial),
        endpoint=simulate_suspension_endpoint(
            walker.system,
            initial,
            first_flow_time + 0.5,
            max_jumps=3,
            max_step=0.01,
        ),
    )
    cemetery = CemeteryCell("garcia_failure_cemetery")
    diagnostics = {
        "source": "Garcia et al. (1998), DOI 10.1115/1.2798313",
        "gamma": float(gamma),
        "guard_delta": float(guard_delta),
        "transversality_eta": float(transversality_eta),
        "period_two_point_a": PERIOD_TWO_POINT_A,
        "period_two_point_b": PERIOD_TWO_POINT_B,
        "first_postimpact_residual_to_b": float(
            np.linalg.norm(first_reset[:2] - np.asarray(PERIOD_TWO_POINT_B))
        ),
        "second_postimpact_residual_to_a": float(
            np.linalg.norm(second_reset[:2] - np.asarray(PERIOD_TWO_POINT_A))
        ),
        "first_flow_time": first_flow_time,
        "second_flow_time": second_flow_time,
        "first_guard_gap": heelstrike_gap(first_guard),
        "second_guard_gap": heelstrike_gap(second_guard),
        "first_guard_transversality": heelstrike_transversality(first_guard),
        "second_guard_transversality": heelstrike_transversality(second_guard),
        "failure_policy": "one absorbing cemetery cell",
        "raw_failure_labels": RAW_FAILURE_LABELS,
        "failure_cell_status": (
            "explicit compactification smoke model; not generated by ODE propagation"
        ),
    }
    return materialize_periodic_orbit_relation(
        model_name=MODEL_NAME,
        system=walker.system,
        initial_state=initial,
        grid=grid,
        t_star=t_star,
        orbit_period=orbit_period,
        handle_slabs=handle_slabs,
        sample_count=sample_count,
        max_jumps=8,
        max_step=0.01,
        skeleton_flow_arcs=2,
        witnesses=(fixed_time_witness, handle_witness),
        diagnostics=diagnostics,
        failure_cells=(cemetery,),
        compute_cmgdb=compute_cmgdb,
    )


if __name__ == "__main__":
    print(build_garcia_passive_walker_suspension_pipeline().summary())


__all__ = [
    "MODEL_NAME",
    "DEFAULT_T_STAR",
    "DEFAULT_HANDLE_SLABS",
    "DEFAULT_GAMMA",
    "DEFAULT_GUARD_DELTA",
    "DEFAULT_TRANSVERSALITY_ETA",
    "PERIOD_TWO_POINT_A",
    "PERIOD_TWO_POINT_B",
    "RAW_FAILURE_LABELS",
    "PHYSICAL_DOMAIN_BOUNDS",
    "GUARD_ALIGNED_DOMAIN_BOUNDS",
    "InadmissibleHeelstrikeError",
    "GarciaPassiveWalker",
    "GuardAlignedGarciaPassiveWalker",
    "ode_fun",
    "heelstrike_gap",
    "heelstrike_transversality",
    "reset_map",
    "post_impact_state",
    "guard_aligned_domain_hull",
    "encode_guard_aligned_state",
    "decode_guard_aligned_state",
    "encode_guard_aligned_velocity",
    "guard_aligned_ode_fun",
    "guard_aligned_reset_map",
    "guard_aligned_post_impact_state",
    "build_garcia_passive_walker_suspension_pipeline",
]
