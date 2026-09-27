"""
Core hybrid dynamical system implementation with event detection.

This module provides the main HybridSystem class that handles continuous dynamics,
discrete events, and reset maps.
"""

from typing import Callable

import numpy as np

from .config import config
from .hybrid_time import HybridTime
from .hybrid_trajectory import HybridTrajectory

# Guard membership of a point on the event surface is decided by the sign of
# the event function's change along one short forward-Euler step of the flow.
# The step moves the fastest coordinate by ``_SQRT_EPS * max(1, |x|)``; a
# change below ``_TANGENCY_FRACTION`` of that displacement counts as tangency.
_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))
_TANGENCY_FRACTION = 1.0e-6


class HybridSystem:
    """Represents a hybrid dynamical system with continuous flow and discrete jumps.

    A hybrid system consists of:
    - Continuous dynamics: dx/dt = f(x, t)
    - Event function: g(x) = 0 triggers discrete jumps
    - Reset map: x_new = r(x_old) after event detection

    The guard is the set of points of the event surface ``g = 0`` at which
    the flow crosses (or touches) the surface in the event direction ``d``:
    ``G = {g = 0, d * (dg/dt) >= 0}`` for ``d = +1`` or ``-1``, and the whole
    surface for ``d = 0``.  For the rimless wheel (``g = theta - alpha -
    gamma``, ``d = +1``) this is ``{theta = alpha + gamma, omega >= 0}``; for
    the bouncing ball (``g = h`` if ``v <= 0`` else ``1``, ``d = -1``) it is
    ``{h = 0, v <= 0}``.  An explicit ``guard_predicate`` replaces this rule.
    See :meth:`on_guard` and :meth:`jumps_at_start`.
    """

    def __init__(
        self,
        ode: Callable[[float, np.ndarray], np.ndarray],
        event_function: Callable[[float, np.ndarray], float],
        reset_map: Callable[[np.ndarray], np.ndarray],
        domain_bounds: list[tuple[float, float]] | None = None,
        max_jumps: int = 100,
        event_direction: int = -1,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        domain_predicate: Callable[[np.ndarray], bool] | None = None,
        guard_predicate: Callable[[np.ndarray], bool] | None = None,
    ):
        """Initialize hybrid system.

        Args:
            ode: Continuous dynamics function f(t, x) -> dx/dt
            event_function: Guard condition g(x) -> scalar (zero crossing triggers jump).
            reset_map: Discrete map r(x) -> x_new after event
            domain_bounds: Valid state space bounds [(x1_min, x1_max), ...]
            max_jumps: Maximum allowed discrete transitions
            event_direction: SciPy event-crossing direction
                (-1: positive to negative, 0: both, 1: negative to positive).
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
            domain_predicate: Optional additional membership test for a
                nonrectangular state space.  When both domain descriptions are
                supplied, a state must satisfy both.
            guard_predicate: Optional explicit membership test of the guard.
                When supplied, it alone decides whether an initial state
                jumps before flowing (:meth:`jumps_at_start`) and whether a
                point lies on the guard (:meth:`on_guard`).  Jumps during
                integration are still located by ``event_function``.
        """
        self.ode = ode
        self.event_function = event_function
        self.reset_map = reset_map
        self.domain_bounds = domain_bounds
        self.domain_predicate = domain_predicate
        self.guard_predicate = guard_predicate
        self.max_jumps = max_jumps
        self.rtol = rtol
        self.atol = atol
        self.event_direction = event_direction

        # Configure event for scipy
        self._setup_event_detection()

    def _setup_event_detection(self):
        """Ensure event_function has 'terminal' and 'direction' attributes for scipy."""
        # Add scipy event attributes directly to the event_function if not already present
        if not hasattr(self.event_function, "terminal"):
            self.event_function.terminal = True

        if not hasattr(self.event_function, "direction"):
            self.event_function.direction = self.event_direction

    def _check_domain_bounds(self, state: np.ndarray) -> bool:
        """Check rectangular bounds and any additional domain predicate."""
        point = np.asarray(state, dtype=float)
        if point.ndim != 1 or not np.all(np.isfinite(point)):
            return False

        if self.domain_bounds is not None:
            if len(point) != len(self.domain_bounds):
                raise ValueError("State dimension must match domain bounds dimension")

            if not all(
                lower <= point[i] <= upper
                for i, (lower, upper) in enumerate(self.domain_bounds)
            ):
                return False
        if self.domain_predicate is not None:
            try:
                return bool(self.domain_predicate(point))
            except (TypeError, ValueError, FloatingPointError):
                return False
        return True

    def simulate(
        self,
        initial_state: np.ndarray,
        time_span: tuple[float, float],
        max_jumps: int | None = None,
        dense_output: bool = True,
        max_step: float | None = None,
        debug_info: dict | None = None,
        jump_time_penalty: bool = False,
        jump_time_penalty_epsilon: float | None = None,
    ) -> HybridTrajectory:
        """Simulate hybrid trajectory from initial condition.

        Args:
            initial_state: Initial state vector
            time_span: (t_start, t_end) integration time span
            max_jumps: Override default max_jumps for this simulation
            dense_output: Whether to use dense output for smooth interpolation
            max_step: Maximum step size for integration
            debug_info: An optional dictionary to store debugging information.
            jump_time_penalty: If True, each jump consumes time from the total duration
            jump_time_penalty_epsilon: Time deducted for each jump (defaults to config value)

        Returns:
            HybridTrajectory containing complete simulation results
        """
        return HybridTrajectory.compute_trajectory(
            system=self,
            initial_state=initial_state,
            time_span=time_span,
            max_jumps=max_jumps,
            dense_output=dense_output,
            max_step=max_step,
            debug_info=debug_info,
            jump_time_penalty=jump_time_penalty,
            jump_time_penalty_epsilon=jump_time_penalty_epsilon,
        )

    def simulate_from_hybrid_time(
        self,
        initial_hybrid_time: HybridTime,
        initial_state: np.ndarray,
        duration: float,
        max_jumps: int | None = None,
        jump_time_penalty: bool = False,
        jump_time_penalty_epsilon: float | None = None,
    ) -> HybridTrajectory:
        """Simulate from a specific hybrid time point.

        Args:
            initial_hybrid_time: Starting hybrid time (t, j)
            initial_state: Initial state vector
            duration: How long to simulate in continuous time
            max_jumps: Maximum additional jumps allowed
            jump_time_penalty: If True, each jump consumes time from the total duration
            jump_time_penalty_epsilon: Time deducted for each jump (defaults to config value)

        Returns:
            HybridTrajectory starting from specified hybrid time
        """
        return HybridTrajectory.compute_from_hybrid_time(
            system=self,
            initial_hybrid_time=initial_hybrid_time,
            initial_state=initial_state,
            duration=duration,
            max_jumps=max_jumps,
            jump_time_penalty=jump_time_penalty,
            jump_time_penalty_epsilon=jump_time_penalty_epsilon,
        )

    def is_valid_state(self, state: np.ndarray) -> bool:
        """Check if a state is valid (within domain bounds).

        Args:
            state: State vector to validate

        Returns:
            True if state is valid
        """
        return self._check_domain_bounds(state)

    @property
    def event_crossing_direction(self) -> int:
        """SciPy crossing direction of ``event_function`` (``-1``, ``0``, or ``1``)."""

        return int(np.sign(getattr(self.event_function, "direction", 0)))

    def _event_change_sign(self, t: float, point: np.ndarray, value: float) -> int:
        """Sign of the change of ``g`` along the flow at ``point`` (0 for tangency).

        The event function is evaluated once more, at the end of a short
        forward-Euler step of the vector field.  A one-sided step is used
        because event functions are often defined piecewise off the guard
        (for example ``h if v <= 0 else 1`` for the bouncing ball).
        """

        velocity = np.asarray(self.ode(t, point), dtype=np.float64).reshape(-1)
        speed = float(np.max(np.abs(velocity))) if velocity.size else 0.0
        if not np.isfinite(speed) or speed == 0.0:
            return 0
        displacement = _SQRT_EPS * max(1.0, float(np.max(np.abs(point))))
        step = displacement / speed
        try:
            ahead = float(self.event_function(t + step, point + step * velocity))
        except (TypeError, ValueError, FloatingPointError):
            return 0
        change = ahead - value
        if not np.isfinite(change) or abs(change) <= _TANGENCY_FRACTION * displacement:
            return 0
        return 1 if change > 0.0 else -1

    def on_guard(
        self,
        t: float,
        state: np.ndarray,
        *,
        tolerance: float | None = None,
    ) -> bool:
        """Whether ``state`` lies on the guard ``G`` as the model defines it.

        With ``guard_predicate`` this is ``guard_predicate(state)``.
        Otherwise the point must lie on the event surface,
        ``|g(t, state)| <= tolerance`` (default
        ``config.simulation.event_tolerance``), and, for a directional event
        ``d = +1`` or ``-1``, the flow must not leave the surface against the
        event direction: ``d * (dg/dt) >= 0``, where a tangential flow
        (``dg/dt = 0``) counts as on the guard.  For ``d = 0`` every point of
        the surface is on the guard.
        """

        point = np.asarray(state, dtype=np.float64)
        if self.guard_predicate is not None:
            return bool(self.guard_predicate(point))
        tol = config.simulation.event_tolerance if tolerance is None else float(tolerance)
        value = float(self.evaluate_event_function(t, point))
        if not np.isfinite(value) or abs(value) > tol:
            return False
        direction = self.event_crossing_direction
        if direction == 0:
            return True
        return direction * self._event_change_sign(t, point, value) >= 0

    def jumps_at_start(self, t: float, state: np.ndarray) -> bool:
        """Whether a trajectory starting at ``state`` begins with a jump.

        With ``guard_predicate`` this is ``guard_predicate(state)``.
        Otherwise a point jumps if it lies on the guard (:meth:`on_guard`)
        or strictly past the event surface, on the side the event crosses
        into: ``d * g > tolerance`` for ``d = +1`` or ``-1``, and
        ``g < -tolerance`` for ``d = 0`` (the jump set ``{g <= 0}`` of the
        thermostat).  The side past the surface lies outside the state space
        of the ball, wheel, and neuron examples.

        A point of the event surface at which the flow moves against the
        event direction does not jump: for the rimless wheel,
        ``(alpha + gamma, omega)`` with ``omega < 0`` flows back.
        """

        point = np.asarray(state, dtype=np.float64)
        if self.guard_predicate is not None:
            return bool(self.guard_predicate(point))
        tol = config.simulation.event_tolerance
        value = float(self.evaluate_event_function(t, point))
        if not np.isfinite(value):
            return False
        direction = self.event_crossing_direction
        past = -value if direction == 0 else direction * value
        if past > tol:
            return True
        return self.on_guard(t, point, tolerance=tol)

    def evaluate_event_function(self, t: float, state: np.ndarray) -> float:
        """Evaluate the event function at given time and state.

        Args:
            t: Time point
            state: State vector

        Returns:
            Event function value
        """
        return self.event_function(t, state)

    def evaluate_ode(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the ODE at given time and state.

        Args:
            t: Time point
            state: State vector

        Returns:
            Time derivative of state
        """
        return self.ode(t, state)

    def apply_reset_map(self, state: np.ndarray) -> np.ndarray:
        """Apply reset map to state.

        Args:
            state: State before jump

        Returns:
            State after jump
        """
        return self.reset_map(state)

    def flow_with_jumps(self, point: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
        """
        Computes the state and jump count after a fixed time duration tau.

        This is a convenience wrapper around the simulator for use with `evaluate_grid`.

        Args:
            point: The initial state.
            tau: The time duration to simulate.

        Returns:
            A tuple containing (final_state, num_jumps).
            Returns (NaN vector, -1) on failure.
        """
        try:
            traj = self.simulate(point, (0, tau))
            if traj.total_duration >= tau:
                final_state = traj.interpolate(tau)
                num_jumps = traj.num_jumps
                return (final_state, num_jumps)
            return (np.full(len(point), np.nan), -1)
        except Exception:
            return (np.full(len(point), np.nan), -1)

    def __str__(self) -> str:
        """String representation of hybrid system."""
        bounds_str = (
            "unbounded"
            if self.domain_bounds is None
            else f"{len(self.domain_bounds)}D bounded"
        )
        return f"HybridSystem ({bounds_str}, max_jumps={self.max_jumps})"
