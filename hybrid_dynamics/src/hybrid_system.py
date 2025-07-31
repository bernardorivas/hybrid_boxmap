"""
Core hybrid dynamical system implementation with event detection.

This module provides the main HybridSystem class that handles continuous dynamics,
discrete events, and reset maps.
"""

from typing import Callable

import numpy as np

from .hybrid_time import HybridTime
from .hybrid_trajectory import HybridTrajectory


class HybridSystem:
    """Represents a hybrid dynamical system with continuous flow and discrete jumps.

    A hybrid system consists of:
    - Continuous dynamics: dx/dt = f(x, t)
    - Event function: g(x) = 0 triggers discrete jumps
    - Reset map: x_new = r(x_old) after event detection
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
    ):
        """Initialize hybrid system.

        Args:
            ode: Continuous dynamics function f(t, x) -> dx/dt
            event_function: Guard condition g(x) -> scalar (zero crossing triggers jump).
            reset_map: Discrete map r(x) -> x_new after event
            domain_bounds: Valid state space bounds [(x1_min, x1_max), ...]
            max_jumps: Maximum allowed discrete transitions
            event_direction: Direction of event detection (-1: neg to pos, 0: both, 1: pos to neg)
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
        """
        self.ode = ode
        self.event_function = event_function
        self.reset_map = reset_map
        self.domain_bounds = domain_bounds
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
        """Check if state is within domain bounds."""
        if self.domain_bounds is None:
            return True

        if len(state) != len(self.domain_bounds):
            raise ValueError("State dimension must match domain bounds dimension")

        return all(
            lower <= state[i] <= upper
            for i, (lower, upper) in enumerate(self.domain_bounds)
        )

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
