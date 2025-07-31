"""
Hybrid trajectory representation with complete time domain tracking.

This module provides classes for representing hybrid trajectories, which consist
of continuous trajectory segments connected by discrete jumps.
"""

import pickle
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .config import config
from .hybrid_time import HybridTime, HybridTimeInterval

if TYPE_CHECKING:
    from scipy.integrate._ivp.ivp import OdeResult

    from .hybrid_system import HybridSystem


@dataclass
class TrajectorySegment:
    """Represents a continuous piece of a hybrid trajectory

    Can represent either a normal segment with scipy solution or a trivial
    segment (single point) for instant jumps.

    Attributes:
        scipy_solution: The actual solve_ivp solution object (optional)
        jump_index: Which jump index this segment corresponds to
        trivial_time: Time for trivial segment (optional)
        trivial_state: State for trivial segment (optional)
        t_start: Start time of this segment
        t_end: End time of this segment
    """

    jump_index: int
    scipy_solution: Optional["OdeResult"] = None
    trivial_time: Optional[float] = None
    trivial_state: Optional[np.ndarray] = None
    t_start: float = field(init=False)
    t_end: float = field(init=False)

    def __post_init__(self):
        """Initialize derived attributes and validate data."""
        if self.trivial_time is not None and self.trivial_state is not None:
            # Trivial segment case
            self.t_start = float(self.trivial_time)
            self.t_end = float(self.trivial_time)
        elif self.scipy_solution is not None:
            # Normal segment case
            if not hasattr(self.scipy_solution, "t") or not hasattr(
                self.scipy_solution, "y",
            ):
                raise ValueError(
                    "Invalid scipy solution object: missing 't' or 'y' attributes",
                )
            if (
                not hasattr(self.scipy_solution, "sol")
                or self.scipy_solution.sol is None
            ):
                raise ValueError("Scipy solution object must have a 'sol' attribute.")
            if len(self.scipy_solution.t) == 0:
                raise ValueError("Empty solution time array in scipy_solution")

            self.t_start = float(self.scipy_solution.t[0])
            self.t_end = float(self.scipy_solution.t[-1])
        else:
            raise ValueError(
                "Must provide either scipy_solution or both trivial_time and trivial_state",
            )

        if self.jump_index < 0:
            raise ValueError("Jump index must be non-negative")

    def duration(self) -> float:
        """Get the duration of this segment."""
        return self.t_end - self.t_start

    @property
    def time_values(self) -> np.ndarray:
        """Get time values from scipy solution or trivial segment."""
        if self.trivial_time is not None:
            return np.array([self.trivial_time])
        return self.scipy_solution.t

    @property
    def state_values(self) -> np.ndarray:
        """Get state values from scipy solution with shape (n_points, state_dim)."""
        if self.trivial_state is not None:
            return self.trivial_state.reshape(1, -1)
        return self.scipy_solution.y.T

    @property
    def solution(self) -> Optional[Callable]:
        """Get interpolation function for this segment."""
        if self.trivial_state is not None:
            # Return a function that always returns the trivial state
            return lambda t: self.trivial_state
        return self.scipy_solution.sol if self.scipy_solution else None

    def to_hybrid_time_interval(self) -> HybridTimeInterval:
        """Convert this segment to a HybridTimeInterval."""
        return HybridTimeInterval(
            t_start=self.t_start, t_end=self.t_end, jump_index=self.jump_index,
        )


@dataclass
class HybridTrajectory:
    """Complete hybrid trajectory with full time domain tracking.

    Provides scipy-like interface leveraging dense output from solve_ivp.

    Attributes:
        segments: list of continuous trajectory segments (each with scipy solution)
        jump_times: list of times when jumps occurred
        jump_states: list of tuples (state_before, state_after) for each jump
        time_domain: list of HybridTimeInterval objects representing domain
    """

    segments: list[TrajectorySegment] = field(default_factory=list)
    jump_times: list[float] = field(default_factory=list)
    jump_states: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    time_domain: list[HybridTimeInterval] = field(default_factory=list)

    def __post_init__(self):
        """Validate trajectory consistency."""
        if self.segments:
            self._update_time_domain()
            self._validate_consistency()

    def _update_time_domain(self):
        """Update time domain from segments."""
        self.time_domain = (
            [segment.to_hybrid_time_interval() for segment in self.segments]
            if self.segments
            else []
        )

    def _validate_consistency(self):
        """Validate internal consistency of trajectory data."""
        if not self.segments:
            return

        for i in range(len(self.segments) - 1):
            current_seg, next_seg = self.segments[i], self.segments[i + 1]
            if current_seg.jump_index >= next_seg.jump_index:
                raise ValueError("Segments must be ordered by increasing jump index")
            if next_seg.jump_index != current_seg.jump_index + 1:
                raise ValueError("Jump indices must be consecutive")

        expected_jumps = len(self.segments) - 1
        if len(self.jump_times) != expected_jumps:
            raise ValueError(
                f"Expected {expected_jumps} jump times, got {len(self.jump_times)}",
            )
        if len(self.jump_states) != expected_jumps:
            raise ValueError(
                f"Expected {expected_jumps} jump states, got {len(self.jump_states)}",
            )

    @property
    def t(self) -> np.ndarray:
        """Scipy-like time array for smooth trajectory access (concatenated segment times)."""
        return self.get_all_times()

    @property
    def y(self) -> np.ndarray:
        """Scipy-like state array with shape (state_dim, n_points)."""
        states = self.get_all_states()
        if states.size == 0:
            return np.array([])
        return states.T if states.ndim > 1 else states.reshape(1, -1)

    @property
    def x(self) -> np.ndarray:
        """Alternative name for state array (same as y)."""
        return self.y

    @property
    def t_events(self) -> list[np.ndarray]:
        """Scipy-like event times array (returns list with jump times)."""
        return [np.array(self.jump_times)]

    def interpolate(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate trajectory at given time(s) using scipy dense output from segments."""
        if np.isscalar(t):
            return self._interpolate_single_time(float(t))

        t_array = np.asarray(t)
        results = [
            self._interpolate_single_time(float(time_point)) for time_point in t_array
        ]
        return np.array(results)

    def _interpolate_single_time(self, t: float) -> np.ndarray:
        """Interpolate state at a single time point by finding the correct segment."""
        for segment in self.segments:
            if segment.t_start <= t <= segment.t_end:
                if segment.solution is not None:
                    return segment.solution(t)
                raise ValueError(f"Segment at time {t} has no solution function")
        raise ValueError(f"Time {t} is outside trajectory domain")

    def get_scipy_like_solution(
        self, resample_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get scipy-like (t, y) arrays, optionally resampled at uniform intervals."""
        if not self.segments:
            return np.array([]), np.array([])

        if resample_dt is None:
            return self.get_all_times(), self.get_all_states()

        t_start = self.segments[0].t_start
        t_end = self.segments[-1].t_end
        new_times = np.arange(t_start, t_end + resample_dt * 0.5, resample_dt)
        new_states = self.interpolate(new_times)
        return new_times, new_states

    @property
    def num_jumps(self) -> int:
        """Number of discrete jumps in trajectory."""
        return len(self.jump_times)

    @property
    def total_duration(self) -> float:
        """Total continuous time duration across all segments."""
        return (
            sum(segment.duration() for segment in self.segments)
            if self.segments
            else 0.0
        )

    @property
    def initial_state(self) -> np.ndarray | None:
        """Initial state of trajectory."""
        return self.segments[0].state_values[0].copy() if self.segments else None

    @property
    def final_state(self) -> np.ndarray | None:
        """Final state of trajectory."""
        return self.segments[-1].state_values[-1].copy() if self.segments else None

    def add_segment(self, segment: TrajectorySegment):
        """Add a new trajectory segment and update domain/consistency."""
        if self.segments:
            expected_jump_index = self.segments[-1].jump_index + 1
            if segment.jump_index != expected_jump_index:
                raise ValueError(
                    f"Expected jump index {expected_jump_index}, got {segment.jump_index}",
                )
        elif segment.jump_index != 0:
            raise ValueError("First segment must have jump index 0")

        self.segments.append(segment)
        self._update_time_domain()

    def add_jump(
        self, jump_time: float, state_before: np.ndarray, state_after: np.ndarray,
    ):
        """Add jump information."""
        self.jump_times.append(jump_time)
        self.jump_states.append((state_before.copy(), state_after.copy()))

    def get_state_at_hybrid_time(self, ht: HybridTime) -> np.ndarray:
        """Retrieve state at specific hybrid time (t, j)."""
        matching_segment = next(
            (seg for seg in self.segments if seg.jump_index == ht.discrete), None,
        )
        if not matching_segment:
            raise ValueError(f"No segment found for jump index {ht.discrete}")

        if not (matching_segment.t_start <= ht.continuous <= matching_segment.t_end):
            raise ValueError(
                f"Time {ht.continuous} not in segment [{matching_segment.t_start}, {matching_segment.t_end}] for jump index {ht.discrete}",
            )

        return matching_segment.scipy_solution.sol(ht.continuous)

    def get_segment_by_jump_index(self, jump_index: int) -> TrajectorySegment | None:
        """Get trajectory segment by jump index."""
        return next(
            (seg for seg in self.segments if seg.jump_index == jump_index), None,
        )

    def to_hybrid_time_notation(self) -> str:
        """Express trajectory domain as union of HybridTimeInterval notations."""
        return (
            HybridTimeInterval.union_notation(self.time_domain)
            if self.time_domain
            else "∅"
        )

    def get_all_states(self) -> np.ndarray:
        """Get all state values concatenated into single array (N, state_dim)."""
        if not self.segments:
            return np.array([])

        all_states_list = []
        expected_state_dim = None
        for i, segment in enumerate(self.segments):
            state_vals = segment.state_values
            if state_vals.ndim == 1:
                state_vals = state_vals.reshape(-1, 1)

            if expected_state_dim is None:
                expected_state_dim = state_vals.shape[1]
            elif state_vals.shape[1] != expected_state_dim:
                raise ValueError(
                    f"Segment {i} has incompatible state dimension: "
                    f"expected {expected_state_dim}, got {state_vals.shape[1]}",
                )
            all_states_list.append(state_vals)

        return np.vstack(all_states_list) if all_states_list else np.array([])

    def get_all_times(self) -> np.ndarray:
        """Get all time values concatenated into single array."""
        if not self.segments:
            return np.array([])
        return np.concatenate([segment.time_values for segment in self.segments])

    def save(self, filename: str):
        """Save trajectory data to a file using pickle."""
        data_to_save = {
            "segments": self.segments,
            "jump_times": self.jump_times,
            "jump_states": self.jump_states,
            "time_domain": self.time_domain,
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load(cls, filename: str) -> "HybridTrajectory":
        """Load trajectory data from a file."""
        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)

        trajectory = cls(
            segments=loaded_data.get("segments", []),
            jump_times=loaded_data.get("jump_times", []),
            jump_states=loaded_data.get("jump_states", []),
        )
        if not trajectory.segments and "time_domain" in loaded_data:
            trajectory.time_domain = loaded_data["time_domain"]
        elif trajectory.segments:
            trajectory.__post_init__()

        return trajectory

    def __str__(self) -> str:
        """String representation of trajectory."""
        if not self.segments:
            return "Empty HybridTrajectory"
        return f"HybridTrajectory with {len(self.segments)} segments, {self.num_jumps} jumps, domain: {self.to_hybrid_time_notation()}"

    def __len__(self) -> int:
        """Number of segments in trajectory."""
        return len(self.segments)

    @classmethod
    def compute_trajectory(
        cls,
        system: "HybridSystem",
        initial_state: np.ndarray,
        time_span: tuple[float, float],
        max_jumps: int | None = None,
        dense_output: bool = True,
        max_step: float | None = None,
        debug_info: dict | None = None,
        jump_time_penalty: bool = False,
        jump_time_penalty_epsilon: float | None = None,
    ) -> "HybridTrajectory":
        """Compute hybrid trajectory from initial condition using given system.
        
        Args:
            system: The hybrid system to simulate
            initial_state: Initial state vector
            time_span: (t_start, t_end) integration time span
            max_jumps: Maximum allowed discrete transitions
            dense_output: Whether to use dense output for smooth interpolation
            max_step: Maximum step size for integration
            debug_info: Optional dictionary to store debugging information
            jump_time_penalty: If True, each jump consumes time from the total duration
            jump_time_penalty_epsilon: Time deducted for each jump (defaults to config value)
        
        Returns:
            HybridTrajectory containing complete simulation results
        """
        if max_jumps is None:
            max_jumps = system.max_jumps

        # Use default epsilon from config if not provided
        if jump_time_penalty_epsilon is None:
            jump_time_penalty_epsilon = config.simulation.jump_time_penalty_epsilon

        t_start, t_end = time_span
        current_state = initial_state.copy()
        current_time = t_start
        jump_count = 0
        
        # Track remaining time for jump penalty mode
        total_duration = t_end - t_start
        remaining_time = total_duration

        trajectory = cls()

        if not system._check_domain_bounds(current_state):
            raise ValueError("Initial state outside domain bounds")

        # Check if initial state already satisfies event condition
        initial_event_value = system.evaluate_event_function(
            current_time, current_state,
        )
        # Determine if we should jump based on event value and direction
        event_direction = getattr(system.event_function, "direction", 0)
        should_jump = False

        if event_direction == -1:  # Trigger on negative to positive crossing
            # Jump if we're on the negative side (event condition satisfied)
            should_jump = initial_event_value <= 0
        elif event_direction == 1:  # Trigger on positive to negative crossing
            # Jump if we're on the positive side (event condition satisfied)
            should_jump = initial_event_value >= 0
        else:  # event_direction == 0, trigger on any crossing
            # For bidirectional events, we need to be more careful
            # Jump if event value is already negative (in the jump region)
            # This handles cases like thermostat where being below threshold should trigger jump
            should_jump = initial_event_value < -config.simulation.event_tolerance

        if should_jump:
            # Create trivial initial segment before jumping
            trivial_segment = TrajectorySegment(
                jump_index=0,
                trivial_time=current_time,
                trivial_state=np.array(current_state, dtype=np.float64),
            )
            trajectory.add_segment(trivial_segment)

            # Apply reset immediately
            try:
                # Check if we have enough time for initial jump in penalty mode
                if jump_time_penalty and remaining_time < jump_time_penalty_epsilon:
                    # Not enough time for jump, just return initial segment
                    return trajectory
                
                state_after_jump = system.reset_map(current_state)
                if not system._check_domain_bounds(state_after_jump):
                    warnings.warn(
                        "Post-jump state outside domain bounds", RuntimeWarning,
                    )

                # Deduct time for initial jump if in penalty mode
                if jump_time_penalty:
                    remaining_time -= jump_time_penalty_epsilon

                trajectory.add_jump(current_time, current_state, state_after_jump)
                current_state = state_after_jump
                jump_count = 1
            except Exception as e:
                warnings.warn(f"Initial reset map failed: {str(e)}", RuntimeWarning)

        while current_time < t_end and jump_count <= max_jumps:
            # Calculate effective end time based on jump penalty mode
            effective_t_end = t_end
            if jump_time_penalty:
                # Adjust end time based on remaining time
                effective_t_end = current_time + remaining_time
                if remaining_time <= 0:
                    break
            
            integration_span = (current_time, effective_t_end)

            try:
                solve_ivp_args = {
                    "fun": system.ode,
                    "t_span": integration_span,
                    "y0": current_state,
                    "events": system.event_function,
                    "dense_output": dense_output,
                    "rtol": system.rtol,
                    "atol": system.atol,
                }
                if max_step is not None:
                    solve_ivp_args["max_step"] = max_step

                sol = solve_ivp(**solve_ivp_args)

                if not sol.success:
                    warnings.warn(f"Integration failed: {sol.message}", RuntimeWarning)
                    break
            except Exception as e:
                warnings.warn(f"Integration error: {str(e)}", RuntimeWarning)
                break

            segment = TrajectorySegment(scipy_solution=sol, jump_index=jump_count)

            num_samples = min(10, len(segment.state_values))
            sample_indices = np.linspace(
                0, len(segment.state_values) - 1, num_samples, dtype=int,
            )
            for idx in sample_indices:
                if not system._check_domain_bounds(segment.state_values[idx]):
                    break

            trajectory.add_segment(segment)

            if sol.t_events and sol.t_events[0].size > 0:
                event_time = sol.t_events[0][0]
                event_time = min(event_time, sol.t[-1])
                state_before_jump = (
                    sol.sol(event_time)
                    if hasattr(sol, "sol") and sol.sol is not None
                    else sol.y_events[0][0]
                )

                try:
                    state_after_jump = system.reset_map(state_before_jump)
                except Exception as e:
                    warnings.warn(f"Reset map failed: {str(e)}", RuntimeWarning)
                    break

                if not system._check_domain_bounds(state_after_jump):
                    warnings.warn(
                        "Post-jump state outside domain bounds", RuntimeWarning,
                    )
                    # Continue simulation even outside domain bounds

                # Handle jump time penalty
                if jump_time_penalty:
                    # Calculate time elapsed in this segment
                    time_elapsed = event_time - current_time
                    remaining_time -= time_elapsed
                    
                    # Check if we have enough time for the jump penalty
                    if remaining_time < jump_time_penalty_epsilon:
                        # Not enough time for jump, stop here
                        break
                    
                    # Deduct jump penalty epsilon time for the jump
                    remaining_time -= jump_time_penalty_epsilon

                trajectory.add_jump(event_time, state_before_jump, state_after_jump)
                current_state = state_after_jump
                current_time = event_time
                jump_count += 1

                if jump_count > max_jumps:
                    break
            else:
                break

        return trajectory

    @classmethod
    def compute_from_hybrid_time(
        cls,
        system: "HybridSystem",
        initial_hybrid_time: HybridTime,
        initial_state: np.ndarray,
        duration: float,
        max_jumps: int | None = None,
        jump_time_penalty: bool = False,
        jump_time_penalty_epsilon: float | None = None,
    ) -> "HybridTrajectory":
        """Compute trajectory from a specific hybrid time point, adjusting jump indices.
        
        Args:
            system: The hybrid system to simulate
            initial_hybrid_time: Starting hybrid time (t, j)
            initial_state: State at the initial hybrid time
            duration: How long to simulate forward in continuous time
            max_jumps: Maximum allowed discrete transitions
            jump_time_penalty: If True, each jump consumes time from the total duration
            jump_time_penalty_epsilon: Time deducted for each jump (defaults to config value)
        
        Returns:
            HybridTrajectory with jump indices offset by initial_hybrid_time.discrete
        """
        if max_jumps is None:
            max_jumps = system.max_jumps

        time_span = (
            initial_hybrid_time.continuous,
            initial_hybrid_time.continuous + duration,
        )
        trajectory = cls.compute_trajectory(
            system, initial_state, time_span, max_jumps, 
            jump_time_penalty=jump_time_penalty,
            jump_time_penalty_epsilon=jump_time_penalty_epsilon
        )

        offset = initial_hybrid_time.discrete
        if offset > 0:
            for segment in trajectory.segments:
                segment.jump_index += offset
            trajectory._update_time_domain()

        return trajectory

    def validate(self) -> bool:
        """Check for consistency in trajectory data."""
        if not self.segments:
            return True

        # Check: Ensure all state vectors have the same dimension
        expected_state_dim = None
        for segment in self.segments:
            state_vals = segment.state_values
            if state_vals.ndim != 2:
                raise ValueError(
                    f"Segment state_values must be 2D, got {state_vals.ndim}D",
                )

            if expected_state_dim is None:
                expected_state_dim = state_vals.shape[1]
            elif state_vals.shape[1] != expected_state_dim:
                raise ValueError(
                    "State dimension mismatch in segments: "
                    f"expected {expected_state_dim}, got {state_vals.shape[1]}",
                )

        return True
