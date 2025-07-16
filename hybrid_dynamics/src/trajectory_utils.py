"""
Utility functions for trajectory manipulation and analysis.
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

from ..src.hybrid_trajectory import HybridTrajectory, TrajectorySegment


def create_perturbed_trajectory(
    system: "HybridSystem",
    initial_condition: np.ndarray,
    time_span: Tuple[float, float] = (0.0, 10.0),
    max_jumps: int = 50,
    epsilon: float = 0.005,
    n_dense_points: int = 100,
    n_perturbations: int = 8,
    segment_selection: str = "second_to_last",
) -> Tuple[HybridTrajectory, np.ndarray, int]:
    """
    Create a perturbed trajectory for cubical coverage by adding epsilon-neighborhood
    points around a selected segment of the original trajectory.

    Args:
        system: The HybridSystem to simulate
        initial_condition: Starting state for simulation
        time_span: (start_time, end_time) for simulation
        max_jumps: Maximum number of discrete jumps allowed
        epsilon: Perturbation radius for epsilon-neighborhood
        n_dense_points: Number of dense evaluation points along the segment
        n_perturbations: Number of perturbation directions around each point
        segment_selection: Strategy for selecting segment ("second_to_last", "first", "last")

    Returns:
        Tuple of (perturbed_trajectory, original_segment_states, jump_index)
    """

    # Simulate original trajectory
    trajectory = system.simulate(
        initial_condition, time_span, max_jumps=max_jumps, dense_output=True,
    )

    # Select target segment based on strategy
    if segment_selection == "second_to_last" and len(trajectory.segments) >= 2:
        target_segment = trajectory.segments[-2]
    elif segment_selection == "last":
        target_segment = trajectory.segments[-1]
    else:  # "first" or fallback
        target_segment = trajectory.segments[0]

    segment_start = target_segment.t_start
    segment_end = target_segment.t_end
    jump_index = target_segment.jump_index

    # Create dense trajectory evaluation for this specific segment
    t_dense = np.linspace(segment_start, segment_end, n_dense_points)
    states_dense = []

    for t in t_dense:
        try:
            state = target_segment.scipy_solution.sol(t)
            states_dense.append(state)
        except Exception:
            continue

    if states_dense:
        states_dense = np.array(states_dense)
    else:
        states_dense = target_segment.state_values

    # Create epsilon-neighborhood perturbations
    segment_states = (
        states_dense if len(states_dense) > 0 else target_segment.state_values
    )
    perturbed_points = []

    # Add original trajectory points
    perturbed_points.extend(segment_states)

    # Add perturbed points around each trajectory point
    angles = np.linspace(0, 2 * np.pi, n_perturbations, endpoint=False)
    state_dim = len(segment_states[0]) if len(segment_states) > 0 else 2

    for state in segment_states[::2]:  # Sample every 2nd point to avoid too many
        for angle in angles:
            # Create perturbation vector with same dimension as state
            perturbation = np.zeros(state_dim)
            perturbation[0] = epsilon * np.cos(angle)  # Perturb first dimension
            perturbation[1] = epsilon * np.sin(angle)  # Perturb second dimension
            perturbed_state = state + perturbation
            perturbed_points.append(perturbed_state)

    perturbed_points = np.array(perturbed_points)

    # Create trajectory with perturbed points
    class PerturbedSolution:
        def __init__(self, points: np.ndarray, times: np.ndarray) -> None:
            self.t = times
            self.y = points.T
            self.sol = interp1d(
                times,
                points.T,
                axis=1,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    perturbed_times = np.linspace(segment_start, segment_end, len(perturbed_points))
    perturbed_solution = PerturbedSolution(perturbed_points, perturbed_times)
    perturbed_segment = TrajectorySegment(perturbed_solution, jump_index)
    perturbed_trajectory = HybridTrajectory(segments=[perturbed_segment])

    return perturbed_trajectory, states_dense, jump_index


def format_ic_string(ic: np.ndarray) -> str:
    """
    Format initial condition array as a filename-safe string.

    Args:
        ic: Initial condition array

    Returns:
        Formatted string suitable for filenames
    """
    ic_str = f"ic_{ic[0]}_{ic[1]:g}".replace("-", "neg").replace(".", "p")
    return ic_str
