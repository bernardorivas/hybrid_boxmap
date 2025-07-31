"""
Evaluation over grids.

This module provides utilities for evaluating functions over Grid instances
with support for different sampling strategies.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .config import config
from .grid import Grid

logger = config.get_logger(__name__)


def evaluate_box(
    grid: Grid,
    box_index: int,
    function: Callable[
        [npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]],
    ],
    sampling_mode: str = "center",
    num_points: int = 1,
    subdivision_level: int = 1,
) -> List[Union[float, npt.NDArray[np.float64]]]:
    """
    Evaluate function on sample points within a single box.

    Args:
        grid: Grid instance
        box_index: Index of box to evaluate
        function: Function to evaluate f(point) -> value
        sampling_mode: How to sample points ('center', 'corners', 'random', 'subdivision')
        num_points: Number of points for random sampling
        subdivision_level: Level of subdivision for 'subdivision' mode

    Returns:
        List of function values at sample points
    """
    points = grid.get_sample_points(
        box_index, sampling_mode, num_points, subdivision_level,
    )
    results = []
    for point in points:
        try:
            value = function(point)
            results.append(value)
        except Exception as e:
            # Store NaN for failed evaluations
            logger.debug(f"Failed to evaluate point {point}: {e}")
            results.append(np.nan)

    return results


def evaluate_grid_sequential(
    grid: Grid,
    function: Callable[[np.ndarray], Union[float, np.ndarray]],
    sampling_mode: str = "center",
    num_points: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    subdivision_level: int = 1,
) -> Dict[int, List[Union[float, np.ndarray]]]:
    """
    Evaluate function over entire grid sequentially.

    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy
        num_points: Number of points for random sampling
        progress_callback: Optional callback for progress updates (completed, total)
        subdivision_level: Level of subdivision for 'subdivision' mode

    Returns:
        Dictionary mapping box_index -> list of function values
    """
    results = {}

    for i, box_index in enumerate(grid.box_indices):
        box_results = evaluate_box(
            grid, box_index, function, sampling_mode, num_points, subdivision_level,
        )
        results[box_index] = box_results

        if progress_callback:
            progress_callback(i + 1, grid.total_boxes)

    return results


def evaluate_grid(
    grid: Grid,
    function: Callable[
        [npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]],
    ],
    sampling_mode: str = "center",
    num_points: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,  # Accept and ignore any extra parameters for backward compatibility
) -> Dict[int, List[Union[float, npt.NDArray[np.float64]]]]:
    """
    Evaluate function over entire grid.

    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy ('center', 'corners', 'random', 'subdivision')
        num_points: Number of points for random sampling
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping box_index -> list of function values
    """
    subdivision_level = kwargs.get("subdivision_level", 1)
    return evaluate_grid_sequential(
        grid, function, sampling_mode, num_points, progress_callback, subdivision_level,
    )


# Worker function for parallel evaluation - must be at module level for pickling
def _evaluate_points_worker(
    args: Union[
        Tuple[np.ndarray, Callable, tuple, float],
        Tuple[np.ndarray, Callable, tuple, float, bool],
        Tuple[np.ndarray, Callable, tuple, float, bool, Optional[float]]
    ],
) -> List[Tuple[int, Any]]:
    """
    Worker function for parallel point evaluation.

    Args:
        args: Tuple of (points_batch, system_factory, system_args, tau) or
              (points_batch, system_factory, system_args, tau, jump_time_penalty) or
              (points_batch, system_factory, system_args, tau, jump_time_penalty, jump_time_penalty_epsilon)

    Returns:
        List of (point_index, result) tuples
    """
    # Handle both old and new argument formats
    if len(args) == 4:
        points_batch, system_factory, system_args, tau = args
        jump_time_penalty = False
        jump_time_penalty_epsilon = None
    elif len(args) == 5:
        points_batch, system_factory, system_args, tau, jump_time_penalty = args
        jump_time_penalty_epsilon = None
    else:
        points_batch, system_factory, system_args, tau, jump_time_penalty, jump_time_penalty_epsilon = args

    # Create system instance in worker process
    system = system_factory(*system_args)

    results = []
    for i, point in enumerate(points_batch):
        try:
            # Create the evaluation function that simulates the system
            def evaluate_flow(pt):
                traj = system.simulate(pt, (0, tau), jump_time_penalty=jump_time_penalty,
                                     jump_time_penalty_epsilon=jump_time_penalty_epsilon)
                
                # Get first guard state if any jumps occurred
                guard_state = None
                if traj.jump_states:
                    guard_state = traj.jump_states[0][0]  # state_before from first jump
                
                if traj.total_duration >= tau:
                    final_state = traj.interpolate(tau)
                    num_jumps = traj.num_jumps
                    return (final_state, num_jumps, guard_state)

                # If tau not reached, use last available point
                if traj.segments and traj.segments[-1].state_values.size > 0:
                    final_state = traj.segments[-1].state_values[-1]
                    num_jumps = traj.num_jumps
                    return (final_state, num_jumps, guard_state)

                return (np.full(len(pt), np.nan), -1, None)

            result = evaluate_flow(point)
            results.append((i, result))
        except Exception as e:
            logger.debug(f"Failed to evaluate point {point}: {e}")
            results.append((i, (np.full(len(point), np.nan), -1, None)))

    return results


def evaluate_unique_points_parallel(
    points: np.ndarray,
    system_factory: Callable,
    system_args: tuple,
    tau: float,
    max_workers: Optional[int] = None,
    batch_size: int = 1000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    jump_time_penalty: bool = False,
    jump_time_penalty_epsilon: Optional[float] = None,
) -> Dict[int, Tuple[np.ndarray, int, Optional[np.ndarray]]]:
    """
    Evaluate flow map for unique points using parallel processing.

    Args:
        points: Array of points to evaluate, shape (n_points, ndim)
        system_factory: Function that creates a HybridSystem instance
        system_args: Arguments to pass to system_factory
        tau: Time horizon for simulation
        max_workers: Maximum number of parallel workers (None = CPU count)
        batch_size: Number of points per batch
        progress_callback: Optional callback for progress updates
        jump_time_penalty: If True, each jump consumes time from the total duration
        jump_time_penalty_epsilon: Time deducted for each jump (defaults to config value)

    Returns:
        Dictionary mapping point_index -> (final_state, num_jumps, last_pre_jump_state)
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    n_points = len(points)
    results = {}
    completed = 0

    # Split points into batches
    batches = []
    for i in range(0, n_points, batch_size):
        batch = points[i : i + batch_size]
        batches.append((batch, system_factory, system_args, tau, jump_time_penalty, jump_time_penalty_epsilon))

    # Process batches in parallel
    with Pool(processes=max_workers) as pool:
        # Submit all batches
        batch_results = pool.map(_evaluate_points_worker, batches)

        # Collect results
        for batch_idx, batch_result in enumerate(batch_results):
            batch_start = batch_idx * batch_size

            for local_idx, result in batch_result:
                global_idx = batch_start + local_idx
                results[global_idx] = result

            completed += len(batch_result)
            if progress_callback:
                progress_callback(completed, n_points)

    return results
