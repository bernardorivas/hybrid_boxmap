"""
Evaluation over grids.

This module provides utilities for evaluating functions over Grid instances
with support for different sampling strategies.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .grid import Grid
from .config import config

logger = config.get_logger(__name__)


def evaluate_box(
    grid: Grid,
    box_index: int,
    function: Callable[[npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]],
    sampling_mode: str = 'center',
    num_points: int = 1
) -> List[Union[float, npt.NDArray[np.float64]]]:
    """
    Evaluate function on sample points within a single box.
    
    Args:
        grid: Grid instance
        box_index: Index of box to evaluate
        function: Function to evaluate f(point) -> value
        sampling_mode: How to sample points ('center', 'corners', 'random')
        num_points: Number of points for random sampling
        
    Returns:
        List of function values at sample points
    """
    # Get sample points
    points = grid.get_sample_points(box_index, sampling_mode, num_points)
    
    # Evaluate function at each point
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
    sampling_mode: str = 'center',
    num_points: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[int, List[Union[float, np.ndarray]]]:
    """
    Evaluate function over entire grid sequentially.
    
    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy
        num_points: Number of points for random sampling
        progress_callback: Optional callback for progress updates (completed, total)
        
    Returns:
        Dictionary mapping box_index -> list of function values
    """
    results = {}
    
    for i, box_index in enumerate(grid.box_indices):
        box_results = evaluate_box(grid, box_index, function, sampling_mode, num_points)
        results[box_index] = box_results
        
        if progress_callback:
            progress_callback(i + 1, grid.total_boxes)
    
    return results




def evaluate_grid(
    grid: Grid,
    function: Callable[[npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]],
    sampling_mode: str = 'center',
    num_points: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs  # Accept and ignore any extra parameters for backward compatibility
) -> Dict[int, List[Union[float, npt.NDArray[np.float64]]]]:
    """
    Evaluate function over entire grid.
    
    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy ('center', 'corners', 'random')
        num_points: Number of points for random sampling
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary mapping box_index -> list of function values
    """
    return evaluate_grid_sequential(
        grid, function, sampling_mode, num_points, progress_callback
    )