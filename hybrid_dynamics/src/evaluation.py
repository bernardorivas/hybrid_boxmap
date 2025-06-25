"""
Evaluation over grids with parallel processing support.

This module provides utilities for evaluating functions over Grid instances
with support for different sampling strategies and parallel execution.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union, Iterator

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


def _evaluate_box_worker(args):
    """
    Worker function for parallel evaluation.
    
    Args:
        args: Tuple of (grid, box_index, function, sampling_mode, num_points)
        
    Returns:
        Tuple of (box_index, results)
    """
    grid, box_index, function, sampling_mode, num_points = args
    results = evaluate_box(grid, box_index, function, sampling_mode, num_points)
    return box_index, results


def _generate_evaluation_args(
    grid: Grid, 
    function: Callable,
    sampling_mode: str,
    num_points: int
) -> Iterator[tuple]:
    """Generate evaluation arguments lazily to save memory."""
    for box_index in grid.box_indices:
        yield (grid, box_index, function, sampling_mode, num_points)


def _process_chunk(
    args_chunk: List[tuple],
    executor,
    progress_callback: Optional[Callable[[int, int], None]],
    completed_counter: List[int],
    total_boxes: int
) -> Dict[int, List]:
    """Process a chunk of evaluation tasks."""
    chunk_results = {}
    
    # Submit tasks for this chunk
    future_to_box = {
        executor.submit(_evaluate_box_worker, args): args[1]
        for args in args_chunk
    }
    
    # Collect results as they complete
    for future in as_completed(future_to_box):
        box_index, box_results = future.result()
        chunk_results[box_index] = box_results
        completed_counter[0] += 1
        
        if progress_callback:
            progress_callback(completed_counter[0], total_boxes)
    
    return chunk_results


def evaluate_grid_parallel(
    grid: Grid,
    function: Callable[[npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]],
    sampling_mode: str = 'center',
    num_points: int = 1,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    initializer: Optional[Callable] = None,
    initargs: tuple = (),
    chunk_size: Optional[int] = None
) -> Dict[int, List[Union[float, npt.NDArray[np.float64]]]]:
    """
    Evaluate function over entire grid using parallel processing.
    
    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy
        num_points: Number of points for random sampling
        max_workers: Maximum number of parallel workers (None = CPU count)
        progress_callback: Optional callback for progress updates
        initializer: A function to run at the start of each worker process.
        initargs: Arguments to pass to the initializer function.
        
    Returns:
        Dictionary mapping box_index -> list of function values
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Determine chunk size for memory efficiency
    if chunk_size is None:
        # Use a reasonable chunk size based on grid size and workers
        chunk_size = max(100, grid.total_boxes // (max_workers * 4))
        chunk_size = min(chunk_size, 1000)  # Cap at 1000 for memory
    
    logger.debug(f"Using chunk size: {chunk_size} for {grid.total_boxes} boxes")
    
    results = {}
    completed_counter = [0]  # Use list to allow modification in nested function
    
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=initargs
    ) as executor:
        # Process in chunks to avoid memory explosion
        args_generator = _generate_evaluation_args(grid, function, sampling_mode, num_points)
        chunk = []
        
        for args in args_generator:
            chunk.append(args)
            
            if len(chunk) >= chunk_size:
                # Process this chunk
                chunk_results = _process_chunk(
                    chunk, executor, progress_callback, 
                    completed_counter, grid.total_boxes
                )
                results.update(chunk_results)
                chunk = []  # Reset chunk
        
        # Process any remaining items
        if chunk:
            chunk_results = _process_chunk(
                chunk, executor, progress_callback,
                completed_counter, grid.total_boxes
            )
            results.update(chunk_results)
    
    return results


def evaluate_grid(
    grid: Grid,
    function: Callable[[npt.NDArray[np.float64]], Union[float, npt.NDArray[np.float64]]],
    sampling_mode: str = 'center',
    num_points: int = 1,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    initializer: Optional[Callable] = None,
    initargs: tuple = (),
    chunk_size: Optional[int] = None
) -> Dict[int, List[Union[float, npt.NDArray[np.float64]]]]:
    """
    Evaluate function over entire grid with automatic parallel/sequential selection.
    
    Args:
        grid: Grid instance
        function: Function to evaluate
        sampling_mode: Sampling strategy ('center', 'corners', 'random')
        num_points: Number of points for random sampling
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        progress_callback: Optional callback for progress updates
        initializer: A function to run at the start of each worker process.
        initargs: Arguments to pass to the initializer function.
        
    Returns:
        Dictionary mapping box_index -> list of function values
    """
    if parallel and grid.total_boxes > 1:
        return evaluate_grid_parallel(
            grid, function, sampling_mode, num_points, max_workers, progress_callback,
            initializer=initializer, initargs=initargs, chunk_size=chunk_size
        )
    else:
        return evaluate_grid_sequential(
            grid, function, sampling_mode, num_points, progress_callback
        )