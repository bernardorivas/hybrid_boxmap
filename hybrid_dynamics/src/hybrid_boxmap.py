""" 
This module provides the HybridBoxMap class, which is a dictionary that maps
source boxes to destination boxes.

The key is the source box index, and the value is a set of destination box indices.

The HybridBoxMap is used to store the box map for a hybrid system.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
import itertools
import numpy.typing as npt

from .grid import Grid
from .hybrid_system import HybridSystem
from .evaluation import evaluate_grid
from .config import get_default_bloat_factor, config
from .parallel_utils import create_parallel_evaluator, test_parallel_capability


class HybridBoxMap(dict):
    """
    A dictionary mapping source grid box indices to sets of destination box indices.

    This class computes and stores the mapping that describes how a hybrid system
    moves sets of states from one region of a grid to another over a given time tau.
    """

    def __init__(self, grid: Grid, system: HybridSystem, tau: float):
        super().__init__()
        self.grid = grid
        self.system = system
        self.tau = tau
        self.metadata = {
            'tau': tau,
            'grid_bounds': grid.bounds.tolist(),
            'grid_subdivisions': grid.subdivisions.tolist()
        }

    @classmethod
    def compute(
        cls,
        grid: Grid,
        system: HybridSystem,
        tau: float,
        sampling_mode: str = 'corners',
        bloat_factor: Optional[float] = None,
        discard_out_of_bounds_destinations: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> HybridBoxMap:
        """
        Computes the box map for a given hybrid system using a sample-and-bloat method.

        Args:
            grid: The grid discretizing the state space.
            system: The hybrid system to analyze.
            tau: The time horizon for the flow map.
            sampling_mode: How to sample points in each box ('corners', 'center', etc.).
            bloat_factor: A factor to "bloat" the bounding box of the destination
                          points to ensure coverage. The bloating is relative to the
                          grid box size. A value of 0 means no bloating.
            discard_out_of_bounds_destinations: If True, destination points that
                                                land outside the grid are ignored.
            parallel: Whether to use parallel processing.
            max_workers: The number of workers for parallel processing.
            progress_callback: Optional callback function for progress updates.
                              Called with (completed, total) box counts.

        Returns:
            An instance of HybridBoxMap containing the computed map.
        """
        # Use default bloat factor if not provided
        if bloat_factor is None:
            bloat_factor = get_default_bloat_factor()
        
        # Get logger
        logger = config.get_logger(__name__)
        
        if config.logging.verbose:
            logger.info("Computing Hybrid Box Map...")
        
        # Dictionary to store debugging info, e.g., points causing excessive jumps
        debug_info = {}

        # Define the function to be evaluated on the grid for each sample point
        def evaluate_flow_with_jumps(point: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], int]:
            try:
                traj = system.simulate(point, (0, tau), debug_info=debug_info)
                # Check if the simulation reached tau
                if traj.total_duration >= tau:
                    final_state = traj.interpolate(tau)
                    num_jumps = traj.num_jumps
                    return (final_state, num_jumps)
                
                # If tau is not reached, use the last available point from the trajectory.
                if traj.segments and traj.segments[-1].state_values.size > 0:
                    final_state = traj.segments[-1].state_values[-1]
                    num_jumps = traj.num_jumps
                    return (final_state, num_jumps)

                return (np.full(grid.ndim, np.nan), -1)
            except Exception as e:
                logger.debug(f"Failed to evaluate flow at point {point}: {e}")
                return (np.full(grid.ndim, np.nan), -1)

        # 1. Evaluate the flow for all sample points in the grid.
        # This gives us a dictionary from box_idx -> list of (final_state, num_jumps)
        if config.logging.verbose:
            logger.info(f"Step 1: Evaluating flow map for all '{sampling_mode}' points...")
        
        # Check if we can use parallel processing
        can_parallelize = parallel and test_parallel_capability(system)
        
        if can_parallelize:
            # Use the parallel-safe evaluator
            evaluate_function = create_parallel_evaluator(system, tau)
            logger.debug("Using parallel processing for flow evaluation")
        else:
            # Use the local closure-based evaluator
            evaluate_function = evaluate_flow_with_jumps
            if parallel:
                logger.warning("Parallel processing requested but system cannot be pickled. Falling back to sequential.")
        
        raw_flow_data = evaluate_grid(
            grid=grid,
            function=evaluate_function,
            sampling_mode=sampling_mode,
            parallel=can_parallelize,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

        # Initialize the box map instance
        box_map = cls(grid, system, tau)

        # 2. For each source box, process the evaluation results
        if config.logging.verbose:
            logger.info("Step 2: Constructing map from evaluation data...")
        for box_idx in grid.box_indices:
            results_for_box = raw_flow_data.get(box_idx)
            if not results_for_box:
                continue

            # Group final points by the number of jumps
            points_by_jumps = defaultdict(list)
            for final_state, num_jumps in results_for_box:
                if num_jumps != -1 and not np.any(np.isnan(final_state)):
                    points_by_jumps[num_jumps].append(final_state)

            destination_indices: Set[int] = set()

            # 3. For each group of points, compute a bloated bounding box
            for _jump_count, points in points_by_jumps.items():
                if not points:
                    continue
                
                points_arr = np.array(points)

                if discard_out_of_bounds_destinations:
                    # Filter out points that landed outside the grid domain
                    points_in_domain = []
                    for p in points_arr:
                        if np.all((p >= grid.bounds[:, 0]) & (p <= grid.bounds[:, 1])):
                            points_in_domain.append(p)
                    
                    if not points_in_domain:
                        continue # All points for this jump count were out of bounds
                    points_arr = np.array(points_in_domain)

                # Create a tight bounding box around the destination points
                min_coords = np.min(points_arr, axis=0)
                max_coords = np.max(points_arr, axis=0)

                # Bloat the box to account for the image of the interior
                bloat_amount = grid.box_widths * bloat_factor
                bloated_min = min_coords - bloat_amount
                bloated_max = max_coords + bloat_amount
                
                # 4. Find all grid cells that intersect the bloated bounding box
                # Clip the coordinates to be within the grid's overall bounds
                clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
                clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

                # Find the multi-indices of the corners of the clipped bloated box
                min_multi_index = np.floor((clipped_min - grid.bounds[:, 0]) / grid.box_widths).astype(int)
                max_multi_index = np.floor((clipped_max - grid.bounds[:, 0]) / grid.box_widths).astype(int)
                
                # Make sure indices are within the valid range
                min_multi_index = np.maximum(0, min_multi_index)
                max_multi_index = np.minimum(grid.subdivisions - 1, max_multi_index)
                
                # Ensure max_multi_index is not smaller than min_multi_index
                max_multi_index = np.maximum(min_multi_index, max_multi_index)

                # Iterate through the grid coordinates from min to max
                # Create a range for each dimension
                iter_ranges = [range(min_multi_index[d], max_multi_index[d] + 1) for d in range(grid.ndim)]
                
                for current_multi_index in itertools.product(*iter_ranges):
                    dest_idx = int(np.ravel_multi_index(current_multi_index, grid.subdivisions, mode='clip'))
                    destination_indices.add(dest_idx)

            if destination_indices:
                box_map[box_idx] = destination_indices
        
        if config.logging.verbose:
            logger.info("✓ Hybrid Box Map computation complete.")
        
        # Attach the raw point data for visualization purposes
        box_map.raw_flow_data = raw_flow_data
        box_map.debug_info = debug_info
        
        return box_map

    def to_networkx(self):
        """
        Converts the HybridBoxMap to a networkx.DiGraph object.

        Returns:
            A networkx.DiGraph where nodes are box indices and edges represent
            the transitions between them.
        """
        import networkx as nx
        return nx.DiGraph(self)


def _evaluate_flow_map(
    points: np.ndarray,
    grid: Grid,
    system: HybridSystem,
    tau: float,
    sampling_mode: str = 'corners',
    bloat_factor: float = 0.1,
    discard_out_of_bounds_destinations: bool = True,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> np.ndarray:
    pass

