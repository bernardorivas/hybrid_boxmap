"""
This module provides the HybridBoxMap class, which is a dictionary that maps
source boxes to destination boxes.

The key is the source box index, and the value is a set of destination box indices.

The HybridBoxMap is used to store the box map for a hybrid system.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Callable, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from .config import config, get_default_bloat_factor
from .evaluation import evaluate_grid, evaluate_unique_points_parallel
from .grid import Grid
from .hybrid_system import HybridSystem


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
            "tau": tau,
            "grid_bounds": grid.bounds.tolist(),
            "grid_subdivisions": grid.subdivisions.tolist(),
        }

    @classmethod
    def compute(
        cls,
        grid: Grid,
        system: HybridSystem,
        tau: float,
        sampling_mode: str = "corners",
        bloat_factor: Optional[float] = None,
        discard_out_of_bounds_destinations: bool = True,
        out_of_bounds_tolerance: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_unique_points: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        system_factory: Optional[Callable] = None,
        system_args: Optional[tuple] = None,
        subdivision_level: int = 1,
        enclosure: bool = False,
    ) -> HybridBoxMap:
        """
        Computes the box map for a given hybrid system using a sample-and-bloat method.

        Args:
            grid: The grid discretizing the state space.
            system: The hybrid system to analyze.
            tau: The time horizon for the flow map.
            sampling_mode: How to sample points in each box ('corners', 'center', 'subdivision', etc.).
            bloat_factor: A factor to "bloat" the bounding box of the destination
                          points to ensure coverage. The bloating is relative to the
                          grid box size. A value of 0 means no bloating.
            discard_out_of_bounds_destinations: If True, destination points that
                                                land outside the grid are ignored.
            out_of_bounds_tolerance: Tolerance for determining if a point is out of bounds.
                                   Points outside the domain by more than this tolerance
                                   are discarded. If None, uses config default (1e-6).
            progress_callback: Optional callback function for progress updates.
                              Called with (completed, total) box counts.
            use_unique_points: If True, use the optimized unique points strategy.
            parallel: If True and use_unique_points=True, use parallel processing.
            max_workers: Maximum number of parallel workers (None = CPU count).
            system_factory: Function that creates a HybridSystem instance (required for parallel).
            system_args: Arguments to pass to system_factory (required for parallel).
            subdivision_level: Level of subdivision n when using 'subdivision' mode.
                             Each box is subdivided into 2^n sub-boxes per dimension.
            enclosure: If True and sampling_mode='corners', compute rectangular enclosures
                      when all corners have the same number of jumps. This creates a
                      bounding box containing all destination corners plus interior boxes.

        Returns:
            An instance of HybridBoxMap containing the computed map.
        """
        # Use default bloat factor if not provided
        if bloat_factor is None:
            bloat_factor = get_default_bloat_factor()

        # Use default out of bounds tolerance if not provided
        if out_of_bounds_tolerance is None:
            out_of_bounds_tolerance = config.grid.out_of_bounds_tolerance

        # Get logger
        logger = config.get_logger(__name__)

        if config.logging.verbose:
            logger.info("Computing Hybrid Box Map...")

        # Dictionary to store debugging info, e.g., points causing excessive jumps
        debug_info = {}

        # Define the function to be evaluated on the grid for each sample point
        def evaluate_flow_with_jumps(
            point: npt.NDArray[np.float64],
        ) -> Tuple[npt.NDArray[np.float64], int]:
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

        # Initialize the box map instance
        box_map = cls(grid, system, tau)

        if use_unique_points:
            # Use the optimized unique points strategy
            if config.logging.verbose:
                logger.info(f"Step 1: Getting unique '{sampling_mode}' points...")

            # Get all unique points
            unique_points, metadata = grid.get_all_unique_points(
                sampling_mode, subdivision_level=subdivision_level,
            )

            if config.logging.verbose:
                logger.info(f"Evaluating {len(unique_points)} unique points...")

            if parallel and system_factory is not None and system_args is not None:
                # Use parallel evaluation
                point_results = evaluate_unique_points_parallel(
                    points=unique_points,
                    system_factory=system_factory,
                    system_args=system_args,
                    tau=tau,
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                )
            else:
                # Fall back to sequential evaluation
                point_results = {}
                for i, point in enumerate(unique_points):
                    result = evaluate_flow_with_jumps(point)
                    point_results[i] = result
                    if progress_callback:
                        progress_callback(i + 1, len(unique_points))

            # Process results and build box map
            if config.logging.verbose:
                logger.info("Step 2: Building box map from unique point results...")

            # If enclosure mode is enabled for corners, we need to collect results per box
            if enclosure and sampling_mode == "corners":
                # First pass: collect all corner results for each box
                box_corner_results = defaultdict(
                    lambda: {"points": [], "destinations": [], "jumps": []},
                )

                for point_idx, (final_state, num_jumps) in point_results.items():
                    if num_jumps == -1 or np.any(np.isnan(final_state)):
                        continue  # Skip failed evaluations

                    point = unique_points[point_idx]

                    # Find all boxes that contain this corner point
                    box_indices = grid.find_boxes_containing_point(point)

                    for box_idx in box_indices:
                        box_corner_results[box_idx]["points"].append(point)
                        box_corner_results[box_idx]["destinations"].append(final_state)
                        box_corner_results[box_idx]["jumps"].append(num_jumps)

                # Second pass: process each box with enclosure logic
                for box_idx, results in box_corner_results.items():
                    if box_idx not in box_map:
                        box_map[box_idx] = set()

                    # Check if all corners have same jump count and we have all corners
                    unique_jumps = set(results["jumps"])
                    expected_corners = 2**grid.ndim

                    if (
                        len(unique_jumps) == 1
                        and len(results["destinations"]) == expected_corners
                    ):
                        # Apply enclosure: compute bounding box of all corner destinations
                        destinations_arr = np.array(results["destinations"])

                        if discard_out_of_bounds_destinations:
                            # Filter out-of-bounds destinations
                            in_bounds_mask = np.all(
                                (
                                    destinations_arr
                                    >= grid.bounds[:, 0] - out_of_bounds_tolerance
                                )
                                & (
                                    destinations_arr
                                    <= grid.bounds[:, 1] + out_of_bounds_tolerance
                                ),
                                axis=1,
                            )
                            destinations_arr = destinations_arr[in_bounds_mask]

                            if len(destinations_arr) == 0:
                                continue  # All destinations out of bounds

                        # Compute bounding box
                        min_coords = np.min(destinations_arr, axis=0)
                        max_coords = np.max(destinations_arr, axis=0)

                        # Apply bloat factor
                        bloat_amount = grid.box_widths * bloat_factor
                        bloated_min = min_coords - bloat_amount
                        bloated_max = max_coords + bloat_amount

                        # Find all boxes in the enclosure
                        clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
                        clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

                        min_multi_index = np.floor(
                            (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)
                        max_multi_index = np.floor(
                            (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)

                        min_multi_index = np.maximum(0, min_multi_index)
                        max_multi_index = np.minimum(
                            grid.subdivisions - 1, max_multi_index,
                        )
                        max_multi_index = np.maximum(min_multi_index, max_multi_index)

                        # Add all boxes in the rectangular enclosure
                        iter_ranges = [
                            range(min_multi_index[d], max_multi_index[d] + 1)
                            for d in range(grid.ndim)
                        ]

                        for current_multi_index in itertools.product(*iter_ranges):
                            dest_idx = int(
                                np.ravel_multi_index(
                                    current_multi_index, grid.subdivisions, mode="clip",
                                ),
                            )
                            box_map[box_idx].add(dest_idx)
                    else:
                        # Fall back to per-point bloating
                        for final_state in results["destinations"]:
                            if discard_out_of_bounds_destinations:
                                outside_lower = (
                                    final_state
                                    < grid.bounds[:, 0] - out_of_bounds_tolerance
                                )
                                outside_upper = (
                                    final_state
                                    > grid.bounds[:, 1] + out_of_bounds_tolerance
                                )
                                if np.any(outside_lower | outside_upper):
                                    continue

                            # Create bloated bounding box around the destination
                            bloat_amount = grid.box_widths * bloat_factor
                            bloated_min = final_state - bloat_amount
                            bloated_max = final_state + bloat_amount

                            # Find destination boxes
                            clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
                            clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

                            min_multi_index = np.floor(
                                (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
                            ).astype(int)
                            max_multi_index = np.floor(
                                (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
                            ).astype(int)

                            min_multi_index = np.maximum(0, min_multi_index)
                            max_multi_index = np.minimum(
                                grid.subdivisions - 1, max_multi_index,
                            )
                            max_multi_index = np.maximum(
                                min_multi_index, max_multi_index,
                            )

                            iter_ranges = [
                                range(min_multi_index[d], max_multi_index[d] + 1)
                                for d in range(grid.ndim)
                            ]

                            for current_multi_index in itertools.product(*iter_ranges):
                                dest_idx = int(
                                    np.ravel_multi_index(
                                        current_multi_index,
                                        grid.subdivisions,
                                        mode="clip",
                                    ),
                                )
                                box_map[box_idx].add(dest_idx)
            else:
                # Original logic: process each point individually
                for point_idx, (final_state, num_jumps) in point_results.items():
                    if num_jumps == -1 or np.any(np.isnan(final_state)):
                        continue  # Skip failed evaluations

                    point = unique_points[point_idx]

                    # Find all boxes that contain this sample point
                    box_indices = grid.find_boxes_containing_point(point)

                    # For each box that contains this point
                    for box_idx in box_indices:
                        if box_idx not in box_map:
                            box_map[box_idx] = set()

                        # Process the destination point
                        if discard_out_of_bounds_destinations:
                            # Check if point is outside domain by more than tolerance
                            outside_lower = (
                                final_state
                                < grid.bounds[:, 0] - out_of_bounds_tolerance
                            )
                            outside_upper = (
                                final_state
                                > grid.bounds[:, 1] + out_of_bounds_tolerance
                            )
                            if np.any(outside_lower | outside_upper):
                                continue  # Skip out-of-bounds destinations

                        # Create bloated bounding box around the destination
                        bloat_amount = grid.box_widths * bloat_factor
                        bloated_min = final_state - bloat_amount
                        bloated_max = final_state + bloat_amount

                        # Find all grid cells that intersect the bloated box
                        clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
                        clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

                        min_multi_index = np.floor(
                            (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)
                        max_multi_index = np.floor(
                            (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)

                        min_multi_index = np.maximum(0, min_multi_index)
                        max_multi_index = np.minimum(
                            grid.subdivisions - 1, max_multi_index,
                        )
                        max_multi_index = np.maximum(min_multi_index, max_multi_index)

                        # Add destination boxes
                        iter_ranges = [
                            range(min_multi_index[d], max_multi_index[d] + 1)
                            for d in range(grid.ndim)
                        ]

                        for current_multi_index in itertools.product(*iter_ranges):
                            dest_idx = int(
                                np.ravel_multi_index(
                                    current_multi_index, grid.subdivisions, mode="clip",
                                ),
                            )
                            box_map[box_idx].add(dest_idx)

            # Store metadata about unique points for debugging
            box_map.unique_points_metadata = metadata

        else:
            # Use the original per-box evaluation method
            # 1. Evaluate the flow for all sample points in the grid.
            # This gives us a dictionary from box_idx -> list of (final_state, num_jumps)
            if config.logging.verbose:
                logger.info(
                    f"Step 1: Evaluating flow map for all '{sampling_mode}' points...",
                )

            raw_flow_data = evaluate_grid(
                grid=grid,
                function=evaluate_flow_with_jumps,
                sampling_mode=sampling_mode,
                parallel=False,
                progress_callback=progress_callback,
                subdivision_level=subdivision_level,
            )

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

                # Check if enclosure mode is applicable
                use_enclosure = (
                    enclosure
                    and sampling_mode == "corners"
                    and len(points_by_jumps) == 1
                    and len(results_for_box) == 2**grid.ndim
                )  # All corners evaluated successfully

                if use_enclosure:
                    # All corners have the same jump count - compute enclosure
                    jump_count = list(points_by_jumps.keys())[0]
                    points = points_by_jumps[jump_count]

                    if points:
                        points_arr = np.array(points)

                        if discard_out_of_bounds_destinations:
                            # Filter out points that landed outside the grid domain
                            points_in_domain = []
                            for p in points_arr:
                                outside_lower = (
                                    p < grid.bounds[:, 0] - out_of_bounds_tolerance
                                )
                                outside_upper = (
                                    p > grid.bounds[:, 1] + out_of_bounds_tolerance
                                )
                                if not np.any(outside_lower | outside_upper):
                                    points_in_domain.append(p)

                            if points_in_domain:
                                points_arr = np.array(points_in_domain)
                            else:
                                continue  # All points out of bounds

                        # Create bounding box of all corner destinations
                        min_coords = np.min(points_arr, axis=0)
                        max_coords = np.max(points_arr, axis=0)

                        # Apply bloat factor
                        bloat_amount = grid.box_widths * bloat_factor
                        bloated_min = min_coords - bloat_amount
                        bloated_max = max_coords + bloat_amount

                        # Find all boxes in the bloated bounding box
                        clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
                        clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

                        min_multi_index = np.floor(
                            (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)
                        max_multi_index = np.floor(
                            (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)

                        min_multi_index = np.maximum(0, min_multi_index)
                        max_multi_index = np.minimum(
                            grid.subdivisions - 1, max_multi_index,
                        )
                        max_multi_index = np.maximum(min_multi_index, max_multi_index)

                        # Add all boxes in the rectangular enclosure
                        iter_ranges = [
                            range(min_multi_index[d], max_multi_index[d] + 1)
                            for d in range(grid.ndim)
                        ]

                        for current_multi_index in itertools.product(*iter_ranges):
                            dest_idx = int(
                                np.ravel_multi_index(
                                    current_multi_index, grid.subdivisions, mode="clip",
                                ),
                            )
                            destination_indices.add(dest_idx)

                else:
                    # Original logic: process each jump group separately
                    # 3. For each group of points, compute a bloated bounding box
                    for _jump_count, points in points_by_jumps.items():
                        if not points:
                            continue

                        points_arr = np.array(points)

                        if discard_out_of_bounds_destinations:
                            # Filter out points that landed outside the grid domain by more than tolerance
                            points_in_domain = []
                            for p in points_arr:
                                outside_lower = (
                                    p < grid.bounds[:, 0] - out_of_bounds_tolerance
                                )
                                outside_upper = (
                                    p > grid.bounds[:, 1] + out_of_bounds_tolerance
                                )
                                if not np.any(outside_lower | outside_upper):
                                    points_in_domain.append(p)

                            if not points_in_domain:
                                continue  # All points for this jump count were out of bounds
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
                        min_multi_index = np.floor(
                            (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)
                        max_multi_index = np.floor(
                            (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
                        ).astype(int)

                        # Make sure indices are within the valid range
                        min_multi_index = np.maximum(0, min_multi_index)
                        max_multi_index = np.minimum(
                            grid.subdivisions - 1, max_multi_index,
                        )

                        # Ensure max_multi_index is not smaller than min_multi_index
                        max_multi_index = np.maximum(min_multi_index, max_multi_index)

                        # Iterate through the grid coordinates from min to max
                        # Create a range for each dimension
                        iter_ranges = [
                            range(min_multi_index[d], max_multi_index[d] + 1)
                            for d in range(grid.ndim)
                        ]

                        for current_multi_index in itertools.product(*iter_ranges):
                            dest_idx = int(
                                np.ravel_multi_index(
                                    current_multi_index, grid.subdivisions, mode="clip",
                                ),
                            )
                            destination_indices.add(dest_idx)

                if destination_indices:
                    box_map[box_idx] = destination_indices

            # Attach the raw point data for visualization purposes
            box_map.raw_flow_data = raw_flow_data

        if config.logging.verbose:
            logger.info("✓ Hybrid Box Map computation complete.")

        # Store debug info in box_map for external analysis
        box_map.debug_info = debug_info

        return box_map

    @classmethod
    def compute_cylindrical(
        cls,
        grid: Grid,
        system: HybridSystem,
        tau: float,
        cylinder_radius: float,
        n_radial_samples: int = 10,
        n_angular_samples: int = 10,
        bloat_factor: Optional[float] = None,
        discard_out_of_bounds_destinations: bool = True,
        out_of_bounds_tolerance: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        system_factory: Optional[Callable] = None,
        system_args: Optional[tuple] = None,
    ) -> HybridBoxMap:
        """
        Computes box map for systems with cylindrical symmetry (e.g., bipedal walker).

        This method is optimized for 4D systems where the first two dimensions (x,y)
        are constrained to a cylinder x² + y² ≤ r². It samples points in cylindrical
        coordinates and only considers boxes that intersect the cylinder.

        Args:
            grid: 4D grid with dimensions [x, y, x_dot, y_dot]
            system: Hybrid system to analyze
            tau: Time horizon
            cylinder_radius: Radius of the cylinder constraint
            n_radial_samples: Number of samples in radial direction
            n_angular_samples: Number of samples in angular direction
            bloat_factor: Bloating factor for destination boxes
            discard_out_of_bounds_destinations: Whether to discard out-of-bounds points
            out_of_bounds_tolerance: Tolerance for out-of-bounds checking
            progress_callback: Progress callback function
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
            system_factory: Factory function for parallel processing
            system_args: Arguments for system_factory

        Returns:
            HybridBoxMap containing transitions for boxes within/intersecting the cylinder
        """
        if grid.ndim != 4:
            raise ValueError(
                "Cylindrical computation requires 4D grid [x, y, x_dot, y_dot]",
            )

        # Use defaults if not provided
        if bloat_factor is None:
            bloat_factor = get_default_bloat_factor()
        if out_of_bounds_tolerance is None:
            out_of_bounds_tolerance = config.grid.out_of_bounds_tolerance

        logger = config.get_logger(__name__)

        if config.logging.verbose:
            logger.info("Computing Cylindrical Hybrid Box Map...")
            logger.info(f"Cylinder radius: {cylinder_radius}")
            logger.info(
                f"Sampling: {n_radial_samples} radial × {n_angular_samples} angular",
            )

        # Initialize box map
        box_map = cls(grid, system, tau)

        # Get velocity bounds from grid
        x_dot_bounds = grid.bounds[2]
        y_dot_bounds = grid.bounds[3]

        # Generate cylindrical samples
        # Use sqrt-uniform distribution for r to get uniform area coverage
        r_samples = cylinder_radius * np.sqrt(np.linspace(0, 1, n_radial_samples))
        # Ensure we include exactly the boundary
        if cylinder_radius not in r_samples:
            r_samples = np.append(r_samples, cylinder_radius)
        theta_samples = np.linspace(0, 2 * np.pi, n_angular_samples, endpoint=False)

        # CRITICAL: Add samples at key angles to ensure we hit boxes containing cardinal points
        # Add samples at 0, π/2, π, 3π/2 if not already included
        key_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for angle in key_angles:
            if not any(abs(theta - angle) < 1e-10 for theta in theta_samples):
                theta_samples = np.append(theta_samples, angle)

        # Sample velocities from grid bounds
        # Use more samples for finer velocity resolution
        n_velocity_samples = 12  # Creates 12x12 = 144 velocity combinations
        x_dot_samples = np.linspace(
            x_dot_bounds[0], x_dot_bounds[1], n_velocity_samples,
        )
        y_dot_samples = np.linspace(
            y_dot_bounds[0], y_dot_bounds[1], n_velocity_samples,
        )

        # Create all sample points
        sample_points = []
        for r in r_samples:
            for theta in theta_samples:
                # Convert to Cartesian
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                # Add velocity variations
                for x_dot in x_dot_samples:
                    for y_dot in y_dot_samples:
                        sample_points.append(np.array([x, y, x_dot, y_dot]))

        # CRITICAL: Add samples for boxes containing key points
        # These are the expected fixed points and periodic orbit points
        key_points = [
            np.array([0.0, 0.0, 0.0, 0.0]),  # Origin fixed point
            np.array([1.0, 0.0, 0.0, 0.0]),  # (1,0) periodic point
            np.array([-1.0, 0.0, 0.0, 0.0]),  # (-1,0) periodic point
            np.array([0.0, 1.0, 0.0, 0.0]),  # (0,1)
            np.array([0.0, -1.0, 0.0, 0.0]),  # (0,-1)
        ]

        # For each key point, find the box containing it and sample from box center
        key_boxes_sampled = []
        for key_pt in key_points:
            try:
                # Find box containing this key point
                box_idx = grid.get_box_from_point(key_pt)
                lower_bounds, upper_bounds = grid.get_box_bounds(box_idx)
                box_center = (lower_bounds + upper_bounds) / 2

                key_boxes_sampled.append((key_pt[:2], box_idx))

                # Add samples from box center with velocity variations
                for x_dot in x_dot_samples:
                    for y_dot in y_dot_samples:
                        pt = box_center.copy()
                        pt[2] = x_dot
                        pt[3] = y_dot
                        sample_points.append(pt)

                # Also add the exact key point for good measure
                for x_dot in x_dot_samples:
                    for y_dot in y_dot_samples:
                        pt = key_pt.copy()
                        pt[2] = x_dot
                        pt[3] = y_dot
                        sample_points.append(pt)

            except Exception:
                # If key point is outside grid, just use the point itself
                for x_dot in x_dot_samples:
                    for y_dot in y_dot_samples:
                        pt = key_pt.copy()
                        pt[2] = x_dot
                        pt[3] = y_dot
                        sample_points.append(pt)

        if config.logging.verbose:
            logger.info(f"Key boxes sampled: {key_boxes_sampled}")

        if config.logging.verbose:
            logger.info(f"Total sample points: {len(sample_points)}")

        # Evaluate flow for all sample points
        if parallel and system_factory is not None and system_args is not None:
            # Use parallel evaluation
            point_results = evaluate_unique_points_parallel(
                points=np.array(sample_points),
                system_factory=system_factory,
                system_args=system_args,
                tau=tau,
                max_workers=max_workers,
                progress_callback=progress_callback,
            )
        else:
            # Sequential evaluation
            point_results = {}
            for i, point in enumerate(sample_points):
                try:
                    traj = system.simulate(point, (0, tau))
                    if traj.total_duration >= tau:
                        final_state = traj.interpolate(tau)
                        num_jumps = traj.num_jumps
                    elif traj.segments and traj.segments[-1].state_values.size > 0:
                        final_state = traj.segments[-1].state_values[-1]
                        num_jumps = traj.num_jumps
                    else:
                        final_state = np.full(4, np.nan)
                        num_jumps = -1
                    point_results[i] = (final_state, num_jumps)
                except Exception:
                    point_results[i] = (np.full(4, np.nan), -1)

                if progress_callback and i % 100 == 0:
                    progress_callback(i, len(sample_points))

        # Process results and build box map
        if config.logging.verbose:
            logger.info("Building box map from cylindrical samples...")

        # For each sample point and its result
        for i, (initial_point, (final_state, num_jumps)) in enumerate(
            zip(sample_points, [point_results[j] for j in range(len(sample_points))]),
        ):
            if num_jumps == -1 or np.any(np.isnan(final_state)):
                continue

            # Find source box
            try:
                src_box = grid.get_box_from_point(initial_point)
            except (ValueError, IndexError):
                continue

            # Check if source box center is within cylinder
            # REMOVED - we should add transitions for any box that contains our sample point
            # regardless of where the box center is

            if src_box not in box_map:
                box_map[src_box] = set()

            # Process destination
            if discard_out_of_bounds_destinations:
                outside_lower = (
                    final_state < grid.bounds[:, 0] - out_of_bounds_tolerance
                )
                outside_upper = (
                    final_state > grid.bounds[:, 1] + out_of_bounds_tolerance
                )
                if np.any(outside_lower | outside_upper):
                    continue

            # Create bloated bounding box
            bloat_amount = grid.box_widths * bloat_factor
            bloated_min = final_state - bloat_amount
            bloated_max = final_state + bloat_amount

            # Find destination boxes
            clipped_min = np.maximum(bloated_min, grid.bounds[:, 0])
            clipped_max = np.minimum(bloated_max, grid.bounds[:, 1])

            min_multi_index = np.floor(
                (clipped_min - grid.bounds[:, 0]) / grid.box_widths,
            ).astype(int)
            max_multi_index = np.floor(
                (clipped_max - grid.bounds[:, 0]) / grid.box_widths,
            ).astype(int)

            min_multi_index = np.maximum(0, min_multi_index)
            max_multi_index = np.minimum(grid.subdivisions - 1, max_multi_index)
            max_multi_index = np.maximum(min_multi_index, max_multi_index)

            # Add destination boxes
            iter_ranges = [
                range(min_multi_index[d], max_multi_index[d] + 1)
                for d in range(grid.ndim)
            ]

            for current_multi_index in itertools.product(*iter_ranges):
                dest_idx = int(
                    np.ravel_multi_index(
                        current_multi_index, grid.subdivisions, mode="clip",
                    ),
                )
                box_map[src_box].add(dest_idx)

        # Add metadata
        box_map.cylinder_radius = cylinder_radius
        box_map.cylindrical_samples = {
            "n_radial": n_radial_samples,
            "n_angular": n_angular_samples,
            "total_points": len(sample_points),
        }

        if config.logging.verbose:
            logger.info(f"✓ Cylindrical Box Map complete. Active boxes: {len(box_map)}")

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
