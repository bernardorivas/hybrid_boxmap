"""
MultiGrid framework for hybrid systems with discrete modes.

This module provides the MultiGrid class which represents hybrid systems
as a directed graph of modes, where each mode has its own grid discretization.
This enables analysis of systems with discrete state transitions between
different operational modes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .grid import Grid
from .hybrid_system import HybridSystem
from .print_utils import vprint


class MultiGridBoxMap(dict):
    """
    BoxMap for multi-grid systems that can transition between modes.

    Keys are (mode, box_index) tuples, values are sets of (mode, box_index) destinations.
    This allows representation of both intra-mode and inter-mode transitions.
    """

    def __init__(self, multigrid: MultiGrid, system: HybridSystem, tau: float):
        super().__init__()
        self.multigrid = multigrid
        self.system = system
        self.tau = tau
        self.metadata = {
            "tau": tau,
            "modes": list(multigrid.mode_grids.keys()),
            "total_boxes": sum(
                grid.total_boxes for grid in multigrid.mode_grids.values()
            ),
        }

    def get_transitions_from_mode(self, mode: int) -> Dict[int, Set[Tuple[int, int]]]:
        """Get all transitions originating from a specific mode."""
        transitions = defaultdict(set)

        for (src_mode, src_box), destinations in self.items():
            if src_mode == mode:
                for dest_mode, dest_box in destinations:
                    transitions[dest_mode].add((dest_mode, dest_box))

        return dict(transitions)

    def get_intra_mode_transitions(self, mode: int) -> Dict[int, Set[int]]:
        """Get transitions within a single mode (mode -> mode)."""
        intra_transitions = {}

        for (src_mode, src_box), destinations in self.items():
            if src_mode == mode:
                same_mode_destinations = {
                    dest_box
                    for dest_mode, dest_box in destinations
                    if dest_mode == mode
                }
                if same_mode_destinations:
                    intra_transitions[src_box] = same_mode_destinations

        return intra_transitions

    def get_inter_mode_transitions(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Get transitions between different modes."""
        inter_transitions = {}

        for (src_mode, src_box), destinations in self.items():
            different_mode_destinations = {
                (dest_mode, dest_box)
                for dest_mode, dest_box in destinations
                if dest_mode != src_mode
            }
            if different_mode_destinations:
                inter_transitions[(src_mode, src_box)] = different_mode_destinations

        return inter_transitions


class MultiGrid:
    """
    Multi-grid system for hybrid systems with discrete modes.

    Represents a hybrid system as a directed graph where each vertex corresponds
    to a mode with its own grid discretization. Edges represent possible transitions
    between modes.
    """

    def __init__(self, mode_graph: nx.DiGraph, mode_grids: Dict[int, Grid]):
        """
        Initialize MultiGrid.

        Args:
            mode_graph: Directed graph representing possible mode transitions
            mode_grids: Dictionary mapping mode_id -> Grid for that mode
        """
        self.mode_graph = mode_graph
        self.mode_grids = mode_grids
        self.embedding_grid = None

        # Validate that all graph nodes have corresponding grids
        graph_nodes = set(mode_graph.nodes())
        grid_modes = set(mode_grids.keys())
        if graph_nodes != grid_modes:
            raise ValueError(
                f"Mode graph nodes {graph_nodes} must match grid modes {grid_modes}",
            )

    @property
    def modes(self) -> List[int]:
        """List of all modes in the system."""
        return list(self.mode_grids.keys())

    @property
    def total_boxes(self) -> int:
        """Total number of boxes across all modes."""
        return sum(grid.total_boxes for grid in self.mode_grids.values())

    def get_mode_from_state(self, state: np.ndarray) -> int:
        """
        Determine which mode a state belongs to.

        For systems with discrete mode variables, this extracts the mode
        from the state vector. Override this method for system-specific logic.

        Args:
            state: State vector

        Returns:
            Mode identifier
        """
        if len(state) < 2:
            raise ValueError(
                "State must have at least 2 dimensions for mode extraction",
            )

        # Default: assume last component is the discrete mode variable
        mode_value = state[-1]

        # Round to nearest integer and ensure it's a valid mode
        mode = int(round(mode_value))
        if mode not in self.mode_grids:
            raise ValueError(
                f"Invalid mode {mode}, available modes: {list(self.mode_grids.keys())}",
            )

        return mode

    def get_continuous_state(self, state: np.ndarray) -> np.ndarray:
        """
        Extract continuous part of state (remove discrete mode component).

        Args:
            state: Full state vector including mode

        Returns:
            Continuous state components only
        """
        # Default: assume last component is discrete mode, rest is continuous
        return state[:-1]

    def get_box_from_state(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Get (mode, box_index) from a state vector.

        Args:
            state: State vector

        Returns:
            Tuple of (mode, box_index)
        """
        mode = self.get_mode_from_state(state)
        continuous_state = self.get_continuous_state(state)

        grid = self.mode_grids[mode]
        box_index = grid.get_box_from_point(continuous_state)

        return mode, box_index

    def get_sample_points_for_mode_box(
        self, mode: int, box_index: int, sampling_mode: str = "corners",
    ) -> np.ndarray:
        """
        Get sample points for a specific (mode, box_index) pair.

        Args:
            mode: Mode identifier
            box_index: Box index within the mode's grid
            sampling_mode: Sampling strategy ('corners', 'center', 'random')

        Returns:
            Array of sample points in full state space (including mode component)
        """
        if mode not in self.mode_grids:
            raise ValueError(f"Invalid mode {mode}")

        grid = self.mode_grids[mode]
        continuous_points = grid.get_sample_points(box_index, sampling_mode)

        # Add mode component to each point
        num_points = continuous_points.shape[0]
        mode_column = np.full((num_points, 1), mode, dtype=float)

        full_state_points = np.hstack([continuous_points, mode_column])

        return full_state_points

    def get_all_sample_points(
        self, sampling_mode: str = "corners",
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get sample points for all (mode, box_index) pairs.

        Args:
            sampling_mode: Sampling strategy

        Returns:
            Dictionary mapping (mode, box_index) -> sample points array
        """
        all_points = {}

        for mode, grid in self.mode_grids.items():
            for box_index in grid.box_indices:
                points = self.get_sample_points_for_mode_box(
                    mode, box_index, sampling_mode,
                )
                all_points[(mode, box_index)] = points

        return all_points

    def create_embedding_grid(
        self, y_spacing: float = 1.0, y_subdivisions: Optional[int] = None,
    ) -> Grid:
        """
        Create a 2D embedding grid for visualization.

        This stacks the 1D mode grids vertically at different y-levels
        to create a 2D visualization space. The y-axis can have more subdivisions
        than the number of modes for better visualization aesthetics.

        Args:
            y_spacing: Vertical spacing between mode levels
            y_subdivisions: Number of subdivisions in y-direction for visualization.
                          If None, defaults to number of modes (original behavior).
                          For better visualization, consider using 10x the number of modes.

        Returns:
            2D Grid containing all modes
        """
        if not self.mode_grids:
            raise ValueError("No mode grids available")

        # Get bounds from the first grid (assume all have same continuous bounds)
        first_grid = next(iter(self.mode_grids.values()))
        if first_grid.ndim != 1:
            raise ValueError("Embedding currently only supports 1D mode grids")

        x_bounds = first_grid.bounds[0]

        # Create y bounds to accommodate all modes
        num_modes = len(self.modes)
        y_min = -0.5 * y_spacing
        y_max = (num_modes - 0.5) * y_spacing
        y_bounds = [y_min, y_max]

        # Use same x subdivision as mode grids
        x_subdivisions = first_grid.subdivisions[0]

        # Allow customizable y subdivisions for better visualization
        if y_subdivisions is None:
            y_subdivisions = num_modes  # Original behavior: one subdivision per mode

        self.embedding_grid = Grid(
            bounds=[x_bounds, y_bounds], subdivisions=[x_subdivisions, y_subdivisions],
        )

        return self.embedding_grid

    def compute_multi_boxmap(
        self,
        system: HybridSystem,
        tau: float,
        bloat_factor: Optional[float] = None,
        parallel: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MultiGridBoxMap:
        """
        Compute MultiGrid BoxMap that can transition between modes.

        Args:
            system: Hybrid system to analyze
            tau: Time horizon
            bloat_factor: Bloating factor for destination boxes
            parallel: Whether to use parallel processing
            progress_callback: Progress callback function

        Returns:
            MultiGridBoxMap containing inter- and intra-mode transitions
        """
        from .config import get_default_bloat_factor

        if bloat_factor is None:
            bloat_factor = get_default_bloat_factor()

        vprint("Computing MultiGrid BoxMap...")
        vprint(f"  Modes: {self.modes}")
        vprint(f"  Total boxes: {self.total_boxes}")

        # Initialize the MultiGrid BoxMap
        multi_boxmap = MultiGridBoxMap(self, system, tau)

        # Get all sample points
        all_sample_points = self.get_all_sample_points("corners")

        total_mode_boxes = len(all_sample_points)
        completed = 0

        # Process each (mode, box_index) pair
        for (src_mode, src_box_index), sample_points in all_sample_points.items():

            destination_mode_boxes: Set[Tuple[int, int]] = set()

            # Simulate from each sample point
            for point in sample_points:
                try:
                    trajectory = system.simulate(point, (0, tau))

                    if trajectory.total_duration >= tau:
                        final_state = trajectory.interpolate(tau)
                    elif (
                        trajectory.segments
                        and trajectory.segments[-1].state_values.size > 0
                    ):
                        final_state = trajectory.segments[-1].state_values[-1]
                    else:
                        continue  # Skip failed simulations

                    # Determine destination mode and box
                    try:
                        dest_mode, dest_box_index = self.get_box_from_state(final_state)
                        destination_mode_boxes.add((dest_mode, dest_box_index))
                    except (ValueError, IndexError):
                        # Skip points that land outside valid regions
                        continue

                except Exception:
                    # Skip failed simulations
                    continue

            # Apply bloating within each destination mode
            if destination_mode_boxes:
                bloated_destinations: Set[Tuple[int, int]] = set()

                # Group destinations by mode
                destinations_by_mode = defaultdict(set)
                for dest_mode, dest_box in destination_mode_boxes:
                    destinations_by_mode[dest_mode].add(dest_box)

                # Apply bloating within each mode
                for dest_mode, dest_boxes in destinations_by_mode.items():
                    dest_grid = self.mode_grids[dest_mode]

                    # Get coordinates of destination boxes
                    dest_coords = []
                    for dest_box in dest_boxes:
                        box_bounds = dest_grid.get_box_bounds(dest_box)
                        center = (box_bounds[0] + box_bounds[1]) / 2
                        dest_coords.append(center)

                    if dest_coords:
                        dest_coords = np.array(dest_coords)

                        # Create bloated bounding box
                        min_coords = np.min(dest_coords, axis=0)
                        max_coords = np.max(dest_coords, axis=0)

                        bloat_amount = dest_grid.box_widths * bloat_factor
                        bloated_min = min_coords - bloat_amount
                        bloated_max = max_coords + bloat_amount

                        # Find all boxes intersecting bloated region
                        grid_bounds = dest_grid.bounds
                        clipped_min = np.maximum(bloated_min, grid_bounds[:, 0])
                        clipped_max = np.minimum(bloated_max, grid_bounds[:, 1])

                        # Convert to box indices
                        min_indices = np.floor(
                            (clipped_min - grid_bounds[:, 0]) / dest_grid.box_widths,
                        ).astype(int)
                        max_indices = np.floor(
                            (clipped_max - grid_bounds[:, 0]) / dest_grid.box_widths,
                        ).astype(int)

                        min_indices = np.maximum(0, min_indices)
                        max_indices = np.minimum(
                            dest_grid.subdivisions - 1, max_indices,
                        )
                        max_indices = np.maximum(min_indices, max_indices)

                        # Add all boxes in bloated region
                        if dest_grid.ndim == 1:
                            for box_idx in range(min_indices[0], max_indices[0] + 1):
                                bloated_destinations.add((dest_mode, box_idx))
                        else:
                            # Handle multi-dimensional case
                            from itertools import product

                            ranges = [
                                range(min_indices[d], max_indices[d] + 1)
                                for d in range(dest_grid.ndim)
                            ]
                            for multi_index in product(*ranges):
                                box_idx = int(
                                    np.ravel_multi_index(
                                        multi_index, dest_grid.subdivisions, mode="clip",
                                    ),
                                )
                                bloated_destinations.add((dest_mode, box_idx))

                if bloated_destinations:
                    multi_boxmap[(src_mode, src_box_index)] = bloated_destinations

            completed += 1
            if progress_callback:
                progress_callback(completed, total_mode_boxes)

        vprint("✓ MultiGrid BoxMap computation complete.", level="always")

        return multi_boxmap

    def compute_multi_boxmap_interval(
        self,
        system: HybridSystem,
        tau: float,
        bloat_factor: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MultiGridBoxMap:
        """
        Compute MultiGrid BoxMap using interval-based approach for 1D systems.

        This method is designed for 1D hybrid systems (like thermostats) where each mode
        has a 1D continuous state. Instead of just using corner points, it computes
        the image of the entire interval, properly handling mode switches.

        Args:
            system: Hybrid system to analyze
            tau: Time horizon
            bloat_factor: Bloating factor for destination boxes
            progress_callback: Progress callback function

        Returns:
            MultiGridBoxMap containing inter- and intra-mode transitions
        """
        from .config import get_default_bloat_factor

        # Verify this is a 1D system in each mode
        for mode, grid in self.mode_grids.items():
            if grid.ndim != 1:
                raise ValueError(
                    f"Interval method only supports 1D grids, but mode {mode} has {grid.ndim}D grid",
                )

        if bloat_factor is None:
            bloat_factor = get_default_bloat_factor()

        vprint("Computing MultiGrid BoxMap using interval method...")
        vprint(f"  Modes: {self.modes}")
        vprint(f"  Total boxes: {self.total_boxes}")

        # Initialize the MultiGrid BoxMap
        multi_boxmap = MultiGridBoxMap(self, system, tau)

        total_mode_boxes = sum(grid.total_boxes for grid in self.mode_grids.values())
        completed = 0

        # Process each (mode, box_index) pair
        for src_mode, src_grid in self.mode_grids.items():
            for src_box_index in src_grid.box_indices:
                # Get box bounds [a, b]
                box_bounds = src_grid.get_box_bounds(src_box_index)
                a, b = (
                    box_bounds[0][0],
                    box_bounds[1][0],
                )  # Extract scalars from 1D arrays

                # Create corner states (temperature, mode)
                state_a = np.array([a, float(src_mode)])
                state_b = np.array([b, float(src_mode)])

                try:
                    # Simulate from both corners
                    traj_a = system.simulate(state_a, (0, tau))
                    traj_b = system.simulate(state_b, (0, tau))

                    # Get final states
                    if traj_a.total_duration >= tau:
                        final_a = traj_a.interpolate(tau)
                    elif traj_a.segments and traj_a.segments[-1].state_values.size > 0:
                        final_a = traj_a.segments[-1].state_values[-1]
                    else:
                        continue

                    if traj_b.total_duration >= tau:
                        final_b = traj_b.interpolate(tau)
                    elif traj_b.segments and traj_b.segments[-1].state_values.size > 0:
                        final_b = traj_b.segments[-1].state_values[-1]
                    else:
                        continue

                    # Extract temperatures and modes from final states
                    temp_final_a, mode_final_a = final_a[0], int(round(final_a[1]))
                    temp_final_b, mode_final_b = final_b[0], int(round(final_b[1]))

                    # Check if there was a mode switch between the corners
                    if (
                        mode_final_a == mode_final_b
                        and traj_a.num_jumps == traj_b.num_jumps
                    ):
                        # No mode switch - use simple interval
                        temp_min = min(temp_final_a, temp_final_b)
                        temp_max = max(temp_final_a, temp_final_b)

                        # Add boxes for this interval with bloating
                        self._add_boxes_for_interval(
                            multi_boxmap,
                            (src_mode, src_box_index),
                            mode_final_a,
                            temp_min,
                            temp_max,
                            bloat_factor,
                        )
                    else:
                        # Mode switch occurred - need to handle jump point
                        jump_info = self._find_jump_between_trajectories(
                            traj_a, traj_b, a, b, src_mode,
                        )

                        if jump_info is not None:
                            jump_temp, jump_mode = jump_info

                            # Add interval from a to jump point in original mode
                            if mode_final_a == src_mode:
                                # Trajectory a didn't jump
                                self._add_boxes_for_interval(
                                    multi_boxmap,
                                    (src_mode, src_box_index),
                                    src_mode,
                                    temp_final_a,
                                    jump_temp,
                                    bloat_factor,
                                )
                            # Trajectory a jumped - use pre-jump temperature
                            elif traj_a.segments and len(traj_a.segments) > 1:
                                pre_jump_state = traj_a.segments[0].state_values[-1]
                                pre_jump_temp = pre_jump_state[0]
                                self._add_boxes_for_interval(
                                    multi_boxmap,
                                    (src_mode, src_box_index),
                                    src_mode,
                                    pre_jump_temp,
                                    jump_temp,
                                    bloat_factor,
                                )

                            # Add interval from jump point to b in switched mode
                            if mode_final_b != src_mode:
                                # Trajectory b jumped
                                self._add_boxes_for_interval(
                                    multi_boxmap,
                                    (src_mode, src_box_index),
                                    mode_final_b,
                                    jump_temp,
                                    temp_final_b,
                                    bloat_factor,
                                )
                            # Trajectory b didn't jump - this shouldn't happen if jump_info exists
                            # But handle it anyway
                            elif traj_b.segments and len(traj_b.segments) > 1:
                                post_jump_state = traj_b.segments[1].state_values[0]
                                post_jump_temp = post_jump_state[0]
                                self._add_boxes_for_interval(
                                    multi_boxmap,
                                    (src_mode, src_box_index),
                                    1 - src_mode,
                                    jump_temp,
                                    post_jump_temp,
                                    bloat_factor,
                                )
                        else:
                            # Couldn't find jump point - fall back to adding both endpoints
                            self._add_boxes_for_interval(
                                multi_boxmap,
                                (src_mode, src_box_index),
                                mode_final_a,
                                temp_final_a,
                                temp_final_a,
                                bloat_factor,
                            )
                            self._add_boxes_for_interval(
                                multi_boxmap,
                                (src_mode, src_box_index),
                                mode_final_b,
                                temp_final_b,
                                temp_final_b,
                                bloat_factor,
                            )

                except Exception as e:
                    # Skip failed simulations
                    vprint(
                        f"Warning: Failed to process box ({src_mode}, {src_box_index}): {e}",
                        level="always",
                    )
                    continue

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_mode_boxes)

        vprint(
            "✓ MultiGrid BoxMap (interval method) computation complete.", level="always",
        )
        return multi_boxmap

    def _find_jump_between_trajectories(
        self, traj_a, traj_b, temp_a: float, temp_b: float, src_mode: int,
    ) -> Optional[Tuple[float, int]]:
        """
        Find the temperature where a jump occurs between two trajectories.

        Args:
            traj_a, traj_b: Trajectories from corner points
            temp_a, temp_b: Initial temperatures
            src_mode: Source mode

        Returns:
            Tuple of (jump_temperature, destination_mode) or None if no clear jump found
        """
        # Check if trajectories have different number of jumps
        if traj_a.num_jumps != traj_b.num_jumps:
            # Find which trajectory jumped
            if traj_a.num_jumps > traj_b.num_jumps:
                # Trajectory a jumped
                if traj_a.segments and len(traj_a.segments) > 1:
                    # Get the state just before the jump
                    pre_jump_state = traj_a.segments[0].state_values[-1]
                    jump_temp = pre_jump_state[0]
                    dest_mode = 1 - src_mode
                    return (jump_temp, dest_mode)
            # Trajectory b jumped
            elif traj_b.segments and len(traj_b.segments) > 1:
                # Get the state just before the jump
                pre_jump_state = traj_b.segments[0].state_values[-1]
                jump_temp = pre_jump_state[0]
                dest_mode = 1 - src_mode
                return (jump_temp, dest_mode)

        return None

    def _add_boxes_for_interval(
        self,
        multi_boxmap: MultiGridBoxMap,
        src_key: Tuple[int, int],
        dest_mode: int,
        temp_min: float,
        temp_max: float,
        bloat_factor: float,
    ):
        """
        Add all boxes that intersect the given temperature interval.

        Args:
            multi_boxmap: The MultiGridBoxMap to update
            src_key: Source (mode, box_index) tuple
            dest_mode: Destination mode
            temp_min, temp_max: Temperature interval bounds
            bloat_factor: Bloating factor
        """
        if src_key not in multi_boxmap:
            multi_boxmap[src_key] = set()

        dest_grid = self.mode_grids[dest_mode]

        # Apply bloating
        bloat_amount = dest_grid.box_widths[0] * bloat_factor
        bloated_min = temp_min - bloat_amount
        bloated_max = temp_max + bloat_amount

        # Clip to grid bounds
        grid_bounds = dest_grid.bounds
        clipped_min = max(bloated_min, grid_bounds[0, 0])
        clipped_max = min(bloated_max, grid_bounds[0, 1])

        if clipped_min > clipped_max:
            return  # No intersection with grid

        # Find box indices
        min_idx = int(
            np.floor((clipped_min - grid_bounds[0, 0]) / dest_grid.box_widths[0]),
        )
        max_idx = int(
            np.floor((clipped_max - grid_bounds[0, 0]) / dest_grid.box_widths[0]),
        )

        # Ensure indices are valid
        min_idx = max(0, min_idx)
        max_idx = min(dest_grid.subdivisions[0] - 1, max_idx)

        # Add all boxes in the range
        for box_idx in range(min_idx, max_idx + 1):
            multi_boxmap[src_key].add((dest_mode, box_idx))

    def __repr__(self) -> str:
        """String representation."""
        mode_info = {
            mode: f"{grid.total_boxes} boxes" for mode, grid in self.mode_grids.items()
        }
        return (
            f"MultiGrid(modes={mode_info}, transitions={list(self.mode_graph.edges())})"
        )
