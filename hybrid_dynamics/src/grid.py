"""
Grid Manager for systematic domain discretization.

This module provides the Grid class for creating systematic rectangular grids
that leverage the existing Box geometric primitives.
"""

from __future__ import annotations

import itertools
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .box import Box


class Grid:
    """
    Manages systematic discretization of N-dimensional rectangular domains.

    Creates regular grids with systematic integer indexing, designed for
    parallel function evaluation over discretized domains.
    """

    def __init__(self, bounds: List[List[float]], subdivisions: List[int]):
        """
        Initialize grid with systematic box layout.

        Args:
            bounds: List of [min, max] pairs for each dimension
                   e.g., [[-2, 2], [-1, 1]] for 2D domain
            subdivisions: Number of divisions per dimension
                         e.g., [50, 50] creates 2500 boxes total
        """
        if len(bounds) != len(subdivisions):
            raise ValueError("bounds and subdivisions must have same length")

        if any(s <= 0 for s in subdivisions):
            raise ValueError("All subdivisions must be positive")

        if any(b[1] <= b[0] for b in bounds):
            raise ValueError("All bounds must have max > min")

        self.bounds = np.array(bounds)
        self.subdivisions = np.array(subdivisions)
        self.ndim = len(bounds)
        self.total_boxes = int(np.prod(subdivisions))

        # Calculate grid spacing
        self.box_widths = (self.bounds[:, 1] - self.bounds[:, 0]) / self.subdivisions

    @classmethod
    def from_center_and_size(
        cls,
        center: Union[List[float], np.ndarray],
        total_size: Union[List[float], np.ndarray],
        subdivisions: List[int],
    ) -> Grid:
        """
        Create grid from center point and total domain size.

        Args:
            center: Center point of domain
            total_size: Total size in each dimension
            subdivisions: Number of divisions per dimension
        """
        center = np.asarray(center)
        total_size = np.asarray(total_size)

        half_size = total_size / 2
        bounds = [[c - hs, c + hs] for c, hs in zip(center, half_size)]

        return cls(bounds, subdivisions)

    @classmethod
    def from_domain_box(cls, domain_box: Box, subdivisions: List[int]) -> Grid:
        """
        Create grid from existing Box defining the domain.

        Args:
            domain_box: Box instance defining the domain
            subdivisions: Number of divisions per dimension
        """
        bounds = [
            [domain_box.bounds[i, 0], domain_box.bounds[i, 1]]
            for i in range(domain_box.dimension)
        ]

        return cls(bounds, subdivisions)

    def get_box_from_point(self, point: Union[List[float], np.ndarray]) -> int:
        """
        Find the box index containing the given point.

        Args:
            point: N-dimensional coordinates

        Returns:
            Linear box index
        """
        point = np.asarray(point)

        if len(point) != self.ndim:
            raise ValueError(f"Point must have {self.ndim} dimensions")

        if not np.all((point >= self.bounds[:, 0]) & (point <= self.bounds[:, 1])):
            raise ValueError("Point outside grid domain")

        # Convert to grid coordinates
        grid_coords = np.floor((point - self.bounds[:, 0]) / self.box_widths).astype(
            int,
        )
        grid_coords = np.minimum(grid_coords, self.subdivisions - 1)

        # Convert to linear index
        return int(np.ravel_multi_index(grid_coords, self.subdivisions))

    def get_box_bounds(self, box_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds of a box by its index.

        Args:
            box_index: Linear box index

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        if not (0 <= box_index < self.total_boxes):
            raise ValueError(f"Box index {box_index} out of range")

        # Convert linear index to grid coordinates
        grid_coords = np.array(np.unravel_index(box_index, self.subdivisions))

        # Calculate bounds
        lower_bounds = self.bounds[:, 0] + grid_coords * self.box_widths
        upper_bounds = lower_bounds + self.box_widths

        return lower_bounds, upper_bounds

    def get_box(self, box_index: int) -> Box:
        """
        Get Box instance for given index.

        Args:
            box_index: Linear box index

        Returns:
            Box instance
        """
        lower_bounds, upper_bounds = self.get_box_bounds(box_index)

        # Create bounds in Box format: [x1_min, x1_max, x2_min, x2_max, ...]
        bounds_flat = []
        for i in range(self.ndim):
            bounds_flat.extend([lower_bounds[i], upper_bounds[i]])

        return Box(bounds_flat)

    def get_sample_points(
        self,
        box_index: int,
        mode: str = "center",
        num_points: int = 1,
        subdivision_level: int = 1,
    ) -> np.ndarray:
        """
        Generate sample points within a box.

        Args:
            box_index: Linear box index
            mode: Sampling strategy - 'center', 'corners', 'random', or 'subdivision'
            num_points: Number of random points (only for 'random' mode)
            subdivision_level: Level of subdivision (only for 'subdivision' mode)

        Returns:
            Array of sample points with shape (n_points, ndim)
        """
        if mode not in ["center", "corners", "random", "subdivision"]:
            raise ValueError(
                "mode must be 'center', 'corners', 'random', or 'subdivision'",
            )

        lower_bounds, upper_bounds = self.get_box_bounds(box_index)

        if mode == "center":
            center = (lower_bounds + upper_bounds) / 2
            return center.reshape(1, -1)

        if mode == "corners":
            box = self.get_box(box_index)
            return box.corners

        if mode == "random":
            if num_points <= 0:
                raise ValueError("num_points must be positive for random sampling")

            return np.random.uniform(
                low=lower_bounds, high=upper_bounds, size=(num_points, self.ndim),
            )

        if mode == "subdivision":
            # Generate subdivision points for this box
            n_points_per_dim = 2**subdivision_level + 1

            dim_points = []
            for d in range(self.ndim):
                dim_points.append(
                    np.linspace(lower_bounds[d], upper_bounds[d], n_points_per_dim),
                )

            # Create all combinations
            if self.ndim == 1:
                return dim_points[0].reshape(-1, 1)
            grids = np.meshgrid(*dim_points, indexing="ij")
            return np.column_stack([g.ravel() for g in grids])

    @property
    def box_indices(self) -> Iterator[int]:
        """Iterator over all valid box indices."""
        return range(self.total_boxes)

    def __len__(self) -> int:
        """Number of boxes in the grid."""
        return self.total_boxes

    def __getitem__(self, box_index: int) -> Box:
        """Get box by index using [] notation."""
        return self.get_box(box_index)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Grid(ndim={self.ndim}, bounds={self.bounds.tolist()}, "
            f"subdivisions={self.subdivisions.tolist()}, total_boxes={self.total_boxes}"
        )

    def get_all_unique_points(
        self,
        mode: str = "corners",
        num_random: int = 1,
        data_points: Optional[np.ndarray] = None,
        subdivision_level: int = 1,
    ) -> Tuple[np.ndarray, dict]:
        """
        Get all unique sample points for the entire grid.

        Args:
            mode: Sampling mode - 'corners', 'center', 'random', 'data', or 'subdivision'
            num_random: Number of random points per box (only for 'random' mode)
            data_points: User-provided points (only for 'data' mode)
            subdivision_level: Level of subdivision n (only for 'subdivision' mode)
                             Each box is subdivided into 2^n sub-boxes per dimension

        Returns:
            points: Array of unique points with shape (n_points, ndim)
            metadata: Dictionary with mode-specific information
        """
        if mode == "corners":
            # Generate all grid intersection points
            # For subdivisions [n1, n2, ...], we have (n1+1) x (n2+1) x ... points
            ranges = []
            for i in range(self.ndim):
                ranges.append(
                    np.linspace(
                        self.bounds[i, 0], self.bounds[i, 1], self.subdivisions[i] + 1,
                    ),
                )

            # Create meshgrid and flatten to get all corner points
            grids = np.meshgrid(*ranges, indexing="ij")
            points = np.column_stack([g.ravel() for g in grids])

            metadata = {
                "mode": "corners",
                "grid_shape": tuple(s + 1 for s in self.subdivisions),
            }

        elif mode == "center":
            # Generate center points for all boxes
            ranges = []
            for i in range(self.ndim):
                # Centers are at half-steps
                centers = np.linspace(
                    self.bounds[i, 0] + self.box_widths[i] / 2,
                    self.bounds[i, 1] - self.box_widths[i] / 2,
                    self.subdivisions[i],
                )
                ranges.append(centers)

            grids = np.meshgrid(*ranges, indexing="ij")
            points = np.column_stack([g.ravel() for g in grids])

            metadata = {"mode": "center", "grid_shape": tuple(self.subdivisions)}

        elif mode == "random":
            # Generate unique random points for each box
            points_list = []
            np.random.seed(42)  # For reproducibility

            for box_idx in range(self.total_boxes):
                lower_bounds, upper_bounds = self.get_box_bounds(box_idx)
                box_points = np.random.uniform(
                    low=lower_bounds, high=upper_bounds, size=(num_random, self.ndim),
                )
                points_list.extend(box_points)

            points = np.array(points_list)
            metadata = {"mode": "random", "num_per_box": num_random, "seed": 42}

        elif mode == "data":
            if data_points is None:
                raise ValueError("data_points must be provided for 'data' mode")

            # Remove duplicates while preserving order
            unique_points, unique_indices = np.unique(
                data_points, axis=0, return_index=True,
            )
            points = unique_points[np.argsort(unique_indices)]

            metadata = {
                "mode": "data",
                "original_count": len(data_points),
                "unique_count": len(points),
            }

        elif mode == "subdivision":
            # Generate subdivision points for all boxes
            # Each box is subdivided into 2^n sub-boxes per dimension
            points_set = set()  # Use set to automatically handle duplicates

            # Number of subdivision points per dimension (including boundaries)
            n_points_per_dim = 2**subdivision_level + 1

            # Process each box
            for box_idx in range(self.total_boxes):
                lower_bounds, upper_bounds = self.get_box_bounds(box_idx)

                # Create subdivision points for this box
                dim_points = []
                for d in range(self.ndim):
                    # Create evenly spaced points in this dimension
                    dim_points.append(
                        np.linspace(lower_bounds[d], upper_bounds[d], n_points_per_dim),
                    )

                # Create all combinations of points
                if self.ndim == 1:
                    box_points = dim_points[0].reshape(-1, 1)
                else:
                    grids = np.meshgrid(*dim_points, indexing="ij")
                    box_points = np.column_stack([g.ravel() for g in grids])

                # Add to set (automatically handles duplicates at boundaries)
                for point in box_points:
                    points_set.add(tuple(point))

            # Convert set back to array
            points = np.array(list(points_set))

            metadata = {
                "mode": "subdivision",
                "subdivision_level": subdivision_level,
                "points_per_box_dim": n_points_per_dim,
                "total_unique_points": len(points),
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return points, metadata

    def find_boxes_containing_point(
        self, point: np.ndarray, tolerance: float = 1e-10,
    ) -> List[int]:
        """
        Find all box indices that contain or share this point.

        Args:
            point: N-dimensional coordinates
            tolerance: Numerical tolerance for boundary detection

        Returns:
            List of box indices that contain the point
        """
        point = np.asarray(point)

        if len(point) != self.ndim:
            raise ValueError(f"Point must have {self.ndim} dimensions")

        # Check if point is within grid bounds
        if not np.all(
            (point >= self.bounds[:, 0] - tolerance)
            & (point <= self.bounds[:, 1] + tolerance),
        ):
            return []  # Point outside grid

        # Compute which box the point would be in (if interior)
        relative_pos = (point - self.bounds[:, 0]) / self.box_widths
        base_indices = np.floor(relative_pos).astype(int)

        # Check if point is on any boundary
        # A point is on a lower boundary if it's very close to a grid line
        on_lower_boundary = np.abs(relative_pos - base_indices) < tolerance
        # A point is on an upper boundary if it's very close to the next grid line
        on_upper_boundary = np.abs(relative_pos - (base_indices + 1)) < tolerance

        # Handle edge cases at domain boundaries
        at_lower_domain = np.abs(point - self.bounds[:, 0]) < tolerance
        at_upper_domain = np.abs(point - self.bounds[:, 1]) < tolerance

        # Adjust for domain boundaries
        for i in range(self.ndim):
            if at_lower_domain[i]:
                on_lower_boundary[i] = True
                base_indices[i] = 0
            if at_upper_domain[i]:
                on_upper_boundary[i] = True
                base_indices[i] = min(base_indices[i], self.subdivisions[i] - 1)

        # Generate all adjacent boxes that share this point
        adjacent_boxes = []

        # Create offset combinations
        offsets = []
        for i in range(self.ndim):
            if on_lower_boundary[i] and base_indices[i] > 0:
                if on_upper_boundary[i]:
                    # Point is exactly on a grid line
                    offsets.append([-1, 0])
                else:
                    offsets.append([-1, 0])
            elif on_upper_boundary[i] and base_indices[i] < self.subdivisions[i] - 1:
                offsets.append([0, 1])
            else:
                offsets.append([0])

        # Generate all combinations
        for offset_combo in itertools.product(*offsets):
            box_indices = base_indices + np.array(offset_combo)

            # Check bounds
            if np.all(box_indices >= 0) and np.all(box_indices < self.subdivisions):
                box_idx = int(np.ravel_multi_index(box_indices, self.subdivisions))
                if box_idx not in adjacent_boxes:
                    adjacent_boxes.append(box_idx)

        return adjacent_boxes
