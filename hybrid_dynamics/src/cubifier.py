"""
This module provides tools for cubifying datasets (trajectories, point clouds)
into grid-aligned boxes for analysis and visualization.
"""

import itertools
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patches
from matplotlib.collections import PatchCollection

from ..src.hybrid_trajectory import HybridTrajectory
from .box import Box, SquareBox
from .config import config

logger = config.get_logger(__name__)


class DatasetCubifier:
    """Cubifies datasets into grid-aligned boxes.

    This class provides methods to find minimal sets of boxes that contain
    given datasets like trajectories or point clouds.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        base_resolution: int = 10,
        subdivision_levels: int = 3,
    ):
        """Initialize cubifier with domain and resolution parameters.

        Args:
            bounds: Domain bounds for each dimension [(x1_min, x1_max), ...]
            base_resolution: Initial grid divisions per dimension
            subdivision_levels: How many refinement levels to allow
        """
        self.bounds = bounds
        self.dimension = len(bounds)
        self.base_resolution = base_resolution
        self.subdivision_levels = subdivision_levels

        # Validate bounds
        for i, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(
                    f"Invalid bounds for dimension {i}: [{lower}, {upper}]",
                )

        # Create base grid
        self._create_base_grid()

    def _create_base_grid(self):
        """Create the base grid of boxes efficiently using numpy."""
        self.base_boxes = []

        # Generate grid coordinates efficiently
        grid_coords = []
        for lower, upper in self.bounds:
            # Create evenly spaced coordinates
            coords = np.linspace(lower, upper, self.base_resolution + 1)
            grid_coords.append(coords)

        # Calculate grid spacing
        grid_sizes = [(coords[1] - coords[0]) for coords in grid_coords]

        # Generate all grid boxes using itertools for arbitrary dimensions
        # Create index ranges for each dimension
        index_ranges = [range(self.base_resolution) for _ in range(self.dimension)]

        # Use itertools.product for efficient iteration
        for indices in itertools.product(*index_ranges):
            # Create bounds for this box
            box_bounds = []
            for dim, idx in enumerate(indices):
                lower = grid_coords[dim][idx]
                upper = grid_coords[dim][idx + 1]
                box_bounds.append((lower, upper))

            box = Box.from_bounds_list(box_bounds)
            self.base_boxes.append(box)

        logger.debug(
            f"Created {len(self.base_boxes)} base grid boxes for {self.dimension}D space",
        )

    def _make_boxes_square(self, boxes: List[Box]) -> List[SquareBox]:
        """Convert regular boxes to square boxes.

        Args:
            boxes: List of regular Box instances

        Returns:
            List of SquareBox instances
        """
        square_boxes = []

        for box in boxes:
            # Find the maximum side length
            max_side = np.max(box.sizes)

            # Create square box centered at original box center
            square_box = SquareBox.from_center(box.center, max_side)
            square_boxes.append(square_box)

        return square_boxes

    def cubify_trajectory(
        self,
        trajectory: HybridTrajectory,
        make_square: bool = True,
        use_subdivision: bool = True,
    ) -> List[Union[Box, SquareBox]]:
        """Find all boxes containing the trajectory.

        Args:
            trajectory: HybridTrajectory to cubify
            make_square: Force boxes to be square/cubic
            use_subdivision: Whether to use subdivision for better fit

        Returns:
            List of boxes containing trajectory points
        """
        if not trajectory.segments:
            return []

        # Get all trajectory points
        all_states = trajectory.get_all_states()

        if all_states.size == 0:
            return []

        # Ensure correct dimensionality
        if all_states.ndim == 1:
            all_states = all_states.reshape(-1, 1)

        if all_states.shape[1] != self.dimension:
            raise ValueError(
                f"Trajectory dimension {all_states.shape[1]} != cubifier dimension {self.dimension}",
            )

        return self.cubify_points(
            all_states, make_square=make_square, use_subdivision=use_subdivision,
        )

    def cubify_points(
        self,
        points: npt.NDArray[np.float64],
        tolerance: float = 0.0,
        make_square: bool = True,
        use_subdivision: bool = True,
    ) -> List[Union[Box, SquareBox]]:
        """Cubify a cloud of points with optional tolerance.

        Args:
            points: Array of shape (n_points, dimension)
            tolerance: Optional expansion tolerance for boxes
            make_square: Force boxes to be square/cubic
            use_subdivision: Whether to use subdivision for better fit

        Returns:
            List of boxes containing all points
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if points.shape[1] != self.dimension:
            raise ValueError(
                f"Points dimension {points.shape[1]} != cubifier dimension {self.dimension}",
            )

        if len(points) == 0:
            return []

        # Find base boxes that intersect with points efficiently
        containing_boxes = []

        # Pre-compute which boxes contain points using vectorized operations
        for box in self.base_boxes:
            if np.any(box.contains_points(points)):
                containing_boxes.append(box)

        logger.debug(f"Found {len(containing_boxes)} boxes containing points")

        # Optionally subdivide for better fit
        if use_subdivision and self.subdivision_levels > 0:
            refined_boxes = []

            for box in containing_boxes:
                # Get points within this box
                mask = box.contains_points(points)
                box_points = points[mask]

                if len(box_points) > 0:
                    # Check if subdivision would help (points occupy small portion of box)
                    point_range = np.ptp(box_points, axis=0)  # peak-to-peak range
                    box_sizes = box.sizes

                    # Only subdivide if points occupy less than 50% of box in any dimension
                    if np.any(point_range < 0.5 * box_sizes):
                        # Try subdivision
                        subdivided = box.subdivide(1)

                        # Keep only subdivided boxes that contain points
                        for sub_box in subdivided:
                            if np.any(sub_box.contains_points(points)):
                                refined_boxes.append(sub_box)
                    else:
                        # Keep original box if subdivision wouldn't help
                        refined_boxes.append(box)

            if refined_boxes:
                containing_boxes = refined_boxes
                logger.debug(
                    f"Refined to {len(containing_boxes)} boxes after subdivision",
                )

        # Apply tolerance expansion if specified
        if tolerance > 0:
            expanded_boxes = []
            for box in containing_boxes:
                expanded_sizes = box.sizes + 2 * tolerance
                expanded_box = Box.from_center_and_size(box.center, expanded_sizes)
                expanded_boxes.append(expanded_box)
            containing_boxes = expanded_boxes

        # Convert to square boxes if requested
        if make_square:
            containing_boxes = self._make_boxes_square(containing_boxes)

        return containing_boxes

    def cubify_bounding_box(
        self, points: np.ndarray, make_square: bool = True,
    ) -> Union[Box, SquareBox]:
        """Create a single box that bounds all points.

        Args:
            points: Array of shape (n_points, dimension)
            make_square: Force box to be square/cubic

        Returns:
            Single box containing all points
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if points.shape[1] != self.dimension:
            raise ValueError(
                f"Points dimension {points.shape[1]} != cubifier dimension {self.dimension}",
            )

        if len(points) == 0:
            raise ValueError("Cannot create bounding box for empty point set")

        # Find bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        bounds_list = [(min_coords[i], max_coords[i]) for i in range(self.dimension)]
        box = Box.from_bounds_list(bounds_list)

        if make_square:
            box = self._make_boxes_square([box])[0]

        return box

    def plot_cubification(
        self,
        boxes: List[Union[Box, SquareBox]],
        dataset: Union[HybridTrajectory, np.ndarray],
        ax: Optional[plt.Axes] = None,
        box_style: dict = None,
        trajectory_style: dict = None,
    ) -> plt.Axes:
        """Visualize cubification with dataset overlay.

        Args:
            boxes: List of boxes to plot
            dataset: HybridTrajectory or point array to overlay
            ax: Matplotlib axes to plot on
            box_style: Style dictionary for box plotting
            trajectory_style: Style dictionary for trajectory plotting

        Returns:
            Matplotlib axes
        """
        if self.dimension != 2:
            raise NotImplementedError("Visualization only implemented for 2D")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Default styles
        default_box_style = {
            "facecolor": "lightblue",
            "edgecolor": "black",
            "alpha": 0.3,
            "linewidth": 1,
        }
        default_trajectory_style = {"color": "red", "linewidth": 2, "alpha": 0.8}

        if box_style is not None:
            default_box_style.update(box_style)
        if trajectory_style is not None:
            default_trajectory_style.update(trajectory_style)

        # Plot boxes
        box_patches = []
        for box in boxes:
            rect = patches.Rectangle(
                (box.lower_bounds[0], box.lower_bounds[1]), box.sizes[0], box.sizes[1],
            )
            box_patches.append(rect)

        box_collection = PatchCollection(box_patches, **default_box_style)
        ax.add_collection(box_collection)

        # Plot dataset
        if isinstance(dataset, HybridTrajectory):
            # Plot trajectory segments
            for segment in dataset.segments:
                if segment.state_values.shape[1] >= 2:
                    ax.plot(
                        segment.state_values[:, 0],
                        segment.state_values[:, 1],
                        **default_trajectory_style,
                    )

            # Plot jump transitions as dashed lines
            for jump_time, (state_before, state_after) in zip(
                dataset.jump_times, dataset.jump_states,
            ):
                if len(state_before) >= 2 and len(state_after) >= 2:
                    # Draw reset map as dashed line segment
                    ax.plot(
                        [state_before[0], state_after[0]],
                        [state_before[1], state_after[1]],
                        color="red",
                        alpha=0.7,
                        linewidth=1.5,
                        linestyle="--",
                    )

        else:
            # Plot point cloud
            points = np.asarray(dataset)
            if points.ndim == 1:
                points = points.reshape(-1, 1)

            if points.shape[1] >= 2:
                ax.scatter(points[:, 0], points[:, 1], **default_trajectory_style)

        # Set axis properties
        domain_box = Box.from_bounds_list(self.bounds)
        ax.set_xlim(domain_box.lower_bounds[0], domain_box.upper_bounds[0])
        ax.set_ylim(domain_box.lower_bounds[1], domain_box.upper_bounds[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

        return ax

    def get_coverage_statistics(
        self, boxes: List[Union[Box, SquareBox]], points: np.ndarray,
    ) -> dict:
        """Compute statistics about how well boxes cover points.

        Args:
            boxes: List of boxes
            points: Points to analyze coverage for

        Returns:
            Dictionary with coverage statistics
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        total_points = len(points)
        covered_points = 0
        total_volume = 0

        # Check coverage
        for box in boxes:
            covered_points += np.sum(box.contains_points(points))
            total_volume += box.volume

        # Domain volume
        domain_box = Box.from_bounds_list(self.bounds)
        domain_volume = domain_box.volume

        return {
            "num_boxes": len(boxes),
            "total_points": total_points,
            "covered_points": covered_points,
            "coverage_ratio": covered_points / total_points if total_points > 0 else 0,
            "total_volume": total_volume,
            "domain_volume": domain_volume,
            "volume_ratio": total_volume / domain_volume,
            "efficiency": (
                (covered_points / total_points) / (total_volume / domain_volume)
                if total_volume > 0
                else 0
            ),
        }

    def __str__(self) -> str:
        """String representation of cubifier."""
        return (
            f"DatasetCubifier({self.dimension}D, resolution={self.base_resolution}, "
            f"subdivision_levels={self.subdivision_levels})"
        )
