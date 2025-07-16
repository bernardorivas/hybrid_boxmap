"""
Box and SquareBox classes for spatial cubification.

This module provides geometric primitives for representing rectangular regions
in state space, with special support for square/cubic boxes.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Box:
    """Represents a rectangular box in n-dimensional space.

    A box is defined by lower and upper bounds in each dimension.
    """

    def __init__(self, bounds: Union[np.ndarray, List[float]]):
        """Initialize box from bounds array.

        Args:
            bounds: Array of [x1_min, x1_max, x2_min, x2_max, ...] or
                   List of (min, max) tuples
        """
        bounds = np.asarray(bounds)

        if len(bounds) % 2 != 0:
            raise ValueError("Bounds array must have even length")

        self.dimension = len(bounds) // 2
        self.bounds = bounds.reshape(self.dimension, 2)

        # Validate bounds
        for i in range(self.dimension):
            if self.bounds[i, 0] > self.bounds[i, 1]:
                raise ValueError(f"Lower bound must be <= upper bound in dimension {i}")

    @classmethod
    def from_bounds_list(cls, bounds_list: List[Tuple[float, float]]) -> "Box":
        """Create box from list of (min, max) tuples.

        Args:
            bounds_list: List of (min, max) tuples for each dimension

        Returns:
            Box instance
        """
        bounds_array = np.array(bounds_list).flatten()
        return cls(bounds_array)

    @classmethod
    def from_center_and_size(cls, center: np.ndarray, sizes: np.ndarray) -> "Box":
        """Create box from center point and sizes.

        Args:
            center: Center point of box
            sizes: Size in each dimension

        Returns:
            Box instance
        """
        center = np.asarray(center)
        sizes = np.asarray(sizes)

        if len(center) != len(sizes):
            raise ValueError("Center and sizes must have same dimension")

        half_sizes = sizes / 2
        bounds_list = [(c - hs, c + hs) for c, hs in zip(center, half_sizes)]
        return cls.from_bounds_list(bounds_list)

    @property
    def lower_bounds(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper_bounds(self) -> np.ndarray:
        return self.bounds[:, 1]

    @property
    def center(self) -> np.ndarray:
        return (self.lower_bounds + self.upper_bounds) / 2

    @property
    def sizes(self) -> np.ndarray:
        return self.upper_bounds - self.lower_bounds

    @property
    def volume(self) -> float:
        return np.prod(self.sizes)

    @property
    def corners(self) -> np.ndarray:
        """All corner points of the box.

        Returns:
            Array of shape (2^dimension, dimension) with all corners
        """
        if self.dimension > 10:  # Avoid memory issues
            raise ValueError("Too many dimensions for corner enumeration")

        # Generate all combinations of lower/upper bounds
        corner_coords = []
        for i in range(2**self.dimension):
            corner = []
            for d in range(self.dimension):
                if (i >> d) & 1:
                    corner.append(self.upper_bounds[d])
                else:
                    corner.append(self.lower_bounds[d])
            corner_coords.append(corner)

        return np.array(corner_coords)

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside box.

        Args:
            point: Point to check

        Returns:
            True if point is inside box
        """
        point = np.asarray(point)
        if len(point) != self.dimension:
            raise ValueError(
                f"Point dimension {len(point)} != box dimension {self.dimension}",
            )

        return np.all((point >= self.lower_bounds) & (point <= self.upper_bounds))

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside box.

        Args:
            points: Array of shape (n_points, dimension)

        Returns:
            Boolean array of shape (n_points,)
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] != self.dimension:
            raise ValueError(
                f"Points dimension {points.shape[1]} != box dimension {self.dimension}",
            )

        inside = np.all(
            (points >= self.lower_bounds) & (points <= self.upper_bounds), axis=1,
        )
        return inside

    def intersects(self, other: "Box") -> bool:
        """Check if this box intersects with another box.

        Args:
            other: Another Box instance

        Returns:
            True if boxes intersect
        """
        if self.dimension != other.dimension:
            raise ValueError("Boxes must have same dimension")

        # Boxes intersect if they overlap in all dimensions
        for d in range(self.dimension):
            if (
                self.upper_bounds[d] < other.lower_bounds[d]
                or other.upper_bounds[d] < self.lower_bounds[d]
            ):
                return False

        return True

    def intersection(self, other: "Box") -> Optional["Box"]:
        """Compute intersection with another box.

        Args:
            other: Another Box instance

        Returns:
            Box representing intersection, or None if no intersection
        """
        if not self.intersects(other):
            return None

        # Intersection bounds
        lower = np.maximum(self.lower_bounds, other.lower_bounds)
        upper = np.minimum(self.upper_bounds, other.upper_bounds)

        bounds_list = [(lower[i], upper[i]) for i in range(self.dimension)]
        return Box.from_bounds_list(bounds_list)

    def subdivide(self, levels: int = 1) -> List["Box"]:
        """Subdivide box into smaller boxes.

        Args:
            levels: Number of subdivision levels (each level divides each dimension in half)

        Returns:
            List of subdivided boxes
        """
        if levels <= 0:
            return [self]

        # Divide each dimension in half
        boxes = [self]

        for level in range(levels):
            new_boxes = []
            for box in boxes:
                # Split along each dimension
                current_boxes = [box]

                for d in range(box.dimension):
                    split_boxes = []
                    for b in current_boxes:
                        mid = (b.lower_bounds[d] + b.upper_bounds[d]) / 2

                        # Create two boxes by splitting at midpoint
                        bounds1 = b.bounds.copy()
                        bounds1[d, 1] = mid

                        bounds2 = b.bounds.copy()
                        bounds2[d, 0] = mid

                        split_boxes.append(Box(bounds1.flatten()))
                        split_boxes.append(Box(bounds2.flatten()))

                    current_boxes = split_boxes

                new_boxes.extend(current_boxes)

            boxes = new_boxes

        return boxes

    def plot_2d(self, ax: plt.Axes, **kwargs):
        """
        Plot 2D box as a rectangle on the given axes.

        Args:
            ax: Matplotlib axes to plot on.
            **kwargs: Additional arguments for the Rectangle patch.
        """
        if self.dimension != 2:
            raise ValueError("Can only plot 2D boxes.")

        rect = patches.Rectangle(
            xy=self.lower_bounds, width=self.sizes[0], height=self.sizes[1], **kwargs,
        )
        ax.add_patch(rect)

    def plot_2d_projection(
        self, ax: plt.Axes, dims: Tuple[int, int] = (0, 1), **kwargs,
    ):
        """
        Plot 2D projection of n-dimensional box as a rectangle on the given axes.

        Args:
            ax: Matplotlib axes to plot on.
            dims: Tuple of (x_dim, y_dim) indices for projection dimensions.
            **kwargs: Additional arguments for the Rectangle patch.
        """
        if self.dimension < 2:
            raise ValueError("Box must have at least 2 dimensions for projection.")

        x_dim, y_dim = dims
        if x_dim >= self.dimension or y_dim >= self.dimension:
            raise ValueError(
                f"Projection dimensions {dims} exceed box dimension {self.dimension}",
            )

        # Extract bounds for the specified dimensions
        x_min, x_max = self.bounds[x_dim]
        y_min, y_max = self.bounds[y_dim]

        rect = patches.Rectangle(
            xy=(x_min, y_min), width=x_max - x_min, height=y_max - y_min, **kwargs,
        )
        ax.add_patch(rect)

    def plot_3d(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Plot 3D box as cuboid.

        Args:
            ax: Matplotlib 3D axes to plot on
            **kwargs: Additional arguments for plotting

        Returns:
            Matplotlib axes
        """
        if self.dimension != 3:
            raise ValueError("plot_3d only works for 3D boxes")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Get all corners
        corners = self.corners

        # Define faces of the cube (indices into corners array)
        faces = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],  # bottom, top
            [0, 1, 5, 4],
            [2, 3, 7, 6],  # front, back
            [0, 2, 6, 4],
            [1, 3, 7, 5],  # left, right
        ]

        face_vertices = [[corners[j] for j in face] for face in faces]

        # Default styling
        plot_kwargs = {
            "alpha": 0.3,
            "facecolor": "lightblue",
            "edgecolor": "black",
            "linewidth": 1,
        }
        plot_kwargs.update(kwargs)

        collection = Poly3DCollection(face_vertices, **plot_kwargs)
        ax.add_collection3d(collection)

        ax.set_xlim(self.lower_bounds[0], self.upper_bounds[0])
        ax.set_ylim(self.lower_bounds[1], self.upper_bounds[1])
        ax.set_zlim(self.lower_bounds[2], self.upper_bounds[2])

        return ax

    def __str__(self) -> str:
        bounds_str = ", ".join(
            [
                f"[{self.bounds[i,0]:.3f}, {self.bounds[i,1]:.3f}]"
                for i in range(self.dimension)
            ],
        )
        return f"Box({bounds_str})"

    def __eq__(self, other: "Box") -> bool:
        if not isinstance(other, Box):
            return False
        return np.allclose(self.bounds, other.bounds)


class SquareBox(Box):
    """Box with equal side lengths in all dimensions."""

    def __init__(self, bounds: Union[np.ndarray, List[float]]):
        """Initialize square box and validate equal sides.

        Args:
            bounds: Array of [x1_min, x1_max, x2_min, x2_max, ...] or
                   List of (min, max) tuples
        """
        super().__init__(bounds)

        if not self.is_square():
            raise ValueError("All dimensions must have equal size for SquareBox")

    @classmethod
    def from_center(cls, center: np.ndarray, side_length: float) -> "SquareBox":
        """Create square box from center point and side length.

        Args:
            center: Center point of box
            side_length: Side length (same for all dimensions)

        Returns:
            SquareBox instance
        """
        center = np.asarray(center)
        half_side = side_length / 2

        bounds_list = [(c - half_side, c + half_side) for c in center]
        return cls.from_bounds_list(bounds_list)

    @classmethod
    def from_corner(cls, corner: np.ndarray, side_length: float) -> "SquareBox":
        """Create square box from corner point and side length.

        Args:
            corner: Corner point (lower bounds)
            side_length: Side length (same for all dimensions)

        Returns:
            SquareBox instance
        """
        corner = np.asarray(corner)
        bounds_list = [(c, c + side_length) for c in corner]
        return cls.from_bounds_list(bounds_list)

    def is_square(self) -> bool:
        """Check if box has equal sides.

        Returns:
            True if all dimensions have equal size
        """
        sizes = self.sizes
        return np.allclose(sizes, sizes[0])

    @property
    def side_length(self) -> float:
        """Side length of square box."""
        return self.sizes[0]

    def subdivide_square(self, levels: int = 1) -> List["SquareBox"]:
        """Subdivide into smaller square boxes.

        Args:
            levels: Number of subdivision levels

        Returns:
            List of SquareBox instances
        """
        regular_boxes = self.subdivide(levels)
        return [SquareBox(box.bounds.flatten()) for box in regular_boxes]

    def __str__(self) -> str:
        """String representation of square box."""
        center_str = ", ".join([f"{c:.3f}" for c in self.center])
        return f"SquareBox(center=({center_str}), side={self.side_length:.3f})"
