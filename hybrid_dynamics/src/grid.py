"""
Grid Manager for systematic domain discretization.

This module provides the Grid class for creating systematic rectangular grids
that leverage the existing Box geometric primitives.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple, Union

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
        subdivisions: List[int]
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
        bounds = [[domain_box.bounds[i, 0], domain_box.bounds[i, 1]] 
                  for i in range(domain_box.dimension)]
        
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
        grid_coords = np.floor((point - self.bounds[:, 0]) / self.box_widths).astype(int)
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
    
    def get_sample_points(self, box_index: int, mode: str = 'center', num_points: int = 1) -> np.ndarray:
        """
        Generate sample points within a box.
        
        Args:
            box_index: Linear box index
            mode: Sampling strategy - 'center', 'corners', or 'random'
            num_points: Number of random points (only for 'random' mode)
            
        Returns:
            Array of sample points with shape (n_points, ndim)
        """
        if mode not in ['center', 'corners', 'random']:
            raise ValueError("mode must be 'center', 'corners', or 'random'")
        
        lower_bounds, upper_bounds = self.get_box_bounds(box_index)
        
        if mode == 'center':
            center = (lower_bounds + upper_bounds) / 2
            return center.reshape(1, -1)
        
        elif mode == 'corners':
            box = self.get_box(box_index)
            return box.corners
        
        elif mode == 'random':
            if num_points <= 0:
                raise ValueError("num_points must be positive for random sampling")
            
            return np.random.uniform(
                low=lower_bounds,
                high=upper_bounds,
                size=(num_points, self.ndim)
            )
    
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
        return (f"Grid(ndim={self.ndim}, bounds={self.bounds.tolist()}, "
                f"subdivisions={self.subdivisions.tolist()}, total_boxes={self.total_boxes})")