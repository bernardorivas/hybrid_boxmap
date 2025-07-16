"""
Utilities for grid-based analysis of datasets, such as identifying
covered regions by a trajectory.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CoveredGrid:
    """
    Represents a 2D grid and which of its cells are covered by a dataset.

    Attributes:
        domain_bounds: The ((xmin, xmax), (ymin, ymax)) of the gridded area.
        grid_resolution: The (nx, ny) number of cells in each dimension.
        x_edges: 1D array of x-axis bin edges (length nx + 1).
        y_edges: 1D array of y-axis bin edges (length ny + 1).
        covered_mask: A 2D boolean NumPy array of shape (nx, ny).
                      True if the cell (i, j) is covered.
                      Indexing corresponds to x_edges[i] to x_edges[i+1] (rows)
                      and y_edges[j] to y_edges[j+1] (columns).
                      Note: This matches the output of np.histogram2d if a transpose
                      is not applied for typical image/mesh plotting conventions.
                      We store it in a way that covered_mask[ix, iy] corresponds to
                      the ix-th bin in x and iy-th bin in y.
    """

    domain_bounds: tuple[tuple[float, float], tuple[float, float]]
    grid_resolution: tuple[int, int]
    x_edges: np.ndarray
    y_edges: np.ndarray
    covered_mask: np.ndarray  # Boolean, shape (nx, ny)

    def __post_init__(self):
        nx, ny = self.grid_resolution
        if self.x_edges.shape != (nx + 1,):
            raise ValueError(
                f"x_edges shape {self.x_edges.shape} does not match grid_resolution nx={nx}",
            )
        if self.y_edges.shape != (ny + 1,):
            raise ValueError(
                f"y_edges shape {self.y_edges.shape} does not match grid_resolution ny={ny}",
            )
        if self.covered_mask.shape != (nx, ny):
            raise ValueError(
                f"covered_mask shape {self.covered_mask.shape} does not match grid_resolution {(nx, ny)}",
            )
        if self.covered_mask.dtype != bool:
            raise ValueError("covered_mask must be a boolean array.")

    def get_rectangle_patches(self, **kwargs) -> List["matplotlib.patches.Rectangle"]:
        """
        Generates a list of matplotlib.patches.Rectangle objects for covered cells.

        Args:
            **kwargs: Keyword arguments to pass to matplotlib.patches.Rectangle
                      (e.g., facecolor, alpha, edgecolor).
                      Default if not provided: facecolor='skyblue', alpha=0.6.

        Returns:
            List of matplotlib.patches.Rectangle objects.
        """
        import matplotlib.pyplot as plt  # Lazy import for plotting dependency

        patches = []
        nx, ny = self.grid_resolution

        # Default styling
        style_args = {"facecolor": "skyblue", "alpha": 0.6}
        style_args.update(kwargs)

        for i in range(nx):
            for j in range(ny):
                if self.covered_mask[i, j]:
                    x0 = self.x_edges[i]
                    y0 = self.y_edges[j]
                    width = self.x_edges[i + 1] - x0
                    height = self.y_edges[j + 1] - y0
                    rect = plt.Rectangle((x0, y0), width, height, **style_args)
                    patches.append(rect)
        return patches


def compute_covered_grid(
    dataset_points: np.ndarray,
    domain_bounds: tuple[tuple[float, float], tuple[float, float]],
    grid_resolution: tuple[int, int],
) -> CoveredGrid:
    """
    Computes which cells of a 2D grid are covered by a given dataset of points.

    Args:
        dataset_points: A 2D NumPy array of shape (N, 2) containing [x, y] points.
        domain_bounds: The ((xmin, xmax), (ymin, ymax)) for the grid.
        grid_resolution: The (nx, ny) number of cells in each dimension.

    Returns:
        An instance of CoveredGrid.
    """
    if dataset_points.ndim != 2 or dataset_points.shape[1] != 2:
        raise ValueError("dataset_points must be a 2D array of shape (N, 2).")

    if (
        len(domain_bounds) != 2
        or len(domain_bounds[0]) != 2
        or len(domain_bounds[1]) != 2
    ):
        raise ValueError(
            "domain_bounds must be of the form ((xmin, xmax), (ymin, ymax)).",
        )

    if len(grid_resolution) != 2 or not all(
        isinstance(n, int) and n > 0 for n in grid_resolution
    ):
        raise ValueError(
            "grid_resolution must be a tuple of two positive integers (nx, ny).",
        )

    (xmin, xmax), (ymin, ymax) = domain_bounds
    nx, ny = grid_resolution

    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)

    # np.histogram2d bins data[0] against bins[0] and data[1] against bins[1].
    # The returned histogram H has H[i,j] as the number of points within
    # bins[0][i] <= x < bins[0][i+1] and bins[1][j] <= y < bins[1][j+1].
    # So, H has shape (len(bins[0])-1, len(bins[1])-1), which is (nx, ny).
    counts, _, _ = np.histogram2d(
        dataset_points[:, 0],
        dataset_points[:, 1],
        bins=[x_edges, y_edges],
    )

    covered_mask = counts > 0

    return CoveredGrid(
        domain_bounds=domain_bounds,
        grid_resolution=grid_resolution,
        x_edges=x_edges,
        y_edges=y_edges,
        covered_mask=covered_mask,
    )


"""
# Example Usage (can be run if this file is executed directly)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Sample dataset (e.g., from a trajectory)
    # Lissajous curve for interesting coverage
    t_vals = np.linspace(0, 2 * np.pi, 500)
    sample_points = np.array([
        np.sin(3 * t_vals + np.pi/2),  # x values
        np.sin(2 * t_vals)             # y values
    ]).T

    # Define domain and resolution
    bounds = ((-1.1, 1.1), (-1.1, 1.1))
    resolution = (20, 20) # 20x20 grid

    # Compute covered grid
    covered_info = compute_covered_grid(sample_points, bounds, resolution)
    
    # Grid resolution: {covered_info.grid_resolution}
    # Number of covered cells: {np.sum(covered_info.covered_mask)}

    # Visualization
    fig, ax = plt.subplots()
    ax.set_title(f'Covered Grid ({resolution[0]}x{resolution[1]}) for Sample Data')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal', adjustable='box') # Make cells square if bounds are same range

    # Add grid lines for clarity
    ax.set_xticks(covered_info.x_edges, minor=False)
    ax.set_yticks(covered_info.y_edges, minor=False)
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
    
    # Plot the original data points
    ax.plot(sample_points[:, 0], sample_points[:, 1], 'o', color='navy', markersize=2, alpha=0.3, label='Dataset Points')

    # Get and add rectangle patches for covered cells
    patches = covered_info.get_rectangle_patches(facecolor='lightcoral', alpha=0.7, edgecolor='black')
    for patch in patches:
        ax.add_patch(patch)
    
    # Create a dummy patch for the legend of covered cells
    if patches:
        legend_patch = plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.7, edgecolor='black', label='Covered Cells')
        ax.legend(handles=[legend_patch, ax.lines[0]]) # Combine with dataset points legend
    else:
        ax.legend()

    plt.show()
    # Example finished
"""
