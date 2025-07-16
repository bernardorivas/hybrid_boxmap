"""
Core modules for hybrid dynamics analysis.

This module provides the foundational classes and utilities for hybrid system simulation,
grid-based analysis, cubification, graph analysis, and visualization.
"""

# Core classes
# Grid and cubification
from .box import Box, SquareBox
from .cubifier import DatasetCubifier
from .data_utils import GridEvaluationResult
from .evaluation import evaluate_grid
from .grid import Grid

# Utilities
from .grid_utils import *
from .hybrid_boxmap import HybridBoxMap
from .hybrid_system import HybridSystem
from .hybrid_time import HybridTime, HybridTimeInterval
from .hybrid_trajectory import HybridTrajectory, TrajectorySegment

# Graph analysis
from .morse_graph import create_morse_graph

# MultiGrid framework
from .multigrid import MultiGrid, MultiGridBoxMap

# Visualization
from .plot_utils import (
    HybridPlotter,
    plot_morse_graph_viz,
    plot_morse_sets_3d,
    plot_morse_sets_on_grid,
    plot_morse_sets_projections,
    visualize_box_map,
    visualize_box_map_entry,
    visualize_flow_map,
)
from .trajectory_utils import *

__all__ = [
    # Core classes
    "HybridSystem",
    "HybridTrajectory",
    "TrajectorySegment",
    "HybridTime",
    "HybridTimeInterval",
    "HybridBoxMap",
    # Grid and cubification
    "Box",
    "SquareBox",
    "Grid",
    "DatasetCubifier",
    "evaluate_grid",
    "GridEvaluationResult",
    # Graph analysis
    "create_morse_graph",
    # MultiGrid framework
    "MultiGrid",
    "MultiGridBoxMap",
    # Visualization
    "HybridPlotter",
    "visualize_flow_map",
    "visualize_box_map",
    "visualize_box_map_entry",
    "plot_morse_graph_viz",
    "plot_morse_sets_on_grid",
    "plot_morse_sets_3d",
    "plot_morse_sets_projections",
]
