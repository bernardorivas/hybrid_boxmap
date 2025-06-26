"""
Core modules for hybrid dynamics analysis.

This module provides the foundational classes and utilities for hybrid system simulation,
grid-based analysis, cubification, graph analysis, and visualization.
"""

# Core classes
from .hybrid_system import HybridSystem
from .hybrid_trajectory import HybridTrajectory, TrajectorySegment
from .hybrid_time import HybridTime, HybridTimeInterval
from .hybrid_boxmap import HybridBoxMap

# Grid and cubification
from .box import Box, SquareBox
from .grid import Grid
from .cubifier import DatasetCubifier
from .evaluation import evaluate_grid
from .data_utils import GridEvaluationResult

# Graph analysis
from .morse_graph import create_morse_graph

# MultiGrid framework
from .multigrid import MultiGrid, MultiGridBoxMap

# Utilities
from .grid_utils import *
from .trajectory_utils import *

# Visualization
from .plot_utils import (
    HybridPlotter,
    visualize_flow_map,
    visualize_box_map, 
    visualize_box_map_entry,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    plot_morse_sets_3d,
    plot_morse_sets_projections,
)

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