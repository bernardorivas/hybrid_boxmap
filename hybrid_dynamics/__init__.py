"""
Hybrid Dynamics - A Python library for hybrid dynamical systems analysis.

This library provides tools for:
- Hybrid system simulation with event detection
- Enhanced trajectory tracking with hybrid time domains
- Box map computation for global dynamics approximation
- Morse graph analysis of recurrent components
- Visualization of hybrid systems, box maps, and analysis results
"""

# Global configuration
from .src.config import config

# Cubical grid components
from .src.box import Box
from .src.grid import Grid
from .src.hybrid_boxmap import HybridBoxMap
from .src.hybrid_system import HybridSystem
from .src.hybrid_time import HybridTime, HybridTimeInterval
from .src.hybrid_trajectory import HybridTrajectory, TrajectorySegment

# Graph analysis functions
from .src.morse_graph import create_morse_graph

# MultiGrid framework
from .src.multigrid import MultiGrid, MultiGridBoxMap

# Visualization functions
from .src.plot_utils import (
    HybridPlotter,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    visualize_box_map,
    visualize_box_map_entry,
    visualize_flow_map,
)

__version__ = "0.2.0"
__author__ = "Your Name/Team"

__all__ = [
    # Core
    "HybridTime",
    "HybridTimeInterval",
    "HybridTrajectory",
    "TrajectorySegment",
    "HybridSystem",
    "HybridBoxMap",
    # Cubical
    "Box",
    "Grid",
    # Graph
    "create_morse_graph",
    # MultiGrid
    "MultiGrid",
    "MultiGridBoxMap",
    # Plotting
    "HybridPlotter",
    "visualize_flow_map",
    "visualize_box_map",
    "visualize_box_map_entry",
    "plot_morse_sets_on_grid",
    "plot_morse_graph_viz",
    # Config
    "config",
]