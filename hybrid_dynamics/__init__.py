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
# Cubical grid components
from .src.box import Box
from .src.config import config
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
    plot_morse_sets_on_grid_fast,
    plot_morse_sets_with_roa,
    plot_morse_sets_with_roa_fast,
    visualize_box_map,
    visualize_box_map_entry,
    visualize_flow_map,
)

# Region of attraction analysis
from .src.roa_utils import (
    analyze_roa_coverage,
    compute_regions_of_attraction,
    compute_roa,
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
    # ROA
    "compute_roa",
    "compute_regions_of_attraction",
    "analyze_roa_coverage",
    # MultiGrid
    "MultiGrid",
    "MultiGridBoxMap",
    # Plotting
    "HybridPlotter",
    "visualize_flow_map",
    "visualize_box_map",
    "visualize_box_map_entry",
    "plot_morse_sets_on_grid",
    "plot_morse_sets_with_roa",
    "plot_morse_sets_on_grid_fast",
    "plot_morse_sets_with_roa_fast",
    "plot_morse_graph_viz",
    # Config
    "config",
]
