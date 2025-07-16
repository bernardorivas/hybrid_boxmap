#!/usr/bin/env python3
"""
Bipedal Walker Demo: Morse Set Analysis and Visualization

Generates visualizations of Morse sets and the Morse graph
for the bipedal walker hybrid dynamical system.
"""
from pathlib import Path
import numpy as np
import warnings
import time

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.bipedal import BipedalWalker
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    plot_morse_sets_3d,
    plot_morse_sets_projections,
)
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# Factory function for parallel processing
def create_bipedal_system(x0=1.0, y0=0.0, z0=1.0, g=1.0, max_jumps=20):
    """Factory function to create BipedalWalker system for parallel processing."""
    biped = BipedalWalker(
        x0=x0,
        y0=y0,
        z0=z0,
        g=g,
        max_jumps=max_jumps
    )
    return biped.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 0.5                    # Integration time horizon
SUBDIVISIONS = [20, 20, 5, 5] # Grid subdivisions [x, y, x_dot, y_dot]
USE_CYLINDRICAL = False      # Use cylindrical sampling method
N_RADIAL = 250               # Number of radial samples for cylindrical method
N_ANGULAR = 360              # Number of angular samples for cylindrical method (every degree)
SAMPLING_MODE = 'corners'    # Sampling mode for non-cylindrical method
ENCLOSURE = True             # Use enclosure mode for corners sampling
# ===============================================


def run_bipedal_demo():
    """Bipedal walker Morse set analysis and visualization."""
    
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("bipedal")
    box_map_file = data_dir / "bipedal_box_map.pkl"

    # 1. Define configuration
    biped = BipedalWalker(
        x0=1.0,             # Initial x-coordinate for radius
        y0=0.0,             # Initial y-coordinate for radius
        z0=1.0,             # Height of center of mass
        g=1.0,              # Gravitational acceleration
        max_jumps=20
    )
    grid = Grid(bounds=biped.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Initial conditions for phase portrait (4D system, we'll plot first 2 dimensions)
    initial_conditions = [
        # Key dynamical points
        [1.0, 0.0, 0.0, 0.0],         # Start at (1,0)
        [-1.0, 0.0, 0.0, 0.0],        # Start at (-1,0)
        [0.0, 0.0, 0.0, 0.0],         # Origin (fixed point)
        # Points on the circle
        [0.0, 1.0, 0.0, 0.0],         # Start at (0,1)
        [0.0, -1.0, 0.0, 0.0],        # Start at (0,-1)
        [0.707, 0.707, 0.0, 0.0],     # 45 degrees
        [-0.707, 0.707, 0.0, 0.0],    # 135 degrees
        # Various radii to show flow
        [0.5, 0.0, 0.0, 0.0],         # Half radius
        [0.3, 0.0, 0.0, 0.0],         # Inner circle
        [0.8, 0.0, 0.0, 0.0],         # Near boundary
        # With initial velocities
        [0.5, 0.5, 0.2, -0.2],        # With velocity
        [-0.5, 0.5, -0.2, 0.2],       # With velocity
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    biped_params = {
        "x0": biped.x0, 
        "y0": biped.y0, 
        "z0": biped.z0, 
        "g": biped.g, 
        "max_jumps": biped.max_jumps,
        "use_cylindrical": USE_CYLINDRICAL,
        "sampling_mode": SAMPLING_MODE,
        "enclosure": ENCLOSURE
    }
    current_config_hash = create_config_hash(
        biped_params, biped.domain_bounds, grid.subdivisions.tolist(), tau
    )

    # Get cylinder radius from system parameters
    cylinder_radius = np.sqrt(biped.x0**2 + biped.y0**2)
    
    # 2. Try to load from cache
    cache_start_time = time.time()
    box_map = load_box_map_from_cache(grid, biped.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        compute_start_time = time.time()
        progress_callback = create_progress_callback()
        
        if USE_CYLINDRICAL:
            
            box_map = HybridBoxMap.compute_cylindrical(
                grid=grid,
                system=biped.system,
                tau=tau,
                cylinder_radius=cylinder_radius,
                n_radial_samples=N_RADIAL,
                n_angular_samples=N_ANGULAR,
                discard_out_of_bounds_destinations=False,
                progress_callback=progress_callback,
                parallel=True,
                system_factory=create_bipedal_system,
                system_args=(biped.x0, biped.y0, biped.z0, biped.g, biped.max_jumps)
            )
        else:
            box_map = HybridBoxMap.compute(
                grid=grid,
                system=biped.system,
                tau=tau,
                sampling_mode=SAMPLING_MODE,
                enclosure=ENCLOSURE,
                discard_out_of_bounds_destinations=True,
                progress_callback=progress_callback,
                parallel=True,
                system_factory=create_bipedal_system,
                system_args=(biped.x0, biped.y0, biped.z0, biped.g, biped.max_jumps)
            )

        # Save to cache
        config_details = {
            "biped_params": biped_params,
            "domain_bounds": biped.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
            "use_cylindrical": USE_CYLINDRICAL,
            "sampling_mode": SAMPLING_MODE,
            "enclosure": ENCLOSURE
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
        
        compute_time = time.time() - compute_start_time
    else:
        cache_time = time.time() - cache_start_time
    
    # Convert to NetworkX and analyze
    morse_start_time = time.time()
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    morse_time = time.time() - morse_start_time
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print(f"Morse graph has {morse_graph.number_of_nodes()} nodes and {morse_graph.number_of_edges()} edges")
        print(f"Found {len(morse_sets)} Morse sets")
    else:
        print("Warning: Morse graph computation resulted in empty graph")

    # Generate visualizations
    viz_start_time = time.time()
    
    # Check system dimensionality and generate appropriate visualizations
    if grid.ndim == 2:
        # Standard 2D visualization
        plot_morse_sets_on_grid(
            grid, 
            morse_sets, 
            str(run_dir / "morse_sets.png"),
            xlabel="x",
            ylabel="y"
        )
    elif grid.ndim == 3:
        # 3D system: generate both 3D view and projections
        plot_morse_sets_3d(
            grid,
            morse_sets,
            str(run_dir / "morse_sets_3d.png"),
            xlabel="x",
            ylabel="y", 
            zlabel="z"
        )
        # Also create 2D projections
        plot_morse_sets_projections(
            grid,
            morse_sets,
            str(run_dir)
        )
    else:
        # Higher dimensional systems: only projections
        # For 4D bipedal, project to x vs y dimensions
        plot_morse_sets_on_grid(
            grid, 
            morse_sets, 
            str(run_dir / "morse_sets.png"),
            plot_dims=(0, 1),  # Project to x vs y dimensions
            xlabel="x",
            ylabel="y"
        )
    
    # Morse graph visualization remains the same for all dimensions
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    viz_time = time.time() - viz_start_time
    total_time = time.time() - total_start_time
    
    # Final summary


if __name__ == "__main__":
    run_bipedal_demo()