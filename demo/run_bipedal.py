#!/usr/bin/env python3
"""
Bipedal Walker Demo: Morse Set Analysis and Visualization

Generates visualizations of Morse sets and the Morse graph
for the bipedal walker hybrid dynamical system.
"""
from pathlib import Path

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.bipedal import BipedalWalker
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
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


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 0.6                    # Integration time horizon
SUBDIVISIONS = [50, 50, 10, 10] # Grid subdivisions [x, y, x_dot, y_dot]
# ===============================================


def run_bipedal_demo():
    """Bipedal walker Morse set analysis and visualization."""
    
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
        [-0.866, 0.5, 0.5, -0.5],     # Start at target, moving inside
        [0.707, 0.707, -0.3, -0.3],   # Another point on circle
        [-0.5, -0.866, 0.2, 0.4],     # Third point
        [0.0, 1.0, -0.1, -0.6],       # Fourth point
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    biped_params = {
        "x0": biped.x0, 
        "y0": biped.y0, 
        "z0": biped.z0, 
        "g": biped.g, 
        "max_jumps": biped.max_jumps
    }
    current_config_hash = create_config_hash(
        biped_params, biped.domain_bounds, grid.subdivisions.tolist(), tau
    )

    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, biped.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        print("Computing new box map...")
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=biped.system,
            tau=tau,
            discard_out_of_bounds_destinations=False,
            parallel=False,
            progress_callback=progress_callback,
        )

        # Save to cache
        config_details = {
            "biped_params": biped_params,
            "domain_bounds": biped.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)

    # Convert to NetworkX and analyze
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print("✓ Morse graph and Morse sets were computed.")
    else:
        print("⚠ Warning: Morse graph is empty. No recurrent sets were found.")

    # Generate visualizations
    
    # Phase portrait (plotting first 2 dimensions: x vs y)
    plotter = HybridPlotter()
    plotter.create_phase_portrait_with_trajectories(
        system=biped.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 2.0),
        output_path=str(run_dir / "phase_portrait.png"),
        max_jumps=8,
        title="Phase Portrait",
        xlabel="x",
        ylabel="y",
        domain_bounds=[(biped.domain_bounds[0]), (biped.domain_bounds[1])],  # Only first 2 dims
        figsize=(10, 8),
        dpi=150,
        show_legend=False
    )
    
    plot_morse_sets_on_grid(
        grid, 
        morse_sets, 
        str(run_dir / "morse_sets.png"),
        plot_dims=(0, 1),  # Project to x vs y dimensions
        xlabel="x",
        ylabel="y"
    )
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Final summary
    print(f"✓ Figures are saved in {run_dir}")


if __name__ == "__main__":
    run_bipedal_demo()