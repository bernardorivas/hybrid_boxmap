#!/usr/bin/env python3
"""
Generates a phase portrait, morse sets, and morse graph for the bouncing ball hybrid dynamical system.
"""
from pathlib import Path

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.bouncing_ball import BouncingBall
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
)
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    plot_box_containing_point,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 0.3                # Integration time horizon
SUBDIVISIONS = [51, 51]  # Grid subdivisions [height_subdivisions, velocity_subdivisions]
# ===============================================


def run_bouncing_ball_demo():
    """Bouncing ball phase portrait, morse sets, and morse graph."""
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("bouncing_ball")
    box_map_file = data_dir / "bouncing_ball_box_map.pkl"

    # 1. Define configuration
    ball = BouncingBall(
        domain_bounds=[(0.0, 2.0), (-5.0, 5.0)],
        g=9.81,
        c=0.5,
        max_jumps=50,
    )
    grid = Grid(bounds=ball.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Initial conditions for phase portrait
    initial_conditions = [
        [1.0, 0.0],   # Drop from height 1.0
        [1.5, 1.0],   # Start with upward velocity
        [0.5, -1.0],  # Start with downward velocity
        [2.0, 0.5],   # Drop from height 2.0 with small initial velocity
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    ball_params = {"g": ball.g, "c": ball.c, "max_jumps": ball.max_jumps}
    current_config_hash = create_config_hash(
        ball_params,
        ball.domain_bounds,
        grid.subdivisions.tolist(),
        tau,
    )

    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, ball.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute if cache miss
    if box_map is None:
        print("Computing box map...")
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=ball.system,
            tau=tau,
            discard_out_of_bounds_destinations=False,
            parallel=False,
            progress_callback=progress_callback,
        )

        # Save to cache
        config_details = {
            "ball_params": ball_params,
            "domain_bounds": ball.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)

    # Convert to NetworkX and compute morse graph and sets
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print("✓ Morse graph and Morse sets were computed.")
    else:
        print("⚠ Warning: Morse graph is empty. No recurrent sets were found.")

    # Generate visualizations
    
    # Phase portrait
    plotter = HybridPlotter()
    plotter.create_phase_portrait_with_trajectories(
        system=ball.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 1.0),
        output_path=str(run_dir / "phase_portrait.png"),
        max_jumps=50,
        max_step=0.01,
        title="Phase Portrait",
        xlabel="Height h (m)",
        ylabel="Velocity v (m/s)",
        domain_bounds=ball.domain_bounds,
        figsize=(10, 8),
        dpi=150,
        show_legend=False
    )

    # The boxes of the recurrent components on the state space grid
    plot_morse_sets_on_grid(grid, morse_sets, str(run_dir / "morse_sets.png"), xlabel="Height h (m)", ylabel="Velocity v (m/s)")

    # The Morse graph
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Plot box containing origin (0,0) as an example
    plot_box_containing_point(box_map, grid, [0.0, 0.0], run_dir, "origin_box_map")
    
    # Final summary
    print(f"✓ Figures are saved in {run_dir}")


if __name__ == "__main__":
    run_bouncing_ball_demo()
