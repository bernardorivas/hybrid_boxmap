#!/usr/bin/env python3
"""
Generates a phase portrait, morse sets, and morse graph for the bouncing ball hybrid dynamical system.
"""
from pathlib import Path
import warnings
import time

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.bouncing_ball import BouncingBall
from hybrid_dynamics import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid_fast,
    plot_morse_sets_with_roa_fast,
    compute_roa,
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
def create_bouncing_ball_system(g=9.81, c=0.5, max_jumps=50):
    """Factory function to create BouncingBall system for parallel processing."""
    ball = BouncingBall(
        domain_bounds=[(0.0, 2.0), (-5.0, 5.0)],
        g=g,
        c=c,
        max_jumps=max_jumps,
    )
    return ball.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 2.0                # Integration time horizon
SUBDIVISIONS = [21, 101]  # Grid subdivisions [height_subdivisions, velocity_subdivisions]
SAMPLING_MODE = 'corners'  # Sampling mode: 'corners', 'center', 'subdivision'
SUBDIVISION_LEVEL = 1     # Subdivision level for sampling (only used if SAMPLING_MODE='subdivision')
                         # Level n means 2^n subdivisions per dimension
ENCLOSURE = True         # Use enclosure mode for corners sampling
# ===============================================


def run_bouncing_ball_demo():
    """Bouncing ball phase portrait, morse sets, and morse graph."""
    
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
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
    # Include sampling parameters in configuration
    config_params = {
        **ball_params,
        "sampling_mode": SAMPLING_MODE,
        "enclosure": ENCLOSURE,
    }
    # Add subdivision level only if using subdivision mode
    if SAMPLING_MODE == 'subdivision':
        config_params["subdivision_level"] = SUBDIVISION_LEVEL
    
    current_config_hash = create_config_hash(
        config_params,
        ball.domain_bounds,
        grid.subdivisions.tolist(),
        tau,
    )

    # 2. Try to load from cache
    cache_start_time = time.time()
    box_map = load_box_map_from_cache(grid, ball.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute if cache miss
    if box_map is None:
        compute_start_time = time.time()
        progress_callback = create_progress_callback()
        
        # Build compute arguments
        compute_kwargs = {
            'grid': grid,
            'system': ball.system,
            'tau': tau,
            'sampling_mode': SAMPLING_MODE,
            'enclosure': ENCLOSURE,
            'discard_out_of_bounds_destinations': True,
            'progress_callback': progress_callback,
            'parallel': True,
            'system_factory': create_bouncing_ball_system,
            'system_args': (ball.g, ball.c, ball.max_jumps)
        }
        
        # Add subdivision_level only if using subdivision mode
        if SAMPLING_MODE == 'subdivision':
            compute_kwargs['subdivision_level'] = SUBDIVISION_LEVEL
        
        box_map = HybridBoxMap.compute(**compute_kwargs)

        # Save to cache
        config_details = {
            "ball_params": ball_params,
            "domain_bounds": ball.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
            "sampling_mode": SAMPLING_MODE,
            "enclosure": ENCLOSURE,
        }
        # Add subdivision level to details if using subdivision mode
        if SAMPLING_MODE == 'subdivision':
            config_details["subdivision_level"] = SUBDIVISION_LEVEL
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
        
        compute_time = time.time() - compute_start_time
    else:
        cache_time = time.time() - cache_start_time

    # Convert to NetworkX and compute morse graph and sets
    morse_start_time = time.time()
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    morse_time = time.time() - morse_start_time
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print(f"Morse graph has {morse_graph.number_of_nodes()} nodes and {morse_graph.number_of_edges()} edges")
        print(f"Found {len(morse_sets)} Morse sets")
        # Compute regions of attraction
        roa_dict = compute_roa(graph, morse_sets)
    else:
        print("Warning: Morse graph is empty")
        roa_dict = {}

    # Generate visualizations
    viz_start_time = time.time()
    
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
    plot_morse_sets_on_grid_fast(grid, morse_sets, str(run_dir / "morse_sets.png"), xlabel="Height h (m)", ylabel="Velocity v (m/s)")

    # The Morse graph
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Plot Morse sets with regions of attraction
    if morse_sets and roa_dict:
        plot_morse_sets_with_roa_fast(
            grid, morse_sets, roa_dict, 
            str(run_dir / "morse_sets_with_roa.png"),
            xlabel="Height h (m)", 
            ylabel="Velocity v (m/s)"
        )
    
    viz_time = time.time() - viz_start_time
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"\nResults saved to: {run_dir}")
    print(f"Box map computed with tau={tau}, grid subdivisions={SUBDIVISIONS}")
    print(f"Enclosure mode: {ENCLOSURE}")
    if morse_sets:
        for i in range(min(3, len(morse_sets))):
            roa_size = len(roa_dict.get(i, []))
            print(f"  - Morse set {i}: {len(morse_sets[i])} boxes, ROA: {roa_size} boxes")
        if len(morse_sets) > 3:
            print(f"  ... and {len(morse_sets) - 3} more Morse sets")


if __name__ == "__main__":
    run_bouncing_ball_demo()
