#!/usr/bin/env python3
"""
Rimless Wheel Demo: Morse Set Analysis and Visualization

Generates visualizations of Morse sets and the Morse graph
for the rimless wheel hybrid dynamical system.
"""
from pathlib import Path
import warnings
import time

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
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


# Factory function for parallel processing
def create_rimless_wheel_system(alpha=0.4, gamma=0.2, max_jumps=50):
    """Factory function to create RimlessWheel system for parallel processing."""
    wheel = RimlessWheel(alpha=alpha, gamma=gamma, max_jumps=max_jumps)
    return wheel.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 0.5                 # Integration time horizon
SUBDIVISIONS = [100, 100] # Grid subdivisions [x_subdivisions, y_subdivisions]
# ===============================================


def run_rimless_wheel_demo():
    """Rimless wheel Morse set analysis and visualization."""
    
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("rimless_wheel")
    box_map_file = data_dir / "rimless_wheel_box_map.pkl"

    # 1. Define configuration
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Initial conditions for phase portrait
    initial_conditions = [
        [0.0, 0.4], [0.0, 0.2], [0.0, -0.2], [0.0, -0.4]
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    wheel_params = {"alpha": wheel.alpha, "gamma": wheel.gamma, "max_jumps": wheel.max_jumps}
    current_config_hash = create_config_hash(
        wheel_params, wheel.domain_bounds, grid.subdivisions.tolist(), tau
    )

    # 2. Try to load from cache
    cache_start_time = time.time()
    box_map = load_box_map_from_cache(grid, wheel.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        print("Computing new box map...")
        compute_start_time = time.time()
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=wheel.system,
            tau=tau,
            discard_out_of_bounds_destinations=True,
            progress_callback=progress_callback,
            parallel=True,
            system_factory=create_rimless_wheel_system,
            system_args=(wheel.alpha, wheel.gamma, wheel.max_jumps)
        )

        # Save to cache
        config_details = {
            "wheel_params": wheel_params,
            "domain_bounds": wheel.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
        
        compute_time = time.time() - compute_start_time
        print(f"  Box map computation time: {compute_time:.2f} seconds")
    else:
        cache_time = time.time() - cache_start_time
        print(f"  Cache load time: {cache_time:.3f} seconds")

    # Convert to NetworkX and analyze
    morse_start_time = time.time()
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    morse_time = time.time() - morse_start_time
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print("✓ Morse graph and Morse sets were computed.")
        print(f"  Number of Morse sets: {len(morse_sets)}")
        print(f"  Morse graph analysis time: {morse_time:.3f} seconds")
    else:
        print("⚠ Warning: Morse graph is empty. No recurrent sets were found.")

    # Generate visualizations
    viz_start_time = time.time()
    
    # Phase portrait
    plotter = HybridPlotter()
    plotter.create_phase_portrait_with_trajectories(
        system=wheel.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 3.0),
        output_path=str(run_dir / "phase_portrait.png"),
        max_jumps=8,
        title="Phase Portrait",
        xlabel="Angle θ (rad)",
        ylabel="Angular Velocity ω (rad/s)",
        domain_bounds=wheel.domain_bounds,
        figsize=(10, 8),
        dpi=150,
        show_legend=False
    )
    
    plot_morse_sets_on_grid(grid, morse_sets, str(run_dir / "morse_sets.png"), xlabel="Angle θ (rad)", ylabel="Angular Velocity ω (rad/s)")
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Final summary
    viz_time = time.time() - viz_start_time
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"✓ Figures are saved in {run_dir}")
    print(f"  - phase_portrait.png")
    print(f"  - morse_sets.png")
    print(f"  - morse_graph.png")
    print(f"  Visualization time: {viz_time:.2f} seconds")
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    run_rimless_wheel_demo()