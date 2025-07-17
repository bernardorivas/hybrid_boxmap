#!/usr/bin/env python3
"""
Rimless Wheel Demo: Morse Set Analysis and Visualization

Generates visualizations of Morse sets and the Morse graph
for the rimless wheel hybrid dynamical system.
"""
from pathlib import Path
import warnings
import time

from hybrid_dynamics import (
    Grid, 
    HybridBoxMap, 
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid_fast,
    plot_morse_sets_with_roa_fast,
    compute_roa, 
    analyze_roa_coverage
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
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
TAU = 2.0                       # Integration time horizon
SUBDIVISIONS = [201, 201]       # Grid subdivisions [x_subdivisions, y_subdivisions]
BLOAT_FACTOR = 0.12             # Bloat factor for box map computation
SAMPLING_MODE = 'corners'       # Sampling mode: 'corners', 'center', 'subdivision'
SUBDIVISION_LEVEL = 2           # Subdivision level (only used if SAMPLING_MODE='subdivision', 2^n subdivisions per dimension)
ENCLOSURE = True                # Use enclosure mode for corners sampling
# ===============================================


def run_rimless_wheel_demo():
    """Rimless wheel Morse set analysis and visualization with comprehensive timing analysis."""
    
    # Initialize performance profiler
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
    bloat_factor = BLOAT_FACTOR
    
    # Initial conditions for phase portrait
    initial_conditions = [
        [0.0, 0.4], [0.0, 0.2], [0.0, -0.2], [0.0, -0.4]
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist(), bloat_factor)

    # Create configuration hash for validation
    wheel_params = {"alpha": wheel.alpha, "gamma": wheel.gamma, "max_jumps": wheel.max_jumps}
    # Include bloat factor and sampling parameters in configuration to ensure proper caching
    config_params = {
        **wheel_params, 
        "bloat_factor": bloat_factor,
        "sampling_mode": SAMPLING_MODE,
        "enclosure": ENCLOSURE,
    }
    # Add subdivision level only if using subdivision mode
    if SAMPLING_MODE == 'subdivision':
        config_params["subdivision_level"] = SUBDIVISION_LEVEL
    
    current_config_hash = create_config_hash(
        config_params, wheel.domain_bounds, grid.subdivisions.tolist(), tau
    )

    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, wheel.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        # Create progress callback (more frequent updates for subdivision mode)
        progress_callback = create_progress_callback(update_interval=5000)
        
        # Build compute arguments
        compute_kwargs = {
            'grid': grid,
            'system': wheel.system,
            'tau': tau,
            'sampling_mode': SAMPLING_MODE,
            'bloat_factor': bloat_factor,
            'enclosure': ENCLOSURE,
            'discard_out_of_bounds_destinations': True,
            'progress_callback': progress_callback,
            'parallel': True,
            'system_factory': create_rimless_wheel_system,
            'system_args': (wheel.alpha, wheel.gamma, wheel.max_jumps)
        }
        
        # Add subdivision_level only if using subdivision mode
        if SAMPLING_MODE == 'subdivision':
            compute_kwargs['subdivision_level'] = SUBDIVISION_LEVEL
        
        # Time the box map computation
        compute_start_time = time.time()
        box_map = HybridBoxMap.compute(**compute_kwargs)
        compute_time = time.time() - compute_start_time
        print(f"Time to compute box map: {compute_time:.2f} seconds")

        # Save to cache
        config_details = {
                "wheel_params": wheel_params,
                "domain_bounds": wheel.domain_bounds,
                "grid_subdivisions": grid.subdivisions.tolist(),
                "tau": tau,
                "bloat_factor": bloat_factor,
                "sampling_mode": SAMPLING_MODE,
                "enclosure": ENCLOSURE,
            }
        # Add subdivision level to details if using subdivision mode
        if SAMPLING_MODE == 'subdivision':
            config_details["subdivision_level"] = SUBDIVISION_LEVEL
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
            

    # Convert to NetworkX
    graph = box_map.to_networkx()

    morse_graph, morse_sets = create_morse_graph(graph)
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        # Compute regions of attraction
        roa_dict = compute_roa(graph, morse_sets)
        
        # Analyze ROA coverage
        statistics, uncovered_boxes = analyze_roa_coverage(roa_dict, len(grid))
    else:
        roa_dict = {}  # Empty ROA dictionary for failed case

    # Generate visualizations
    
    # Phase portrait
    plotter = HybridPlotter()
    viz_start_time = time.time()
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
    viz_time = time.time() - viz_start_time
    print(f"Phase portrait visualization computed in {viz_time:.2f} seconds")
    
    viz_start_time = time.time()
    plot_morse_sets_on_grid_fast(grid, morse_sets, str(run_dir / "morse_sets.png"), 
                                 xlabel="Angle θ (rad)", ylabel="Angular Velocity ω (rad/s)")
    viz_time = time.time() - viz_start_time
    print(f"Morse sets visualization computed in {viz_time:.2f} seconds")
    
    viz_start_time = time.time()
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    viz_time = time.time() - viz_start_time
    print(f"Morse graph visualization computed in {viz_time:.2f} seconds")
    
    # Generate ROA visualization if we have Morse sets
    if morse_sets and roa_dict:
        viz_start_time = time.time()
        plot_morse_sets_with_roa_fast(
            grid, morse_sets, roa_dict, 
            str(run_dir / "morse_set_roa.png"),
            xlabel="Angle θ (rad)", 
            ylabel="Angular Velocity ω (rad/s)"
        )
        viz_time = time.time() - viz_start_time
        print(f"Morse sets with ROA visualization computed in {viz_time:.2f} seconds")
    
    # Final summary
    total_time = time.time() - total_start_time


if __name__ == "__main__":
    run_rimless_wheel_demo()