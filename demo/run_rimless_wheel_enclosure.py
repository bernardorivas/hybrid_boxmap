#!/usr/bin/env python3
"""
Rimless Wheel Demo with Enclosure: Morse Set Analysis and Visualization

Generates visualizations of Morse sets and the Morse graph
for the rimless wheel hybrid dynamical system using the enclosure option
for more conservative reachability analysis.
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
SUBDIVISIONS = [101, 101]       # Grid subdivisions [x_subdivisions, y_subdivisions]
BLOAT_FACTOR = 0.20             # Bloat factor for box map computation
SAMPLING_MODE = 'corners'       # MUST be 'corners' for enclosure mode
ENCLOSURE = True                # Enable enclosure mode for conservative analysis
# ===============================================


def run_rimless_wheel_demo():
    """Rimless wheel Morse set analysis and visualization with enclosure mode."""
    
    # Initialize performance profiler
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("rimless_wheel")
    box_map_file = data_dir / "rimless_wheel_enclosure_box_map.pkl"

    # 1. Define configuration
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    bloat_factor = BLOAT_FACTOR
    
    # Initial conditions for phase portrait
    initial_conditions = [
        [0.0, 0.4], [0.0, 0.2], [0.0, -0.2], [0.0, -0.4]
    ]

    # Create run directory with proper enclosure suffix
    run_name_parts = [
        f"run_tau_{int(tau*100):03d}",
        f"subdiv_{grid.subdivisions[0]}_{grid.subdivisions[1]}",
        f"bloat_{int(bloat_factor*100):02d}",
    ]
    base_run_name = "_".join(run_name_parts)
    
    # Find next available run number
    existing_runs = list(figures_base_dir.glob(f"{base_run_name}_*"))
    if existing_runs:
        run_numbers = []
        for run_dir in existing_runs:
            try:
                # Extract run number, accounting for possible "with_enclosure" suffix
                dir_name = run_dir.name
                if dir_name.endswith("_with_enclosure"):
                    # Remove the suffix before extracting number
                    dir_name = dir_name[:-len("_with_enclosure")]
                run_numbers.append(int(dir_name.split('_')[-1]))
            except ValueError:
                pass
        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1
    
    # Create final run directory name with enclosure suffix
    run_dir_name = f"{base_run_name}_{next_run:03d}_with_enclosure"
    run_dir = figures_base_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created run directory: {run_dir}")

    # Create configuration hash for validation
    wheel_params = {"alpha": wheel.alpha, "gamma": wheel.gamma, "max_jumps": wheel.max_jumps}
    # Include bloat factor, sampling mode, and enclosure in configuration
    config_params = {
        **wheel_params, 
        "bloat_factor": bloat_factor,
        "sampling_mode": SAMPLING_MODE,
        "enclosure": ENCLOSURE,
    }
    
    current_config_hash = create_config_hash(
        config_params, wheel.domain_bounds, grid.subdivisions.tolist(), tau
    )

    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, wheel.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        print("\n" + "="*50)
        print("Computing box map with ENCLOSURE mode enabled")
        print("This creates more conservative reachability approximations")
        print("="*50 + "\n")
        
        # Create progress callback
        progress_callback = create_progress_callback(update_interval=5000)
        
        # Build compute arguments with enclosure enabled
        compute_kwargs = {
            'grid': grid,
            'system': wheel.system,
            'tau': tau,
            'sampling_mode': SAMPLING_MODE,
            'bloat_factor': bloat_factor,
            'enclosure': ENCLOSURE,  # Enable enclosure mode
            'discard_out_of_bounds_destinations': True,
            'progress_callback': progress_callback,
            'parallel': True,
            'system_factory': create_rimless_wheel_system,
            'system_args': (wheel.alpha, wheel.gamma, wheel.max_jumps)
        }
        
        # Time the box map computation
        compute_start_time = time.time()
        box_map = HybridBoxMap.compute(**compute_kwargs)
        compute_time = time.time() - compute_start_time
        print(f"Time to compute box map (with enclosure): {compute_time:.2f} seconds")
        
        # Print statistics about enclosure effect
        total_transitions = sum(len(dests) for dests in box_map.values())
        print(f"Total transitions with enclosure: {total_transitions}")
        print(f"Average destinations per box: {total_transitions/len(box_map):.2f}")

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
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
            

    # Convert to NetworkX and analyze
    graph = box_map.to_networkx()

    morse_graph, morse_sets = create_morse_graph(graph)
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        # Compute regions of attraction
        roa_dict = compute_roa(graph, morse_sets)
        
        # Analyze ROA coverage
        statistics, uncovered_boxes = analyze_roa_coverage(roa_dict, len(grid))
        
        print("\n" + "="*50)
        print("ENCLOSURE MODE ANALYSIS RESULTS")
        print("="*50)
        print(f"Number of Morse sets: {len(morse_sets)}")
        print(f"ROA coverage: {statistics['coverage_percentage']:.1f}%")
        print("="*50 + "\n")
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
            title="Phase Portrait (Enclosure Mode)",
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
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    run_rimless_wheel_demo()