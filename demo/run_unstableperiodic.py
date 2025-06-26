#!/usr/bin/env python3
"""
Unstable Periodic Demo: SCC Analysis and Visualization

Generates visualizations of strongly connected components and condensation graph
for the unstable periodic hybrid dynamical system.
"""
from pathlib import Path
import networkx as nx
import warnings

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
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
def create_unstableperiodic_system(max_jumps=15):
    """Factory function to create UnstablePeriodicSystem for parallel processing."""
    system = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=max_jumps
    )
    return system.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 2.0                 # Integration time horizon
SUBDIVISIONS = [100, 100] # Grid subdivisions [x_subdivisions, y_subdivisions]
# ===============================================


def run_unstableperiodic_demo():
    """Unstable periodic Morse set analysis and visualization."""
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("unstable_periodic")
    box_map_file = data_dir / "unstable_periodic_box_map.pkl"

    # 1. Define configuration
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Initial conditions for phase portrait
    initial_conditions = [
        [0.1, 0.6],    # Start near left boundary
        [0.3, -0.4],   # Start with negative y
        [0.8, 0.2],    # Start near right boundary 
        [0.5, 0.0]     # Start in middle
    ]
    
    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())
    
    # Create configuration hash for validation
    system_params = {"max_jumps": system_obj.max_jumps}
    current_config_hash = create_config_hash(
        system_params, 
        system_obj.domain_bounds, 
        grid.subdivisions.tolist(), 
        tau
    )
    
    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, system_obj.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=system_obj.system,
            tau=tau,
            discard_out_of_bounds_destinations=True,
            progress_callback=progress_callback,
            parallel=True,
            system_factory=create_unstableperiodic_system,
            system_args=(system_obj.max_jumps,)
        )
        
        # Save to cache
        config_details = {
            'system_params': system_params,
            'domain_bounds': system_obj.domain_bounds,
            'grid_subdivisions': grid.subdivisions.tolist(),
            'tau': tau
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)

    # Convert to NetworkX and analyze
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
    else:

    # Generate visualizations
    
    # Phase portrait
    plotter = HybridPlotter()
    plotter.create_phase_portrait_with_trajectories(
        system=system_obj.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 4.0),
        output_path=str(run_dir / "phase_portrait.png"),
        max_jumps=8,
        title="Phase Portrait",
        xlabel="x",
        ylabel="y",
        domain_bounds=system_obj.domain_bounds,
        figsize=(10, 8),
        dpi=150,
        show_legend=False
    )
    
    plot_morse_sets_on_grid(grid, morse_sets, str(run_dir / "morse_sets.png"), xlabel="x", ylabel="y")
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Final summary


if __name__ == "__main__":
    run_unstableperiodic_demo()