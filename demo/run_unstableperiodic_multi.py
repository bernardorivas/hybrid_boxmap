#!/usr/bin/env python3
"""
Unstable Periodic Demo: Multiple Parameter Analysis

Runs the unstable periodic system analysis with various tau values and grid subdivisions.
"""
import itertools
from pathlib import Path
import networkx as nx

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


# Parameter combinations to test
TAU_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
SUBDIVISION_VALUES = [10, 15, 20, 30, 50, 100]


def run_unstableperiodic_multi():
    """Run unstable periodic analysis for multiple parameter combinations."""
    
    # Setup base directories
    data_dir, figures_base_dir = setup_demo_directories("unstable_periodic")
    
    # Track results
    results_summary = []
    
    # Run all combinations
    total_combinations = len(TAU_VALUES) * len(SUBDIVISION_VALUES)
    current_run = 0
    
    for tau, n in itertools.product(TAU_VALUES, SUBDIVISION_VALUES):
        current_run += 1
        subdivisions = [n, n]  # n=m
        
        print(f"\n{'='*60}")
        print(f"Run {current_run}/{total_combinations}: tau={tau}, subdivisions={subdivisions}")
        print(f"{'='*60}")
        
        # Create specific cache file for this configuration
        cache_file = data_dir / f"unstable_periodic_tau_{int(tau*100):03d}_sub_{n}x{n}.pkl"
        
        # Define system
        system_obj = UnstablePeriodicSystem(
            domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
            max_jumps=15
        )
        grid = Grid(bounds=system_obj.domain_bounds, subdivisions=subdivisions)
        
        # Initial conditions for phase portrait
        initial_conditions = [
            [0.1, 0.6],    # Start near left boundary
            [0.3, -0.4],   # Start with negative y
            [0.8, 0.2],    # Start near right boundary 
            [0.5, 0.0]     # Start in middle
        ]
        
        # Create run directory
        run_dir = create_next_run_dir(figures_base_dir, tau, subdivisions)
        
        # Create configuration hash
        system_params = {"max_jumps": system_obj.max_jumps}
        current_config_hash = create_config_hash(
            system_params, 
            system_obj.domain_bounds, 
            subdivisions, 
            tau
        )
        
        # Try to load from cache
        box_map = load_box_map_from_cache(grid, system_obj.system, tau, cache_file, current_config_hash)
        
        # Compute new if cache miss
        if box_map is None:
            print(f"Computing new box map for tau={tau}, subdivisions={subdivisions}...")
            progress_callback = create_progress_callback()
            box_map = HybridBoxMap.compute(
                grid=grid,
                system=system_obj.system,
                tau=tau,
                discard_out_of_bounds_destinations=False,
                progress_callback=progress_callback,
            )
            
            # Save to cache
            config_details = {
                'system_params': system_params,
                'domain_bounds': system_obj.domain_bounds,
                'grid_subdivisions': subdivisions,
                'tau': tau
            }
            save_box_map_with_config(box_map, current_config_hash, config_details, cache_file)
        else:
            print(f"Loaded box map from cache")
        
        # Convert to NetworkX and analyze
        graph = box_map.to_networkx()
        morse_graph, morse_sets = create_morse_graph(graph)
        
        # Record results
        num_morse_sets = len(morse_sets)
        num_morse_edges = morse_graph.number_of_edges()
        total_boxes_in_morse_sets = sum(len(ms) for ms in morse_sets)
        
        result = {
            'tau': tau,
            'subdivisions': subdivisions,
            'num_morse_sets': num_morse_sets,
            'num_morse_edges': num_morse_edges,
            'total_boxes_in_morse_sets': total_boxes_in_morse_sets,
            'run_dir': run_dir
        }
        results_summary.append(result)
        
        print(f"  Morse sets: {num_morse_sets}")
        print(f"  Morse graph edges: {num_morse_edges}")
        print(f"  Total boxes in Morse sets: {total_boxes_in_morse_sets}")
        
        # Generate visualizations
        print("  Generating visualizations...")
        
        # Phase portrait
        plotter = HybridPlotter()
        plotter.create_phase_portrait_with_trajectories(
            system=system_obj.system,
            initial_conditions=initial_conditions,
            time_span=(0.0, 4.0),
            output_path=str(run_dir / "phase_portrait.png"),
            max_jumps=8,
            title=f"Phase Portrait (τ={tau}, grid={n}×{n})",
            xlabel="x",
            ylabel="y",
            domain_bounds=system_obj.domain_bounds,
            figsize=(10, 8),
            dpi=150,
            show_legend=False
        )
        
        plot_morse_sets_on_grid(
            grid, morse_sets, 
            str(run_dir / "morse_sets.png"), 
            xlabel="x", ylabel="y"
        )
        
        plot_morse_graph_viz(
            morse_graph, morse_sets, 
            str(run_dir / "morse_graph.png")
        )
        
        print(f"  ✓ Figures saved to {run_dir}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'='*60}")
    print(f"{'tau':<6} {'grid':<10} {'# Sets':<8} {'# Edges':<8} {'# Boxes':<10}")
    print(f"{'-'*50}")
    
    for r in results_summary:
        grid_str = f"{r['subdivisions'][0]}×{r['subdivisions'][1]}"
        print(f"{r['tau']:<6.1f} {grid_str:<10} {r['num_morse_sets']:<8} "
              f"{r['num_morse_edges']:<8} {r['total_boxes_in_morse_sets']:<10}")
    
    print("\n✓ All runs completed successfully!")


if __name__ == "__main__":
    run_unstableperiodic_multi()