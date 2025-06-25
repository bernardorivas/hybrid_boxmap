#!/usr/bin/env python3
"""
Test script for finding and visualizing recurrent sets in the HybridBoxMap.
"""
from pathlib import Path
import networkx as nx

from hybrid_dynamics import (
    Grid,
    HybridBoxMap,
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel

def get_next_run_dir(base_dir: Path) -> Path:
    """Finds the next available run directory."""
    base_dir.mkdir(exist_ok=True)
    existing_runs = [int(d.name.split('_')[1]) for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = base_dir / f"run_{next_run_num:04d}"
    run_dir.mkdir()
    return run_dir

def find_and_visualize_recurrent_sets():
    """
    Computes the box map, finds recurrent sets, and visualizes them.
    """
    print("--- Running Recurrent Set Analysis Test ---")

    # 1. Compute the Box Map
    print("Computing box map...")
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[100, 100])
    tau = 0.5
    
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
    )

    # Check for and report any problematic points from the computation
    if hasattr(box_map, 'debug_info'):
        exceeded_jump_points = box_map.debug_info.get('max_jumps_exceeded_points')
        if exceeded_jump_points:
            print("\n--- DEBUG: Points that exceeded max jumps ---")
            for point in exceeded_jump_points:
                print(f"  - Initial state: {point}")
            print("---------------------------------------------\n")

    # 2. Convert to NetworkX graph
    print("\nConverting to graph and analyzing...")
    graph = box_map.to_networkx()

    # 3. Find recurrent components via the Morse graph
    morse_graph, recurrent_components = create_morse_graph(graph)

    print(f"Found {len(recurrent_components)} recurrent component(s) to plot.")
    for i, scc in enumerate(recurrent_components):
        print(f"  - Component {i}: {len(scc)} boxes")

    # 4. Visualize the results
    print("\nGenerating visualizations...")
    output_base_dir = Path(__file__).parent / "recurrent_sets_figures"
    run_dir = get_next_run_dir(output_base_dir)
    
    # Plot 1: The boxes of the recurrent components on the state space grid
    scc_plot_path = run_dir / "recurrent_components_on_grid.png"
    plot_morse_sets_on_grid(grid, recurrent_components, scc_plot_path)

    # Plot 2: The morse graph
    morse_plot_path = run_dir / "morse_graph.png"
    plot_morse_graph_viz(morse_graph, recurrent_components, morse_plot_path)

    print(f"\n✓ Recurrent set analysis completed. Plots saved to: {run_dir}")

if __name__ == "__main__":
    find_and_visualize_recurrent_sets() 