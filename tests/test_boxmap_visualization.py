#!/usr/bin/env python3
"""
Test script for visualizing the HybridBoxMap for the rimless wheel example.

This script generates a directory of plots, where each plot shows the
destination boxes for a single source box.
"""
import random
from pathlib import Path
import numpy as np

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics import visualize_box_map_entry

def get_next_run_dir(base_dir: Path) -> Path:
    """Finds the next available run directory."""
    base_dir.mkdir(exist_ok=True)
    existing_runs = [int(d.name.split('_')[1]) for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = base_dir / f"run_{next_run_num:04d}"
    run_dir.mkdir()
    return run_dir

def visualize_hybrid_boxmap(
    selection_mode: str = 'random',
    max_plots: int = 10,
    every_n: int = 5
):
    """
    Computes and visualizes the HybridBoxMap.

    Args:
        selection_mode: How to select boxes to plot.
                        'all': plot all mapped boxes.
                        'random': plot a random sample of `max_plots`.
                        'every_n': plot every nth box.
        max_plots: The number of random plots to generate for 'random' mode.
        every_n: The 'n' for 'every_n' mode.
    """
    print("--- Running HybridBoxMap Visualization Test ---")
    
    # 1. System and Grid Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[30, 30])
    tau = 0.2
    sampling_mode = 'corners'

    # 2. Compute the Box Map
    print("Computing box map (this may take a moment)...")
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode=sampling_mode,
        parallel=False,
    )
    print(f"Box map computed with {len(box_map)} mapped source boxes.")

    # 3. Set up output directories
    output_base_dir = Path(__file__).parent / "boxmap_figures"
    run_dir = get_next_run_dir(output_base_dir)
    print(f"Saving plots to: {run_dir}")

    # 4. Select which boxes to plot based on the mode
    source_indices = sorted(list(box_map.keys()))
    
    if selection_mode == 'all':
        selected_indices = source_indices
    elif selection_mode == 'random':
        num_to_select = min(max_plots, len(source_indices))
        selected_indices = random.sample(source_indices, num_to_select)
    elif selection_mode == 'every_n':
        selected_indices = source_indices[::every_n]
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    print(f"Generating {len(selected_indices)} plots for mode '{selection_mode}'...")

    # 5. Generate and save a plot for each selected box
    for i, source_idx in enumerate(selected_indices):
        destination_indices = box_map[source_idx]
        
        # Get the initial sample points for the source box
        initial_points = grid.get_sample_points(source_idx, mode=sampling_mode)
        
        # Compute the final points just for these initial points
        final_points = []
        for p in initial_points:
            try:
                # This logic mirrors the flow_with_jumps method
                traj = wheel.system.simulate(p, (0, tau))
                if traj.total_duration >= tau:
                    final_points.append(traj.interpolate(tau))
                else:
                    final_points.append(np.full(grid.ndim, np.nan))
            except Exception:
                final_points.append(np.full(grid.ndim, np.nan))
        
        final_points = np.array(final_points)

        save_path = run_dir / f"source_box_{source_idx}.png"
        visualize_box_map_entry(grid, source_idx, destination_indices, initial_points, final_points, save_path)
        print(f"  ({i+1}/{len(selected_indices)}) Plot saved for source box {source_idx}.")

    print("\n✓ Visualization script completed successfully!")

if __name__ == "__main__":
    # You can change the mode here to control the output
    visualize_hybrid_boxmap(selection_mode='random', max_plots=5)
    # visualize_hybrid_boxmap(selection_mode='every_n', every_n=10)
    # visualize_hybrid_boxmap(selection_mode='all') 