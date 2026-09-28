#!/usr/bin/env python3
"""
Test Boundary-Bridged Stratified Enclosure Method

This demo compares the new boundary-bridged enclosure method with the original
per-point bloating method for boxes that have corners with different jump counts.
"""
import time
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from hybrid_dynamics import (
    Grid, 
    HybridBoxMap, 
    create_morse_graph,
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.demo_utils import (
    create_next_run_dir,
    create_progress_callback,
)


def find_boxes_with_mixed_jumps(box_map_old, box_map_new, grid, system, tau):
    """Find boxes where corners have different numbers of jumps."""
    mixed_jump_boxes = []
    
    # Test a sample of boxes
    test_boxes = np.random.choice(len(grid), min(100, len(grid)), replace=False)
    
    for box_idx in test_boxes:
        # Get corners of this box
        corners = grid.get_sample_points(box_idx, mode='corners')
        
        # Simulate from each corner
        jump_counts = []
        for corner in corners:
            try:
                traj = system.simulate(corner, (0, tau))
                jump_counts.append(traj.num_jumps)
            except:
                jump_counts.append(-1)
        
        # Check if there are different jump counts
        unique_jumps = set(j for j in jump_counts if j >= 0)
        if len(unique_jumps) > 1:
            mixed_jump_boxes.append({
                'box_idx': box_idx,
                'jump_counts': jump_counts,
                'unique_jumps': sorted(unique_jumps),
                'old_destinations': box_map_old.get(box_idx, set()),
                'new_destinations': box_map_new.get(box_idx, set())
            })
    
    return mixed_jump_boxes


def visualize_box_destinations(grid, box_info, old_destinations, new_destinations, 
                              output_path):
    """Visualize the destination sets for a specific box."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get box bounds
    box_idx = box_info['box_idx']
    lower_bounds, upper_bounds = grid.get_box_bounds(box_idx)
    
    # Common setup for both axes
    for ax, destinations, title in [
        (ax1, old_destinations, "Original Per-Point Bloating"),
        (ax2, new_destinations, "Boundary-Bridged Enclosure")
    ]:
        ax.set_xlim(grid.bounds[0])
        ax.set_ylim(grid.bounds[1])
        ax.set_aspect('equal')
        ax.set_xlabel('Angle θ (rad)')
        ax.set_ylabel('Angular Velocity ω (rad/s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Plot source box
        source_rect = patches.Rectangle(
            (lower_bounds[0], lower_bounds[1]),
            upper_bounds[0] - lower_bounds[0],
            upper_bounds[1] - lower_bounds[1],
            facecolor='red', alpha=0.3, edgecolor='red', linewidth=2
        )
        ax.add_patch(source_rect)
        
        # Plot destination boxes
        dest_rectangles = []
        for dest_idx in destinations:
            dest_lower, dest_upper = grid.get_box_bounds(dest_idx)
            rect = patches.Rectangle(
                (dest_lower[0], dest_lower[1]),
                dest_upper[0] - dest_lower[0],
                dest_upper[1] - dest_lower[1]
            )
            dest_rectangles.append(rect)
        
        if dest_rectangles:
            pc = PatchCollection(dest_rectangles, facecolor='blue', 
                               edgecolor='darkblue', alpha=0.5, linewidth=0.5)
            ax.add_collection(pc)
        
        # Add text info
        ax.text(0.02, 0.98, f"Source box: {box_idx}\nJumps: {box_info['unique_jumps']}\n"
                           f"Dest boxes: {len(destinations)}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f"Destination Set Comparison for Box {box_idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_enclosure_methods():
    """Compare boundary-bridged vs per-point bloating methods."""
    
    # Suppress warnings
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[41, 41])
    tau = 1.5  # Choose tau that likely creates mixed jump counts
    bloat_factor = 0.1
    
    print("=" * 80)
    print("BOUNDARY-BRIDGED ENCLOSURE METHOD COMPARISON")
    print("=" * 80)
    
    # Create output directory
    figures_base_dir = Path("figures/boundary_bridged_enclosure")
    figures_base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist(), bloat_factor)
    
    print(f"\nCreated run directory: {run_dir.name}")
    print(f"Parameters: tau={tau}, grid={grid.subdivisions.tolist()}, bloat={bloat_factor}")
    
    progress_callback = create_progress_callback(update_interval=500)
    
    # Compute box map with OLD method (per-point bloating)
    print("\n1. Computing box map with ORIGINAL per-point bloating...")
    start_time = time.time()
    box_map_old = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        progress_callback=progress_callback,
        parallel=False,
        use_boundary_bridging=False  # Use old method
    )
    time_old = time.time() - start_time
    print(f"Completed in {time_old:.2f} seconds")
    print(f"Total transitions: {sum(len(dests) for dests in box_map_old.values())}")
    
    # Compute box map with NEW method (boundary-bridged)
    print("\n2. Computing box map with NEW boundary-bridged enclosure...")
    start_time = time.time()
    box_map_new = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        progress_callback=progress_callback,
        parallel=False,
        use_boundary_bridging=True  # Use new method
    )
    time_new = time.time() - start_time
    print(f"Completed in {time_new:.2f} seconds")
    print(f"Total transitions: {sum(len(dests) for dests in box_map_new.values())}")
    
    # Find boxes with mixed jump counts
    print("\n3. Finding boxes with mixed jump counts...")
    mixed_boxes = find_boxes_with_mixed_jumps(box_map_old, box_map_new, grid, wheel.system, tau)
    print(f"Found {len(mixed_boxes)} boxes with corners having different jump counts")
    
    if mixed_boxes:
        # Sort by difference in destination count
        mixed_boxes.sort(key=lambda x: abs(len(x['new_destinations']) - len(x['old_destinations'])), 
                        reverse=True)
        
        print("\nTop boxes with largest differences:")
        for i, box_info in enumerate(mixed_boxes[:5]):
            old_count = len(box_info['old_destinations'])
            new_count = len(box_info['new_destinations'])
            print(f"  Box {box_info['box_idx']}: jumps={box_info['unique_jumps']}, "
                  f"old_dests={old_count}, new_dests={new_count}, diff={new_count-old_count}")
        
        # Visualize top examples
        print("\n4. Generating visualizations...")
        for i, box_info in enumerate(mixed_boxes[:3]):
            output_path = run_dir / f"box_{box_info['box_idx']}_comparison.png"
            visualize_box_destinations(
                grid, box_info, 
                box_info['old_destinations'], 
                box_info['new_destinations'],
                output_path
            )
            print(f"  Generated visualization for box {box_info['box_idx']}")
    
    # Compute and compare Morse graphs
    print("\n5. Computing Morse graphs...")
    graph_old = box_map_old.to_networkx()
    morse_graph_old, morse_sets_old = create_morse_graph(graph_old)
    
    graph_new = box_map_new.to_networkx()
    morse_graph_new, morse_sets_new = create_morse_graph(graph_new)
    
    print(f"Old method: {len(morse_sets_old)} Morse sets")
    print(f"New method: {len(morse_sets_new)} Morse sets")
    
    # Overall comparison plot
    print("\n6. Creating overall comparison plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot transition counts
    ax1.bar(['Old Method', 'New Method'], 
            [sum(len(dests) for dests in box_map_old.values()),
             sum(len(dests) for dests in box_map_new.values())])
    ax1.set_ylabel('Total Transitions')
    ax1.set_title('Transition Count Comparison')
    
    # Plot computation time
    ax2.bar(['Old Method', 'New Method'], [time_old, time_new])
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time')
    
    # Plot Morse set counts
    ax3.bar(['Old Method', 'New Method'], [len(morse_sets_old), len(morse_sets_new)])
    ax3.set_ylabel('Number of Morse Sets')
    ax3.set_title('Morse Decomposition')
    
    # Plot distribution of destination counts for mixed boxes
    if mixed_boxes:
        old_dest_counts = [len(box['old_destinations']) for box in mixed_boxes]
        new_dest_counts = [len(box['new_destinations']) for box in mixed_boxes]
        
        ax4.scatter(old_dest_counts, new_dest_counts, alpha=0.6)
        ax4.plot([0, max(max(old_dest_counts), max(new_dest_counts))],
                [0, max(max(old_dest_counts), max(new_dest_counts))], 
                'k--', alpha=0.3)
        ax4.set_xlabel('Old Method Destinations')
        ax4.set_ylabel('New Method Destinations')
        ax4.set_title('Destination Count Comparison (Mixed Jump Boxes)')
    else:
        ax4.text(0.5, 0.5, 'No mixed jump boxes found', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle('Boundary-Bridged Enclosure Method Comparison')
    plt.tight_layout()
    plt.savefig(run_dir / 'overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary_file = run_dir / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Boundary-Bridged Enclosure Method Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  - tau: {tau}\n")
        f.write(f"  - grid: {grid.subdivisions.tolist()}\n")
        f.write(f"  - bloat_factor: {bloat_factor}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Old method (per-point bloating):\n")
        f.write(f"    - Transitions: {sum(len(dests) for dests in box_map_old.values())}\n")
        f.write(f"    - Morse sets: {len(morse_sets_old)}\n")
        f.write(f"    - Time: {time_old:.2f}s\n\n")
        f.write(f"  New method (boundary-bridged):\n")
        f.write(f"    - Transitions: {sum(len(dests) for dests in box_map_new.values())}\n")
        f.write(f"    - Morse sets: {len(morse_sets_new)}\n")
        f.write(f"    - Time: {time_new:.2f}s\n\n")
        f.write(f"  Mixed jump boxes found: {len(mixed_boxes)}\n")
        if mixed_boxes:
            f.write(f"  Average destination difference: "
                   f"{np.mean([len(b['new_destinations'])-len(b['old_destinations']) for b in mixed_boxes]):.1f}\n")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    compare_enclosure_methods()