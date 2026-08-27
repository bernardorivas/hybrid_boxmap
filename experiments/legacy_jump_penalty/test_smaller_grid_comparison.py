#!/usr/bin/env python3
"""
Test with smaller grid to better understand the boundary-bridged method.
"""
import time
import warnings
import numpy as np
from pathlib import Path

from hybrid_dynamics import Grid, HybridBoxMap, create_morse_graph
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.demo_utils import create_progress_callback


def count_mixed_jump_boxes(box_map, grid, system, tau):
    """Count boxes with mixed jump counts."""
    mixed_count = 0
    
    # Sample some boxes
    sample_boxes = list(box_map.keys())[:100]
    
    for box_idx in sample_boxes:
        corners = grid.get_sample_points(box_idx, mode='corners')
        jump_counts = []
        
        for corner in corners:
            try:
                traj = system.simulate(corner, (0, tau))
                jump_counts.append(traj.num_jumps)
            except:
                jump_counts.append(-1)
        
        unique_jumps = set(j for j in jump_counts if j >= 0)
        if len(unique_jumps) > 1:
            mixed_count += 1
    
    return mixed_count


def compare_methods_small_grid():
    """Compare methods on a smaller grid for clarity."""
    
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Smaller grid and tau
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[21, 21])
    tau = 1.2
    bloat_factor = 0.05  # Smaller bloat
    
    print("=" * 80)
    print("SMALL GRID COMPARISON")
    print("=" * 80)
    print(f"Grid: {grid.subdivisions.tolist()}, tau={tau}, bloat={bloat_factor}")
    
    progress_callback = create_progress_callback(update_interval=100)
    
    # Method 1: Standard enclosure
    print("\n1. Standard enclosure (no bridging)...")
    start_time = time.time()
    box_map_standard = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        progress_callback=progress_callback,
        parallel=False,
        use_boundary_bridging=False
    )
    time_standard = time.time() - start_time
    
    graph_standard = box_map_standard.to_networkx()
    _, morse_sets_standard = create_morse_graph(graph_standard)
    
    print(f"  Time: {time_standard:.2f}s")
    print(f"  Transitions: {sum(len(dests) for dests in box_map_standard.values())}")
    print(f"  Active boxes: {len(box_map_standard)}")
    print(f"  Morse sets: {len(morse_sets_standard)}")
    
    # Count mixed boxes
    mixed_standard = count_mixed_jump_boxes(box_map_standard, grid, wheel.system, tau)
    print(f"  Mixed jump boxes (sample): {mixed_standard}")
    
    # Method 2: Boundary-bridged
    print("\n2. Boundary-bridged enclosure...")
    start_time = time.time()
    box_map_bridged = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        progress_callback=progress_callback,
        parallel=False,
        use_boundary_bridging=True
    )
    time_bridged = time.time() - start_time
    
    graph_bridged = box_map_bridged.to_networkx()
    _, morse_sets_bridged = create_morse_graph(graph_bridged)
    
    print(f"  Time: {time_bridged:.2f}s")
    print(f"  Transitions: {sum(len(dests) for dests in box_map_bridged.values())}")
    print(f"  Active boxes: {len(box_map_bridged)}")
    print(f"  Morse sets: {len(morse_sets_bridged)}")
    
    # Analyze differences
    print("\n3. Detailed Analysis:")
    print("-" * 40)
    
    # Find boxes with biggest differences
    max_diff = 0
    max_diff_box = None
    
    for box_idx in box_map_standard.keys():
        std_dests = len(box_map_standard.get(box_idx, set()))
        brd_dests = len(box_map_bridged.get(box_idx, set()))
        diff = brd_dests - std_dests
        
        if diff > max_diff:
            max_diff = diff
            max_diff_box = box_idx
    
    print(f"Box with largest difference: {max_diff_box} ({max_diff} more destinations)")
    
    # Look at a specific box with mixed jumps
    print("\nAnalyzing specific mixed-jump box...")
    for box_idx in box_map_standard.keys():
        corners = grid.get_sample_points(box_idx, mode='corners')
        jump_counts = []
        
        for corner in corners:
            traj = wheel.system.simulate(corner, (0, tau))
            jump_counts.append(traj.num_jumps)
        
        if len(set(jump_counts)) > 1:
            print(f"\nBox {box_idx}:")
            print(f"  Jump counts: {jump_counts}")
            print(f"  Standard destinations: {len(box_map_standard.get(box_idx, set()))}")
            print(f"  Bridged destinations: {len(box_map_bridged.get(box_idx, set()))}")
            break
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_methods_small_grid()