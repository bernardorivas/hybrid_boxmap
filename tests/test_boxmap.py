#!/usr/bin/env python3
"""
Test script for the HybridBoxMap class using the rimless wheel example.
"""
import numpy as np
from pathlib import Path
import pickle

from hybrid_dynamics import Grid, HybridBoxMap, config
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics import visualize_box_map

def test_compute_rimless_wheel_boxmap():
    """
    Tests the computation of the HybridBoxMap for the RimlessWheel system.
    """
    print("Testing HybridBoxMap computation for the Rimless Wheel...")

    # 1. System and Grid Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    bounds = wheel.domain_bounds
    subdivisions = [20, 20]  # Using a smaller grid for a faster test
    grid = Grid(bounds, subdivisions)
    tau = 0.2

    print(f"Grid: {grid.total_boxes} boxes, Tau: {tau}")

    # 2. Compute the Hybrid Box Map
    # This single call performs the evaluation of the flow and constructs the map.
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        # bloat_factor will use default from config if not specified
        parallel=False # Keep it sequential for now to avoid pickling issues
    )

    # 3. Verify and Save Results
    assert box_map, "The computed box map should not be empty."
    
    # Print some statistics
    source_boxes_count = len(box_map)
    total_destination_links = sum(len(dests) for dests in box_map.values())
    avg_dest_per_source = total_destination_links / source_boxes_count if source_boxes_count > 0 else 0

    print(f"✓ Computation complete.")
    print(f"  - {source_boxes_count} source boxes have at least one destination.")
    print(f"  - {avg_dest_per_source:.2f} average destination boxes per source box.")

    # Save the computed box map
    output_dir = Path(__file__).parent
    output_path = output_dir / "rimless_wheel_boxmap.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(box_map, f)
    print(f"Box map saved to {output_path}")

    # 4. Visualize the box map
    viz_path = output_dir / "rimless_wheel_boxmap.png"
    visualize_box_map(grid, box_map, viz_path)

    return box_map

if __name__ == "__main__":
    test_compute_rimless_wheel_boxmap() 