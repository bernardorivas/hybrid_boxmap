#!/usr/bin/env python3
"""
Investigate why boxes on the right boundary have self-loops.
"""
import numpy as np

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    setup_demo_directories,
    load_box_map_from_cache,
)


# Configuration
TAU = 1.42
SUBDIVISIONS = [100, 100]


def investigate_self_loops():
    """Detailed investigation of self-loops on right boundary."""
    
    # Setup
    data_dir, _ = setup_demo_directories("unstable_periodic")
    box_map_file = data_dir / "unstable_periodic_box_map.pkl"
    
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    
    # Load box map
    system_params = {"max_jumps": system_obj.max_jumps}
    current_config_hash = create_config_hash(
        system_params, 
        system_obj.domain_bounds, 
        grid.subdivisions.tolist(), 
        TAU
    )
    
    box_map = load_box_map_from_cache(grid, system_obj.system, TAU, box_map_file, current_config_hash)
    
    print("=== Self-Loop Investigation ===\n")
    
    # Get a specific box on right boundary
    right_boundary_x_index = grid.subdivisions[0] - 1
    
    # Test a few boxes
    test_cases = [
        (right_boundary_x_index, 0, "Bottom corner"),
        (right_boundary_x_index, 50, "Middle"),
        (right_boundary_x_index, 99, "Top corner"),
        (right_boundary_x_index, 25, "Lower middle"),
    ]
    
    for x_idx, y_idx, label in test_cases:
        box_idx = int(np.ravel_multi_index([x_idx, y_idx], grid.subdivisions))
        box = grid.get_box(box_idx)
        
        print(f"\n--- {label}: Box {box_idx} ---")
        print(f"Box bounds: x ∈ [{box.lower_bounds[0]:.4f}, {box.upper_bounds[0]:.4f}], "
              f"y ∈ [{box.lower_bounds[1]:.4f}, {box.upper_bounds[1]:.4f}]")
        
        # Check if it has self-loop
        has_self_loop = box_idx in box_map.get(box_idx, set())
        print(f"Has self-loop: {has_self_loop}")
        
        if box_idx in box_map:
            destinations = box_map[box_idx]
            print(f"Total destinations: {len(destinations)}")
            
            # Analyze destinations
            dest_regions = {"left": 0, "middle": 0, "right": 0, "self": 0}
            
            for dest_idx in destinations:
                if dest_idx == box_idx:
                    dest_regions["self"] += 1
                else:
                    dest_box = grid.get_box(dest_idx)
                    x_center = (dest_box.lower_bounds[0] + dest_box.upper_bounds[0]) / 2
                    if x_center < 0.3:
                        dest_regions["left"] += 1
                    elif x_center > 0.7:
                        dest_regions["right"] += 1
                    else:
                        dest_regions["middle"] += 1
            
            print(f"Destination breakdown: {dest_regions}")
            
            # Simulate from corners and center to understand why self-loop exists
            print("\nTrajectory analysis:")
            
            # Get sample points (corners + center)
            sample_points = grid.get_sample_points(box_idx, mode='corners')
            center_point = grid.get_sample_points(box_idx, mode='center')[0]
            
            # Add center to sample points
            test_points = list(sample_points) + [center_point]
            point_labels = [f"Corner {i}" for i in range(len(sample_points))] + ["Center"]
            
            for point, point_label in zip(test_points[:3], point_labels[:3]):  # Test first 3
                print(f"\n  From {point_label} at ({point[0]:.4f}, {point[1]:.4f}):")
                
                # Simulate
                try:
                    trajectory = system_obj.system.simulate(
                        initial_state=point,
                        time_span=(0.0, TAU),
                        max_jumps=10
                    )
                    
                    # Check final state
                    final_state = trajectory(TAU)
                    print(f"    Final state: ({final_state[0]:.4f}, {final_state[1]:.4f})")
                    
                    # Which box does it end up in?
                    try:
                        final_box_idx = grid.get_box_from_point(final_state)
                        if final_box_idx == box_idx:
                            print(f"    → Stays in same box!")
                        else:
                            print(f"    → Ends in box {final_box_idx}")
                    except ValueError:
                        print(f"    → Ends outside domain")
                    
                    # Count jumps
                    print(f"    Number of jumps: {len(trajectory.jumps)}")
                    
                    # Show jump details
                    for i, jump in enumerate(trajectory.jumps[:2]):  # First 2 jumps
                        print(f"    Jump {i+1}: ({jump['state_minus'][0]:.4f}, {jump['state_minus'][1]:.4f}) "
                              f"→ ({jump['state_plus'][0]:.4f}, {jump['state_plus'][1]:.4f})")
                        
                except Exception as e:
                    print(f"    Error: {e}")
    
    # Now let's understand the bloat factor effect
    print("\n\n=== Understanding Bloat Factor ===")
    print(f"Current bloat factor in box map computation: {box_map.metadata.get('bloat_factor', 'Not stored')}")
    
    # The self-loops likely occur because:
    # 1. Some trajectories from corners stay within time tau
    # 2. The bloat factor expands the destination region
    # 3. This expanded region includes the source box
    
    print("\nPossible reasons for self-loops on right boundary:")
    print("1. Bloat factor: After flowing and jumping, the bloated destination region")
    print("   might overlap with the source box, especially for boxes at extreme y-values")
    print("2. Corner sampling: Different corners of a box may have different behaviors")
    print("3. Time horizon: τ = 1.42 allows for one jump and some flow afterwards")
    
    # Check the reset map behavior for extreme y
    print("\n=== Reset Map Analysis for Extreme Y ===")
    for y in [-0.99, -0.5, 0.0, 0.5, 0.99]:
        reset_state = system_obj.system.reset_map(np.array([1.0, y]))
        print(f"Reset from (1.0, {y:5.2f}) → ({reset_state[0]:.3f}, {reset_state[1]:.3f})")
        
        # After reset, how long to flow back to right boundary?
        time_to_return = (1.0 - reset_state[0]) / 1.0
        print(f"  Time to return to x=1: {time_to_return:.3f}")
        print(f"  Remaining time after return: {TAU - time_to_return:.3f}")


if __name__ == "__main__":
    investigate_self_loops()