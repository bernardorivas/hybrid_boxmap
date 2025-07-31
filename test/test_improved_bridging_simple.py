#!/usr/bin/env python3
"""
Simple test to verify the improved boundary-bridged method works correctly.
"""
import numpy as np
from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel


def test_single_box():
    """Test the improved method on a single box."""
    
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[41, 41])
    tau = 1.5
    
    # Search for a box with mixed jumps
    box_idx = None
    for idx in range(100, min(500, len(grid))):
        corners_test = grid.get_sample_points(idx, mode='corners')
        jumps_test = []
        for c in corners_test:
            traj = wheel.system.simulate(c, (0, tau))
            jumps_test.append(traj.num_jumps)
        if len(set(jumps_test)) > 1:
            box_idx = idx
            break
    
    if box_idx is None:
        print("No mixed jump box found!")
        return
    corners = grid.get_sample_points(box_idx, mode='corners')
    
    print(f"Testing box {box_idx} with {len(corners)} corners")
    print("-" * 60)
    
    # Simulate each corner to check jump counts
    jump_counts = []
    for i, corner in enumerate(corners):
        traj = wheel.system.simulate(corner, (0, tau))
        jump_counts.append(traj.num_jumps)
        print(f"Corner {i}: {corner} → {traj.num_jumps} jumps")
    
    if len(set(jump_counts)) > 1:
        print(f"\nMixed jumps detected: {set(jump_counts)}")
        
        # Test finding next guard for corners with fewer jumps
        min_jumps = min(jump_counts)
        print(f"\nTesting _find_next_guard_state for corners with {min_jumps} jumps:")
        
        for i, (corner, jumps) in enumerate(zip(corners, jump_counts)):
            if jumps == min_jumps:
                # Get final state
                traj = wheel.system.simulate(corner, (0, tau))
                final_state = traj.interpolate(tau)
                
                # Find next guard
                next_guard = HybridBoxMap._find_next_guard_state(
                    wheel.system, final_state, 0.0, 5.0
                )
                
                if next_guard is not None:
                    print(f"\nCorner {i}:")
                    print(f"  Final state after {tau}s: {final_state}")
                    print(f"  Next guard state found: {next_guard}")
                    
                    # Check reset map
                    try:
                        post_jump = wheel.system.reset_map(next_guard)
                        print(f"  Post-jump state: {post_jump}")
                    except Exception as e:
                        print(f"  Reset failed: {e}")
                else:
                    print(f"\nCorner {i}: No next guard found")
    else:
        print("\nNo mixed jumps in this box")
    
    # Now compute box maps to compare transition counts
    print("\n" + "=" * 60)
    print("Comparing box map methods...")
    
    # Standard method
    box_map_standard = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=0.1,
        enclosure=True,
        use_boundary_bridging=False,
        parallel=False
    )
    
    # Improved method
    box_map_improved = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=0.1,
        enclosure=True,
        use_boundary_bridging=True,
        parallel=False
    )
    
    # Compare results for our test box
    std_dests = len(box_map_standard.get(box_idx, set()))
    imp_dests = len(box_map_improved.get(box_idx, set()))
    
    print(f"\nBox {box_idx} destinations:")
    print(f"  Standard method: {std_dests}")
    print(f"  Improved method: {imp_dests}")
    print(f"  Difference: {imp_dests - std_dests}")
    
    # Overall statistics
    total_std = sum(len(dests) for dests in box_map_standard.values())
    total_imp = sum(len(dests) for dests in box_map_improved.values())
    
    print(f"\nTotal transitions:")
    print(f"  Standard method: {total_std}")
    print(f"  Improved method: {total_imp}")
    print(f"  Increase: {100 * (total_imp - total_std) / total_std:.1f}%")


if __name__ == "__main__":
    test_single_box()