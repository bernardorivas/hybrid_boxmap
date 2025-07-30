#!/usr/bin/env python3
"""
Analyze behavior near the reset boundary for the rimless wheel with jump penalty.
This script investigates why transitions seem to get stuck at the reset region.
"""
import numpy as np
import matplotlib.pyplot as plt
from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel

def analyze_reset_boundary():
    # Create system and grid
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[51, 51])
    
    # Test parameters
    tau_values = [1.5, 3.0, 5.0]
    bloat_factor = 0.12
    
    # The reset event occurs when theta = alpha + gamma = 0.6
    reset_theta = wheel.alpha + wheel.gamma
    
    print("Rimless Wheel Reset Boundary Analysis")
    print("=" * 60)
    print(f"Reset occurs at theta = {reset_theta}")
    print(f"Domain bounds: theta ∈ [{wheel.domain_bounds[0][0]}, {wheel.domain_bounds[0][1]}]")
    print(f"               omega ∈ [{wheel.domain_bounds[1][0]}, {wheel.domain_bounds[1][1]}]")
    print()
    
    # Find boxes near the reset boundary
    theta_idx_reset = int((reset_theta - wheel.domain_bounds[0][0]) / grid.box_widths[0])
    print(f"Reset boundary is approximately at theta index {theta_idx_reset}")
    
    # Test specific points near the boundary
    test_theta_values = [0.55, 0.58, 0.59, 0.595, 0.598, 0.599]
    test_omega = 0.5
    
    print("\nTesting individual trajectories near reset boundary:")
    print("-" * 60)
    
    for tau in tau_values:
        print(f"\nτ = {tau}:")
        for theta in test_theta_values:
            state = np.array([theta, test_omega])
            
            # Simulate without penalty
            traj_no_penalty = wheel.system.simulate(
                state, (0, tau), jump_time_penalty=False
            )
            
            # Simulate with penalty
            traj_with_penalty = wheel.system.simulate(
                state, (0, tau), jump_time_penalty=True
            )
            
            print(f"  θ={theta:.3f}: no_penalty={traj_no_penalty.num_jumps} jumps, "
                  f"with_penalty={traj_with_penalty.num_jumps} jumps")
            
            # Check if trajectory gets stuck near reset
            if traj_with_penalty.final_state is not None:
                final_theta = traj_with_penalty.final_state[0]
                if abs(final_theta - reset_theta) < 0.05:
                    print(f"    → With penalty: stuck near reset (final θ={final_theta:.3f})")
    
    # Analyze box transitions near the boundary
    print("\n\nAnalyzing box map transitions near reset boundary:")
    print("-" * 60)
    
    for tau in tau_values:
        print(f"\nComputing box maps for τ = {tau}...")
        
        # Compute box maps
        box_map_no_penalty = HybridBoxMap.compute(
            grid=grid,
            system=wheel.system,
            tau=tau,
            sampling_mode='corners',
            bloat_factor=bloat_factor,
            enclosure=True,
            jump_time_penalty=False
        )
        
        box_map_with_penalty = HybridBoxMap.compute(
            grid=grid,
            system=wheel.system,
            tau=tau,
            sampling_mode='corners',
            bloat_factor=bloat_factor,
            enclosure=True,
            jump_time_penalty=True
        )
        
        # Count boxes that map to themselves (self-loops)
        self_loops_no_penalty = 0
        self_loops_with_penalty = 0
        boxes_near_reset = 0
        
        for box_idx in range(len(grid)):
            center = grid.get_sample_points(box_idx, mode="center")[0]
            theta = center[0]
            
            # Check if box is near reset boundary
            if abs(theta - reset_theta) < 0.05:
                boxes_near_reset += 1
                
                # Check for self-loops
                if box_idx in box_map_no_penalty and box_idx in box_map_no_penalty[box_idx]:
                    self_loops_no_penalty += 1
                if box_idx in box_map_with_penalty and box_idx in box_map_with_penalty[box_idx]:
                    self_loops_with_penalty += 1
        
        print(f"  Boxes near reset boundary: {boxes_near_reset}")
        print(f"  Self-loops (no penalty): {self_loops_no_penalty}")
        print(f"  Self-loops (with penalty): {self_loops_with_penalty}")
        
        # Check a specific box near the boundary
        test_theta = 0.59
        test_omega = 0.5
        test_point = np.array([test_theta, test_omega])
        # Find which box contains this point
        test_box_idx = None
        for idx in range(len(grid)):
            bounds = grid.get_box_bounds(idx)
            if (bounds[0][0] <= test_point[0] <= bounds[0][1] and 
                bounds[1][0] <= test_point[1] <= bounds[1][1]):
                test_box_idx = idx
                break
        
        if test_box_idx is not None:
            print(f"\n  Detailed analysis for box at θ={test_theta}, ω={test_omega}:")
            
            if test_box_idx in box_map_no_penalty:
                dests_no_penalty = box_map_no_penalty[test_box_idx]
                print(f"    No penalty: maps to {len(dests_no_penalty)} boxes")
                
                # Check if it maps to itself
                if test_box_idx in dests_no_penalty:
                    print("    → Contains self-loop (no penalty)")
            
            if test_box_idx in box_map_with_penalty:
                dests_with_penalty = box_map_with_penalty[test_box_idx]
                print(f"    With penalty: maps to {len(dests_with_penalty)} boxes")
                
                # Check if it maps to itself
                if test_box_idx in dests_with_penalty:
                    print("    → Contains self-loop (with penalty)")
    
    # Visualize the effect
    print("\n\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Reset Boundary Behavior Analysis", fontsize=14)
    
    for i, tau in enumerate(tau_values):
        # Compute box maps again for visualization
        box_map_no_penalty = HybridBoxMap.compute(
            grid=grid, system=wheel.system, tau=tau,
            sampling_mode='corners', bloat_factor=bloat_factor,
            enclosure=True, jump_time_penalty=False
        )
        
        box_map_with_penalty = HybridBoxMap.compute(
            grid=grid, system=wheel.system, tau=tau,
            sampling_mode='corners', bloat_factor=bloat_factor,
            enclosure=True, jump_time_penalty=True
        )
        
        # Plot boxes near reset that have self-loops
        for row, (box_map, title_suffix) in enumerate([
            (box_map_no_penalty, "No Penalty"),
            (box_map_with_penalty, "With Penalty")
        ]):
            ax = axes[row, i]
            ax.set_title(f"τ={tau} - {title_suffix}")
            ax.set_xlabel("Angle θ (rad)")
            ax.set_ylabel("Angular Velocity ω (rad/s)")
            ax.set_xlim(wheel.domain_bounds[0])
            ax.set_ylim(wheel.domain_bounds[1])
            
            # Mark boxes with self-loops near reset
            for box_idx in box_map:
                if box_idx in box_map[box_idx]:  # Self-loop
                    center = grid.get_sample_points(box_idx, mode="center")[0]
                    theta = center[0]
                    
                    if abs(theta - reset_theta) < 0.1:  # Near reset
                        rect = plt.Rectangle(
                            (center[0] - grid.box_widths[0]/2, 
                             center[1] - grid.box_widths[1]/2),
                            grid.box_widths[0], grid.box_widths[1],
                            facecolor='red', alpha=0.5
                        )
                        ax.add_patch(rect)
            
            # Draw reset boundary
            ax.axvline(x=reset_theta, color='black', linestyle='--', alpha=0.5, label='Reset boundary')
            
            # Sample trajectories
            for theta_offset in [-0.02, -0.01, -0.005]:
                state = np.array([reset_theta + theta_offset, 0.5])
                # Make sure state is within bounds
                if state[0] >= wheel.domain_bounds[0][0] and state[0] <= wheel.domain_bounds[0][1]:
                    traj = wheel.system.simulate(
                        state, (0, tau), 
                        jump_time_penalty=(row == 1)
                    )
                    if traj.segments:
                        states = traj.get_all_states()
                        if states.size > 0:
                            ax.plot(states[:, 0], states[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/reset_boundary_analysis.png", dpi=150)
    plt.close()
    print("Saved visualization to figures/reset_boundary_analysis.png")
    
    print("\n\nKey Insights:")
    print("-" * 60)
    print("1. With jump penalty, trajectories that jump consume 1 unit of time per jump")
    print("2. Near the reset boundary (θ ≈ 0.6), trajectories may:")
    print("   - Jump immediately if they start very close to the boundary")
    print("   - Flow for a short time before jumping")
    print("3. With penalty, if τ < 2, trajectories can make at most 1 jump")
    print("4. This creates 'sticky' behavior near reset: boxes can map to themselves")
    print("   because trajectories don't have enough time budget to jump away")

if __name__ == "__main__":
    analyze_reset_boundary()