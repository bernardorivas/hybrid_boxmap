#!/usr/bin/env python3
"""
Test the unstable periodic system without domain bounds to see natural behavior.
"""
import numpy as np
import matplotlib.pyplot as plt

from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics import HybridSystem


def test_without_bounds():
    """Compare behavior with and without domain bounds."""
    
    # Create two versions: one with bounds, one without
    system_with_bounds = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=20
    )
    
    # Create version without y-bounds (keep x bounds for jump detection)
    system_no_y_bounds = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-10.0, 10.0)],  # Very large y bounds
        max_jumps=20
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test initial conditions near boundary
    initial_conditions = [
        [0.5, 0.99],
        [0.5, 0.95],
        [0.5, 0.9],
        [0.5, 0.0],
    ]
    colors = ['red', 'blue', 'green', 'purple']
    
    for ax, system, title in [(ax1, system_with_bounds.system, "With Domain Bounds [-1, 1]"),
                               (ax2, system_no_y_bounds.system, "With Extended Bounds [-10, 10]")]:
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        
        # Draw original domain bounds
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Original domain y=±1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Jump boundary')
        
        for ic, color in zip(initial_conditions, colors):
            try:
                trajectory = system.simulate(
                    initial_state=np.array(ic),
                    time_span=(0.0, 5.0),
                    max_jumps=20
                )
                
                # Plot trajectory
                all_states = []
                for segment in trajectory.segments:
                    if segment.t_end > segment.t_start:
                        t_vals = np.linspace(segment.t_start, segment.t_end, 100)
                        for t in t_vals:
                            try:
                                state = segment.scipy_solution.sol(t)
                                all_states.append(state)
                            except:
                                pass
                
                if all_states:
                    states = np.array(all_states)
                    ax.plot(states[:, 0], states[:, 1], '-', color=color, 
                           alpha=0.7, linewidth=2, label=f'y₀={ic[1]}')
                
                # Mark jumps
                for state_minus, state_plus in trajectory.jump_states:
                    ax.plot(state_minus[0], state_minus[1], 'o', color=color, markersize=5)
                    ax.plot(state_plus[0], state_plus[1], 's', color=color, markersize=5)
                    
            except Exception as e:
                print(f"Error simulating from {ic} with {title}: {e}")
        
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('test_results/comparison_with_without_bounds.png', dpi=150, bbox_inches='tight')
    print("Saved to: test_results/comparison_with_without_bounds.png")
    
    # Analyze the difference
    print("\n=== Analysis ===")
    print("\n1. With strict domain bounds [-1, 1]:")
    print("   - Simulation stops when trajectory would exceed y = ±1")
    print("   - Trajectories get 'stuck' (simulation terminates)")
    print("   - This creates artificial behavior\n")
    
    print("2. With extended bounds [-10, 10]:")
    print("   - Trajectories can follow the natural cubic reset map")
    print("   - y-values can grow/shrink according to y_new = -y³ + 2y")
    print("   - Shows the true unstable behavior")
    
    # Show what happens with iteration of the reset map
    print("\n3. Iterating the reset map from y = 0.99:")
    y = 0.99
    print(f"   Start: y = {y}")
    for i in range(5):
        y_new = -y**3 + 2*y
        print(f"   Jump {i+1}: y = {y:.6f} → y_new = {y_new:.6f}")
        y = y_new
    
    print("\n4. The y = ±1 fixed points are unstable!")
    print("   Derivative at y = 1: dy_new/dy = -3(1)² + 2 = -1")
    print("   This is marginally stable, but numerical errors can cause divergence")


if __name__ == "__main__":
    test_without_bounds()