#!/usr/bin/env python3
"""
Simple demonstration of boundary trapping behavior.
"""
import numpy as np
import matplotlib.pyplot as plt

from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem


def demonstrate_trapping():
    """Show how trajectories get trapped at boundaries."""
    
    # Create system
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=10
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Phase space trajectories
    ax1.set_title("Trajectories Starting Near y = 1 Boundary")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(0.8, 1.15)
    ax1.grid(True, alpha=0.3)
    
    # Domain boundary
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Domain boundary y=1')
    ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Jump boundary x=1')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Simulate from different starting points near y=1
    initial_y_values = [0.98, 0.99, 0.995, 0.999]
    colors = ['green', 'orange', 'purple', 'brown']
    
    for y0, color in zip(initial_y_values, colors):
        initial_state = np.array([0.5, y0])
        
        # Simulate
        trajectory = system_obj.system.simulate(
            initial_state=initial_state,
            time_span=(0.0, 3.0),
            max_jumps=10
        )
        
        # Plot segments
        for segment in trajectory.segments:
            if segment.t_end > segment.t_start:
                t_vals = np.linspace(segment.t_start, segment.t_end, 50)
                states = []
                for t in t_vals:
                    try:
                        state = segment.scipy_solution.sol(t)
                        states.append(state)
                    except:
                        pass
                
                if states:
                    states = np.array(states)
                    ax1.plot(states[:, 0], states[:, 1], '-', color=color, 
                            alpha=0.7, linewidth=2, label=f'y₀={y0}' if segment == trajectory.segments[0] else '')
        
        # Mark jumps
        for state_minus, state_plus in trajectory.jump_states:
            ax1.plot(state_minus[0], state_minus[1], 'o', color=color, markersize=6)
            ax1.plot(state_plus[0], state_plus[1], 's', color=color, markersize=6)
            
            # Show where it would go without clipping
            y_unclamped = -state_minus[1]**3 + 2*state_minus[1]
            if y_unclamped > 1:
                ax1.plot(state_plus[0], y_unclamped, 'x', color=color, markersize=8, alpha=0.5)
    
    ax1.legend(loc='lower left')
    
    # Right plot: Reset map behavior
    ax2.set_title("Reset Map Near y = 1")
    ax2.set_xlabel('y (before reset)')
    ax2.set_ylabel('y (after reset)')
    ax2.grid(True, alpha=0.3)
    
    # Plot reset map
    y_vals = np.linspace(0.8, 1.0, 200)
    y_new_vals = -y_vals**3 + 2*y_vals
    
    ax2.plot(y_vals, y_new_vals, 'b-', linewidth=2, label='y_new = -y³ + 2y')
    ax2.plot(y_vals, y_vals, 'k--', alpha=0.5, label='Identity')
    
    # Show domain boundary
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Domain boundary')
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.3)
    
    # Highlight the problematic region
    problem_y = y_vals[y_new_vals > 1]
    problem_y_new = y_new_vals[y_new_vals > 1]
    ax2.fill_between(problem_y, 1, problem_y_new, alpha=0.3, color='red', 
                     label='Values that exceed domain')
    
    # Mark specific points
    for y0 in initial_y_values:
        y_new = -y0**3 + 2*y0
        ax2.plot(y0, y_new, 'o', markersize=8, color='red')
        ax2.plot(y0, np.clip(y_new, -1, 1), 's', markersize=8, color='green')
    
    ax2.set_xlim(0.8, 1.02)
    ax2.set_ylim(0.95, 1.12)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test_results/boundary_trapping_explained.png', dpi=150, bbox_inches='tight')
    print("Saved to: test_results/boundary_trapping_explained.png")
    
    # Print analysis
    print("\n=== Boundary Trapping Mechanism ===")
    print("\n1. The reset map y_new = -y³ + 2y has a fixed point at y = 1")
    print("2. For y close to 1, the map produces y_new > 1:")
    for y in [0.98, 0.99, 0.995, 0.999]:
        y_new = -y**3 + 2*y
        print(f"   y = {y} → y_new = {y_new:.6f}")
    
    print("\n3. The system clips values outside [-1, 1] back to the boundary")
    print("4. This creates a 'sticky' boundary where trajectories get trapped")
    print("\n5. Once at y = 1, the trajectory:")
    print("   - Flows rightward (dx/dt = 1)")
    print("   - Hits x = 1 and jumps back to x = 0")
    print("   - Reset map keeps y at 1 (since it's a fixed point)")
    print("   - Cycle repeats indefinitely")
    
    print("\nThis explains the vertical Morse sets at the right boundary!")


if __name__ == "__main__":
    demonstrate_trapping()