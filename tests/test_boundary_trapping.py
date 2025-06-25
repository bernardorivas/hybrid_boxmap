#!/usr/bin/env python3
"""
Visualize trajectories that get trapped at the boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src.hybrid_trajectory import HybridTrajectory


def plot_trapped_trajectories():
    """Plot trajectories starting near y = ±1 to show trapping behavior."""
    
    # Create system
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=20
    )
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Common setup for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
        
        # Draw domain bounds
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Domain boundary')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Jump boundary')
    
    # Test cases showing different behaviors
    test_cases = [
        # (initial_state, time_span, title, ax)
        ([0.5, 0.99], 5.0, "Starting near y=1 boundary", ax1),
        ([0.5, -0.99], 5.0, "Starting near y=-1 boundary", ax2),
        ([0.5, 0.85], 5.0, "Starting at y=0.85 (near critical point)", ax3),
        ([0.5, 0.0], 5.0, "Starting at y=0 (unstable fixed point)", ax4),
    ]
    
    # For tracking what happens
    trajectory_summaries = []
    
    for initial_state, t_max, title, ax in test_cases:
        ax.set_title(title)
        
        # Simulate
        trajectory = system_obj.system.simulate(
            initial_state=np.array(initial_state),
            time_span=(0.0, t_max),
            max_jumps=20
        )
        
        # Plot continuous segments
        all_x = []
        all_y = []
        
        for i, segment in enumerate(trajectory.segments):
            # Sample points along segment
            if segment.t_end > segment.t_start:
                t_vals = np.linspace(segment.t_start, segment.t_end, 100)
                states = []
                for t in t_vals:
                    try:
                        # Get state at time t
                        sol = segment.scipy_solution.sol
                        state = sol(t)
                        states.append(state)
                    except:
                        pass
                
                if states:
                    states = np.array(states)
                    ax.plot(states[:, 0], states[:, 1], 'b-', alpha=0.7, linewidth=1.5)
                    all_x.extend(states[:, 0])
                    all_y.extend(states[:, 1])
        
        # Mark jumps
        for i, (jump_time, (state_minus, state_plus)) in enumerate(zip(trajectory.jump_times, trajectory.jump_states)):
            # Pre-jump state
            ax.plot(state_minus[0], state_minus[1], 
                   'ro', markersize=6, label='Pre-jump' if i == 0 else '')
            
            # Post-jump state
            ax.plot(state_plus[0], state_plus[1], 
                   'go', markersize=6, label='Post-jump' if i == 0 else '')
            
            # Jump arrow
            ax.annotate('', xy=(state_plus[0], state_plus[1]),
                       xytext=(state_minus[0], state_minus[1]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1))
        
        # Mark initial condition
        ax.plot(initial_state[0], initial_state[1], 'k*', markersize=12, label='Initial')
        
        # Mark final state
        try:
            final_time = min(t_max, trajectory.segments[-1].t_end)
            final_state = trajectory.segments[-1].scipy_solution.sol(final_time)
            ax.plot(final_state[0], final_state[1], 'ms', markersize=10, label='Final')
        except:
            pass
        
        # Add legend only to first subplot
        if ax == ax1:
            ax.legend(loc='lower left', fontsize=9)
        
        # Analyze trajectory
        summary = {
            'initial': initial_state,
            'num_jumps': len(trajectory.jump_times),
            'y_values': all_y,
            'title': title
        }
        trajectory_summaries.append(summary)
    
    plt.suptitle('Trajectory Behavior Near Domain Boundaries', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('test_results/boundary_trapping_trajectories.png', dpi=150, bbox_inches='tight')
    print("Trajectory plot saved to: test_results/boundary_trapping_trajectories.png")
    
    # Now create a detailed y-evolution plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (summary, ax) in enumerate(zip(trajectory_summaries, axes)):
        ax.set_title(summary['title'])
        ax.set_xlabel('Point index along trajectory')
        ax.set_ylabel('y value')
        ax.grid(True, alpha=0.3)
        
        y_vals = summary['y_values']
        ax.plot(y_vals, 'b-', alpha=0.7, linewidth=1)
        
        # Mark domain bounds
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='y = ±1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        
        # Add statistics
        y_array = np.array(y_vals)
        if len(y_array) > 0:
            textstr = f'Jumps: {summary["num_jumps"]}\n'
            textstr += f'y range: [{np.min(y_array):.3f}, {np.max(y_array):.3f}]\n'
            textstr += f'Final y: {y_array[-1]:.3f}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        ax.set_ylim(-1.3, 1.3)
    
    plt.suptitle('Y-coordinate Evolution Along Trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_results/y_evolution_analysis.png', dpi=150, bbox_inches='tight')
    print("Y-evolution plot saved to: test_results/y_evolution_analysis.png")
    
    # Analyze the reset map behavior in detail
    print("\n=== Detailed Analysis of Boundary Behavior ===")
    
    # Show what happens when y values are near ±1
    test_y_values = [0.99, 0.995, 0.999, 1.0]
    print("\nReset map behavior near y = 1:")
    for y in test_y_values:
        y_new = -y**3 + 2*y
        print(f"  y = {y:.3f} → y_new = {y_new:.6f} (excess: {y_new - 1:.6f})")
    
    print("\nReset map behavior near y = -1:")
    for y in [-val for val in test_y_values]:
        y_new = -y**3 + 2*y
        print(f"  y = {y:.3f} → y_new = {y_new:.6f} (excess: {-1 - y_new:.6f})")
    
    # Show iterative behavior
    print("\n=== Iterative Behavior When Clipped ===")
    print("Starting at y = 0.99, applying reset map and clipping to [-1, 1]:")
    y = 0.99
    for i in range(10):
        y_new = -y**3 + 2*y
        y_clipped = np.clip(y_new, -1, 1)
        print(f"  Iteration {i}: y = {y:.6f} → y_new = {y_new:.6f} → clipped = {y_clipped:.6f}")
        y = y_clipped
        if abs(y - 1.0) < 1e-6:
            print("  → Converged to boundary!")
            break


if __name__ == "__main__":
    plot_trapped_trajectories()