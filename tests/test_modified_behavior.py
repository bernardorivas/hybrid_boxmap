#!/usr/bin/env python3
"""
Test the modified behavior where simulation continues outside domain bounds.
"""
import numpy as np
import matplotlib.pyplot as plt

from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.src import create_morse_graph


def test_modified_behavior():
    """Test that trajectories now continue outside domain bounds."""
    
    # Create system with original bounds
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=20
    )
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Individual trajectories showing behavior outside bounds
    ax1.set_title("Trajectories Can Now Exit Domain Bounds")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.3)
    
    # Domain boundaries
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Domain bounds')
    ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Jump boundary')
    
    # Test trajectories starting near boundaries
    test_ics = [[0.5, 0.99], [0.5, 0.95], [0.5, -0.99], [0.5, 0.0]]
    colors = ['green', 'orange', 'purple', 'brown']
    
    for ic, color in zip(test_ics, colors):
        trajectory = system_obj.system.simulate(
            initial_state=np.array(ic),
            time_span=(0.0, 10.0),
            max_jumps=20
        )
        
        # Collect all states
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
            ax1.plot(states[:, 0], states[:, 1], '-', color=color, 
                    alpha=0.7, linewidth=1.5, label=f'IC: y₀={ic[1]}')
            
            # Mark if it goes outside bounds
            outside = np.any(np.abs(states[:, 1]) > 1.0)
            if outside:
                max_y = np.max(np.abs(states[:, 1]))
                ax1.text(0.1, ic[1], f'Max |y|={max_y:.2f}', color=color, fontsize=8)
    
    ax1.legend()
    
    # Plot 2: Y-evolution over time
    ax2.set_title("Y-coordinate Evolution (No Clipping)")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    
    for ic, color in zip(test_ics, colors):
        trajectory = system_obj.system.simulate(
            initial_state=np.array(ic),
            time_span=(0.0, 10.0),
            max_jumps=20
        )
        
        times = []
        y_vals = []
        
        for segment in trajectory.segments:
            if segment.t_end > segment.t_start:
                t_vals = np.linspace(segment.t_start, segment.t_end, 50)
                for t in t_vals:
                    try:
                        state = segment.scipy_solution.sol(t)
                        times.append(t)
                        y_vals.append(state[1])
                    except:
                        pass
        
        if times:
            ax2.plot(times, y_vals, '-', color=color, alpha=0.7, linewidth=1.5)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(-1.5, 1.5)
    
    # Plot 3: Phase portrait with many trajectories
    ax3.set_title("Phase Portrait (Many Trajectories)")
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-1.3, 1.3)
    ax3.grid(True, alpha=0.3)
    
    # Domain boundaries
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    
    # Grid of initial conditions
    y_vals = np.linspace(-0.95, 0.95, 15)
    for y0 in y_vals:
        try:
            trajectory = system_obj.system.simulate(
                initial_state=np.array([0.1, y0]),
                time_span=(0.0, 3.0),
                max_jumps=10
            )
            
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
                        # Color based on whether it goes outside bounds
                        if np.any(np.abs(states[:, 1]) > 1.0):
                            ax3.plot(states[:, 0], states[:, 1], 'r-', alpha=0.3, linewidth=0.8)
                        else:
                            ax3.plot(states[:, 0], states[:, 1], 'b-', alpha=0.3, linewidth=0.8)
        except:
            pass
    
    # Plot 4: Analysis of reset map near boundaries
    ax4.set_title("Reset Map Behavior")
    ax4.set_xlabel('y (before reset)')
    ax4.set_ylabel('y (after reset)')
    ax4.grid(True, alpha=0.3)
    
    y = np.linspace(-1.2, 1.2, 500)
    y_new = -y**3 + 2*y
    
    ax4.plot(y, y_new, 'b-', linewidth=2, label='y_new = -y³ + 2y')
    ax4.plot(y, y, 'k--', alpha=0.5, label='Identity')
    
    # Mark domain bounds
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=-1, color='red', linestyle='--', alpha=0.5)
    
    # Highlight regions that map outside
    outside_mask = np.abs(y_new) > 1
    ax4.fill_between(y[outside_mask], -1.5, 1.5, alpha=0.1, color='red', 
                     label='Maps outside [-1,1]')
    
    # Show actual trajectory points from jump states
    for ic, color in zip(test_ics[:2], colors[:2]):
        trajectory = system_obj.system.simulate(
            initial_state=np.array(ic),
            time_span=(0.0, 5.0),
            max_jumps=10
        )
        
        for state_minus, state_plus in trajectory.jump_states:
            ax4.plot(state_minus[1], state_plus[1], 'o', color=color, markersize=5, alpha=0.7)
    
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.5, 1.5)
    ax4.legend()
    
    plt.suptitle("Modified Behavior: Simulation Continues Outside Domain Bounds", fontsize=14)
    plt.tight_layout()
    plt.savefig('test_results/modified_behavior_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved to: test_results/modified_behavior_analysis.png")
    
    print("\n=== Analysis of Modified Behavior ===")
    print("\n1. Trajectories now continue outside the domain bounds")
    print("2. The cubic reset map creates oscillatory behavior around y = ±1")
    print("3. This should eliminate the artificial 'trapping' at boundaries")
    print("4. Morse sets should now better reflect the true dynamics")
    
    # Let's compute a small box map to see if Morse sets changed
    print("\n=== Quick Morse Set Check ===")
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=[20, 20])
    
    print("Computing box map with modified behavior...")
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=system_obj.system,
        tau=1.42,
    )
    
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    
    print(f"Number of Morse sets: {len(morse_sets)}")
    
    # Check for vertical structures at boundaries
    right_boundary_x = grid.subdivisions[0] - 1
    for i, morse_set in enumerate(morse_sets):
        boundary_boxes = []
        for box_idx in morse_set:
            multi_idx = np.unravel_index(box_idx, grid.subdivisions)
            if multi_idx[0] == right_boundary_x:
                boundary_boxes.append(multi_idx[1])
        
        if boundary_boxes:
            print(f"Morse set {i}: Contains {len(boundary_boxes)} boxes on right boundary")
            if len(boundary_boxes) > 3:
                print(f"  Still shows vertical structure (y-indices: {min(boundary_boxes)}-{max(boundary_boxes)})")


if __name__ == "__main__":
    test_modified_behavior()