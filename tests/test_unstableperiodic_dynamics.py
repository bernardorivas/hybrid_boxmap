#!/usr/bin/env python3
"""
Analyze the dynamics of the unstable periodic system to understand
the behavior at the right boundary.
"""
import numpy as np
import matplotlib.pyplot as plt

from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src.plot_utils import HybridPlotter


def analyze_system_dynamics():
    """Analyze the ODE and jump dynamics of the unstable periodic system."""
    
    # Create system
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    
    print("=== Unstable Periodic System Analysis ===\n")
    
    # Test the ODE at various points
    print("1. ODE behavior (dx/dt, dy/dt):")
    test_points = [
        (0.1, 0.0, "Left side"),
        (0.5, 0.0, "Center"),
        (0.9, 0.0, "Near right boundary"),
        (0.99, 0.0, "Very near right boundary"),
        (1.0, 0.0, "At right boundary"),
        (0.5, 0.5, "Upper region"),
        (0.5, -0.5, "Lower region")
    ]
    
    for x, y, label in test_points:
        state = np.array([x, y])
        derivatives = system_obj.system.ode(0.0, state)
        print(f"  At ({x:.2f}, {y:.2f}) [{label}]: dx/dt = {derivatives[0]:.3f}, dy/dt = {derivatives[1]:.3f}")
    
    # Test event function
    print("\n2. Event function (guard condition):")
    for x, y, label in test_points:
        state = np.array([x, y])
        event_val = system_obj.system.event_function(0.0, state)
        print(f"  At ({x:.2f}, {y:.2f}) [{label}]: g(x,y) = {event_val:.6f}")
    
    # Test reset map
    print("\n3. Reset map behavior:")
    # Find some states where event is approximately zero
    x_vals = np.linspace(0.0, 1.0, 100)
    for x in x_vals:
        for y in [-0.5, 0.0, 0.5]:
            state = np.array([x, y])
            event_val = system_obj.system.event_function(0.0, state)
            if abs(event_val) < 0.01:  # Near zero
                reset_state = system_obj.system.reset_map(state)
                print(f"  From ({x:.3f}, {y:.3f}): reset to ({reset_state[0]:.3f}, {reset_state[1]:.3f})")
    
    # Create phase portrait with vector field
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vector field plot
    ax1.set_title("Vector Field of ODE")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-1, 1)
    
    # Create grid for vector field
    x_grid = np.linspace(0.05, 0.95, 20)
    y_grid = np.linspace(-0.95, 0.95, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute vector field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.array([X[i,j], Y[i,j]])
            derivatives = system_obj.system.ode(0.0, state)
            U[i,j] = derivatives[0]
            V[i,j] = derivatives[1]
    
    # Normalize for better visualization
    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1  # Avoid division by zero
    scale = 0.05
    U_norm = scale * U / M
    V_norm = scale * V / M
    
    ax1.quiver(X, Y, U_norm, V_norm, M, cmap='viridis', alpha=0.6)
    
    # Plot event function zero level set
    x_fine = np.linspace(0, 1, 200)
    y_fine = np.linspace(-1, 1, 200)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z = np.zeros_like(X_fine)
    
    for i in range(X_fine.shape[0]):
        for j in range(X_fine.shape[1]):
            state = np.array([X_fine[i,j], Y_fine[i,j]])
            Z[i,j] = system_obj.system.event_function(0.0, state)
    
    # Draw zero contour (guard set)
    contour = ax1.contour(X_fine, Y_fine, Z, levels=[0], colors='red', linewidths=2)
    ax1.clabel(contour, inline=True, fontsize=10, fmt='g=0')
    
    # Highlight boundaries
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Left boundary')
    ax1.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Right boundary')
    ax1.legend()
    
    # Phase portrait with sample trajectories
    ax2.set_title("Sample Trajectories from Right Boundary")
    
    # Test initial conditions at right boundary
    y_values = np.linspace(-0.8, 0.8, 9)
    initial_conditions = [[0.99, y] for y in y_values]
    
    plotter = HybridPlotter()
    # Create a temporary figure for the plotter
    temp_fig, temp_ax = plt.subplots()
    
    for ic in initial_conditions:
        try:
            trajectory = system_obj.system.simulate(
                initial_state=np.array(ic),
                time_span=(0.0, 2.0),
                max_jumps=5
            )
            
            # Plot continuous segments
            for segment in trajectory.segments:
                t_vals = np.linspace(segment.t_start, segment.t_end, 100)
                states = np.array([segment.solution(t) for t in t_vals])
                ax2.plot(states[:, 0], states[:, 1], 'b-', alpha=0.6, linewidth=1)
            
            # Mark jumps
            for jump in trajectory.jumps:
                ax2.plot([jump['state_minus'][0], jump['state_plus'][0]], 
                        [jump['state_minus'][1], jump['state_plus'][1]], 
                        'r--', alpha=0.8, linewidth=1)
                ax2.plot(jump['state_minus'][0], jump['state_minus'][1], 'ro', markersize=4)
                ax2.plot(jump['state_plus'][0], jump['state_plus'][1], 'go', markersize=4)
                
        except Exception as e:
            print(f"Warning: Failed to simulate from ({ic[0]}, {ic[1]}): {e}")
    
    plt.close(temp_fig)  # Close the temporary figure
    
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add boundary lines
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    
    # Draw guard set on phase portrait
    contour2 = ax2.contour(X_fine, Y_fine, Z, levels=[0], colors='red', linewidths=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("test_results/unstableperiodic_dynamics_analysis.png", dpi=150)
    print("\nVisualization saved to: test_results/unstableperiodic_dynamics_analysis.png")
    
    # Analyze behavior near right boundary
    print("\n4. Special behavior at right boundary:")
    print("   Testing what happens when starting exactly at x=1.0...")
    
    for y in [-0.5, 0.0, 0.5]:
        state = np.array([1.0, y])
        ode_val = system_obj.system.ode(0.0, state)
        event_val = system_obj.system.event_function(0.0, state)
        
        print(f"\n   At (1.0, {y}):")
        print(f"     ODE: dx/dt = {ode_val[0]:.3f}, dy/dt = {ode_val[1]:.3f}")
        print(f"     Event: g = {event_val:.6f}")
        
        if abs(event_val) < 1e-6:
            reset_state = system_obj.system.reset_map(state)
            print(f"     Reset: jumps to ({reset_state[0]:.3f}, {reset_state[1]:.3f})")


if __name__ == "__main__":
    analyze_system_dynamics()