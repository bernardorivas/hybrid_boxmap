#!/usr/bin/env python3
"""
Analyze the actual reset map behavior: y_new = -y³ + 2y
"""
import numpy as np
import matplotlib.pyplot as plt


def reset_map_y(y):
    """The y-component of the reset map: y_new = -y³ + 2y"""
    return -y**3 + 2*y


def analyze_reset_map():
    """Analyze the reset map behavior."""
    
    print("=== Reset Map Analysis: y_new = -y³ + 2y ===\n")
    
    # Find fixed points (where y_new = y)
    # -y³ + 2y = y
    # -y³ + y = 0
    # y(-y² + 1) = 0
    # y(1 - y²) = 0
    # y = 0, y = ±1
    
    print("Fixed points of the reset map:")
    print("  y = 0: y_new = 0 (fixed)")
    print("  y = 1: y_new = -1 + 2 = 1 (fixed)")
    print("  y = -1: y_new = -(-1) + 2(-1) = 1 - 2 = -1 (fixed)")
    
    # Find critical points (where dy_new/dy = 0)
    # dy_new/dy = -3y² + 2 = 0
    # y² = 2/3
    # y = ±√(2/3) ≈ ±0.816
    
    y_crit = np.sqrt(2/3)
    print(f"\nCritical points:")
    print(f"  y = ±{y_crit:.3f}")
    print(f"  At y = {y_crit:.3f}: y_new = {reset_map_y(y_crit):.3f}")
    print(f"  At y = -{y_crit:.3f}: y_new = {reset_map_y(-y_crit):.3f}")
    
    # Test specific values
    print("\nReset map at key values:")
    test_values = [-0.99, -0.9, -0.816, -0.5, 0, 0.5, 0.816, 0.9, 0.99]
    for y in test_values:
        y_new = reset_map_y(y)
        print(f"  y = {y:6.3f} → y_new = {y_new:6.3f}")
    
    # Check if any values map outside [-1, 1]
    print("\nChecking for values that map outside domain [-1, 1]:")
    y_vals = np.linspace(-1, 1, 1000)
    y_new_vals = reset_map_y(y_vals)
    
    outside_domain = np.abs(y_new_vals) > 1
    if np.any(outside_domain):
        y_outside = y_vals[outside_domain]
        y_new_outside = y_new_vals[outside_domain]
        print(f"  Found {len(y_outside)} values that map outside domain")
        # Show some examples
        for i in range(min(5, len(y_outside))):
            print(f"    y = {y_outside[i]:.3f} → y_new = {y_new_outside[i]:.3f}")
    else:
        print("  All values map within the domain!")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Reset map function
    ax1.set_title("Reset Map: y_new = -y³ + 2y")
    ax1.set_xlabel("y")
    ax1.set_ylabel("y_new")
    ax1.grid(True, alpha=0.3)
    
    y = np.linspace(-1.2, 1.2, 1000)
    y_new = reset_map_y(y)
    
    ax1.plot(y, y_new, 'b-', linewidth=2, label='Reset map')
    ax1.plot(y, y, 'k--', alpha=0.5, label='Identity (y_new = y)')
    
    # Mark fixed points
    fixed_points = [0, 1, -1]
    for fp in fixed_points:
        ax1.plot(fp, fp, 'ro', markersize=8)
    
    # Mark critical points
    for y_c in [y_crit, -y_crit]:
        ax1.plot(y_c, reset_map_y(y_c), 'go', markersize=8)
    
    # Domain bounds
    ax1.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    ax1.axhline(y=-1, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(x=1, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(x=-1, color='r', linestyle=':', alpha=0.5)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend()
    
    # Plot 2: Iteration behavior
    ax2.set_title("Iteration of Reset Map")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("y value")
    ax2.grid(True, alpha=0.3)
    
    # Test different initial conditions
    initial_conditions = [-0.99, -0.9, -0.5, 0.1, 0.5, 0.9, 0.99]
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))
    
    for y0, color in zip(initial_conditions, colors):
        y_history = [y0]
        y_current = y0
        
        for _ in range(20):
            y_current = reset_map_y(y_current)
            # Clip to domain bounds if necessary
            y_current = np.clip(y_current, -1, 1)
            y_history.append(y_current)
        
        ax2.plot(y_history, 'o-', color=color, alpha=0.7, label=f'y₀ = {y0:.2f}')
    
    ax2.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    ax2.axhline(y=-1, color='r', linestyle=':', alpha=0.5)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("test_results/reset_map_analysis.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: test_results/reset_map_analysis.png")
    
    # Analyze stability of fixed points
    print("\n=== Stability Analysis ===")
    print("Derivative of reset map: dy_new/dy = -3y² + 2")
    
    for fp in fixed_points:
        derivative = -3*fp**2 + 2
        print(f"\nAt fixed point y = {fp}:")
        print(f"  dy_new/dy = {derivative:.3f}")
        if abs(derivative) < 1:
            print("  → Stable (attracting)")
        elif abs(derivative) > 1:
            print("  → Unstable (repelling)")
        else:
            print("  → Marginally stable")


if __name__ == "__main__":
    analyze_reset_map()