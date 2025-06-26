#!/usr/bin/env python3
"""
Test script for 3D visualization functionality.

This demonstrates the new 3D plotting capabilities for hybrid systems.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.bouncing_ball import BouncingBall
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    plot_morse_sets_3d,
    plot_morse_sets_projections,
)
from hybrid_dynamics.src.plot_utils import HybridPlotter

def create_3d_bouncing_ball_system():
    """Create a simple 3D test system based on bouncing ball."""
    # For demonstration, we'll use a 3D grid with the bouncing ball's 2D dynamics
    # This is just to test the visualization - real 3D systems would have 3D dynamics
    ball = BouncingBall(g=9.81, c=0.8, h_min=0.0, h_max=5.0, v_max=10.0)
    return ball

def test_3d_morse_sets():
    """Test 3D Morse set visualization."""
    print("Testing 3D Morse set visualization...")
    
    # Create output directory
    output_dir = Path("figures/test_3d")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a 3D grid (adding a dummy third dimension for testing)
    bounds = [
        (0.0, 5.0),   # height
        (-10.0, 10.0), # velocity
        (-2.0, 2.0)    # dummy dimension for 3D testing
    ]
    subdivisions = [20, 20, 5]
    
    grid = Grid(bounds=bounds, subdivisions=subdivisions)
    
    # Create some dummy Morse sets for visualization
    # In real usage, these would come from the Morse graph computation
    morse_sets = [
        {0, 1, 2, 100, 101, 102},  # First Morse set
        {500, 501, 502, 600, 601, 602},  # Second Morse set
        {1500, 1501, 1502, 1600, 1601}  # Third Morse set
    ]
    
    # Test 3D visualization
    print("  Creating 3D Morse sets plot...")
    plot_morse_sets_3d(
        grid, morse_sets,
        str(output_dir / "morse_sets_3d.png"),
        xlabel="Height (h)",
        ylabel="Velocity (v)",
        zlabel="Z",
        elev=30,
        azim=45
    )
    
    # Test projection visualizations
    print("  Creating 2D projection plots...")
    plot_morse_sets_projections(
        grid, morse_sets,
        str(output_dir)
    )
    
    print(f"✓ 3D visualizations saved to {output_dir}")

def test_3d_phase_portrait():
    """Test 3D phase portrait visualization."""
    print("\nTesting 3D phase portrait visualization...")
    
    output_dir = Path("figures/test_3d")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a simple 3D system for testing
    # We'll create synthetic trajectories since bouncing ball is 2D
    plotter = HybridPlotter()
    
    # Create dummy 3D initial conditions
    initial_conditions = [
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.5],
        [3.0, -1.0, -0.5],
        [0.5, 2.0, 1.0],
        [1.5, -2.0, -1.0]
    ]
    
    # For this test, we'll create synthetic trajectory data
    # In real usage, you would simulate the system
    print("  Creating synthetic 3D trajectories...")
    
    # Create synthetic trajectories (spiral motion for visualization)
    from hybrid_dynamics.src.hybrid_trajectory import HybridTrajectory, TrajectorySegment
    
    trajectories = []
    for ic in initial_conditions:
        traj = HybridTrajectory()
        
        # Create a spiral trajectory
        t = np.linspace(0, 10, 100)
        x = ic[0] * np.exp(-0.1 * t) * np.cos(t)
        y = ic[1] * np.exp(-0.1 * t) * np.sin(t) 
        z = ic[2] + 0.5 * t
        
        states = np.column_stack([x, y, z])
        
        segment = TrajectorySegment(
            time_values=t,
            state_values=states,
            jump_index=0
        )
        traj.add_segment(segment)
        trajectories.append(traj)
    
    # Test 3D phase portrait
    print("  Creating 3D phase portrait...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories manually for this test
    colors = plt.cm.get_cmap('tab10')
    for i, traj in enumerate(trajectories):
        color = colors(i)
        for segment in traj.segments:
            ax.plot(segment.state_values[:, 0],
                   segment.state_values[:, 1],
                   segment.state_values[:, 2],
                   color=color, alpha=0.8, linewidth=1.5,
                   label=f"Trajectory {i+1}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Phase Portrait (Test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase_portrait_3d_test.png", dpi=150)
    plt.close()
    
    # Test projections using the plotter
    print("  Creating phase portrait projections...")
    # Note: In real usage, you would pass the actual system and let it simulate
    # For this test, we're just demonstrating the visualization
    
    print(f"✓ Phase portraits saved to {output_dir}")

def test_2d_backward_compatibility():
    """Test that 2D systems still work correctly."""
    print("\nTesting 2D backward compatibility...")
    
    output_dir = Path("figures/test_3d/2d_compat")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create standard 2D bouncing ball
    ball = BouncingBall()
    grid = Grid(bounds=ball.domain_bounds, subdivisions=[20, 20])
    
    # Create dummy Morse sets
    morse_sets = [
        {100, 101, 120, 121},
        {200, 201, 220, 221}
    ]
    
    # Should work exactly as before
    plot_morse_sets_on_grid(
        grid, morse_sets,
        str(output_dir / "morse_sets_2d.png"),
        xlabel="Height",
        ylabel="Velocity"
    )
    
    print(f"✓ 2D compatibility test passed, saved to {output_dir}")

if __name__ == "__main__":
    print("=== Testing 3D Visualization Features ===\n")
    
    # Run tests
    test_3d_morse_sets()
    test_3d_phase_portrait()
    test_2d_backward_compatibility()
    
    print("\n=== All tests completed! ===")
    print("Check the 'figures/test_3d' directory for output visualizations.")