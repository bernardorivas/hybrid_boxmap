#!/usr/bin/env python3
"""
Test Jump Time Penalty Feature with Rimless Wheel

This demo compares the box map computation with and without the jump time penalty
to verify the implementation works correctly.
"""
from pathlib import Path
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import networkx as nx

from hybrid_dynamics import (
    Grid, 
    HybridBoxMap, 
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid_fast,
    plot_morse_sets_with_roa_fast,
    compute_roa, 
    analyze_roa_coverage
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# Factory function for parallel processing
def create_rimless_wheel_system(alpha=0.4, gamma=0.2, max_jumps=50):
    """Factory function to create RimlessWheel system for parallel processing."""
    wheel = RimlessWheel(alpha=alpha, gamma=gamma, max_jumps=max_jumps)
    return wheel.system


def plot_morse_sets_on_axis(ax, grid, morse_sets, title="Morse Sets"):
    """Plot morse sets on a given axis."""
    # Get colors
    colors = plt.colormaps.get_cmap("turbo").resampled(len(morse_sets)) if morse_sets else []
    
    # Get grid properties
    x_dim, y_dim = (0, 1)
    box_width_x = grid.box_widths[x_dim]
    box_width_y = grid.box_widths[y_dim]
    
    # Set up the plot
    domain_bounds = grid.bounds
    x_min, x_max = domain_bounds[x_dim]
    y_min, y_max = domain_bounds[y_dim]
    margin = 0.05
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    
    # Plot each Morse set
    for i, component in enumerate(morse_sets):
        if not component:
            continue
            
        color = colors(i)
        rectangles = []
        
        for box_idx in component:
            center = grid.get_sample_points(box_idx, mode="center")[0]
            rect = patches.Rectangle(
                (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                box_width_x, box_width_y
            )
            rectangles.append(rect)
        
        pc = PatchCollection(rectangles, facecolor=color, edgecolor="none", alpha=0.7)
        ax.add_collection(pc)
        ax.plot([], [], "s", color=color, label=f"M{i}", markersize=10)
    
    ax.set_xlabel("Angle θ (rad)")
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.set_title(title)
    if len(morse_sets) <= 10:
        ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_trajectories_on_axis(ax, wheel, initial_conditions, tau, jump_time_penalty, title="Trajectories"):
    """Plot trajectories on a given axis."""
    ax.set_title(title)
    ax.set_xlabel("Angle θ (rad)")
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.grid(True, alpha=0.3)
    
    for ic in initial_conditions:
        traj = wheel.system.simulate(ic, (0, tau), jump_time_penalty=jump_time_penalty, max_jumps=10)
        if traj.segments:
            states = traj.get_all_states()
            if states.size > 0:
                ax.plot(states[:, 0], states[:, 1], '-', linewidth=2, alpha=0.7)
                ax.plot(ic[0], ic[1], 'o', markersize=8)
    
    # Set consistent axis limits
    ax.set_xlim(wheel.domain_bounds[0])
    ax.set_ylim(wheel.domain_bounds[1])


def plot_roa_on_axis(ax, grid, morse_sets, roa_dict, title="Regions of Attraction"):
    """Plot regions of attraction on a given axis."""
    # Get colors
    colors = plt.colormaps.get_cmap("turbo").resampled(len(morse_sets)) if morse_sets else []
    
    # Get grid properties
    x_dim, y_dim = (0, 1)
    box_width_x = grid.box_widths[x_dim]
    box_width_y = grid.box_widths[y_dim]
    
    # Set up the plot
    domain_bounds = grid.bounds
    x_min, x_max = domain_bounds[x_dim]
    y_min, y_max = domain_bounds[y_dim]
    margin = 0.05
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    
    # First pass: Plot ROA boxes with dimmer colors
    for morse_idx, roa_boxes in roa_dict.items():
        if morse_idx >= len(morse_sets):
            continue
            
        color = colors(morse_idx)
        morse_set = morse_sets[morse_idx]
        
        # Plot ROA boxes (excluding the Morse set itself)
        roa_only_boxes = roa_boxes - morse_set
        
        if roa_only_boxes:
            # Create rectangles for ROA boxes
            roa_rectangles = []
            for box_idx in roa_only_boxes:
                center = grid.get_sample_points(box_idx, mode="center")[0]
                rect = patches.Rectangle(
                    (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                    box_width_x, box_width_y
                )
                roa_rectangles.append(rect)
            
            # Create PatchCollection for ROA boxes
            pc = PatchCollection(roa_rectangles, facecolor=color, edgecolor="none", alpha=0.15)
            ax.add_collection(pc)
    
    # Second pass: Plot Morse sets with vibrant colors on top
    for morse_idx, morse_set in enumerate(morse_sets):
        if not morse_set:
            continue
            
        color = colors(morse_idx)
        
        # Create rectangles for Morse set boxes
        morse_rectangles = []
        for box_idx in morse_set:
            center = grid.get_sample_points(box_idx, mode="center")[0]
            rect = patches.Rectangle(
                (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                box_width_x, box_width_y
            )
            morse_rectangles.append(rect)
        
        # Create PatchCollection for Morse set
        pc = PatchCollection(morse_rectangles, facecolor=color, edgecolor="none", alpha=0.7)
        ax.add_collection(pc)
        
        # Add label for legend
        ax.plot([], [], "s", color=color, label=f"M{morse_idx} + ROA", markersize=10)
    
    ax.set_xlabel("Angle θ (rad)")
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.set_title(title)
    if len(morse_sets) <= 10:
        ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)


def create_combined_comparison_plot(wheel, grid, morse_sets_no_penalty, morse_sets_with_penalty, 
                                  roa_dict_no_penalty, roa_dict_with_penalty,
                                  initial_conditions, tau, output_path):
    """Create a 2x3 combined comparison plot."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Jump Time Penalty Comparison (τ={tau})", fontsize=16, y=0.98)
    
    # Add row labels
    fig.text(0.02, 0.75, "No Penalty", fontsize=14, weight='bold', rotation=90, va='center')
    fig.text(0.02, 0.25, "With Penalty", fontsize=14, weight='bold', rotation=90, va='center')
    
    # Top row - No penalty
    plot_morse_sets_on_axis(axes[0, 0], grid, morse_sets_no_penalty, 
                           "Morse Sets")
    plot_trajectories_on_axis(axes[0, 1], wheel, initial_conditions, tau, False, 
                             "Trajectories")
    plot_roa_on_axis(axes[0, 2], grid, morse_sets_no_penalty, roa_dict_no_penalty,
                     "Regions of Attraction")
    
    # Bottom row - With penalty
    plot_morse_sets_on_axis(axes[1, 0], grid, morse_sets_with_penalty, 
                           "Morse Sets") 
    plot_trajectories_on_axis(axes[1, 1], wheel, initial_conditions, tau, True, 
                             "Trajectories")
    plot_roa_on_axis(axes[1, 2], grid, morse_sets_with_penalty, roa_dict_with_penalty,
                     "Regions of Attraction")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, left=0.06)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def compare_jump_penalty_behavior():
    """Compare box maps with and without jump time penalty."""
    
    # Suppress warnings
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[51, 51])
    tau = 2.0
    bloat_factor = 0.12
    
    # Test with specific initial conditions
    test_points = [
        np.array([0.0, 0.4]),   # Will likely jump
        np.array([0.2, 0.3]),   # Might jump once
        np.array([-0.1, 0.1]),  # May not jump
    ]
    
    print("=" * 80)
    print("TESTING JUMP TIME PENALTY FEATURE")
    print("=" * 80)
    
    # Test trajectory simulations first
    print("\n1. Testing individual trajectories:")
    print("-" * 40)
    
    for i, initial_state in enumerate(test_points):
        print(f"\nTest point {i+1}: {initial_state}")
        
        # Simulate without penalty
        traj_no_penalty = wheel.system.simulate(
            initial_state, (0, tau), jump_time_penalty=False
        )
        
        # Simulate with penalty
        traj_with_penalty = wheel.system.simulate(
            initial_state, (0, tau), jump_time_penalty=True
        )
        
        print(f"  Without penalty: {traj_no_penalty.num_jumps} jumps, "
              f"duration = {traj_no_penalty.total_duration:.3f}")
        print(f"  With penalty:    {traj_with_penalty.num_jumps} jumps, "
              f"duration = {traj_with_penalty.total_duration:.3f}")
        
        if traj_no_penalty.final_state is not None and traj_with_penalty.final_state is not None:
            state_diff = np.linalg.norm(traj_no_penalty.final_state - traj_with_penalty.final_state)
            print(f"  Final state difference: {state_diff:.6f}")
    
    # Now test box maps
    print("\n\n2. Computing box maps:")
    print("-" * 40)
    
    # Setup base directory for figures
    figures_base_dir = Path("figures/rimless_wheel_jump_penalty")
    figures_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run directory with descriptive name
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist(), bloat_factor)
    
    # Create subdirectories for each penalty mode
    no_penalty_dir = run_dir / "no_penalty"
    with_penalty_dir = run_dir / "with_penalty"
    no_penalty_dir.mkdir(exist_ok=True)
    with_penalty_dir.mkdir(exist_ok=True)
    
    print(f"Created run directory: {run_dir.name}")
    
    progress_callback = create_progress_callback(update_interval=1000)
    
    # Compute box map WITHOUT penalty
    print("\nComputing box map WITHOUT jump penalty...")
    start_time = time.time()
    box_map_no_penalty = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        discard_out_of_bounds_destinations=True,
        progress_callback=progress_callback,
        parallel=False,  # Use sequential for clearer testing
        jump_time_penalty=False
    )
    time_no_penalty = time.time() - start_time
    print(f"Completed in {time_no_penalty:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_no_penalty.values())}")
    
    # Compute box map WITH penalty
    print("\nComputing box map WITH jump penalty...")
    start_time = time.time()
    box_map_with_penalty = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        discard_out_of_bounds_destinations=True,
        progress_callback=progress_callback,
        parallel=False,  # Use sequential for clearer testing
        jump_time_penalty=True
    )
    time_with_penalty = time.time() - start_time
    print(f"Completed in {time_with_penalty:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_with_penalty.values())}")
    
    # Analyze differences
    print("\n\n3. Analyzing differences:")
    print("-" * 40)
    
    # Count boxes with different mappings
    boxes_with_different_maps = 0
    boxes_only_in_no_penalty = 0
    boxes_only_in_with_penalty = 0
    
    all_source_boxes = set(box_map_no_penalty.keys()) | set(box_map_with_penalty.keys())
    
    for box_idx in all_source_boxes:
        dests_no_penalty = box_map_no_penalty.get(box_idx, set())
        dests_with_penalty = box_map_with_penalty.get(box_idx, set())
        
        if dests_no_penalty != dests_with_penalty:
            boxes_with_different_maps += 1
            
            if box_idx not in box_map_with_penalty:
                boxes_only_in_no_penalty += 1
            elif box_idx not in box_map_no_penalty:
                boxes_only_in_with_penalty += 1
    
    print(f"Boxes with different mappings: {boxes_with_different_maps}")
    print(f"Boxes only in no-penalty map: {boxes_only_in_no_penalty}")
    print(f"Boxes only in with-penalty map: {boxes_only_in_with_penalty}")
    
    # Compute and compare Morse graphs
    print("\n\n4. Computing Morse graphs:")
    print("-" * 40)
    
    graph_no_penalty = box_map_no_penalty.to_networkx()
    morse_graph_no_penalty, morse_sets_no_penalty = create_morse_graph(graph_no_penalty)
    
    graph_with_penalty = box_map_with_penalty.to_networkx()
    morse_graph_with_penalty, morse_sets_with_penalty = create_morse_graph(graph_with_penalty)
    
    print(f"No penalty: {len(morse_sets_no_penalty)} Morse sets")
    print(f"With penalty: {len(morse_sets_with_penalty)} Morse sets")
    
    # Compute regions of attraction
    print("\n\n5. Computing regions of attraction:")
    print("-" * 40)
    
    roa_dict_no_penalty = compute_roa(graph_no_penalty, morse_sets_no_penalty)
    roa_dict_with_penalty = compute_roa(graph_with_penalty, morse_sets_with_penalty)
    
    # Analyze ROA coverage
    total_boxes = len(grid)
    stats_no_penalty, _ = analyze_roa_coverage(roa_dict_no_penalty, total_boxes)
    stats_with_penalty, _ = analyze_roa_coverage(roa_dict_with_penalty, total_boxes)
    
    print(f"No penalty ROA coverage: {stats_no_penalty['coverage_percentage']:.1f}%")
    print(f"With penalty ROA coverage: {stats_with_penalty['coverage_percentage']:.1f}%")
    
    # Generate visualizations
    print("\n\n6. Generating visualizations:")
    print("-" * 40)
    
    # Create combined comparison plot
    print("Generating combined comparison plot...")
    initial_conditions_for_plot = [
        [0.0, 0.4], [0.1, 0.3], [-0.1, 0.2]
    ]
    create_combined_comparison_plot(
        wheel, grid,
        morse_sets_no_penalty,
        morse_sets_with_penalty,
        roa_dict_no_penalty,
        roa_dict_with_penalty,
        initial_conditions_for_plot,
        tau,
        run_dir / "combined_comparison.png"
    )
    print("Generated combined_comparison.png")
    
    # Also generate individual plots for detailed viewing
    if morse_sets_no_penalty:
        plot_morse_sets_on_grid_fast(
            grid, morse_sets_no_penalty, 
            str(no_penalty_dir / "morse_sets.png"),
            xlabel="Angle θ (rad)", ylabel="Angular Velocity ω (rad/s)"
        )
        print(f"Generated {no_penalty_dir.name}/morse_sets.png")
    
    if morse_sets_with_penalty:
        plot_morse_sets_on_grid_fast(
            grid, morse_sets_with_penalty, 
            str(with_penalty_dir / "morse_sets.png"),
            xlabel="Angle θ (rad)", ylabel="Angular Velocity ω (rad/s)"
        )
        print(f"Generated {with_penalty_dir.name}/morse_sets.png")
    
    # Generate morse graph visualizations if pygraphviz is available
    try:
        if morse_graph_no_penalty.number_of_nodes() > 0:
            plot_morse_graph_viz(morse_graph_no_penalty, morse_sets_no_penalty, 
                               str(no_penalty_dir / "morse_graph.png"))
            print(f"Generated {no_penalty_dir.name}/morse_graph.png")
        if morse_graph_with_penalty.number_of_nodes() > 0:
            plot_morse_graph_viz(morse_graph_with_penalty, morse_sets_with_penalty,
                               str(with_penalty_dir / "morse_graph.png"))
            print(f"Generated {with_penalty_dir.name}/morse_graph.png")
    except Exception as e:
        print(f"Could not generate morse graph visualizations: {e}")
    
    # Create comparison plot showing a few trajectories
    print("\nGenerating trajectory comparison plot...")
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Initial conditions for visualization
    initial_conditions = [
        [0.0, 0.4], [0.1, 0.3], [-0.1, 0.2]
    ]
    
    # Plot trajectories without penalty
    ax1.set_title("Trajectories - No Jump Penalty")
    ax1.set_xlabel("Angle θ (rad)")
    ax1.set_ylabel("Angular Velocity ω (rad/s)")
    ax1.grid(True, alpha=0.3)
    
    for ic in initial_conditions:
        traj = wheel.system.simulate(ic, (0, tau), jump_time_penalty=False, max_jumps=10)
        if traj.segments:
            states = traj.get_all_states()
            if states.size > 0:
                ax1.plot(states[:, 0], states[:, 1], '-', linewidth=2, alpha=0.7)
                ax1.plot(ic[0], ic[1], 'o', markersize=8)
    
    # Plot trajectories with penalty
    ax2.set_title("Trajectories - With Jump Penalty")
    ax2.set_xlabel("Angle θ (rad)")
    ax2.set_ylabel("Angular Velocity ω (rad/s)")
    ax2.grid(True, alpha=0.3)
    
    for ic in initial_conditions:
        traj = wheel.system.simulate(ic, (0, tau), jump_time_penalty=True, max_jumps=10)
        if traj.segments:
            states = traj.get_all_states()
            if states.size > 0:
                ax2.plot(states[:, 0], states[:, 1], '-', linewidth=2, alpha=0.7)
                ax2.plot(ic[0], ic[1], 'o', markersize=8)
    
    plt.tight_layout()
    plt.savefig(run_dir / "trajectory_comparison.png", dpi=150)
    plt.close()
    print("Generated trajectory_comparison.png")
    
    # Save summary info to run directory
    summary_file = run_dir / "run_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Jump Time Penalty Test Results\n")
        f.write(f"==============================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  - tau: {tau}\n")
        f.write(f"  - grid subdivisions: {grid.subdivisions.tolist()}\n")
        f.write(f"  - bloat factor: {bloat_factor}\n")
        f.write(f"  - alpha: {wheel.alpha}\n")
        f.write(f"  - gamma: {wheel.gamma}\n")
        f.write(f"\nResults:\n")
        f.write(f"  - No penalty: {len(morse_sets_no_penalty)} Morse sets, {sum(len(dests) for dests in box_map_no_penalty.values())} transitions\n")
        f.write(f"  - With penalty: {len(morse_sets_with_penalty)} Morse sets, {sum(len(dests) for dests in box_map_with_penalty.values())} transitions\n")
        f.write(f"  - Boxes with different mappings: {boxes_with_different_maps}\n")
        f.write(f"\nRegions of Attraction:\n")
        f.write(f"  - No penalty ROA coverage: {stats_no_penalty['coverage_percentage']:.1f}%\n")
        f.write(f"  - With penalty ROA coverage: {stats_with_penalty['coverage_percentage']:.1f}%\n")
    print("Generated run_summary.txt")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("Results saved to:", run_dir)
    print("  - Combined comparison: combined_comparison.png")
    print("  - No penalty results:", no_penalty_dir)
    print("  - With penalty results:", with_penalty_dir)
    print("=" * 80)


if __name__ == "__main__":
    compare_jump_penalty_behavior()