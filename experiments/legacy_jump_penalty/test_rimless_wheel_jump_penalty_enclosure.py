#!/usr/bin/env python3
"""
Test Jump Time Penalty with Boundary-Bridged Enclosure

This demo compares four methods:
1. No jump time penalty (standard enclosure)
2. With jump time penalty (standard enclosure)
3. No jump time penalty with boundary-bridged enclosure
4. With jump time penalty AND boundary-bridged enclosure

Creates a 4x3 comparison plot showing the effects of both features.
"""
from pathlib import Path
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from hybrid_dynamics import (
    Grid, 
    HybridBoxMap, 
    create_morse_graph,
    plot_morse_graph_viz,
    compute_roa, 
    analyze_roa_coverage
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
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


def create_combined_comparison_plot(wheel, grid, results_dict, initial_conditions, tau, output_path):
    """Create a 4x3 combined comparison plot."""
    # Create figure with subplots
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle(f"Jump Penalty & Boundary-Bridged Enclosure Comparison (τ={tau})", fontsize=16, y=0.99)
    
    # Add row labels
    fig.text(0.02, 0.86, "No Penalty", fontsize=14, weight='bold', rotation=90, va='center')
    fig.text(0.02, 0.64, "With Penalty", fontsize=14, weight='bold', rotation=90, va='center')
    fig.text(0.02, 0.42, "No Penalty +\nBridged Enclosure", fontsize=14, weight='bold', rotation=90, va='center')
    fig.text(0.02, 0.20, "With Penalty +\nBridged Enclosure", fontsize=14, weight='bold', rotation=90, va='center')
    
    # Row 1 - No penalty
    plot_morse_sets_on_axis(axes[0, 0], grid, results_dict['no_penalty']['morse_sets'], 
                           "Morse Sets")
    plot_trajectories_on_axis(axes[0, 1], wheel, initial_conditions, tau, False, 
                             "Trajectories")
    plot_roa_on_axis(axes[0, 2], grid, results_dict['no_penalty']['morse_sets'], 
                     results_dict['no_penalty']['roa_dict'], "Regions of Attraction")
    
    # Row 2 - With penalty
    plot_morse_sets_on_axis(axes[1, 0], grid, results_dict['with_penalty']['morse_sets'], 
                           "Morse Sets") 
    plot_trajectories_on_axis(axes[1, 1], wheel, initial_conditions, tau, True, 
                             "Trajectories")
    plot_roa_on_axis(axes[1, 2], grid, results_dict['with_penalty']['morse_sets'], 
                     results_dict['with_penalty']['roa_dict'], "Regions of Attraction")
    
    # Row 3 - No penalty with bridged enclosure
    plot_morse_sets_on_axis(axes[2, 0], grid, results_dict['bridged']['morse_sets'], 
                           "Morse Sets")
    plot_trajectories_on_axis(axes[2, 1], wheel, initial_conditions, tau, False, 
                             "Trajectories")  # Same trajectories as no penalty
    plot_roa_on_axis(axes[2, 2], grid, results_dict['bridged']['morse_sets'], 
                     results_dict['bridged']['roa_dict'], "Regions of Attraction")
    
    # Row 4 - With penalty AND bridged enclosure
    plot_morse_sets_on_axis(axes[3, 0], grid, results_dict['penalty_bridged']['morse_sets'], 
                           "Morse Sets")
    plot_trajectories_on_axis(axes[3, 1], wheel, initial_conditions, tau, True, 
                             "Trajectories")  # Same trajectories as with penalty
    plot_roa_on_axis(axes[3, 2], grid, results_dict['penalty_bridged']['morse_sets'], 
                     results_dict['penalty_bridged']['roa_dict'], "Regions of Attraction")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, left=0.08)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def compare_methods():
    """Compare all three methods: no penalty, with penalty, and bridged enclosure."""
    
    # Suppress warnings
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[101, 101])
    tau = 2.0
    bloat_factor = 0.12
    
    print("=" * 80)
    print("JUMP PENALTY & BOUNDARY-BRIDGED ENCLOSURE COMPARISON")
    print("=" * 80)
    
    # Setup base directory for figures
    figures_base_dir = Path("figures/rimless_wheel_jump_penalty_enclosure")
    figures_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run directory with descriptive name
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist(), bloat_factor)
    
    print(f"Created run directory: {run_dir.name}")
    print(f"Parameters: tau={tau}, grid={grid.subdivisions.tolist()}, bloat={bloat_factor}")
    
    progress_callback = create_progress_callback(update_interval=1000)
    
    # Store results
    results = {}
    
    # Method 1: No penalty, standard enclosure
    print("\n1. Computing box map WITHOUT jump penalty (standard enclosure)...")
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
        parallel=False,
        jump_time_penalty=False,
        use_boundary_bridging=False  # Standard enclosure
    )
    time_no_penalty = time.time() - start_time
    print(f"Completed in {time_no_penalty:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_no_penalty.values())}")
    
    # Compute Morse graph
    graph_no_penalty = box_map_no_penalty.to_networkx()
    morse_graph_no_penalty, morse_sets_no_penalty = create_morse_graph(graph_no_penalty)
    roa_dict_no_penalty = compute_roa(graph_no_penalty, morse_sets_no_penalty)
    
    results['no_penalty'] = {
        'box_map': box_map_no_penalty,
        'morse_sets': morse_sets_no_penalty,
        'roa_dict': roa_dict_no_penalty,
        'time': time_no_penalty,
        'transitions': sum(len(dests) for dests in box_map_no_penalty.values())
    }
    
    # Method 2: With penalty
    print("\n2. Computing box map WITH jump penalty...")
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
        parallel=False,
        jump_time_penalty=True,
        use_boundary_bridging=False  # Standard enclosure
    )
    time_with_penalty = time.time() - start_time
    print(f"Completed in {time_with_penalty:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_with_penalty.values())}")
    
    # Compute Morse graph
    graph_with_penalty = box_map_with_penalty.to_networkx()
    morse_graph_with_penalty, morse_sets_with_penalty = create_morse_graph(graph_with_penalty)
    roa_dict_with_penalty = compute_roa(graph_with_penalty, morse_sets_with_penalty)
    
    results['with_penalty'] = {
        'box_map': box_map_with_penalty,
        'morse_sets': morse_sets_with_penalty,
        'roa_dict': roa_dict_with_penalty,
        'time': time_with_penalty,
        'transitions': sum(len(dests) for dests in box_map_with_penalty.values())
    }
    
    # Method 3: No penalty with boundary-bridged enclosure
    print("\n3. Computing box map WITHOUT penalty but WITH boundary-bridged enclosure...")
    start_time = time.time()
    box_map_bridged = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        discard_out_of_bounds_destinations=True,
        progress_callback=progress_callback,
        parallel=False,
        jump_time_penalty=False,
        use_boundary_bridging=True  # Boundary-bridged enclosure
    )
    time_bridged = time.time() - start_time
    print(f"Completed in {time_bridged:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_bridged.values())}")
    
    # Compute Morse graph
    graph_bridged = box_map_bridged.to_networkx()
    morse_graph_bridged, morse_sets_bridged = create_morse_graph(graph_bridged)
    roa_dict_bridged = compute_roa(graph_bridged, morse_sets_bridged)
    
    results['bridged'] = {
        'box_map': box_map_bridged,
        'morse_sets': morse_sets_bridged,
        'roa_dict': roa_dict_bridged,
        'time': time_bridged,
        'transitions': sum(len(dests) for dests in box_map_bridged.values())
    }
    
    # Method 4: With penalty AND boundary-bridged enclosure
    print("\n4. Computing box map WITH penalty AND boundary-bridged enclosure...")
    start_time = time.time()
    box_map_penalty_bridged = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        discard_out_of_bounds_destinations=True,
        progress_callback=progress_callback,
        parallel=False,
        jump_time_penalty=True,
        use_boundary_bridging=True  # Both penalty and bridging
    )
    time_penalty_bridged = time.time() - start_time
    print(f"Completed in {time_penalty_bridged:.2f} seconds")
    print(f"Number of transitions: {sum(len(dests) for dests in box_map_penalty_bridged.values())}")
    
    # Compute Morse graph
    graph_penalty_bridged = box_map_penalty_bridged.to_networkx()
    morse_graph_penalty_bridged, morse_sets_penalty_bridged = create_morse_graph(graph_penalty_bridged)
    roa_dict_penalty_bridged = compute_roa(graph_penalty_bridged, morse_sets_penalty_bridged)
    
    results['penalty_bridged'] = {
        'box_map': box_map_penalty_bridged,
        'morse_sets': morse_sets_penalty_bridged,
        'roa_dict': roa_dict_penalty_bridged,
        'time': time_penalty_bridged,
        'transitions': sum(len(dests) for dests in box_map_penalty_bridged.values())
    }
    
    # Analyze results
    print("\n\n5. Analysis:")
    print("-" * 40)
    print(f"No penalty: {len(morse_sets_no_penalty)} Morse sets, {results['no_penalty']['transitions']} transitions")
    print(f"With penalty: {len(morse_sets_with_penalty)} Morse sets, {results['with_penalty']['transitions']} transitions")
    print(f"Bridged enclosure: {len(morse_sets_bridged)} Morse sets, {results['bridged']['transitions']} transitions")
    print(f"Penalty + Bridged: {len(morse_sets_penalty_bridged)} Morse sets, {results['penalty_bridged']['transitions']} transitions")
    
    # Compute regions of attraction coverage
    total_boxes = len(grid)
    stats_no_penalty, _ = analyze_roa_coverage(roa_dict_no_penalty, total_boxes)
    stats_with_penalty, _ = analyze_roa_coverage(roa_dict_with_penalty, total_boxes)
    stats_bridged, _ = analyze_roa_coverage(roa_dict_bridged, total_boxes)
    stats_penalty_bridged, _ = analyze_roa_coverage(roa_dict_penalty_bridged, total_boxes)
    
    print(f"\nROA Coverage:")
    print(f"No penalty: {stats_no_penalty['coverage_percentage']:.1f}%")
    print(f"With penalty: {stats_with_penalty['coverage_percentage']:.1f}%")
    print(f"Bridged enclosure: {stats_bridged['coverage_percentage']:.1f}%")
    print(f"Penalty + Bridged: {stats_penalty_bridged['coverage_percentage']:.1f}%")
    
    # Generate visualizations
    print("\n\n6. Generating visualizations:")
    print("-" * 40)
    
    # Create combined comparison plot
    print("Generating combined comparison plot...")
    initial_conditions_for_plot = [
        np.array([0.0, 0.4]), 
        np.array([0.1, 0.3]), 
        np.array([-0.1, 0.2])
    ]
    create_combined_comparison_plot(
        wheel, grid,
        results,
        initial_conditions_for_plot,
        tau,
        run_dir / "combined_comparison.png"
    )
    print("Generated combined_comparison.png")
    
    # Generate individual Morse graphs if pygraphviz is available
    try:
        morse_graphs = {
            'no_penalty': morse_graph_no_penalty,
            'with_penalty': morse_graph_with_penalty,
            'bridged': morse_graph_bridged,
            'penalty_bridged': morse_graph_penalty_bridged
        }
        for name, data in results.items():
            if data['morse_sets']:
                filename = run_dir / f"morse_graph_{name}.png"
                plot_morse_graph_viz(
                    morse_graphs[name],
                    data['morse_sets'],
                    str(filename)
                )
                print(f"Generated morse_graph_{name}.png")
    except Exception as e:
        print(f"Could not generate morse graph visualizations: {e}")
    
    # Save summary info to run directory
    summary_file = run_dir / "run_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Jump Penalty & Boundary-Bridged Enclosure Comparison\n")
        f.write(f"====================================================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  - tau: {tau}\n")
        f.write(f"  - grid subdivisions: {grid.subdivisions.tolist()}\n")
        f.write(f"  - bloat factor: {bloat_factor}\n")
        f.write(f"  - alpha: {wheel.alpha}\n")
        f.write(f"  - gamma: {wheel.gamma}\n")
        f.write(f"\nResults:\n")
        f.write(f"  - No penalty: {len(morse_sets_no_penalty)} Morse sets, "
                f"{results['no_penalty']['transitions']} transitions, "
                f"{stats_no_penalty['coverage_percentage']:.1f}% ROA coverage\n")
        f.write(f"  - With penalty: {len(morse_sets_with_penalty)} Morse sets, "
                f"{results['with_penalty']['transitions']} transitions, "
                f"{stats_with_penalty['coverage_percentage']:.1f}% ROA coverage\n")
        f.write(f"  - Bridged enclosure: {len(morse_sets_bridged)} Morse sets, "
                f"{results['bridged']['transitions']} transitions, "
                f"{stats_bridged['coverage_percentage']:.1f}% ROA coverage\n")
        f.write(f"  - Penalty + Bridged: {len(morse_sets_penalty_bridged)} Morse sets, "
                f"{results['penalty_bridged']['transitions']} transitions, "
                f"{stats_penalty_bridged['coverage_percentage']:.1f}% ROA coverage\n")
        f.write(f"\nComputation times:\n")
        f.write(f"  - No penalty: {time_no_penalty:.2f}s\n")
        f.write(f"  - With penalty: {time_with_penalty:.2f}s\n")
        f.write(f"  - Bridged enclosure: {time_bridged:.2f}s\n")
        f.write(f"  - Penalty + Bridged: {time_penalty_bridged:.2f}s\n")
    print("Generated run_summary.txt")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print(f"Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    compare_methods()