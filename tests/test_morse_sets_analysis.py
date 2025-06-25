#!/usr/bin/env python3
"""
Analyze Morse sets to understand why vertical structures appear on the right boundary.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src import create_morse_graph
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    setup_demo_directories,
    load_box_map_from_cache,
)


# Configuration
TAU = 1.42
SUBDIVISIONS = [100, 100]


def analyze_morse_sets():
    """Detailed analysis of Morse sets and SCCs."""
    
    # Setup
    data_dir, _ = setup_demo_directories("unstable_periodic")
    box_map_file = data_dir / "unstable_periodic_box_map.pkl"
    
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    
    # Load box map
    system_params = {"max_jumps": system_obj.max_jumps}
    current_config_hash = create_config_hash(
        system_params, 
        system_obj.domain_bounds, 
        grid.subdivisions.tolist(), 
        TAU
    )
    
    box_map = load_box_map_from_cache(grid, system_obj.system, TAU, box_map_file, current_config_hash)
    
    # Convert to NetworkX graph
    graph = box_map.to_networkx()
    
    # Compute Morse graph and sets
    morse_graph, morse_sets = create_morse_graph(graph)
    
    print("=== Morse Sets Analysis ===")
    print(f"Number of Morse sets (SCCs): {len(morse_sets)}")
    print(f"Total nodes in graph: {graph.number_of_nodes()}")
    print(f"Total edges in graph: {graph.number_of_edges()}")
    
    # Analyze each Morse set
    right_boundary_x_index = grid.subdivisions[0] - 1
    
    for i, morse_set in enumerate(morse_sets):
        print(f"\n--- Morse Set {i} ---")
        print(f"Size: {len(morse_set)} boxes")
        
        # Check if this set contains right boundary boxes
        right_boundary_boxes_in_set = []
        for box_idx in morse_set:
            multi_idx = np.unravel_index(box_idx, grid.subdivisions)
            if multi_idx[0] == right_boundary_x_index:
                right_boundary_boxes_in_set.append(box_idx)
        
        if right_boundary_boxes_in_set:
            print(f"Contains {len(right_boundary_boxes_in_set)} boxes on right boundary")
            
            # Analyze y-distribution
            y_indices = []
            for box_idx in right_boundary_boxes_in_set:
                multi_idx = np.unravel_index(box_idx, grid.subdivisions)
                y_indices.append(multi_idx[1])
            
            y_indices = sorted(y_indices)
            print(f"Y-index range: {min(y_indices)} to {max(y_indices)}")
            
            # Check if it's a vertical strip
            if len(y_indices) > 1:
                gaps = [y_indices[i+1] - y_indices[i] for i in range(len(y_indices)-1)]
                if all(gap == 1 for gap in gaps):
                    print("Forms a continuous vertical strip")
                else:
                    print(f"Has gaps: {set(gaps)}")
        
        # Analyze x-distribution of entire Morse set
        x_indices = []
        for box_idx in morse_set:
            multi_idx = np.unravel_index(box_idx, grid.subdivisions)
            x_indices.append(multi_idx[0])
        
        x_counts = {}
        for x in x_indices:
            x_counts[x] = x_counts.get(x, 0) + 1
        
        print(f"X-distribution: spans x-indices {min(x_indices)} to {max(x_indices)}")
        if len(morse_set) < 50:  # Only show details for smaller sets
            print(f"X-index counts: {dict(sorted(x_counts.items())[:10])}...")  # Show first 10
    
    # Create detailed visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: All Morse sets with different colors
    ax1.set_title("All Morse Sets")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(system_obj.domain_bounds[0])
    ax1.set_ylim(system_obj.domain_bounds[1])
    
    # Draw all boxes in light gray
    for i in range(grid.total_boxes):
        box = grid.get_box(i)
        rect = Rectangle(
            (box.lower_bounds[0], box.lower_bounds[1]),
            box.upper_bounds[0] - box.lower_bounds[0],
            box.upper_bounds[1] - box.lower_bounds[1],
            linewidth=0.1,
            edgecolor='gray',
            facecolor='lightgray',
            alpha=0.3
        )
        ax1.add_patch(rect)
    
    # Color each Morse set differently
    colors = plt.cm.tab20(np.linspace(0, 1, len(morse_sets)))
    for i, morse_set in enumerate(morse_sets):
        for box_idx in morse_set:
            box = grid.get_box(box_idx)
            rect = Rectangle(
                (box.lower_bounds[0], box.lower_bounds[1]),
                box.upper_bounds[0] - box.lower_bounds[0],
                box.upper_bounds[1] - box.lower_bounds[1],
                linewidth=0.5,
                edgecolor='black',
                facecolor=colors[i],
                alpha=0.7
            )
            ax1.add_patch(rect)
    
    # Highlight right boundary
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 2: Focus on right boundary region
    ax2.set_title("Right Boundary Region (x > 0.9)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(0.9, 1.0)
    ax2.set_ylim(system_obj.domain_bounds[1])
    
    # Draw boxes in right region
    for i in range(grid.total_boxes):
        box = grid.get_box(i)
        if box.lower_bounds[0] >= 0.9:
            rect = Rectangle(
                (box.lower_bounds[0], box.lower_bounds[1]),
                box.upper_bounds[0] - box.lower_bounds[0],
                box.upper_bounds[1] - box.lower_bounds[1],
                linewidth=0.2,
                edgecolor='gray',
                facecolor='lightgray',
                alpha=0.3
            )
            ax2.add_patch(rect)
    
    # Color Morse sets in right region
    for i, morse_set in enumerate(morse_sets):
        for box_idx in morse_set:
            box = grid.get_box(box_idx)
            if box.lower_bounds[0] >= 0.9:
                rect = Rectangle(
                    (box.lower_bounds[0], box.lower_bounds[1]),
                    box.upper_bounds[0] - box.lower_bounds[0],
                    box.upper_bounds[1] - box.lower_bounds[1],
                    linewidth=0.5,
                    edgecolor='black',
                    facecolor=colors[i],
                    alpha=0.7
                )
                ax2.add_patch(rect)
    
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 3: Connectivity analysis
    ax3.set_title("Self-loops and Strong Connectivity")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_xlim(system_obj.domain_bounds[0])
    ax3.set_ylim(system_obj.domain_bounds[1])
    
    # Find boxes with self-loops
    self_loop_boxes = []
    for box_idx in box_map:
        if box_idx in box_map[box_idx]:
            self_loop_boxes.append(box_idx)
    
    print(f"\n=== Self-loop Analysis ===")
    print(f"Total boxes with self-loops: {len(self_loop_boxes)}")
    
    # Check how many are on right boundary
    right_boundary_self_loops = []
    for box_idx in self_loop_boxes:
        multi_idx = np.unravel_index(box_idx, grid.subdivisions)
        if multi_idx[0] == right_boundary_x_index:
            right_boundary_self_loops.append(box_idx)
    
    print(f"Self-loops on right boundary: {len(right_boundary_self_loops)}")
    
    # Draw all boxes
    for i in range(grid.total_boxes):
        box = grid.get_box(i)
        rect = Rectangle(
            (box.lower_bounds[0], box.lower_bounds[1]),
            box.upper_bounds[0] - box.lower_bounds[0],
            box.upper_bounds[1] - box.lower_bounds[1],
            linewidth=0.1,
            edgecolor='gray',
            facecolor='lightgray',
            alpha=0.3
        )
        ax3.add_patch(rect)
    
    # Highlight self-loop boxes
    for box_idx in self_loop_boxes:
        box = grid.get_box(box_idx)
        color = 'red' if box_idx in right_boundary_self_loops else 'blue'
        rect = Rectangle(
            (box.lower_bounds[0], box.lower_bounds[1]),
            box.upper_bounds[0] - box.lower_bounds[0],
            box.upper_bounds[1] - box.lower_bounds[1],
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax3.add_patch(rect)
    
    ax3.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Self-loop (interior)'),
        Patch(facecolor='red', alpha=0.7, label='Self-loop (right boundary)')
    ]
    ax3.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "morse_sets_detailed_analysis.png", dpi=150)
    print(f"\nVisualization saved to: {output_dir / 'morse_sets_detailed_analysis.png'}")
    
    # Additional analysis: Why are there vertical strips?
    print("\n=== Understanding Vertical Structures ===")
    
    # For boxes on right boundary with self-loops, check their dynamics
    print("\nAnalyzing self-loop dynamics on right boundary:")
    sample_boxes = right_boundary_self_loops[:5]  # First 5
    
    for box_idx in sample_boxes:
        box = grid.get_box(box_idx)
        center_x = (box.lower_bounds[0] + box.upper_bounds[0]) / 2
        center_y = (box.lower_bounds[1] + box.upper_bounds[1]) / 2
        
        print(f"\nBox {box_idx} centered at ({center_x:.3f}, {center_y:.3f}):")
        
        # Simulate from center
        initial_state = np.array([center_x, center_y])
        
        # Check if it can reach the boundary in time tau
        time_to_boundary = (1.0 - center_x) / 1.0  # Since dx/dt = 1
        print(f"  Time to reach boundary: {time_to_boundary:.3f}")
        print(f"  Time horizon tau: {TAU}")
        
        if time_to_boundary > TAU:
            print(f"  → Cannot reach boundary in time tau, stays in same box")
        else:
            print(f"  → Should reach boundary and jump")


if __name__ == "__main__":
    analyze_morse_sets()