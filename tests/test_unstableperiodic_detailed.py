#!/usr/bin/env python3
"""
Detailed analysis of box map behavior for the unstable periodic system,
focusing on understanding the dynamics at the right boundary.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# Configuration
TAU = 1.42
SUBDIVISIONS = [100, 100]


def visualize_boundary_mapping():
    """Create detailed visualization of right boundary box mappings."""
    
    # Setup
    data_dir, _ = setup_demo_directories("unstable_periodic")
    box_map_file = data_dir / "unstable_periodic_box_map.pkl"
    
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    
    # Load or compute box map
    system_params = {"max_jumps": system_obj.max_jumps}
    current_config_hash = create_config_hash(
        system_params, 
        system_obj.domain_bounds, 
        grid.subdivisions.tolist(), 
        TAU
    )
    
    box_map = load_box_map_from_cache(grid, system_obj.system, TAU, box_map_file, current_config_hash)
    
    if box_map is None:
        print("Computing new box map...")
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=system_obj.system,
            tau=TAU,
            discard_out_of_bounds_destinations=False,
            progress_callback=progress_callback,
        )
        
        config_details = {
            'system_params': system_params,
            'domain_bounds': system_obj.domain_bounds,
            'grid_subdivisions': grid.subdivisions.tolist(),
            'tau': TAU
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Source boxes (right boundary) colored by number of destinations
    ax1.set_title(f"Right Boundary Boxes\nColored by Number of Image Boxes")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(system_obj.domain_bounds[0])
    ax1.set_ylim(system_obj.domain_bounds[1])
    
    # Find right boundary boxes
    right_boundary_x_index = grid.subdivisions[0] - 1
    right_boundary_boxes = []
    num_destinations = []
    
    for y_idx in range(grid.subdivisions[1]):
        box_index = int(np.ravel_multi_index([right_boundary_x_index, y_idx], grid.subdivisions))
        right_boundary_boxes.append(box_index)
        
        if box_index in box_map:
            num_destinations.append(len(box_map[box_index]))
        else:
            num_destinations.append(0)
    
    # Create colormap
    vmin = min(num_destinations)
    vmax = max(num_destinations)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('viridis')
    
    # Draw all boxes in light gray first
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
    
    # Highlight right boundary boxes
    for i, (box_idx, n_dest) in enumerate(zip(right_boundary_boxes, num_destinations)):
        box = grid.get_box(box_idx)
        color = cmap(norm(n_dest))
        
        rect = Rectangle(
            (box.lower_bounds[0], box.lower_bounds[1]),
            box.upper_bounds[0] - box.lower_bounds[0],
            box.upper_bounds[1] - box.lower_bounds[1],
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.8
        )
        ax1.add_patch(rect)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1)
    cbar1.set_label('Number of Image Boxes')
    
    # Plot 2: Destination distribution
    ax2.set_title(f"Image Box Distribution\nfrom Right Boundary")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(system_obj.domain_bounds[0])
    ax2.set_ylim(system_obj.domain_bounds[1])
    
    # Count how many times each box appears as a destination
    destination_counts = {}
    for box_idx in right_boundary_boxes:
        if box_idx in box_map:
            for dest_idx in box_map[box_idx]:
                destination_counts[dest_idx] = destination_counts.get(dest_idx, 0) + 1
    
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
        ax2.add_patch(rect)
    
    # Highlight destination boxes
    if destination_counts:
        vmin2 = min(destination_counts.values())
        vmax2 = max(destination_counts.values())
        norm2 = plt.Normalize(vmin=vmin2, vmax=vmax2)
        cmap2 = cm.get_cmap('hot')
        
        for dest_idx, count in destination_counts.items():
            box = grid.get_box(dest_idx)
            color = cmap2(norm2(count))
            
            rect = Rectangle(
                (box.lower_bounds[0], box.lower_bounds[1]),
                box.upper_bounds[0] - box.lower_bounds[0],
                box.upper_bounds[1] - box.lower_bounds[1],
                linewidth=0.5,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax2.add_patch(rect)
        
        # Add colorbar
        sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2)
        cbar2.set_label('Times Appearing as Image')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "boundary_mapping_visualization.png", dpi=150)
    print(f"Visualization saved to: {output_dir / 'boundary_mapping_visualization.png'}")
    
    # Analyze patterns
    print("\n=== Detailed Analysis ===")
    
    # Group by y-coordinate behavior
    print("\nBehavior by y-coordinate:")
    y_ranges = [
        (-1.0, -0.5, "Lower half (y < -0.5)"),
        (-0.5, 0.0, "Lower middle (-0.5 ≤ y < 0)"),
        (0.0, 0.5, "Upper middle (0 ≤ y < 0.5)"),
        (0.5, 1.0, "Upper half (y ≥ 0.5)")
    ]
    
    for y_min, y_max, label in y_ranges:
        boxes_in_range = []
        for y_idx in range(grid.subdivisions[1]):
            y_coord = -1.0 + (y_idx + 0.5) * (2.0 / grid.subdivisions[1])
            if y_min <= y_coord < y_max:
                box_index = int(np.ravel_multi_index([right_boundary_x_index, y_idx], grid.subdivisions))
                if box_index in box_map:
                    boxes_in_range.append(len(box_map[box_index]))
        
        if boxes_in_range:
            print(f"\n{label}:")
            print(f"  Average image boxes: {np.mean(boxes_in_range):.1f}")
            print(f"  Min/Max: {min(boxes_in_range)}/{max(boxes_in_range)}")
    
    # Find special cases
    print("\n=== Special Cases ===")
    
    # Boxes that map to themselves
    self_mapping = []
    for box_idx in right_boundary_boxes:
        if box_idx in box_map and box_idx in box_map[box_idx]:
            self_mapping.append(box_idx)
    
    if self_mapping:
        print(f"\nBoxes that map to themselves: {len(self_mapping)}")
        for box_idx in self_mapping[:5]:  # Show first 5
            multi_idx = np.unravel_index(box_idx, grid.subdivisions)
            box = grid.get_box(box_idx)
            y_center = (box.lower_bounds[1] + box.upper_bounds[1]) / 2
            print(f"  Box {box_idx}: y-center = {y_center:.3f}")
    
    # Boxes with unusually many destinations
    threshold = np.mean(num_destinations) + 2 * np.std(num_destinations)
    outliers = [(i, n) for i, n in enumerate(num_destinations) if n > threshold]
    
    if outliers:
        print(f"\nBoxes with unusually many destinations (> {threshold:.1f}):")
        for idx, n_dest in outliers[:5]:  # Show first 5
            box_idx = right_boundary_boxes[idx]
            box = grid.get_box(box_idx)
            y_center = (box.lower_bounds[1] + box.upper_bounds[1]) / 2
            print(f"  Box {box_idx}: y-center = {y_center:.3f}, destinations = {n_dest}")


if __name__ == "__main__":
    visualize_boundary_mapping()