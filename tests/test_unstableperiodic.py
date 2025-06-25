#!/usr/bin/env python3
"""
Test script to analyze box map behavior for boxes touching the right boundary
of the unstable periodic system.
"""
from pathlib import Path
import networkx as nx
import numpy as np

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
)
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 1.42                 # Integration time horizon
SUBDIVISIONS = [100, 100] # Grid subdivisions [x_subdivisions, y_subdivisions]
# ===============================================


def analyze_right_boundary_boxes():
    """Analyze boxes touching the right boundary and their images under the box map."""
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("unstable_periodic")
    box_map_file = data_dir / "unstable_periodic_box_map.pkl"

    # 1. Define configuration
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-1.0, 1.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Create configuration hash for validation
    system_params = {"max_jumps": system_obj.max_jumps}
    current_config_hash = create_config_hash(
        system_params, 
        system_obj.domain_bounds, 
        grid.subdivisions.tolist(), 
        tau
    )
    
    # 2. Try to load from cache
    box_map = load_box_map_from_cache(grid, system_obj.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute new if cache miss
    if box_map is None:
        print("Computing new box map...")
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=system_obj.system,
            tau=tau,
            discard_out_of_bounds_destinations=False,
            progress_callback=progress_callback,
        )
        
        # Save to cache
        config_details = {
            'system_params': system_params,
            'domain_bounds': system_obj.domain_bounds,
            'grid_subdivisions': grid.subdivisions.tolist(),
            'tau': tau
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)

    # 4. Find boxes touching the right boundary
    print("\n=== Analyzing boxes touching the right boundary ===")
    print(f"Domain bounds: x in [{system_obj.domain_bounds[0][0]}, {system_obj.domain_bounds[0][1]}]")
    print(f"Grid subdivisions: {grid.subdivisions}")
    print(f"Number of boxes in x-direction: {grid.subdivisions[0]}")
    
    # The right boundary corresponds to x = 1.0
    # In a grid with subdivisions[0] boxes in x-direction, the rightmost boxes
    # have x-index = subdivisions[0] - 1
    right_boundary_x_index = grid.subdivisions[0] - 1
    
    right_boundary_boxes = []
    for y_idx in range(grid.subdivisions[1]):
        # Convert multi-index to linear index
        box_index = int(np.ravel_multi_index([right_boundary_x_index, y_idx], grid.subdivisions))
        right_boundary_boxes.append(box_index)
    
    print(f"\nNumber of boxes touching right boundary: {len(right_boundary_boxes)}")
    
    # 5. Analyze the images of these boxes
    print("\n=== Box map analysis for right boundary boxes ===")
    
    # Count statistics
    boxes_with_images = 0
    total_image_boxes = 0
    boxes_mapping_to_left = 0
    boxes_mapping_to_right = 0
    boxes_mapping_to_middle = 0
    
    # Detailed analysis for first few boxes
    detailed_count = min(5, len(right_boundary_boxes))
    
    for i, box_idx in enumerate(right_boundary_boxes):
        if box_idx in box_map:
            image_boxes = box_map[box_idx]
            boxes_with_images += 1
            total_image_boxes += len(image_boxes)
            
            # Get the actual box coordinates
            multi_idx = np.unravel_index(box_idx, grid.subdivisions)
            box = grid.get_box(box_idx)
            
            # Analyze where the images are
            for img_idx in image_boxes:
                img_multi_idx = np.unravel_index(img_idx, grid.subdivisions)
                img_box = grid.get_box(img_idx)
                
                # Check x-coordinate of image
                img_x_center = (img_box.lower_bounds[0] + img_box.upper_bounds[0]) / 2
                
                if img_x_center < 0.2:
                    boxes_mapping_to_left += 1
                elif img_x_center > 0.8:
                    boxes_mapping_to_right += 1
                else:
                    boxes_mapping_to_middle += 1
            
            # Detailed output for first few boxes
            if i < detailed_count:
                print(f"\nBox {box_idx} (multi-index {multi_idx}):")
                print(f"  Box bounds: x in [{box.lower_bounds[0]:.3f}, {box.upper_bounds[0]:.3f}], "
                      f"y in [{box.lower_bounds[1]:.3f}, {box.upper_bounds[1]:.3f}]")
                print(f"  Number of image boxes: {len(image_boxes)}")
                
                if len(image_boxes) <= 10:  # Show all if not too many
                    print(f"  Image box indices: {list(image_boxes)}")
                    for img_idx in image_boxes:
                        img_multi_idx = np.unravel_index(img_idx, grid.subdivisions)
                        img_box = grid.get_box(img_idx)
                        print(f"    Box {img_idx} (multi-index {img_multi_idx}): "
                              f"x in [{img_box.lower_bounds[0]:.3f}, {img_box.upper_bounds[0]:.3f}], "
                              f"y in [{img_box.lower_bounds[1]:.3f}, {img_box.upper_bounds[1]:.3f}]")
                else:
                    print(f"  (Too many image boxes to display individually)")
    
    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total right boundary boxes: {len(right_boundary_boxes)}")
    print(f"Boxes with non-empty images: {boxes_with_images} ({boxes_with_images/len(right_boundary_boxes)*100:.1f}%)")
    print(f"Average number of image boxes per right boundary box: {total_image_boxes/len(right_boundary_boxes):.1f}")
    print(f"\nImage box locations:")
    print(f"  Mapping to left region (x < 0.2): {boxes_mapping_to_left}")
    print(f"  Mapping to middle region (0.2 ≤ x ≤ 0.8): {boxes_mapping_to_middle}")
    print(f"  Mapping to right region (x > 0.8): {boxes_mapping_to_right}")
    
    # 6. Visualize specific trajectories from right boundary
    print("\n=== Creating trajectory visualizations ===")
    
    # Select a few representative boxes from right boundary
    test_y_indices = [0, 25, 50, 75, 99]  # Sample across the y-range
    test_initial_conditions = []
    
    for y_idx in test_y_indices:
        if y_idx < grid.subdivisions[1]:
            box_index = int(np.ravel_multi_index([right_boundary_x_index, y_idx], grid.subdivisions))
            box = grid.get_box(box_index)
            # Use center of the box as initial condition
            center_x = (box.lower_bounds[0] + box.upper_bounds[0]) / 2
            center_y = (box.lower_bounds[1] + box.upper_bounds[1]) / 2
            test_initial_conditions.append([center_x, center_y])
    
    # Create trajectory plot
    run_dir = Path("test_results")
    run_dir.mkdir(exist_ok=True)
    
    plotter = HybridPlotter()
    plotter.create_phase_portrait_with_trajectories(
        system=system_obj.system,
        initial_conditions=test_initial_conditions,
        time_span=(0.0, tau),  # Use same time as box map
        output_path=str(run_dir / "right_boundary_trajectories.png"),
        max_jumps=system_obj.max_jumps,
        title="Trajectories from Right Boundary",
        xlabel="x",
        ylabel="y",
        domain_bounds=system_obj.domain_bounds,
        figsize=(10, 8),
        dpi=150,
        show_legend=True
    )
    
    print(f"\nTrajectory visualization saved to: {run_dir / 'right_boundary_trajectories.png'}")
    
    return box_map, grid, system_obj


if __name__ == "__main__":
    analyze_right_boundary_boxes()