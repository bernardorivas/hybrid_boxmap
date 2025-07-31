#!/usr/bin/env python3
"""
Debug the boundary-bridged enclosure for a single box.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel


def analyze_single_box(wheel, grid, box_idx, tau):
    """Analyze how a single box is processed with different methods."""
    
    # Get corners
    corners = grid.get_sample_points(box_idx, mode='corners')
    
    # Simulate from each corner
    results = []
    for i, corner in enumerate(corners):
        traj = wheel.system.simulate(corner, (0, tau))
        final_state = traj.interpolate(tau)
        guard_state = traj.jump_states[0][0] if traj.jump_states else None
        results.append({
            'corner': corner,
            'corner_label': chr(97 + i),  # a, b, c, d
            'final_state': final_state,
            'num_jumps': traj.num_jumps,
            'guard_state': guard_state,
            'trajectory': traj
        })
    
    # Print analysis
    print(f"\nBox {box_idx} Analysis:")
    print("-" * 60)
    for r in results:
        print(f"Corner {r['corner_label']}: {r['corner']} → {r['num_jumps']} jumps")
        if r['guard_state'] is not None:
            print(f"  Guard state: {r['guard_state']}")
        print(f"  Final state: {r['final_state']}")
    
    # Group by jump count
    jump_counts = [r['num_jumps'] for r in results]
    unique_jumps = sorted(set(jump_counts))
    
    if len(unique_jumps) > 1:
        print(f"\nMixed jump counts detected: {unique_jumps}")
        
        # Compute what boundary-bridged method would do
        print("\nBoundary-Bridged Stratification:")
        strata = {}
        
        for n in unique_jumps:
            strata[n] = {
                'direct_points': [],
                'bridged_points': []
            }
        
        # Add direct points
        for r in results:
            strata[r['num_jumps']]['direct_points'].append(r['final_state'])
        
        # Add bridged points
        for r in results:
            if r['guard_state'] is not None and r['num_jumps'] > 0:
                # Guard state goes to stratum 0
                if 0 not in strata:
                    strata[0] = {'direct_points': [], 'bridged_points': []}
                strata[0]['bridged_points'].append(r['guard_state'])
                
                # Apply reset map iteratively
                current = r['guard_state']
                for j in range(r['num_jumps']):
                    try:
                        current = wheel.system.reset_map(current)
                        if j + 1 in strata:
                            strata[j + 1]['bridged_points'].append(current)
                    except:
                        break
        
        # Print stratum analysis
        for n in sorted(strata.keys()):
            s = strata[n]
            print(f"\nStratum {n}:")
            print(f"  Direct points: {len(s['direct_points'])}")
            print(f"  Bridged points: {len(s['bridged_points'])}")
            print(f"  Total points: {len(s['direct_points']) + len(s['bridged_points'])}")
    
    return results, strata if len(unique_jumps) > 1 else None


def visualize_strata(results, strata, grid, box_idx):
    """Visualize the stratified destination sets."""
    if strata is None:
        print("No mixed jumps to visualize")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Get box bounds
    lower_bounds, upper_bounds = grid.get_box_bounds(box_idx)
    
    # Plot source box
    source_rect = patches.Rectangle(
        (lower_bounds[0], lower_bounds[1]),
        upper_bounds[0] - lower_bounds[0],
        upper_bounds[1] - lower_bounds[1],
        facecolor='red', alpha=0.3, edgecolor='red', linewidth=2
    )
    ax.add_patch(source_rect)
    
    # Plot each stratum
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for n, color in zip(sorted(strata.keys()), colors):
        s = strata[n]
        
        # Direct points
        if s['direct_points']:
            points = np.array(s['direct_points'])
            ax.scatter(points[:, 0], points[:, 1], 
                      s=100, color=color, marker='o', 
                      label=f'Stratum {n} (direct)', edgecolor='black')
        
        # Bridged points
        if s['bridged_points']:
            points = np.array(s['bridged_points'])
            ax.scatter(points[:, 0], points[:, 1], 
                      s=100, color=color, marker='s', alpha=0.5,
                      label=f'Stratum {n} (bridged)', edgecolor='black')
        
        # Draw bounding box for all points in stratum
        all_points = s['direct_points'] + s['bridged_points']
        if all_points:
            all_points = np.array(all_points)
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            
            rect = patches.Rectangle(
                (min_coords[0], min_coords[1]),
                width, height,
                fill=False, edgecolor=color, linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
    
    ax.set_xlim(grid.bounds[0])
    ax.set_ylim(grid.bounds[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Stratified Destination Sets for Box {box_idx}')
    ax.set_xlabel('Angle θ (rad)')
    ax.set_ylabel('Angular Velocity ω (rad/s)')
    
    plt.tight_layout()
    plt.savefig('figures/debug_strata_visualization.png', dpi=150)
    plt.show()


def main():
    # Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[41, 41])
    tau = 1.5
    
    # Find a box with mixed jumps
    print("Searching for boxes with mixed jump counts...")
    mixed_boxes = []
    
    for box_idx in range(100, min(500, len(grid))):
        corners = grid.get_sample_points(box_idx, mode='corners')
        jump_counts = []
        
        for corner in corners:
            traj = wheel.system.simulate(corner, (0, tau))
            jump_counts.append(traj.num_jumps)
        
        if len(set(jump_counts)) > 1:
            mixed_boxes.append((box_idx, jump_counts))
    
    if not mixed_boxes:
        print("No mixed jump boxes found!")
        return
    
    print(f"Found {len(mixed_boxes)} boxes with mixed jumps")
    
    # Analyze the first few
    for box_idx, jump_counts in mixed_boxes[:3]:
        results, strata = analyze_single_box(wheel, grid, box_idx, tau)
        if strata:
            visualize_strata(results, strata, grid, box_idx)
            break  # Just visualize the first one


if __name__ == "__main__":
    main()