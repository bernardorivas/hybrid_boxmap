#!/usr/bin/env python3
"""
Visualize the improved boundary-bridged enclosure method.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel


def analyze_improved_bridging(wheel, grid, box_idx, tau):
    """Analyze how the improved method works for a single box."""
    
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
    
    # Group by jump count
    jump_counts = [r['num_jumps'] for r in results]
    unique_jumps = sorted(set(jump_counts))
    
    print(f"\nBox {box_idx} Analysis:")
    print("-" * 60)
    for r in results:
        print(f"Corner {r['corner_label']}: {r['corner']} → {r['num_jumps']} jumps")
        if r['guard_state'] is not None:
            print(f"  Guard state: {r['guard_state']}")
        print(f"  Final state: {r['final_state']}")
    
    if len(unique_jumps) > 1:
        print(f"\nMixed jump counts detected: {unique_jumps}")
        
        # Find next guard states for corners with fewer jumps
        print("\nFinding next guard states for corners with fewer jumps:")
        additional_guards = []
        
        min_jumps = min(jump_counts)
        for r in results:
            if r['num_jumps'] == min_jumps:
                print(f"\nCorner {r['corner_label']} (had {r['num_jumps']} jumps):")
                
                # Continue integration to find next guard
                next_guard = HybridBoxMap._find_next_guard_state(
                    wheel.system, r['final_state'], 0.0, 5.0
                )
                
                if next_guard is not None:
                    print(f"  Next guard found: {next_guard}")
                    additional_guards.append({
                        'corner_label': r['corner_label'],
                        'guard': next_guard,
                        'from_state': r['final_state']
                    })
                    
                    # Apply reset map
                    try:
                        post_jump = wheel.system.reset_map(next_guard)
                        print(f"  Post-jump state: {post_jump}")
                    except Exception as e:
                        print(f"  Reset map failed: {e}")
                else:
                    print("  No next guard found")
        
        return results, additional_guards
    
    return results, []


def visualize_improved_method(results, additional_guards, grid, box_idx):
    """Visualize the improved boundary-bridged method."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get box bounds
    lower_bounds, upper_bounds = grid.get_box_bounds(box_idx)
    
    # Plot source box on both axes
    for ax in [ax1, ax2]:
        source_rect = patches.Rectangle(
            (lower_bounds[0], lower_bounds[1]),
            upper_bounds[0] - lower_bounds[0],
            upper_bounds[1] - lower_bounds[1],
            facecolor='red', alpha=0.3, edgecolor='red', linewidth=2
        )
        ax.add_patch(source_rect)
    
    # Left plot: Standard method
    ax1.set_title('Standard Enclosure Method', fontsize=14)
    
    # Group results by jump count
    jump_groups = {}
    for r in results:
        n = r['num_jumps']
        if n not in jump_groups:
            jump_groups[n] = []
        jump_groups[n].append(r)
    
    colors = ['blue', 'green', 'orange', 'purple']
    
    for n, color in zip(sorted(jump_groups.keys()), colors):
        points = [r['final_state'] for r in jump_groups[n]]
        if points:
            points = np.array(points)
            ax1.scatter(points[:, 0], points[:, 1], 
                       s=100, color=color, marker='o', 
                       label=f'{n} jumps', edgecolor='black')
            
            # Draw bounding box
            if len(points) > 0:
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                
                rect = patches.Rectangle(
                    (min_coords[0], min_coords[1]),
                    max_coords[0] - min_coords[0],
                    max_coords[1] - min_coords[1],
                    fill=False, edgecolor=color, linewidth=2, linestyle='--'
                )
                ax1.add_patch(rect)
    
    # Right plot: Improved method
    ax2.set_title('Improved Boundary-Bridged Method', fontsize=14)
    
    # Plot original destinations
    for n, color in zip(sorted(jump_groups.keys()), colors):
        points = [r['final_state'] for r in jump_groups[n]]
        if points:
            points = np.array(points)
            ax2.scatter(points[:, 0], points[:, 1], 
                       s=100, color=color, marker='o', 
                       label=f'{n} jumps (direct)', edgecolor='black')
    
    # Plot additional guard states and their resets
    if additional_guards:
        # Guard states
        guard_points = np.array([g['guard'] for g in additional_guards])
        ax2.scatter(guard_points[:, 0], guard_points[:, 1], 
                   s=150, color='red', marker='*', 
                   label='Additional guards', edgecolor='black')
        
        # Post-jump states
        post_jumps = []
        wheel = RimlessWheel()  # Need this for reset map
        for g in additional_guards:
            try:
                post_jump = wheel.system.reset_map(g['guard'])
                post_jumps.append(post_jump)
                
                # Draw arrow from guard to post-jump
                ax2.annotate('', xy=post_jump, xytext=g['guard'],
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
            except:
                pass
        
        if post_jumps:
            post_jumps = np.array(post_jumps)
            ax2.scatter(post_jumps[:, 0], post_jumps[:, 1], 
                       s=150, color='green', marker='s', 
                       label='Post-jump states', edgecolor='black')
            
            # Show the expanded enclosure
            all_points_1 = []
            # Original points with 1 jump
            for r in results:
                if r['num_jumps'] == 1:
                    all_points_1.append(r['final_state'])
            # Add post-jump states
            all_points_1.extend(post_jumps)
            
            if all_points_1:
                all_points_1 = np.array(all_points_1)
                min_coords = np.min(all_points_1, axis=0)
                max_coords = np.max(all_points_1, axis=0)
                
                rect = patches.Rectangle(
                    (min_coords[0], min_coords[1]),
                    max_coords[0] - min_coords[0],
                    max_coords[1] - min_coords[1],
                    fill=False, edgecolor='green', linewidth=3, linestyle='-'
                )
                ax2.add_patch(rect)
                ax2.text(min_coords[0], max_coords[1] + 0.02, 
                        'Expanded enclosure', color='green', fontweight='bold')
    
    # Set limits and labels
    for ax in [ax1, ax2]:
        ax.set_xlim(grid.bounds[0])
        ax.set_ylim(grid.bounds[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('Angle θ (rad)')
        ax.set_ylabel('Angular Velocity ω (rad/s)')
    
    plt.tight_layout()
    plt.savefig('figures/improved_bridging_visualization.png', dpi=150)
    plt.show()


def main():
    # Setup
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[41, 41])
    tau = 1.5
    
    # Find a box with mixed jumps
    print("Searching for boxes with mixed jump counts...")
    mixed_box = None
    
    for box_idx in range(100, min(500, len(grid))):
        corners = grid.get_sample_points(box_idx, mode='corners')
        jump_counts = []
        
        for corner in corners:
            traj = wheel.system.simulate(corner, (0, tau))
            jump_counts.append(traj.num_jumps)
        
        if len(set(jump_counts)) > 1 and min(jump_counts) >= 0:
            mixed_box = box_idx
            break
    
    if mixed_box is None:
        print("No mixed jump boxes found!")
        return
    
    print(f"Found mixed jump box: {mixed_box}")
    
    # Analyze the box
    results, additional_guards = analyze_improved_bridging(wheel, grid, mixed_box, tau)
    
    # Visualize
    visualize_improved_method(results, additional_guards, grid, mixed_box)


if __name__ == "__main__":
    main()