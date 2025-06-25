import numpy as np
import matplotlib.pyplot as plt
import os

def run_bipedal_figure():
    """Generate bipedal walker figure with custom trajectory parametrization."""
    
    # Parameter p = sqrt(2)/2
    p = np.sqrt(2) / 2
    
    # Parametrize arc from (-p,p) to (-p,-p) with more curvature
    n_points = 200
    t = np.linspace(0, 1, n_points)
    
    # Create a more pronounced arc
    x_left_seg = -p + 0.3 * p * np.sin(np.pi * t)  # More x variation
    y_left_seg = p * (1 - 2*t)                     # y goes from p to -p
    
    # Reset map: dashed line from (-p,-p) to (p,p)
    x_dashed1 = np.array([-p, p])
    y_dashed1 = np.array([-p, p])
    
    # Additional dashed line from (p,-p) to (-p,p)
    x_dashed2 = np.array([p, -p])
    y_dashed2 = np.array([-p, p])
    
    # Right trajectory from (p,p) to (p,-p) (mirror of left)
    x_right_seg = -x_left_seg  # Mirror across y-axis
    y_right_seg = y_left_seg   # Same y trajectory
    
    # Lower semicircle (red trajectory)
    theta_lower = np.linspace(-np.pi, 0, n_points)
    radius = p * np.sqrt(2)  # Radius to connect the corners
    x_lower = radius * np.cos(theta_lower)
    y_lower = radius * np.sin(theta_lower)
    
    # Create perturbations of both continuous trajectories
    perturbation_scale = 0.0125
    n_perturbations = 10
    
    # Collect all perturbed points
    all_points = []
    
    # Add perturbed left trajectories
    for i in range(n_perturbations):
        perturb_x = np.random.normal(0, perturbation_scale, len(x_left_seg))
        perturb_y = np.random.normal(0, perturbation_scale, len(y_left_seg))
        
        x_perturbed = x_left_seg + perturb_x
        y_perturbed = y_left_seg + perturb_y
        
        points = np.column_stack([x_perturbed, y_perturbed])
        all_points.append(points)
    
    # Add perturbed right trajectories  
    for i in range(n_perturbations):
        perturb_x = np.random.normal(0, perturbation_scale, len(x_right_seg))
        perturb_y = np.random.normal(0, perturbation_scale, len(y_right_seg))
        
        x_perturbed = x_right_seg + perturb_x
        y_perturbed = y_right_seg + perturb_y
        
        points = np.column_stack([x_perturbed, y_perturbed])
        all_points.append(points)
    
    # Combine all points for gridification
    all_points_combined = np.vstack(all_points)
    
    # Simple grid-based cubification
    # Determine bounds
    x_min, x_max = np.min(all_points_combined[:, 0]), np.max(all_points_combined[:, 0])
    y_min, y_max = np.min(all_points_combined[:, 1]), np.max(all_points_combined[:, 1])
    
    # Add padding
    padding = 0.05
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding
    
    # Create grid
    grid_size = 30
    x_grid = np.linspace(x_min, x_max, grid_size + 1)
    y_grid = np.linspace(y_min, y_max, grid_size + 1)
    
    # Find occupied grid cells
    occupied_cells = set()
    for point in all_points_combined:
        x_idx = np.digitize(point[0], x_grid) - 1
        y_idx = np.digitize(point[1], y_grid) - 1
        x_idx = max(0, min(grid_size - 1, x_idx))
        y_idx = max(0, min(grid_size - 1, y_idx))
        occupied_cells.add((x_idx, y_idx))
    
    # Convert to rectangles
    grid_rectangles = []
    for x_idx, y_idx in occupied_cells:
        x_left = x_grid[x_idx]
        x_right = x_grid[x_idx + 1]
        y_bottom = y_grid[y_idx]
        y_top = y_grid[y_idx + 1]
        grid_rectangles.append((x_left, y_bottom, x_right - x_left, y_top - y_bottom))
    
    # Create output directory
    output_dir = "figures/bipedal"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot figure WITH trajectories
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot grid cover
    for x_left, y_bottom, width, height in grid_rectangles:
        rect = plt.Rectangle((x_left, y_bottom), width, height,
                           linewidth=1, edgecolor='none', facecolor='steelblue', alpha=0.6)
        ax1.add_patch(rect)
    
    # Plot continuous trajectories (solid lines)
    ax1.plot(x_left_seg, y_left_seg, 'b-', linewidth=2)
    ax1.plot(x_right_seg, y_right_seg, 'b-', linewidth=2)
    
    # Plot lower semicircle (red)
    ax1.plot(x_lower, y_lower, 'r-', linewidth=2)
    
    # Plot reset connections (dashed lines in blue)
    ax1.plot(x_dashed1, y_dashed1, 'b--', linewidth=2)
    ax1.plot(x_dashed2, y_dashed2, 'b--', linewidth=2)
    
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Save figure with trajectories
    figure_path1 = os.path.join(output_dir, 'bipedal_with_trajectories.png')
    plt.savefig(figure_path1, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot figure WITHOUT trajectories
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot grid cover only
    for x_left, y_bottom, width, height in grid_rectangles:
        rect = plt.Rectangle((x_left, y_bottom), width, height,
                           linewidth=1, edgecolor='none', facecolor='steelblue', alpha=0.6)
        ax2.add_patch(rect)
    
    # Plot lower semicircle (red) on cover-only figure
    ax2.plot(x_lower, y_lower, 'r-', linewidth=2)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Save figure without trajectories
    figure_path2 = os.path.join(output_dir, 'bipedal_cover_only.png')
    plt.savefig(figure_path2, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot figure with ONLY trajectories (no cubes)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot continuous trajectories (solid lines)
    ax3.plot(x_left_seg, y_left_seg, 'b-', linewidth=2)
    ax3.plot(x_right_seg, y_right_seg, 'b-', linewidth=2)
    
    # Plot lower semicircle (red)
    ax3.plot(x_lower, y_lower, 'r-', linewidth=2)
    
    # Plot reset connections (dashed lines in blue)
    ax3.plot(x_dashed1, y_dashed1, 'b--', linewidth=2)
    ax3.plot(x_dashed2, y_dashed2, 'b--', linewidth=2)
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Save figure with trajectories only
    figure_path3 = os.path.join(output_dir, 'bipedal_trajectories_only.png')
    plt.savefig(figure_path3, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated figures:")
    print(f"  With trajectories: {figure_path1}")
    print(f"  Cover only: {figure_path2}")
    print(f"  Trajectories only: {figure_path3}")
    print(f"Number of grid cells in cover: {len(grid_rectangles)}")

if __name__ == "__main__":
    run_bipedal_figure()