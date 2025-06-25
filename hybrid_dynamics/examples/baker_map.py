"""
Baker's Map Example

The baker's map is a classic chaotic dynamical system that demonstrates
stretching and folding behavior. This example evaluates the baker's map
over a discretized unit square.

Baker's map definition:
f(x,y) = (2x, y/2)         if 0 ≤ x < 1/2
f(x,y) = (2-2x, 1-y/2)     if 1/2 ≤ x < 1
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our new grid evaluation system
from ..src.grid import Grid
from ..src.evaluation import evaluate_grid
from ..src.data_utils import GridEvaluationResult
from ..src.config import config, get_default_grid_subdivisions


def baker_map(point: np.ndarray) -> np.ndarray:
    """
    Baker's map transformation.
    
    Args:
        point: 2D point [x, y] in unit square
        
    Returns:
        Transformed point [x', y']
    """
    x, y = point[0], point[1]
    
    # Ensure point is in unit square
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"Point {point} outside unit square [0,1]x[0,1]")
    
    if x < 0.5:
        # Left half: stretch horizontally, compress vertically
        return np.array([2 * x, y / 2])
    else:
        # Right half: fold and compress
        return np.array([2 - 2 * x, 1 - y / 2])


def run_baker_map_evaluation(
    subdivisions: list = None,
    sampling_mode: str = 'center',
    parallel: bool = True,
    save_results: bool = True,
    plot_results: bool = True
) -> GridEvaluationResult:
    """
    Run Baker's map evaluation over the unit square.
    
    Args:
        subdivisions: Grid subdivision in each dimension
        sampling_mode: Sampling strategy ('center', 'corners', 'random')
        parallel: Whether to use parallel processing
        save_results: Whether to save results to file
        plot_results: Whether to create visualization
        
    Returns:
        GridEvaluationResult object
    """
    # Use default subdivisions if not provided
    if subdivisions is None:
        subdivisions = list(get_default_grid_subdivisions())
    
    print("Baker's Map Grid Evaluation")
    print("=" * 40)
    print(f"Domain: [0,1] x [0,1]")
    print(f"Grid: {subdivisions[0]} x {subdivisions[1]} = {np.prod(subdivisions)} boxes")
    print(f"Sampling: {sampling_mode}")
    print(f"Parallel: {parallel}")
    print()
    
    # Create grid over unit square
    grid = Grid(bounds=[[0, 1], [0, 1]], subdivisions=subdivisions)
    
    print(f"Created grid: {grid}")
    
    # Define progress callback
    def progress_callback(completed: int, total: int):
        if completed % 100 == 0 or completed == total:
            percent = 100 * completed / total
            print(f"Progress: {completed}/{total} ({percent:.1f}%)")
    
    # Evaluate Baker's map over the grid
    print("\nEvaluating Baker's map...")
    results = evaluate_grid(
        grid=grid,
        function=baker_map,
        sampling_mode=sampling_mode,
        parallel=parallel,
        progress_callback=progress_callback
    )
    
    print(f"\nEvaluation complete! Processed {len(results)} boxes.")
    
    # Create result object
    result_obj = GridEvaluationResult(
        results=results,
        grid_bounds=[[0, 1], [0, 1]],
        grid_subdivisions=subdivisions,
        sampling_mode=sampling_mode,
        function_description="Baker's map: f(x,y) = (2x,y/2) if x<0.5, (2-2x,1-y/2) if x>=0.5",
        metadata={
            "example": "baker_map",
            "parallel": parallel,
            "domain": "unit_square"
        }
    )
    
    # Print summary
    summary = result_obj.summary()
    print(f"\nResults summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save results if requested
    if save_results:
        output_dir = Path(__file__).parent.parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        json_file = output_dir / f"baker_map_{subdivisions[0]}x{subdivisions[1]}_{sampling_mode}.json"
        result_obj.save_json(json_file)
        print(f"\nResults saved to: {json_file}")
    
    # Create visualization if requested
    if plot_results:
        create_baker_map_visualization(grid, results, subdivisions, sampling_mode)
    
    return result_obj


def create_baker_map_visualization(
    grid: Grid, 
    results: dict, 
    subdivisions: list,
    sampling_mode: str
):
    """
    Create visualization of Baker's map evaluation.
    
    Args:
        grid: Grid instance
        results: Evaluation results dictionary
        subdivisions: Grid subdivisions
        sampling_mode: Sampling mode used
    """
    print("\nCreating visualization...")
    
    # Extract original and transformed coordinates
    original_points = []
    transformed_points = []
    
    for box_idx, box_results in results.items():
        # Get original sample points
        sample_points = grid.get_sample_points(box_idx, sampling_mode)
        
        for i, transformed_point in enumerate(box_results):
            if isinstance(transformed_point, np.ndarray) and len(transformed_point) == 2:
                original_points.append(sample_points[i])
                transformed_points.append(transformed_point)
    
    original_points = np.array(original_points)
    transformed_points = np.array(transformed_points)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original domain with grid
    ax1.scatter(original_points[:, 0], original_points[:, 1], 
               c='blue', s=1, alpha=0.6, label='Sample points')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Original Domain\n{subdivisions[0]}×{subdivisions[1]} grid, {sampling_mode} sampling')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add grid lines
    for i in range(subdivisions[0] + 1):
        x = i / subdivisions[0]
        ax1.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for i in range(subdivisions[1] + 1):
        y = i / subdivisions[1]
        ax1.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Transformed domain
    ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], 
               c='red', s=1, alpha=0.6, label='Transformed points')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("x'")
    ax2.set_ylabel("y'")
    ax2.set_title("Baker's Map Output\nStretching and Folding")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"baker_map_{subdivisions[0]}x{subdivisions[1]}_{sampling_mode}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    
    plt.show()


def demonstrate_sampling_modes():
    """Demonstrate different sampling modes for Baker's map."""
    print("Demonstrating different sampling modes...")
    
    sampling_modes = ['center', 'corners', 'random']
    subdivisions = [20, 20]  # Smaller grid for demonstration
    
    for mode in sampling_modes:
        print(f"\n--- Sampling mode: {mode} ---")
        result = run_baker_map_evaluation(
            subdivisions=subdivisions,
            sampling_mode=mode,
            parallel=False,  # Sequential for consistent output
            save_results=False,
            plot_results=False
        )
        
        # Show a few sample transformations
        print("Sample transformations:")
        for i, (box_idx, box_results) in enumerate(list(result.results.items())[:5]):
            sample_points = Grid(bounds=[[0, 1], [0, 1]], subdivisions=subdivisions).get_sample_points(box_idx, mode)
            print(f"  Box {box_idx}: {sample_points[0]} -> {box_results[0]}")


if __name__ == "__main__":
    # Run the main Baker's map evaluation
    result = run_baker_map_evaluation(
        subdivisions=[50, 50],
        sampling_mode='center',
        parallel=True,
        save_results=True,
        plot_results=True
    )
    
    print("\n" + "="*50)
    print("Baker's Map evaluation completed successfully!")
    print(f"Processed {result.total_boxes} grid boxes.")
    print("Results saved and visualization created.")
    
    # Uncomment to demonstrate different sampling modes
    # demonstrate_sampling_modes()