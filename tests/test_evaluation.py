#!/usr/bin/env python3
"""
Test script for evaluation.py using the rimless wheel example.

This script demonstrates how to use the evaluation module to evaluate
the flow map of the rimless wheel hybrid system over a grid.
The flow map is f(x) = phi(tau, x), where phi is the trajectory.
"""

import numpy as np
import time
from pathlib import Path
import pickle
import warnings

from hybrid_dynamics import Grid
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.evaluation import evaluate_grid
from hybrid_dynamics import visualize_flow_map

# Global wheel instance for sequential processing
_wheel = None
_tau = 0.2

def evaluate_flow(point: np.ndarray) -> np.ndarray:
    """
    Evaluates the map f(x) = phi(tau, x), where phi is the flow of the
    hybrid system. The state is 2-dimensional for RimlessWheel.
    """
    try:
        # Simulate for a fixed time tau
        traj = _wheel.simulate(point, (0, _tau))
        
        # Try to interpolate at time tau. If tau is outside the trajectory's
        # time domain, it will raise a ValueError, which is caught below.
        return traj.interpolate(_tau)
            
    except ValueError:
        # This will catch interpolation errors if tau is out of bounds
        return np.full(len(_wheel.domain_bounds), np.nan)


def test_rimless_wheel_flow_evaluation():
    """Test evaluating the flow map of the rimless wheel."""
    global _wheel
    print("Evaluating flow map for the rimless wheel...")
    
    # Create rimless wheel system
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    _wheel = wheel
    
    # Create a grid over the domain
    bounds = wheel.domain_bounds
    subdivisions = [50, 50]
    
    print(f"Creating grid with bounds: {bounds}")
    print(f"Subdivisions: {subdivisions}")
    
    grid = Grid(bounds, subdivisions)
    print(f"Grid created with {grid.total_boxes} boxes")
    
    # Evaluate the flow map on the corners of each box
    print("\nEvaluating flow map on grid corners (sequentially)...")
    start_time = time.time()
    
    flow_map_data = evaluate_grid(
        grid=grid,
        function=evaluate_flow,
        sampling_mode='corners',
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.3f} seconds")
    
    # Post-process and save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "rimless_wheel_flow_map.pkl"
    
    results_to_save = {
        'grid': grid,
        'flow_map_data': flow_map_data,
        'tau': _tau,
        'eval_time': eval_time
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(results_to_save, f)
        
    print(f"\nResults saved to {output_path}")

    # Visualize the results
    visualization_path = output_dir / "rimless_wheel_flow_map.png"
    visualize_flow_map(output_path, visualization_path)

    # Print some statistics
    valid_results = 0
    total_points = 0
    if flow_map_data:
        for box_idx in grid.box_indices:
            results_for_box = flow_map_data.get(box_idx)
            if results_for_box is not None:
                total_points += len(results_for_box)
                for result_vector in results_for_box:
                    if not np.isnan(result_vector).any():
                        valid_results += 1
    
    if total_points > 0:
        print(f"  {valid_results} / {total_points} points evaluated successfully.")
    else:
        print("  No points were evaluated.")

    
    print("\n✓ Flow map evaluation completed successfully!")
    return results_to_save

if __name__ == "__main__":
    test_rimless_wheel_flow_evaluation()