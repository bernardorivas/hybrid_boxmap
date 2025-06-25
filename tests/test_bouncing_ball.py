#!/usr/bin/env python3
"""
Bouncing Ball Demo: SCC Analysis and Visualization

Generates visualizations of strongly connected components and condensation graph
for the bouncing ball hybrid dynamical system.
"""
import contextlib
import hashlib
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_dynamics import (
    Grid,
    HybridBoxMap,
    HybridPlotter,
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    visualize_box_map_entry,
)
from hybrid_dynamics.examples.bouncing_ball import BouncingBall


def get_config_hash(ball_params, grid_bounds, grid_subdivisions, tau):
    """Create a hash of the configuration for cache validation."""
    config_str = f"{ball_params}_{grid_bounds}_{grid_subdivisions}_{tau}"
    return hashlib.md5(config_str.encode()).hexdigest()


def get_next_run_dir(base_dir: Path, tau: float, subdivisions: list) -> Path:
    """Create the next available run directory with descriptive naming."""
    tau_str = f"{tau:.2f}".replace(".", "")  # 0.5 -> "05"
    subdiv_str = "_".join(map(str, subdivisions))  # [100, 100] -> "100_100"

    base_name = f"run_tau_{tau_str}_subdiv_{subdiv_str}"

    # Find existing runs with this pattern
    existing_runs = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(base_name):
                try:
                    # Extract run number from pattern like run_tau_05_subdiv_100_100_001
                    run_num = int(d.name.split("_")[-1])
                    existing_runs.append(run_num)
                except (ValueError, IndexError):
                    continue

    next_run_num = max(existing_runs) + 1 if existing_runs else 1
    run_dir = base_dir / f"{base_name}_{next_run_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def create_phase_portrait(system, domain_bounds, output_path):
    """Create a phase portrait for the hybrid system."""
    plotter = HybridPlotter()

    # Create some representative trajectories for bouncing ball
    initial_conditions = [
        [1.0, 0.0],   # Drop from height 1.0
        [1.5, 1.0],   # Start with upward velocity
        [0.5, -1.0],  # Start with downward velocity
        [2.0, 0.5],    # Drop from height 2.0 with small initial velocity
    ]

    trajectories = []
    for ic in initial_conditions:
        try:
            traj = system.simulate(ic, (0.0, 1.0), max_step=0.01,max_jumps=50, dense_output=True)
            if traj.segments:  # Check if trajectory has segments
                trajectories.append(traj)
            else:
                pass
        except Exception:
            pass

    if not trajectories:
        return

    # Create phase portrait
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    plotter.plot_phase_portrait(
        system=system,
        trajectories=trajectories,
        ax=ax,
        show_vector_field=False,
        colors=["blue", "red", "green", "orange"],
    )

    ax.set_xlabel("Height h (m)")
    ax.set_ylabel("Velocity v (m/s)")
    ax.set_title("Bouncing Ball Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set domain bounds if available
    if domain_bounds:
        ax.set_xlim(domain_bounds[0])
        ax.set_ylim(domain_bounds[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_sample_box_map_entries(box_map, grid, output_dir, num_samples=5):
    """Save detailed visualizations of a random sample of box map entries."""

    if not box_map:
        return

    # Get a random sample of source boxes
    source_boxes = list(box_map.keys())
    num_to_sample = min(num_samples, len(source_boxes))

    if num_to_sample == 0:
        return

    sample_source_indices = random.sample(source_boxes, num_to_sample)

    # Generate visualizations for each sample
    for i, source_box in enumerate(sample_source_indices):
        destinations = box_map.get(source_box, set())

        # Get corner points for the source box
        initial_points = grid.get_sample_points(source_box, "corners")

        # For visualization, we create plausible final points by taking the center of destination boxes.
        # In a real analysis, these points would come from the flow map evaluation.
        final_points = []
        if destinations:
            dest_centers = [grid.get_sample_points(dest_box, mode="center")[0] for dest_box in destinations]
            dest_centers = np.array(dest_centers)
            # To match the number of initial points, we sample with replacement from the destination centers.
            indices = np.random.choice(len(dest_centers), len(initial_points))  # noqa: NPY002
            final_points = dest_centers[indices]
        else:
            # If a source box has no destinations, pass an empty array for final points.
            final_points = np.array([])

        output_path = output_dir / f"box_map_entry_sample_{i+1}_box_{source_box}.png"

        grid.get_sample_points(source_box, mode="center")[0]

        with contextlib.suppress(Exception):
            visualize_box_map_entry(
                grid=grid,
                source_box_index=source_box,
                destination_indices=set(destinations),
                initial_points=initial_points,
                final_points=final_points,
                output_path=str(output_path),
            )


def print_zero_box_evaluation(box_map, grid):
    """
    Prints the evaluation results for the corners of the box containing (0,0).
    """
    try:
        zero_box_index = grid.get_box_from_point([0.0, 0.0])
        print(f"--- Evaluation for Box {zero_box_index} containing (0,0) ---")

        corners = grid.get_sample_points(zero_box_index, "corners")
        results = box_map.raw_flow_data.get(zero_box_index)

        if results is None:
            print("No evaluation data found for this box.")
            return

        print(f"{'Corner Point':<25} -> {'Final State':<25} | {'Jumps'}")
        print("-" * 60)

        for i, corner in enumerate(corners):
            final_state, num_jumps = results[i]
            corner_str = np.array2string(corner, precision=3, floatmode='fixed')
            final_str = np.array2string(final_state, precision=3, floatmode='fixed')
            print(f"{corner_str:<25} -> {final_str:<25} | {num_jumps}")
        print("-" * 60)

    except (ValueError, IndexError):
        print("Could not find or print evaluation for the box containing (0,0).")


def plot_zero_box_evaluation(box_map, grid, output_dir):
    """
    Find the box containing the origin (0,0) and plot its box map evaluation.
    """
    try:
        zero_box_index = grid.get_box_from_point([0.0, 0.0])
    except ValueError:
        print("Warning: Point (0,0) is outside the grid bounds and cannot be plotted.")
        return

    try:
        destinations = box_map.get(zero_box_index, set())
        initial_points = grid.get_sample_points(zero_box_index, "corners")
        final_points = np.array([res[0] for res in box_map.raw_flow_data.get(zero_box_index, [])])

        output_path = output_dir / f"box_map_entry_sample_zero_box_{zero_box_index}.png"

        visualize_box_map_entry(
            grid=grid,
            source_box_index=zero_box_index,
            destination_indices=set(destinations),
            initial_points=initial_points,
            final_points=final_points,
            output_path=str(output_path),
        )
        print(f"✓ Zero-box plot saved to {output_path}")

    except Exception as e:
        print(f"Error creating zero-box plot for box {zero_box_index}: {e}")


def run_bouncing_ball_demo():
    """Bouncing ball SCC analysis with strongly connected components visualization."""

    # Setup paths
    data_dir = Path("data/bouncing_ball")
    data_dir.mkdir(exist_ok=True, parents=True)
    figures_base_dir = Path("figures/bouncing_ball")
    figures_base_dir.mkdir(exist_ok=True, parents=True)

    box_map_file = data_dir / "bouncing_ball_box_map.pkl"

    # 1. Define configuration
    ball = BouncingBall(
        domain_bounds=[(0.0, 2.0), (-5.0, 5.0)],
        g=9.81,
        c=0.5,
        max_jumps=50,
    )
    grid = Grid(bounds=ball.domain_bounds, subdivisions=[99, 99])
    tau = 1.0

    # Create run directory
    run_dir = get_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    ball_params = f"g={ball.g}_c={ball.c}_max_jumps={ball.max_jumps}"
    current_config_hash = get_config_hash(
        ball_params,
        ball.domain_bounds,
        grid.subdivisions.tolist(),
        tau,
    )

    # 2. Check if we can load from cache
    load_from_cache = False
    if box_map_file.exists() and False: # Force re-computation to get raw_flow_data
        try:
            with open(box_map_file, "rb") as f:
                box_map_data = pickle.load(f)

            # Check if configuration matches
            saved_config_hash = box_map_data.get("config_hash", "")
            if saved_config_hash == current_config_hash:
                load_from_cache = True
            else:
                pass
        except Exception:
            pass

    # 3. Load from cache or compute new
    if load_from_cache:
        # Reconstruct HybridBoxMap from saved data
        box_map = HybridBoxMap(grid, ball.system, tau)
        box_map.update(box_map_data["mapping"])
        box_map.metadata = box_map_data["metadata"]
    else:
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=ball.system,
            tau=tau,
        )

        # Save mapping data with configuration hash
        box_map_data = {
            "mapping": dict(box_map),  # The actual box mapping
            "metadata": box_map.metadata,
            "config_hash": current_config_hash,
            "config_details": {
                "ball_params": ball_params,
                "domain_bounds": ball.domain_bounds,
                "grid_subdivisions": grid.subdivisions.tolist(),
                "tau": tau,
            },
        }

        with open(box_map_file, "wb") as f:  # noqa: PTH123
            pickle.dump(box_map_data, f)

    # # 4. Analyze the box map
    # analyze_box_map(box_map, grid, run_dir)

    # Print the detailed evaluation for the box at (0,0)
    print_zero_box_evaluation(box_map, grid)

    # 5. Save sample box map entries for detailed inspection
    save_sample_box_map_entries(box_map, grid, run_dir, num_samples=5)

    # Plot the evaluation for the box at (0,0)
    plot_zero_box_evaluation(box_map, grid, run_dir)

    # 6. Convert to NetworkX
    graph = box_map.to_networkx()

    # 7. Find recurrent components and the Morse graph
    morse_graph, morse_sets = create_morse_graph(graph)

    # 8. Generate visualizations

    # Plot 1: Phase portrait
    create_phase_portrait(ball.system, ball.domain_bounds, str(run_dir / "phase_portrait.png"))

    # Plot 2: The boxes of the recurrent components on the state space grid
    plot_morse_sets_on_grid(grid, morse_sets, str(run_dir / "morse_sets.png"))

    # Plot 3: The Morse graph
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))


if __name__ == "__main__":
    run_bouncing_ball_demo()
