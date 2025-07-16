"""
Utility functions for demo scripts and analysis workflows.

This module consolidates common functionality used across demo scripts,
including configuration hashing, run directory management, and visualization utilities.
"""

import hashlib
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .plot_utils import visualize_box_map_entry
from .print_utils import vprint


def timed_operation(description: str):
    """
    Decorator to time function execution and print the duration.

    Args:
        description: Description of the operation being timed (e.g., "Box map computation")

    Example:
        @timed_operation("Visualization")
        def create_plot():
            # plotting code
            pass

        # Will print: "Visualization computed in X.XX seconds"
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"{description} computed in {duration:.2f} seconds")
            return result

        return wrapper

    return decorator


def create_config_hash(
    params_dict: Dict[str, Any],
    grid_bounds: List[Tuple[float, float]],
    grid_subdivisions: List[int],
    tau: float,
) -> str:
    """
    Create a hash of the configuration for cache validation.

    Args:
        params_dict: Dictionary of system parameters (e.g., {'g': 9.81, 'c': 0.5})
        grid_bounds: Domain bounds for the grid
        grid_subdivisions: Grid subdivision counts
        tau: Time horizon parameter

    Returns:
        MD5 hash string of the configuration
    """
    # Convert params_dict to a sorted string representation for consistency
    params_str = "_".join(f"{k}={v}" for k, v in sorted(params_dict.items()))
    config_str = f"{params_str}_{grid_bounds}_{grid_subdivisions}_{tau}"
    return hashlib.md5(config_str.encode()).hexdigest()


def create_next_run_dir(
    base_dir: Path,
    tau: float,
    subdivisions: List[int],
    bloat_factor: Optional[float] = None,
) -> Path:
    """
    Create the next available run directory with descriptive naming.

    Args:
        base_dir: Base directory for run outputs
        tau: Time horizon parameter for directory naming
        subdivisions: Grid subdivisions for directory naming
        bloat_factor: Optional bloat factor to include in directory name

    Returns:
        Path to the created run directory
    """
    tau_str = f"{tau:.2f}".replace(".", "")  # 0.5 -> "05"
    subdiv_str = "_".join(map(str, subdivisions))  # [100, 100] -> "100_100"

    # Include bloat factor in name if provided
    if bloat_factor is not None:
        bloat_percent = int(round(bloat_factor * 100))
        base_name = f"run_tau_{tau_str}_subdiv_{subdiv_str}_bloat_{bloat_percent:02d}"
    else:
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


def plot_box_containing_point(
    box_map: "HybridBoxMap",
    grid: "Grid",
    point: List[float],
    output_dir: Path,
    filename_prefix: str = "box_map_entry",
) -> Optional[Path]:
    """
    Find the box containing a given point and plot its box map evaluation.

    Args:
        box_map: HybridBoxMap instance
        grid: Grid instance
        point: Point coordinates to find containing box for
        output_dir: Directory to save the output plot
        filename_prefix: Prefix for the output filename

    Returns:
        Path to the saved plot, or None if plotting failed
    """
    try:
        box_index = grid.get_box_from_point(point)
    except ValueError:
        vprint(
            f"Warning: Point {point} is outside the grid bounds and cannot be plotted.",
            level="always",
        )
        return None

    try:
        destinations = box_map.get(box_index, set())
        initial_points = grid.get_sample_points(box_index, "corners")

        # Ensure raw_flow_data exists before trying to access it
        if hasattr(box_map, "raw_flow_data") and box_map.raw_flow_data.get(box_index):
            final_points = np.array(
                [res[0] for res in box_map.raw_flow_data.get(box_index, [])],
            )
        else:
            # Fallback if raw_flow_data is not available
            final_points = np.array([])

        point_str = "_".join(f"{coord:.3g}" for coord in point)
        output_path = (
            output_dir / f"{filename_prefix}_point_{point_str}_box_{box_index}.png"
        )

        visualize_box_map_entry(
            grid=grid,
            source_box_index=box_index,
            destination_indices=set(destinations),
            initial_points=initial_points,
            final_points=final_points,
            output_path=str(output_path),
        )
        vprint(f"✓ Box map plot saved to {output_path}", level="always")
        return output_path

    except Exception as e:
        vprint(
            f"Error creating box map plot for point {point}, box {box_index}: {e}",
            level="always",
        )
        return None


def setup_demo_directories(demo_name: str) -> Tuple[Path, Path]:
    """
    Set up standard data and figures directories for a demo.

    Args:
        demo_name: Name of the demo (e.g., 'bouncing_ball', 'rimless_wheel')

    Returns:
        Tuple of (data_dir, figures_base_dir) paths
    """
    data_dir = Path(f"data/{demo_name}")
    data_dir.mkdir(exist_ok=True, parents=True)

    figures_base_dir = Path(f"figures/{demo_name}")
    figures_base_dir.mkdir(exist_ok=True, parents=True)

    return data_dir, figures_base_dir


def save_box_map_with_config(
    box_map: "HybridBoxMap",
    config_hash: str,
    config_details: Dict[str, Any],
    file_path: Path,
) -> None:
    """
    Save a box map with its configuration for caching.

    Args:
        box_map: HybridBoxMap to save
        config_hash: Configuration hash for validation
        config_details: Detailed configuration information
        file_path: Path to save the pickle file
    """
    import pickle

    box_map_data = {
        "mapping": dict(box_map),  # The actual box mapping
        "metadata": box_map.metadata,
        "config_hash": config_hash,
        "config_details": config_details,
    }

    with open(file_path, "wb") as f:
        pickle.dump(box_map_data, f)


def load_box_map_from_cache(
    grid: "Grid",
    system: "HybridSystem",
    tau: float,
    file_path: Path,
    expected_hash: str,
) -> Optional["HybridBoxMap"]:
    """
    Load a box map from cache if configuration matches.

    Args:
        grid: Grid instance for reconstruction
        system: HybridSystem instance for reconstruction
        tau: Time horizon parameter
        file_path: Path to the cached pickle file
        expected_hash: Expected configuration hash

    Returns:
        Loaded HybridBoxMap if valid, None if cache miss or invalid
    """
    import pickle

    from .hybrid_boxmap import HybridBoxMap

    if not file_path.exists():
        return None

    try:
        with open(file_path, "rb") as f:
            box_map_data = pickle.load(f)

        # Check if configuration matches
        saved_config_hash = box_map_data.get("config_hash", "")
        if saved_config_hash != expected_hash:
            vprint("New configuration detected, recomputing box map", level="always")
            return None

        vprint("Previous configuration matches, loading from file", level="always")

        # Reconstruct HybridBoxMap from saved data
        box_map = HybridBoxMap(grid, system, tau)
        box_map.update(box_map_data["mapping"])
        box_map.metadata = box_map_data["metadata"]

        return box_map

    except Exception as e:
        vprint(f"⚠ Error loading cache: {e}, will recompute", level="always")
        return None


def create_progress_callback(update_interval: int = 100):
    """
    Create a progress callback function for box map computation.

    Args:
        update_interval: Update progress every N boxes (default: 100)

    Returns:
        Callable progress callback function
    """
    import time

    start_time = time.time()
    last_update = 0

    def progress_callback(completed: int, total: int):
        nonlocal last_update

        # Calculate update threshold (every 1% or every update_interval boxes, whichever is less frequent)
        update_threshold = max(update_interval, total // 100)

        # Only update if we've crossed the threshold or if we're at the end
        if completed - last_update >= update_threshold or completed == total:
            elapsed = time.time() - start_time
            percent = 100 * completed / total

            # Create ASCII progress bar
            bar_width = 20
            filled = int(bar_width * completed / total)
            bar = "■" * filled + "□" * (bar_width - filled)

            # Estimate time remaining
            if completed > 0:
                eta_seconds = elapsed * (total - completed) / completed
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."

            # Print progress
            print(
                f"\rProgress: {completed}/{total} ({percent:.1f}%) [{bar}] ETA: {eta_str}",
                end="",
                flush=True,
            )

            # Print newline when complete
            if completed == total:
                print()  # New line when done

            last_update = completed

    return progress_callback
