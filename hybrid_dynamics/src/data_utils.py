"""
Results storage and serialization for grid evaluations.

This module provides utilities for saving, loading, and managing
results from grid-based function evaluations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


class GridEvaluationResult:
    """
    Container for grid evaluation results with metadata.

    Stores function evaluation results along with grid configuration
    and evaluation metadata for reproducibility.
    """

    def __init__(
        self,
        results: Dict[int, List[Union[float, np.ndarray]]],
        grid_bounds: List[List[float]],
        grid_subdivisions: List[int],
        sampling_mode: str = "center",
        num_points: int = 1,
        function_description: str = "",
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize evaluation results.

        Args:
            results: Dictionary mapping box_index -> function values
            grid_bounds: Grid domain bounds
            grid_subdivisions: Grid subdivision counts
            sampling_mode: Sampling strategy used
            num_points: Number of sample points per box
            function_description: Description of evaluated function
            metadata: Additional metadata dictionary
        """
        self.results = results
        self.grid_bounds = grid_bounds
        self.grid_subdivisions = grid_subdivisions
        self.sampling_mode = sampling_mode
        self.num_points = num_points
        self.function_description = function_description
        self.metadata = metadata or {}

        # Add timestamp if not provided
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

        # Calculate derived properties
        self.total_boxes = len(results)
        self.ndim = len(grid_bounds)

    def save_json(self, filepath: Union[str, Path]) -> None:
        """
        Save results as JSON file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)

        # Convert results to JSON-serializable format
        serializable_results = {}
        for box_idx, values in self.results.items():
            serializable_values = []
            for val in values:
                if isinstance(val, np.ndarray):
                    serializable_values.append(val.tolist())
                elif isinstance(val, (np.integer, np.floating)):
                    serializable_values.append(float(val))
                else:
                    serializable_values.append(val)
            serializable_results[str(box_idx)] = serializable_values

        # Create complete data structure
        data = {
            "metadata": {
                "timestamp": self.metadata.get("timestamp"),
                "function_description": self.function_description,
                "grid_bounds": self.grid_bounds,
                "grid_subdivisions": self.grid_subdivisions,
                "sampling_mode": self.sampling_mode,
                "num_points": self.num_points,
                "total_boxes": self.total_boxes,
                "ndim": self.ndim,
                **self.metadata,
            },
            "results": serializable_results,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save_numpy(self, filepath: Union[str, Path]) -> None:
        """
        Save results as NumPy compressed archive.

        Args:
            filepath: Path to save file (.npz extension)
        """
        filepath = Path(filepath)

        # Prepare metadata
        metadata = {
            "timestamp": self.metadata.get("timestamp"),
            "function_description": self.function_description,
            "grid_bounds": np.array(self.grid_bounds),
            "grid_subdivisions": np.array(self.grid_subdivisions),
            "sampling_mode": self.sampling_mode,
            "num_points": self.num_points,
            "total_boxes": self.total_boxes,
            "ndim": self.ndim,
            **self.metadata,
        }

        # Convert results to arrays where possible
        results_arrays = {}
        for box_idx, values in self.results.items():
            try:
                results_arrays[f"box_{box_idx}"] = np.array(values)
            except (ValueError, TypeError):
                # Store as object array if conversion fails
                results_arrays[f"box_{box_idx}"] = np.array(values, dtype=object)

        # Save everything
        np.savez_compressed(
            filepath, metadata=np.array([metadata], dtype=object), **results_arrays,
        )

    def save(self, filepath: Union[str, Path], format: str = "auto") -> None:
        """
        Save results with automatic format detection.

        Args:
            filepath: Path to save file
            format: Format to use ('json', 'numpy', or 'auto')
        """
        filepath = Path(filepath)

        if format == "auto":
            if filepath.suffix == ".json":
                format = "json"
            elif filepath.suffix in [".npz", ".npy"]:
                format = "numpy"
            else:
                format = "json"  # Default
                filepath = filepath.with_suffix(".json")

        if format == "json":
            self.save_json(filepath)
        elif format == "numpy":
            self.save_numpy(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> GridEvaluationResult:
        """
        Load results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            GridEvaluationResult instance
        """
        with open(filepath) as f:
            data = json.load(f)

        metadata = data["metadata"]

        # Reconstruct results
        results = {}
        for box_idx_str, values in data["results"].items():
            box_idx = int(box_idx_str)
            results[box_idx] = values

        return cls(
            results=results,
            grid_bounds=metadata["grid_bounds"],
            grid_subdivisions=metadata["grid_subdivisions"],
            sampling_mode=metadata["sampling_mode"],
            num_points=metadata["num_points"],
            function_description=metadata["function_description"],
            metadata={
                k: v
                for k, v in metadata.items()
                if k
                not in [
                    "grid_bounds",
                    "grid_subdivisions",
                    "sampling_mode",
                    "num_points",
                    "function_description",
                    "total_boxes",
                    "ndim",
                ]
            },
        )

    @classmethod
    def load_numpy(cls, filepath: Union[str, Path]) -> GridEvaluationResult:
        """
        Load results from NumPy archive.

        Args:
            filepath: Path to .npz file

        Returns:
            GridEvaluationResult instance
        """
        data = np.load(filepath, allow_pickle=True)
        metadata = data["metadata"].item()

        # Reconstruct results
        results = {}
        for key in data.files:
            if key.startswith("box_"):
                box_idx = int(key[4:])
                results[box_idx] = data[key].tolist()

        return cls(
            results=results,
            grid_bounds=metadata["grid_bounds"].tolist(),
            grid_subdivisions=metadata["grid_subdivisions"].tolist(),
            sampling_mode=metadata["sampling_mode"],
            num_points=metadata["num_points"],
            function_description=metadata["function_description"],
            metadata={
                k: v
                for k, v in metadata.items()
                if k
                not in [
                    "grid_bounds",
                    "grid_subdivisions",
                    "sampling_mode",
                    "num_points",
                    "function_description",
                    "total_boxes",
                    "ndim",
                ]
            },
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> GridEvaluationResult:
        """
        Load results with automatic format detection.

        Args:
            filepath: Path to results file

        Returns:
            GridEvaluationResult instance
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            return cls.load_json(filepath)
        if filepath.suffix in [".npz", ".npy"]:
            return cls.load_numpy(filepath)
        raise ValueError(f"Unknown file format: {filepath.suffix}")

    def get_box_results(self, box_index: int) -> List[Union[float, np.ndarray]]:
        """Get results for a specific box."""
        return self.results.get(box_index, [])

    def get_all_values(self) -> List[Union[float, np.ndarray]]:
        """Get all function values as flat list."""
        all_values = []
        for box_results in self.results.values():
            all_values.extend(box_results)
        return all_values

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the results."""
        all_values = []
        for box_results in self.results.values():
            for val in box_results:
                if isinstance(val, (int, float, np.number)) and not np.isnan(val):
                    all_values.append(float(val))

        if not all_values:
            return {"error": "No numeric values found"}

        all_values = np.array(all_values)

        return {
            "total_boxes": self.total_boxes,
            "total_evaluations": sum(len(vals) for vals in self.results.values()),
            "numeric_values": len(all_values),
            "min_value": float(np.min(all_values)),
            "max_value": float(np.max(all_values)),
            "mean_value": float(np.mean(all_values)),
            "std_value": float(np.std(all_values)),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GridEvaluationResult(boxes={self.total_boxes}, "
            f"ndim={self.ndim}, mode='{self.sampling_mode}')"
        )
