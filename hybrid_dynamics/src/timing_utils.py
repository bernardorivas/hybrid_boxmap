"""
Timing and performance analysis utilities for hybrid dynamics library.

This module provides consistent timing measurement, memory tracking, and
performance analysis tools for identifying computational bottlenecks.
"""

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import psutil


@dataclass
class TimingResult:
    """Container for timing measurement results."""

    name: str
    duration: float
    start_time: float
    end_time: float
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_delta(self) -> Optional[float]:
        """Memory increase during operation (MB)."""
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None

    def __str__(self) -> str:
        result = f"{self.name}: {self.duration:.3f}s"
        if self.memory_delta is not None:
            result += f" (Δmem: {self.memory_delta:+.1f}MB)"
        return result


class PerformanceProfiler:
    """Comprehensive performance profiler for timing and memory analysis."""

    def __init__(self, name: str = "Performance Profile"):
        self.name = name
        self.results: List[TimingResult] = []
        self.active_timers: Dict[str, float] = {}
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    @contextmanager
    def timer(self, name: str, track_memory: bool = True, **metadata):
        """Context manager for timing operations with optional memory tracking."""
        memory_before = self.get_memory_usage() if track_memory else None
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            memory_after = self.get_memory_usage() if track_memory else None

            result = TimingResult(
                name=name,
                duration=end_time - start_time,
                start_time=start_time,
                end_time=end_time,
                memory_before=memory_before,
                memory_after=memory_after,
                metadata=metadata,
            )
            self.results.append(result)

    def start_timer(self, name: str) -> None:
        """Start a named timer (for non-context-manager usage)."""
        self.active_timers[name] = time.time()

    def end_timer(
        self, name: str, track_memory: bool = True, **metadata,
    ) -> TimingResult:
        """End a named timer and record results."""
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        start_time = self.active_timers.pop(name)
        end_time = time.time()
        memory_after = self.get_memory_usage() if track_memory else None

        result = TimingResult(
            name=name,
            duration=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
            memory_after=memory_after,
            metadata=metadata,
        )
        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.results:
            return {"total_operations": 0, "total_time": 0.0}

        total_time = sum(r.duration for r in self.results)
        memory_results = [r for r in self.results if r.memory_delta is not None]

        summary = {
            "total_operations": len(self.results),
            "total_time": total_time,
            "average_time": total_time / len(self.results),
            "slowest_operation": max(self.results, key=lambda x: x.duration),
            "fastest_operation": min(self.results, key=lambda x: x.duration),
        }

        if memory_results:
            total_memory_delta = sum(r.memory_delta for r in memory_results)
            summary.update(
                {
                    "total_memory_delta": total_memory_delta,
                    "average_memory_delta": total_memory_delta / len(memory_results),
                    "max_memory_delta": max(r.memory_delta for r in memory_results),
                },
            )

        return summary

    def print_summary(self, detailed: bool = False) -> None:
        """Print performance summary to stdout."""
        print(f"\n=== {self.name} ===")

        if not self.results:
            print("No timing results recorded.")
            return

        summary = self.get_summary()
        print(f"Total operations: {summary['total_operations']}")
        print(f"Total time: {summary['total_time']:.3f}s")
        print(f"Average time: {summary['average_time']:.3f}s")
        print(f"Slowest: {summary['slowest_operation']}")
        print(f"Fastest: {summary['fastest_operation']}")

        if "total_memory_delta" in summary:
            print(f"Total memory delta: {summary['total_memory_delta']:+.1f}MB")
            print(f"Average memory delta: {summary['average_memory_delta']:+.1f}MB")
            print(f"Max memory delta: {summary['max_memory_delta']:+.1f}MB")

        if detailed:
            print("\nDetailed Results:")
            for result in self.results:
                print(f"  {result}")

    def save_results(self, filepath: str) -> None:
        """Save timing results to text file."""
        with open(filepath, "w") as f:
            f.write(f"Performance Analysis: {self.name}\n")
            f.write("=" * 60 + "\n\n")

            # Summary section
            summary = self.get_summary()
            f.write("SUMMARY:\n")
            f.write(f"  Total operations: {summary['total_operations']}\n")
            f.write(f"  Total time: {summary['total_time']:.3f}s\n")
            f.write(f"  Average time: {summary['average_time']:.3f}s\n")

            if "total_memory_delta" in summary:
                f.write(
                    f"  Total memory delta: {summary['total_memory_delta']:+.1f}MB\n",
                )
                f.write(
                    f"  Average memory delta: {summary['average_memory_delta']:+.1f}MB\n",
                )
                f.write(f"  Max memory delta: {summary['max_memory_delta']:+.1f}MB\n")

            f.write(f"\nSlowest operation: {summary['slowest_operation']}\n")
            f.write(f"Fastest operation: {summary['fastest_operation']}\n\n")

            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")

            for result in self.results:
                f.write(f"Operation: {result.name}\n")
                f.write(f"  Duration: {result.duration:.3f}s\n")
                f.write(f"  Start time: {result.start_time:.3f}\n")
                f.write(f"  End time: {result.end_time:.3f}\n")

                if result.memory_before is not None:
                    f.write(f"  Memory before: {result.memory_before:.1f}MB\n")
                if result.memory_after is not None:
                    f.write(f"  Memory after: {result.memory_after:.1f}MB\n")
                if result.memory_delta is not None:
                    f.write(f"  Memory delta: {result.memory_delta:+.1f}MB\n")

                if result.metadata:
                    f.write("  Metadata:\n")
                    for key, value in result.metadata.items():
                        f.write(f"    {key}: {value}\n")

                f.write("\n")

        print(f"Timing results saved to {filepath}")


@contextmanager
def simple_timer(name: str, print_result: bool = True):
    """Simple context manager timer for quick measurements."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if print_result:
            print(f"{name}: {duration:.3f}s")


def time_function(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Time a function call and return (result, duration)."""
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time
    return result, duration


def estimate_grid_timing(
    grid_size: int, sample_time: float, operation_name: str = "operation",
) -> None:
    """Estimate total time based on sample timing and grid size."""
    total_estimate = sample_time * grid_size

    print(f"\nTiming Estimate for {operation_name}:")
    print(f"  Sample time per box: {sample_time:.6f}s")
    print(f"  Grid size: {grid_size:,} boxes")
    print(
        f"  Estimated total time: {total_estimate:.1f}s ({total_estimate/60:.1f} minutes)",
    )

    if total_estimate > 3600:
        print(f"  WARNING: Estimated time > 1 hour ({total_estimate/3600:.1f} hours)")


class ProgressTimer:
    """Timer with progress tracking for long-running operations."""

    def __init__(
        self, total_items: int, name: str = "Progress", update_interval: int = 100,
    ):
        self.total_items = total_items
        self.name = name
        self.update_interval = update_interval
        self.start_time = time.time()
        self.completed = 0
        self.last_update = 0

    def update(self, completed: int = None) -> None:
        """Update progress counter."""
        if completed is not None:
            self.completed = completed
        else:
            self.completed += 1

        # Only print updates at specified intervals
        if (
            self.completed - self.last_update >= self.update_interval
            or self.completed == self.total_items
        ):
            self._print_progress()
            self.last_update = self.completed

    def _print_progress(self) -> None:
        """Print current progress with timing estimates."""
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed / self.total_items) * 100

        if self.completed > 0:
            avg_time_per_item = elapsed / self.completed
            remaining_items = self.total_items - self.completed
            eta = avg_time_per_item * remaining_items

            print(
                f"{self.name}: {self.completed:,}/{self.total_items:,} ({progress_pct:.1f}%) "
                f"- {elapsed:.1f}s elapsed, ETA: {eta:.1f}s ({avg_time_per_item:.3f}s/item)",
            )
        else:
            print(
                f"{self.name}: {self.completed:,}/{self.total_items:,} ({progress_pct:.1f}%)",
            )


def force_garbage_collection() -> float:
    """Force garbage collection and return memory freed (MB)."""
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
    gc.collect()
    memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
    return memory_before - memory_after
