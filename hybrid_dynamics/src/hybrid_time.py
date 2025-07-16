"""
Hybrid time domain data structures for tracking continuous and discrete time.

This module provides classes for representing points and intervals in hybrid time,
which consists of continuous time t ∈ ℝ≥0 and discrete jump index j ∈ ℕ.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HybridTime:
    """Represents a point in hybrid time (t, j).

    Attributes:
        continuous: Continuous time t ∈ ℝ≥0
        discrete: Discrete jump index j ∈ ℕ
    """

    continuous: float
    discrete: int

    def __post_init__(self):
        """Validate hybrid time constraints."""
        if self.continuous < 0:
            raise ValueError("Continuous time must be non-negative")
        if self.discrete < 0:
            raise ValueError("Discrete index must be non-negative")

    def __str__(self) -> str:
        """String representation of hybrid time."""
        return f"({self.continuous:.3f}, {self.discrete})"

    def __repr__(self) -> str:
        """Detailed representation of hybrid time."""
        return f"HybridTime(continuous={self.continuous}, discrete={self.discrete})"

    def __eq__(self, other: "HybridTime") -> bool:
        """Check equality with tolerance for continuous time."""
        if not isinstance(other, HybridTime):
            return False
        return (
            np.isclose(self.continuous, other.continuous, rtol=1e-10)
            and self.discrete == other.discrete
        )

    def __lt__(self, other: "HybridTime") -> bool:
        """Hybrid time ordering: (t1, j1) < (t2, j2) if j1 < j2 or (j1 = j2 and t1 < t2)."""
        if not isinstance(other, HybridTime):
            raise TypeError("Cannot compare HybridTime with non-HybridTime")

        if self.discrete != other.discrete:
            return self.discrete < other.discrete
        return self.continuous < other.continuous

    def __le__(self, other: "HybridTime") -> bool:
        """Less than or equal for hybrid time."""
        return self < other or self == other


@dataclass
class HybridTimeInterval:
    """Represents an interval in hybrid time domain.

    An interval corresponds to a continuous time interval [t_start, t_end]
    at a fixed jump index j.

    Attributes:
        t_start: Start of continuous time interval
        t_end: End of continuous time interval
        jump_index: Discrete jump index for this interval
    """

    t_start: float
    t_end: float
    jump_index: int

    def __post_init__(self):
        """Validate interval constraints."""
        if self.t_start < 0 or self.t_end < 0:
            raise ValueError("Time values must be non-negative")
        if self.t_start > self.t_end:
            raise ValueError("Start time must be <= end time")
        if self.jump_index < 0:
            raise ValueError("Jump index must be non-negative")

    def contains(self, t: float) -> bool:
        """Check if continuous time t is within this interval."""
        return self.t_start <= t <= self.t_end

    def contains_hybrid_time(self, ht: HybridTime) -> bool:
        """Check if hybrid time point is within this interval."""
        return ht.discrete == self.jump_index and self.contains(ht.continuous)

    def duration(self) -> float:
        """Get the duration of this time interval."""
        return self.t_end - self.t_start

    def to_notation(self) -> str:
        """Returns interval in mathematical notation."""
        return f"[{self.t_start:.3f}, {self.t_end:.3f}] × {{{self.jump_index}}}"

    def __str__(self) -> str:
        """String representation using mathematical notation."""
        return self.to_notation()

    def __repr__(self) -> str:
        """Detailed representation of hybrid time interval."""
        return f"HybridTimeInterval(t_start={self.t_start}, t_end={self.t_end}, jump_index={self.jump_index})"

    def overlaps(self, other: "HybridTimeInterval") -> bool:
        """Check if this interval overlaps with another."""
        if self.jump_index != other.jump_index:
            return False
        return not (self.t_end < other.t_start or other.t_end < self.t_start)

    @staticmethod
    def union_notation(intervals: List["HybridTimeInterval"]) -> str:
        """Create union notation for a list of intervals."""
        if not intervals:
            return "∅"

        sorted_intervals = sorted(intervals, key=lambda x: (x.jump_index, x.t_start))
        notation_parts = [interval.to_notation() for interval in sorted_intervals]
        return " ∪ ".join(notation_parts)
