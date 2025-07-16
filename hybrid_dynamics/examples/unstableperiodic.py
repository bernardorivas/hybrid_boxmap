"""
Unstable Periodic System Example

A simple hybrid system demonstrating unstable periodic behavior.
The system has linear dynamics with a periodic reset map.

System:
- State: [x, y]
- Continuous dynamics: x' = 1, y' = 0
- Jump condition: x = 1
- Reset map: (x, y) → (0, -y³ + 2y) (reset to left with cubic map)
"""

import numpy as np

from ..src.hybrid_system import HybridSystem


def ode_fun(t: float, state: np.ndarray, system) -> np.ndarray:
    """
    Continuous dynamics for unstable periodic system.

    Args:
        t: Time (unused, autonomous system)
        state: [x, y] (position coordinates)
        system: UnstablePeriodicSystem instance

    Returns:
        Time derivatives [x', y'] = [1, 0]
    """
    return np.array([1.0, 0.0])


def event_fun(t: float, state: np.ndarray, system) -> float:
    """
    Event function for boundary crossing detection.

    Args:
        t: Time (unused)
        state: [x, y]
        system: UnstablePeriodicSystem instance

    Returns:
        Distance to boundary (zero when x = 1)
    """
    x, y = state
    return 1.0 - x


def reset_map(state: np.ndarray, system) -> np.ndarray:
    """
    Reset map for boundary crossings.

    Args:
        state: [x, y] at boundary crossing
        system: UnstablePeriodicSystem instance

    Returns:
        Post-crossing state [0, -y³ + 2y]
    """
    x, y = state

    # Reset map: (x, y) → (0, -y³ + 2y)
    y_new = -(y**3) + 2 * y

    return np.array([0.0, y_new])


class UnstablePeriodicSystem:
    """Unstable periodic hybrid system."""

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]] = [(0.0, 1.0), (-1.0, 1.0)],
        max_jumps: int = 50,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        # Domain bounds [x, y]
        self.domain_bounds = domain_bounds

        # Integration parameters
        self.rtol = rtol
        self.atol = atol
        self.max_jumps = max_jumps

        # Create event function with parameters
        def event(t: float, state: np.ndarray) -> float:
            return event_fun(t, state, self)

        # Configure event function
        event.terminal = True
        event.direction = -1  # Detect when approaching boundary
        self.event_function = event

        # Create the hybrid system
        self.system = self._create_system()

    def _create_system(self) -> HybridSystem:
        """Create the hybrid system with closures capturing self."""

        def ode(t: float, state: np.ndarray) -> np.ndarray:
            return ode_fun(t, state, self)

        def reset(state: np.ndarray) -> np.ndarray:
            return reset_map(state, self)

        return HybridSystem(
            ode=ode,
            event_function=self.event_function,
            reset_map=reset,
            domain_bounds=self.domain_bounds,
            max_jumps=self.max_jumps,
            rtol=self.rtol,
            atol=self.atol,
        )

    def simulate(self, initial_state: np.ndarray, time_span: tuple[float, float]):
        """Simulate the unstable periodic system."""
        return self.system.simulate(initial_state, time_span, dense_output=True)
