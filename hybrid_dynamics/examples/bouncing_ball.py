"""
Bouncing Ball Example

A ball bouncing under gravity with energy loss at each impact.

System:
- State: [h, v] (height, velocity)
- Continuous dynamics: h' = v, v' = -g
- Jump condition: h = 0 and v < 0 (ball hits ground while falling)
- Reset map: [h, v] → [0, -c*v] (energy loss with coefficient of restitution c)
"""

import numpy as np

from ..src.hybrid_system import HybridSystem


def ode_fun(t: float, state: np.ndarray, ball) -> np.ndarray:
    """
    Continuous dynamics: h' = v, v' = -g

    Args:
        t: Time (unused, autonomous system)
        state: [height, velocity]
        ball: BouncingBall instance with parameters

    Returns:
        Time derivatives [dh/dt, dv/dt] = [v, -g]
    """
    _, v = state
    return np.array([v, -ball.g])


def event_fun(t: float, state: np.ndarray) -> float:
    """
    Event function for ground impact detection.

    Args:
        t: Time (unused)
        state: [height, velocity]

    Returns:
        Distance to ground (zero when ball hits ground)
    """
    h, v = state
    # Only trigger event when ball is falling (v <= 0)
    return h if v <= 0 else 1.0


# Configure event function
event_fun.terminal = True
event_fun.direction = -1  # Detect zero-crossing when h decreases


def reset_map(state: np.ndarray, ball) -> np.ndarray:
    """
    Reset map for bouncing ball impacts.

    Args:
        state: [height, velocity] at impact
        ball: BouncingBall instance with parameters

    Returns:
        Post-impact state [0, -c*v]
    """
    _, v = state
    return np.array([0.0, -ball.c * v])


class BouncingBall:
    """Bouncing ball hybrid system."""

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]] = [(0.0, 2.0), (-5.0, 5.0)],
        g: float = 9.81,
        c: float = 0.8,
        max_jumps: int = 20,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        # Parameters
        self.g = g  # Acceleration due to gravity (m/s^2)
        self.c = c  # Coefficient of restitution
        self.domain_bounds = domain_bounds
        self.rtol = rtol
        self.atol = atol
        self.max_jumps = max_jumps

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
            event_function=event_fun,
            reset_map=reset,
            domain_bounds=self.domain_bounds,
            max_jumps=self.max_jumps,
            rtol=self.rtol,
            atol=self.atol,
        )

    def simulate(self, initial_state: np.ndarray, time_span: tuple[float, float]):
        """Simulate the bouncing ball."""
        return self.system.simulate(initial_state, time_span, dense_output=True)
