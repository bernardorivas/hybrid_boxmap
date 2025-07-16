"""
3D Bipedal Walking Example

A simplified 3D bipedal walking model based on the Linear Inverted Pendulum (LIP) framework.
The model consists of a 3D LIP biped with massless legs where (x,y,z) denote the coordinate
of the point mass in the inertial coordinate frame centered at the point of support.

System:
- State: [x, y, x', y'] (position and velocity of center of mass in horizontal plane)
- Continuous dynamics: x'' = ω^2 x, y'' = ω^2 y
- Jump condition: x^2+y^2 = x0^2+y0^2 with y <= 0, y' <= 0
- Reset map: (x,y,x',y') -> (-x,-y,x',-y')
"""

import numpy as np

from ..src.hybrid_system import HybridSystem


def ode_fun(t: float, state: np.ndarray, biped) -> np.ndarray:
    """
    Continuous dynamics for 3D LIP bipedal walker.

    Args:
        t: Time (unused, autonomous system)
        state: [x, y, x', y'] (position and velocity in horizontal plane)
        biped: Bipedal parameters

    Returns:
        Time derivatives [x', y', x'', y''] = [x', y',ω^2 x, ω^2 y]
    """
    x, y, x_dot, y_dot = state
    omega_sq = biped.omega**2

    return np.array([x_dot, y_dot, omega_sq * x, omega_sq * y])


def event_fun(t: float, state: np.ndarray, biped) -> float:
    """
    Event function for step detection (circular guard condition).

    Args:
        t: Time (unused)
        state: [x, y, x', y']
        biped: Bipedal parameters

    Returns:
        Distance to guard set (zero when x^2 + y^2 = r^2)
    """
    x, y, x_dot, y_dot = state

    # Only trigger when moving toward guard
    if y > 0 or y_dot > 0:
        return 1.0

    # Guard condition: x^2 + y^2 = r^2
    current_radius_sq = x**2 + y**2
    target_radius_sq = biped.x0**2 + biped.y0**2

    return target_radius_sq - current_radius_sq


def reset_map(state: np.ndarray, biped) -> np.ndarray:
    """
    Reset map for bipedal step transition.

    Args:
        state: [x, y, x', y'] at step event
        biped: Bipedal parameters

    Returns:
        New state (x,y,x',y')
    """
    x, y, x_dot, y_dot = state

    # Reset map: (x,y,x',y') -> (-x,-y,x',-y')
    return np.array([-x, -y, x_dot, -y_dot])


class BipedalWalker:
    """3D Linear Inverted Pendulum bipedal walker hybrid system."""

    def __init__(
        self,
        x0: float = np.sqrt(2) / 2,  # Target x-coordinate for step
        y0: float = np.sqrt(2) / 2,  # Target y-coordinate for step
        z0: float = 1.0,  # Constant height of mass
        g: float = 1.0,  # Gravitational acceleration
        max_jumps: int = 20,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        # Physical parameters
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.g = g
        self.omega = np.sqrt(g / z0)  # LIP frequency parameter

        # Domain bounds [x, y, x_dot, y_dot]
        radius = np.sqrt(x0**2 + y0**2)
        max_vel = self.omega * radius  # Rough velocity bound

        self.domain_bounds = [
            (-radius, radius),  # x bounds
            (-radius, radius),  # y bounds
            (-max_vel, max_vel),  # x_dot bounds
            (-max_vel, max_vel),  # y_dot bounds
        ]

        # Integration parameters
        self.rtol = rtol
        self.atol = atol
        self.max_jumps = max_jumps

        # Create event function with parameters
        def event(t: float, state: np.ndarray) -> float:
            return event_fun(t, state, self)

        # Configure event function
        event.terminal = True
        event.direction = -1
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
        """Simulate the bipedal walker."""
        return self.system.simulate(initial_state, time_span, dense_output=True)
