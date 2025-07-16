"""
Rimless Wheel Example

A rimless wheel rolling down an inclined plane, modeling passive dynamic walking.
The wheel has spokes that alternately contact the ground, creating discrete transitions.

System:
- State: [θ, ω] (leg angle, angular velocity)
- Continuous dynamics: θ' = ω, ω' = sin(θ) (pendulum motion between contacts)
- Jump condition: θ = α + γ (next spoke contacts ground)
- Reset map: (α+γ,ω) -> (γ-α,cos(2α)ω)
"""

import numpy as np

from ..src.hybrid_system import HybridSystem


def ode_fun(t: float, state: np.ndarray, wheel) -> np.ndarray:
    """
    Continuous dynamics for rimless wheel between contacts.

    Args:
        t: Time (unused, autonomous system)
        state: [θ, ω] (inter-leg angle, angular velocity)
        wheel: RimlessWheel instance with parameters

    Returns:
        Time derivatives [dθ/dt, dω/dt] = [ω, sin(θ)]
    """
    theta, omega = state
    return np.array([omega, np.sin(theta)])


def event_fun(t: float, state: np.ndarray, wheel) -> float:
    """
    Event function for spoke-ground contact detection.

    Args:
        t: Time (unused)
        state: [θ, ω]
        wheel: RimlessWheel parameters

    Returns:
        Distance to contact event (zero when next spoke hits ground)
    """
    theta, _ = state
    # The solver's `direction=1` setting will handle ignoring cases where omega < 0.
    return theta - (wheel.alpha + wheel.gamma)


def reset_map(state: np.ndarray, wheel) -> np.ndarray:
    """
    Reset map for rimless wheel spoke contact.
    Args:
        state: [θ, ω] at contact
        wheel: RimlessWheel parameters

    Returns:
       New state [θ_new, ω_new]
    """
    _, omega = state

    # theta_new = 2γ-θ
    # but θ = α+γ, so theta_new = γ-α
    theta_new = wheel.gamma - wheel.alpha

    # omega_new = cos(2α)ω
    omega_new = np.cos(2 * wheel.alpha) * omega

    return np.array([theta_new, omega_new])


class RimlessWheel:
    """Rimless wheel hybrid system."""

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]] = [
            (-0.2, 0.6),  # (γ-α,γ+α)
            (-0.5, 1.0),
        ],  # (-ω_max, ω_max)
        alpha: float = 0.4,  # Half-angle between legs
        gamma: float = 0.2,  # Slope angle
        max_jumps: int = 20,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        # Physical parameters
        self.alpha = alpha
        self.gamma = gamma

        # Domain bounds [theta, omega]
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
        event.direction = (
            1  # Detect when event function is increasing (theta increasing)
        )
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
        """Simulate the rimless wheel."""
        return self.system.simulate(initial_state, time_span, dense_output=True)

    def is_valid_state(self, state: np.ndarray) -> bool:
        """Check if a state is valid (within domain bounds)."""
        return self.system.is_valid_state(state)
