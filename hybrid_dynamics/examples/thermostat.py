"""
Thermostat Example

A temperature control system that switches a heater on/off to maintain temperature
within specified bounds. This is a classic example of a switching control system.

System:
- State: [z, q] (temperature, controller state)
- Continuous dynamics: z' = -z + z₀ + Δz·q, q' = 0
- Jump conditions:
  * q=0 and z≤z_min → switch heater on (q=1)
  * q=1 and z≥z_max → switch heater off (q=0)
- Reset map: [z, q] → [z, 1-q] (toggle controller state)

This demonstrates hybrid systems with guard sets defined by level sets.
"""

import numpy as np

from ..src.hybrid_system import HybridSystem


def ode_fun(t: float, state: np.ndarray, thermostat) -> np.ndarray:
    """
    Continuous dynamics for thermostat system.

    Args:
        t: Time (unused, autonomous system)
        state: [temperature, controller_state]
        thermostat: Thermostat instance with parameters

    Returns:
        Time derivatives [dz/dt, dq/dt] = [-z + z₀ + Δz·q, 0]
    """
    z, q = state

    dzdt = -z + thermostat.z0 + thermostat.zdelta * q
    dqdt = 0.0  # Controller state doesn't change continuously

    return np.array([dzdt, dqdt])


def event_fun(t: float, state: np.ndarray, thermostat) -> float:
    """
    Event function for thermostat switching conditions.

    This function returns zero when a switching event should occur:
    - When heater is off (q=0) and temperature reaches or falls below zmin
    - When heater is on (q=1) and temperature reaches or rises above zmax

    Args:
        t: Time (unused)
        state: [temperature, controller_state]
        thermostat: Thermostat instance with parameters

    Returns:
        Distance to switching event (zero when switching should occur)
    """
    z, q = state

    # Check switching conditions
    if q < 0.5:  # q ≈ 0 (heater off)
        # Switch on when z ≤ zmin
        # Event function is positive when z > zmin, zero at z = zmin, negative when z < zmin
        return z - thermostat.zmin
    # q ≈ 1 (heater on)
    # Switch off when z ≥ zmax
    # Event function is positive when z < zmax, zero at z = zmax, negative when z > zmax
    return thermostat.zmax - z


def reset_map(state: np.ndarray, thermostat) -> np.ndarray:
    """
    Reset map for thermostat switching.

    Args:
        state: [temperature, controller_state] at switching
        thermostat: Thermostat instance with parameters

    Returns:
        Post-switching state [z, 1-q]
    """
    z, q = state

    # Toggle controller state
    q_new = 1.0 - q

    # Ensure q_new is exactly 0 or 1
    q_new = 1.0 if q_new > 0.5 else 0.0

    return np.array([z, q_new])


class Thermostat:
    """Thermostat hybrid system."""

    def __init__(
        self,
        z0: float = 60.0,  # Base temperature
        zdelta: float = 30.0,  # Temperature increment when heater is on
        zmin: float = 70.0,  # Lower temperature threshold
        zmax: float = 80.0,  # Upper temperature threshold
        domain_bounds: list[tuple[float, float]] = [(0.0, 100.0), (-0.0, 1.0)],
        max_jumps: int = 50,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        # Physical parameters
        self.z0 = z0
        self.zdelta = zdelta
        self.zmin = zmin
        self.zmax = zmax

        # Domain bounds [temperature, controller_state]
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
        event.direction = 0  # Bidirectional: trigger on any zero crossing
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
        """Simulate the thermostat system."""
        return self.system.simulate(
            initial_state, time_span, dense_output=True, max_step=0.1,
        )

    def create_multigrid(self, temp_subdivisions: int = 1000):
        """
        Create MultiGrid for thermostat with discrete mode structure.

        Args:
            temp_subdivisions: Number of subdivisions for temperature dimension

        Returns:
            MultiGrid with mode transition graph and 1D grids for each mode
        """
        import networkx as nx

        from ..src.grid import Grid
        from ..src.multigrid import MultiGrid

        # Create mode transition graph
        mode_graph = nx.DiGraph()
        mode_graph.add_nodes_from([0, 1])  # Mode 0: heater off, Mode 1: heater on
        mode_graph.add_edges_from([(0, 1), (1, 0)])  # Bidirectional transitions

        # Create 1D grids for each mode
        # Both modes have same temperature range [0, 100]
        temp_bounds = [self.domain_bounds[0]]  # Extract temperature bounds

        mode_grids = {
            0: Grid(bounds=temp_bounds, subdivisions=[temp_subdivisions]),  # Heater off
            1: Grid(bounds=temp_bounds, subdivisions=[temp_subdivisions]),  # Heater on
        }

        return MultiGrid(mode_graph, mode_grids)
