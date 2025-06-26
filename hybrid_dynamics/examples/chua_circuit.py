"""
Chua's Circuit Example

A 3D hybrid system modeling Chua's circuit with a piecewise-linear nonlinearity.
This creates a hidden attractor - an attractor that doesn't contain any equilibrium points.

System:
- State: [x, y, z] (voltage across C1, voltage across C2, current through L)
- Continuous dynamics: Depends on region (x < -1, -1 ≤ x ≤ 1, x > 1)
- Event functions: x + 1 = 0 and x - 1 = 0 (switching surfaces)
- Reset map: Identity (continuous across boundaries)

The piecewise-linear Chua's diode creates three distinct dynamical regions.
"""

import numpy as np
from ..src.hybrid_system import HybridSystem


def chua_diode(x: float, m0: float, m1: float) -> float:
    """
    Piecewise-linear Chua's diode characteristic.
    
    f(x) = m₁x + 0.5(m₀ - m₁)(|x + 1| - |x - 1|)
    
    Args:
        x: Voltage
        m0: Center slope
        m1: Outer slope
        
    Returns:
        Diode current
    """
    return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))


def ode_fun(t: float, state: np.ndarray, chua) -> np.ndarray:
    """
    Continuous dynamics for Chua's circuit.
    
    Args:
        t: Time (unused, autonomous system)
        state: [x, y, z] 
        chua: ChuaCircuit instance with parameters
        
    Returns:
        Time derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    
    # Compute Chua's diode characteristic
    fx = chua_diode(x, chua.m0, chua.m1)
    
    # System equations
    dxdt = chua.alpha * (y - x - fx)
    dydt = x - y + z
    dzdt = -chua.beta * y
    
    return np.array([dxdt, dydt, dzdt])


def event_fun_1(t: float, state: np.ndarray, chua) -> float:
    """
    Event function for x = -1 switching surface.
    
    Args:
        t: Time (unused)
        state: [x, y, z]
        chua: ChuaCircuit instance
        
    Returns:
        x + 1 (zero when x = -1)
    """
    x, y, z = state
    return x + 1.0


def event_fun_2(t: float, state: np.ndarray, chua) -> float:
    """
    Event function for x = 1 switching surface.
    
    Args:
        t: Time (unused) 
        state: [x, y, z]
        chua: ChuaCircuit instance
        
    Returns:
        x - 1 (zero when x = 1)
    """
    x, y, z = state
    return x - 1.0


def reset_map(state: np.ndarray, chua) -> np.ndarray:
    """
    Reset map for Chua's circuit (identity).
    
    The system is continuous across switching surfaces,
    so the reset map is the identity.
    
    Args:
        state: [x, y, z] at switching
        chua: ChuaCircuit instance
        
    Returns:
        Post-switching state (unchanged)
    """
    return state


class ChuaCircuit:
    """Chua's circuit hybrid system with hidden attractor."""
    
    def __init__(self,
                 alpha: float = 8.4,
                 beta: float = 12.0,
                 gamma: float = -0.005,  # Not used in dimensionless form
                 m0: float = -1.2,       # Center slope
                 m1: float = -0.05,      # Outer slope  
                 domain_bounds: list[tuple[float, float]] = [(-4.0, 4.0), (-4.0, 4.0), (-6.0, 6.0)],
                 max_jumps: int = 50,
                 rtol: float = 1e-10,
                 atol: float = 1e-12):
        # Circuit parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m0 = m0
        self.m1 = m1
        
        # Domain bounds [x, y, z]
        self.domain_bounds = domain_bounds
        
        # Integration parameters
        self.rtol = rtol
        self.atol = atol
        self.max_jumps = max_jumps
        
        # Create event functions with parameters
        def event1(t: float, state: np.ndarray) -> float:
            return event_fun_1(t, state, self)
        
        def event2(t: float, state: np.ndarray) -> float:
            return event_fun_2(t, state, self)
        
        # Configure event functions
        event1.terminal = True
        event1.direction = 0  # Bidirectional
        event2.terminal = True  
        event2.direction = 0  # Bidirectional
        
        # For HybridSystem, we need a single event function
        # We'll use the minimum absolute value to determine which surface is closer
        def combined_event(t: float, state: np.ndarray) -> float:
            e1 = event_fun_1(t, state, self)
            e2 = event_fun_2(t, state, self)
            # Return the event with smaller absolute value
            if abs(e1) < abs(e2):
                return e1
            else:
                return e2
        
        combined_event.terminal = True
        combined_event.direction = 0
        self.event_function = combined_event
        
        # Store individual event functions for analysis
        self.event_functions = [event1, event2]
        
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
            atol=self.atol
        )
    
    def simulate(self, initial_state: np.ndarray, time_span: tuple[float, float]):
        """Simulate Chua's circuit."""
        return self.system.simulate(initial_state, time_span, dense_output=True, max_step=0.01)
    
    def get_initial_condition_hidden(self) -> np.ndarray:
        """
        Get initial condition for hidden attractor.
        
        The hidden attractor exists with stable zero equilibrium,
        so we need to start away from equilibria.
        
        Returns:
            Initial state [x₀, y₀, z₀]
        """
        return np.array([2.0, 0.1, 0.1])
    
    def get_equilibria(self) -> list[np.ndarray]:
        """
        Get equilibrium points of the system.
        
        For the hidden attractor parameters, there are three equilibria:
        - Origin (0, 0, 0) - stable
        - Two symmetric saddle points
        
        Returns:
            List of equilibrium points
        """
        # Origin is always an equilibrium
        equilibria = [np.array([0.0, 0.0, 0.0])]
        
        # For the outer regions, equilibria satisfy:
        # x = y = -z
        # x = -(m₁x + (m₀ - m₁))  for x < -1
        # x = -(m₁x - (m₀ - m₁))  for x > 1
        
        # Left equilibrium (x < -1)
        x_left = -(self.m0 - self.m1) / (1 + self.m1)
        if x_left < -1:
            equilibria.append(np.array([x_left, x_left, -x_left]))
        
        # Right equilibrium (x > 1)  
        x_right = (self.m0 - self.m1) / (1 + self.m1)
        if x_right > 1:
            equilibria.append(np.array([x_right, x_right, -x_right]))
        
        return equilibria