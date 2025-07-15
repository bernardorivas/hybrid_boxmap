# Hybrid Dynamics

A Python library for analyzing hybrid dynamical systems: bouncing balls, walking robots, thermostat, etc.

## Key features

- Simulate hybrid trajectories
- Build combinatorial models (boxmap)
- Identify attractors using graph algorithms
- Visualize: phase portraits, time series, box maps
- Built-in examples of some hybrid systems

## Installation

```bash
pip install -e .
```

## Quick example: Bouncing ball

```python
from hybrid_dynamics import HybridSystem

# Define the bouncing ball
system = HybridSystem(
    ode=lambda t, x: [x[1], -9.8],  # velocity, gravity
    event_function=lambda t, x: x[0],  # hit ground when height = 0
    reset_map=lambda x: [0, -0.8*x[1]],  # bounce with 80% restitution
    domain_bounds=[(0, 10), (-20, 20)],
    max_jumps=10
)

# Simulate hybrid trajectory
trajectory = system.simulate([1.0, 0.0], time_span=(0, 5))
```

## Examples

Library contains examples of 

```bash
python demo/run_bouncing_ball.py    # Bouncing ball with damping
python demo/run_rimless_wheel.py    # Passive walking
python demo/run_thermostat.py       # On/off temperature control
python demo/run_bipedal.py          # Two-legged walker (3D LIP)
```

Each demo creates visualizations in the `figures/` directory.

## Mathematical background

### Hybrid time
Hybrid trajectories evolve in "hybrid time" (t,j) where:
- t = continuous time (when the system flows)
- j = jump counter (increments at discrete transitions)

### Hybrid system
A hybrid system consists of
1. **Flow**: `ẋ = f(x,t)`
2. **Guard**: `g(x) = 0` 
3. **Reset**: `x⁺ = r(x⁻)` 

### Box maps / combinatorial analysis

```python
from hybrid_dynamics import Grid, HybridBoxMap, create_morse_graph

# Discretize the state space
grid = Grid(bounds=[(0, 10), (-5, 5)], subdivisions=[50, 50])

# Compute where boxes map to after time τ
box_map = HybridBoxMap.compute(grid, system, tau=1.0)

# Find attractors and their basins
graph = box_map.to_networkx()
morse_graph, attractors = create_morse_graph(graph)
```

## Main classes

- `HybridSystem` – Define and simulate hybrid systems
- `Grid` – Discretize state space into boxes  (`MultiGrid` available)
- `HybridBoxMap` – Compute discrete dynamics on grid
- `HybridPlotter` – Visualize trajectories and box maps

## Future directions

- Rigorous construction of boxmap
- Cubical homology for topological analysis

## Don't cite this, but

```bibtex
@software{hybrid_boxmap,
  title={Hybrid Dynamics: hybrid implementation of CMGDB},
  author={Bernardo Rivas},
  year={2025},
  url={https://github.com/bernardorivas/hybrid_boxmap}
}
```

## License

MIT License – see [LICENSE](LICENSE) file.