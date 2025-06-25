# Hybrid Dynamics

A Python library for analysis and visualization of hybrid dynamical systems using box maps, Morse graphs, and grid-based methods.

## Features

- **Hybrid System Simulation**: Event-driven dynamics with continuous flows and discrete jumps
- **Box Map Computation**: Grid-based discrete abstraction of hybrid system dynamics
- **Morse Graph Analysis**: Automatic detection of recurrent sets and attractors
- **Multi-Modal Systems**: Support for systems with discrete operational modes via MultiGrid framework
- **Parallel Processing**: Efficient grid evaluation using multiprocessing
- **Rich Visualization**: Phase portraits, time series, box maps, and Morse graphs
- **Example Gallery**: Standard hybrid systems (bouncing ball, thermostat, bipedal walker, rimless wheel, etc.)

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from hybrid_dynamics import HybridSystem, HybridPlotter

# Create a simple bouncing ball system
system = HybridSystem(
    ode=lambda t, x: np.array([x[1], -9.8]),  # [velocity, acceleration]
    event_function=lambda t, x: x[0],         # hits ground when height = 0
    reset_map=lambda x: np.array([0, -0.8*x[1]]),  # bounce with damping
    domain_bounds=[(0, 10), (-20, 20)],
    max_jumps=10
)

# Simulate trajectory
trajectory = system.simulate(
    initial_state=np.array([1.0, 0.0]),  # height=1, velocity=0
    time_span=(0, 5),
    max_jumps=10
)

# Visualize
plotter = HybridPlotter()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Phase portrait
plotter.plot_phase_portrait(system, [trajectory], ax=ax1)
ax1.set_xlabel("Height (m)")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title("Phase Portrait")

# Time series
plotter.plot_time_series(trajectory, ax=ax2, show_jumps=True)
ax2.set_title("Time Series")

plt.tight_layout()
plt.show()
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid_boxmap
cd hybrid_boxmap

# Install the package in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with optional dependencies
pip install -e ".[optimization]"  # Includes numba for JIT compilation
pip install -e ".[parallel]"      # Includes joblib for parallel processing
pip install -e ".[testing]"       # Includes pytest and coverage tools

# Optional: Install pygraphviz for Morse graph visualization
pip install pygraphviz
```

## Examples

The `demo/` directory contains complete examples for various hybrid systems:

```bash
# Run individual demos
python demo/run_bouncing_ball.py
python demo/run_bipedal.py
python demo/run_thermostat.py
python demo/run_rimless_wheel.py
python demo/run_unstableperiodic.py
```

### Available Systems

- **Bouncing Ball** (`demo/run_bouncing_ball.py`): Classic bouncing ball with gravity and damping
- **Bipedal Walker** (`demo/run_bipedal.py`): Walking model with foot impacts  
- **Thermostat** (`demo/run_thermostat.py`): Temperature control with on/off switching
- **Rimless Wheel** (`demo/run_rimless_wheel.py`): Rolling wheel with discrete impacts
- **Unstable Periodic** (`demo/run_unstableperiodic.py`): Periodic system with unstable dynamics

## Core Concepts

### Hybrid Time Domain

Trajectories are tracked in hybrid time (t, j) where:
- t ∈ ℝ≥0 is continuous time
- j ∈ ℕ is the discrete jump index

Example for a bouncing ball:
```
[0, 1.2] × {0} ∪ [1.2, 2.1] × {1} ∪ [2.1, 2.7] × {2} ∪ [2.7, 3.0] × {3}
```

### HybridSystem Class

```python
from hybrid_dynamics import HybridSystem

system = HybridSystem(
    ode=lambda t, x: np.array([x[1], -9.8]),  # Continuous dynamics
    event_function=lambda t, x: x[0],         # Guard condition g(x) = 0
    reset_map=lambda x: np.array([0, -0.8*x[1]]),  # Discrete reset
    domain_bounds=[(0, 10), (-20, 20)],      # State space bounds
    max_jumps=20
)

# Simulate
trajectory = system.simulate(
    initial_state=np.array([1.0, 0.0]),
    time_span=(0, 5)
)
```

### Box Map Computation and Analysis

```python
from hybrid_dynamics import HybridBoxMap, Grid, create_morse_graph
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel

# Create system and grid
wheel = RimlessWheel(alpha=0.4, gamma=0.2)
grid = Grid(bounds=wheel.domain_bounds, subdivisions=[100, 100])

# Compute box map
box_map = HybridBoxMap.compute(
    grid=grid,
    system=wheel.system,
    tau=0.5,  # time horizon
    sampling_mode='corners',
    bloat_factor=0.1
)

# Convert to graph and analyze
graph = box_map.to_networkx()
morse_graph, recurrent_components = create_morse_graph(graph)

print(f"Found {len(recurrent_components)} recurrent components")
```

### Visualization

The `HybridPlotter` class provides comprehensive visualization:

```python
from hybrid_dynamics import HybridPlotter

plotter = HybridPlotter()

# Available plot types
plotter.plot_phase_portrait(system, trajectories)
plotter.plot_time_series(trajectory, show_jumps=True)
plotter.plot_hybrid_trajectory(trajectory)
plotter.plot_cubification(cubifier, trajectory)
plotter.plot_3d_trajectory(trajectory)
plotter.plot_event_region(system)
```

## Architecture

```
hybrid_dynamics/
├── src/                      # Core source code
│   ├── hybrid_system.py      # Main HybridSystem class
│   ├── hybrid_time.py        # Hybrid time (t,j) structures
│   ├── hybrid_trajectory.py  # Trajectory with scipy dense output
│   ├── hybrid_boxmap.py      # Box map computation and analysis
│   ├── box.py                # N-dimensional box primitives
│   ├── grid.py               # Grid discretization
│   ├── cubifier.py           # DatasetCubifier for trajectory analysis
│   ├── evaluation.py         # Parallel grid evaluation
│   ├── morse_graph.py        # Morse graph algorithms
│   ├── plot_utils.py         # HybridPlotter and visualization
│   ├── config.py             # Centralized configuration
│   ├── data_utils.py         # Results storage and serialization
│   ├── demo_utils.py         # Demo script utilities
│   ├── grid_utils.py         # Grid analysis utilities
│   ├── trajectory_utils.py   # Trajectory manipulation
│   └── multigrid.py          # MultiGrid for multi-modal systems
├── examples/                 # Example system implementations
│   ├── bouncing_ball.py      # Classic bouncing ball
│   ├── bipedal.py            # Bipedal walker with impacts
│   ├── thermostat.py         # Temperature control system
│   ├── rimless_wheel.py      # Rolling wheel with impacts
│   ├── unstableperiodic.py   # Unstable periodic orbits
│   └── baker_map.py          # Baker's map (discrete)
└── __init__.py               # Package exports

demo/                         # Demonstration scripts
├── run_bouncing_ball.py      # Bouncing ball demo
├── run_bipedal.py            # Bipedal walker demo
├── run_thermostat.py         # Thermostat demo
├── run_rimless_wheel.py      # Rimless wheel demo
└── run_unstableperiodic.py   # Unstable periodic demo

tests/                        # Test suite
├── test_boxmap.py            # Box map computation tests
├── test_evaluation.py        # Grid evaluation tests
├── test_graph_analysis.py    # Graph algorithms tests
├── test_recurrent_sets.py    # Recurrent component tests
├── test_boxmap_visualization.py  # Visualization tests
└── test_bouncing_ball.py     # Bouncing ball tests
```

## API Reference

### Core Classes

- **`HybridSystem`**: Main class for hybrid dynamical systems with event detection
- **`HybridTrajectory`**: Trajectory storage with scipy dense output interpolation
- **`HybridTime`**: Hybrid time point (t, j) representation
- **`HybridTimeInterval`**: Hybrid time interval for trajectory domains
- **`HybridBoxMap`**: Grid-based discrete abstraction of system dynamics
- **`Grid`**: Systematic domain discretization with box indexing
- **`DatasetCubifier`**: Trajectory to grid-aligned box conversion
- **`HybridPlotter`**: Comprehensive visualization interface
- **`MultiGrid`**: Framework for multi-modal hybrid systems

### Key Methods

```python
# System simulation
trajectory = system.simulate(initial_state, time_span, max_jumps)

# Trajectory analysis
print(f"Domain: {trajectory.to_hybrid_time_notation()}")
print(f"Jumps: {trajectory.num_jumps}")

# Box map computation
box_map = HybridBoxMap.compute(grid, system, tau)
graph = box_map.to_networkx()

# Morse graph analysis
morse_graph, recurrent_components = create_morse_graph(graph)

# Visualization
plotter.plot_phase_portrait(system, trajectories)
visualize_box_map(box_map, grid)
plot_morse_graph_viz(morse_graph, recurrent_components)
```

## Dependencies

- **numpy** ≥ 1.20: Numerical computing
- **scipy** ≥ 1.7: ODE integration with event detection
- **matplotlib** ≥ 3.3: Visualization and plotting

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests with custom framework
python tests/test_boxmap.py
python tests/test_evaluation.py
python tests/test_graph_analysis.py
python tests/test_recurrent_sets.py

# Modern development tools (configured in pyproject.toml)
black hybrid_dynamics/           # Code formatting
ruff check hybrid_dynamics/      # Fast linting
ruff format hybrid_dynamics/     # Fast formatting
mypy hybrid_dynamics/            # Type checking
pytest tests/                    # Run tests with coverage
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TO-DO

### Future Enhancements

- **Cubical Homology Support**: Implement algorithms for computing homology groups of cubical complexes to analyze topological properties of attractors and invariant sets
- **Regions of Attraction**: Add methods to compute and visualize regions of attraction for identified recurrent components using backward reachability analysis
- **Persistence Homology**: Integrate persistent homology computations to track topological features across different time scales
<!-- - **Conley Index Theory**: Implement Conley index calculations for robust topological characterization of isolated invariant sets -->

## Citation

If you use this library in research, please cite:

```bibtex
@software{hybrid_boxmap,
  title={tbd},
  author={Bernardo Rivas},
  year={2024},
  url={https://github.com/bernardorivas/hybrid_boxmap}
}
```