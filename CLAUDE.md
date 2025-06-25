# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Install with development dependencies  
pip install -e .[dev]

# Install with specific optional dependencies
pip install -e ".[optimization]"  # Includes numba for JIT compilation
pip install -e ".[parallel]"      # Includes joblib for parallel processing
pip install -e ".[testing]"       # Includes pytest and coverage tools

# Run individual demo scripts
python demo/run_bouncing_ball.py
python demo/run_bipedal.py
python demo/run_thermostat.py
python demo/run_rimless_wheel.py
python demo/run_unstableperiodic.py
```

### Testing and Development
```bash
# Run specific test modules (custom framework in tests/ directory)
# Note: Fix import issues first - tests need to import from main package
python tests/test_boxmap.py
python tests/test_evaluation.py  
python tests/test_graph_analysis.py
python tests/test_recurrent_sets.py
python tests/test_boxmap_visualization.py

# Install development tools first
pip install -e ".[dev]"                # Install all development dependencies

# Modern development tools (configured in pyproject.toml)
black hybrid_dynamics/           # Code formatting
ruff check hybrid_dynamics/      # Fast linting (replaces flake8)
ruff format hybrid_dynamics/     # Fast formatting (alternative to black)
mypy hybrid_dynamics/            # Type checking
pytest tests/                    # Run tests with coverage

# Optional visualization dependency
pip install pygraphviz           # For Morse graph visualization (optional)
```

## Architecture Overview

### Core Design Pattern
This is a **hybrid dynamical systems** library built around three fundamental mathematical concepts:

1. **Hybrid Time Domain**: (t,j) ∈ ℝ≥0 × ℕ where t is continuous time and j is discrete jump count
2. **Event-Driven Dynamics**: Continuous flow interrupted by discrete jumps when guard conditions are met
3. **Spatial Cubification**: Convert trajectory data into grid-aligned box representations for analysis

### Module Hierarchy and Data Flow

```
User → HybridSystem → HybridTrajectory → DatasetCubifier → Visualization
            ↓              ↓                    ↓              ↓
    scipy.integrate  Dense Output      Grid Analysis    HybridPlotter
```

**Key Architecture Points:**

#### 1. Source Module (`hybrid_dynamics.src/`)
All core functionality is located in the `src/` directory:

- `hybrid_system.py`: Central class combining ODE, events, and reset maps
- `hybrid_trajectory.py`: Stores `TrajectorySegment` objects with scipy dense output solutions  
- `hybrid_time.py`: Represents (t,j) hybrid time points and intervals
- `hybrid_boxmap.py`: Dictionary mapping source grid box indices to destination box indices, computed via sample-and-bloat method for hybrid system flow analysis
- `box.py`: N-dimensional rectangular primitives (`Box`/`SquareBox`) with containment testing
- `grid.py`: Systematic domain discretization with parallel evaluation support
- `cubifier.py`: Converts trajectories to grid-aligned boxes (`DatasetCubifier`) with subdivision support
- `evaluation.py`: Parallel function evaluation over grids
- `morse_graph.py`: Morse graph analysis and creation functions
- `plot_utils.py`: `HybridPlotter` class and visualization functions
- `config.py`: Centralized configuration management
- `data_utils.py`: Results storage and serialization for grid evaluations
- `demo_utils.py`: Utility functions for demo scripts and analysis workflows
- `grid_utils.py`: Utilities for grid-based analysis including covered regions
- `trajectory_utils.py`: Trajectory manipulation and analysis utilities
- `multigrid.py`: MultiGrid framework for multi-modal hybrid systems with discrete modes
- **Integration**: Uses `scipy.integrate.solve_ivp` with event detection

#### 2. Examples Module (`hybrid_dynamics.examples/`)
Each example follows this pattern:
```python
class SystemName:
    def __init__(self, parameters):
        self.system = self._create_system()
    
    def _create_system(self) -> HybridSystem:
        # Use closures to capture self for parameter access
        def ode(t, state): return ode_fun(t, state, self)
        def reset(state): return reset_map(state, self)
        return HybridSystem(ode, event_function, reset, ...)
```

Available examples include:
- `bouncing_ball.py`: Classic bouncing ball with gravity and damping
- `bipedal.py`: Bipedal walking model with foot impacts
- `thermostat.py`: Temperature control with on/off switching
- `rimless_wheel.py`: Rolling wheel with discrete impacts  
- `unstableperiodic.py`: Periodic system with unstable dynamics
- `baker_map.py`: Baker's map for chaotic dynamics analysis

### Configuration Management

The library now includes a centralized configuration system to eliminate magic numbers and provide consistent defaults:

```python
from hybrid_dynamics import config

# Access current configuration
print(f"Default bloat factor: {config.grid.bloat_factor}")
print(f"Default grid subdivisions: {config.grid.default_subdivisions}")
print(f"Default max jumps: {config.simulation.default_max_jumps}")

# Modify configuration at runtime
config.grid.bloat_factor = 0.2
config.visualization.default_figsize = (12, 10)
config.set_output_dir("./my_results")

# Bulk configuration updates
custom_config = {
    'grid': {'bloat_factor': 0.15},
    'simulation': {'default_max_jumps': 50},
    'visualization': {'trajectory_color': 'green'}
}
config.update_from_dict(custom_config)

# Configuration is used automatically throughout the library
box_map = HybridBoxMap.compute(grid, system, tau)  # Uses config.grid.bloat_factor
```

**Key Configuration Sections:**
- `config.grid`: Grid analysis parameters (subdivisions, bloat_factor, sampling_modes)
- `config.simulation`: Simulation parameters (max_jumps, time_horizon, tolerances)  
- `config.visualization`: Plotting styles (colors, sizes, figure settings)

### HybridBoxMap Computational Workflow

The `HybridBoxMap` class provides a systematic approach to analyzing discrete dynamical behavior over grids:

```python
from hybrid_dynamics.src.hybrid_boxmap import HybridBoxMap
from hybrid_dynamics.src.grid import Grid

# 1. Create grid and compute box map
grid = Grid(bounds, subdivisions)
box_map = HybridBoxMap.compute(
    grid=grid, 
    system=hybrid_system, 
    tau=time_horizon,
    sampling_mode='corners',  # or 'center'
    bloat_factor=0.1,
    parallel=False
)

# 2. Convert to NetworkX for graph analysis
graph = box_map.to_networkx()

# 3. Analyze recurrent dynamics using Morse graph
from hybrid_dynamics.src.morse_graph import create_morse_graph
morse_graph = create_morse_graph(graph)
```

**Key Computational Steps:**
1. **Sampling**: Sample points from each grid box (corners, center, etc.)
2. **Flow Evaluation**: Simulate hybrid system for time `tau` from each sample point
3. **Bloating**: Create bounding boxes around destination points with configurable bloat factor
4. **Grid Mapping**: Find all grid boxes intersecting the bloated bounding boxes

### Testing Framework

The codebase uses a custom testing framework with executable test modules in `tests/`:

- **`test_boxmap.py`**: Tests HybridBoxMap computation using rimless wheel
- **`test_evaluation.py`**: Tests grid evaluation functionality  
- **`test_graph_analysis.py`**: Tests graph theory algorithms on box maps
- **`test_recurrent_sets.py`**: Tests recurrent component detection
- **`test_boxmap_visualization.py`**: Tests visualization utilities

Each test module can be run independently and includes result visualization and saving.

**Known Issue**: Some test files have incorrect import statements (e.g., `from hybrid_dynamics.utils import visualize_box_map`). These should be fixed to import from the main package: `from hybrid_dynamics import visualize_box_map`.

### Critical Implementation Details

#### Import Structure
**IMPORTANT**: All core functionality is located in the `src/` directory. Correct import patterns:
- `from .src.box import Box, SquareBox`  
- `from .src.cubifier import DatasetCubifier`
- `from .src.grid import Grid`
- `from .src.hybrid_boxmap import HybridBoxMap`
- `from .src.morse_graph import create_morse_graph`
- `from .src.data_utils import GridEvaluationResult`
- `from .src.demo_utils import create_config_hash, setup_demo_directories`
- `from .src.multigrid import MultiGrid, MultiGridBoxMap`

#### Parallel Processing Constraints
Functions used in `evaluation.py` parallel processing must be:
- Defined at module level (not as local functions)
- Picklable for multiprocessing
- Access global state via module-level variables if needed

#### Trajectory Representation
- Trajectories store segments with `scipy` solution objects for dense output
- Each segment maps to one continuous flow phase (between jumps)
- Jump times and states are explicitly tracked
- Interpolation available through `trajectory(t)` calls

#### Domain Bounds and Validation
- All systems require `domain_bounds` parameter as list of (min, max) tuples
- Bounds are enforced during simulation
- Out-of-bounds trajectories are truncated

### File Modification Guidelines

#### DO NOT MODIFY:
- `hybrid_dynamics/src/hybrid_system.py`
- `hybrid_dynamics/src/hybrid_time.py` 
- `hybrid_dynamics/src/hybrid_trajectory.py`
- Files in `hybrid_dynamics/examples/` directory

#### Key Extension Points:
- Add new systems in `examples/` following existing patterns (see `baker_map.py` for discrete map example)
- Extend `src/evaluation.py` for new grid analysis functions
- Add visualization methods to `HybridPlotter` in `src/plot_utils.py`
- Add new graph analysis algorithms to `src/morse_graph.py`
- Create new test modules in `tests/` directory following existing patterns
- Extend `HybridBoxMap` with additional sampling strategies or analysis methods

### MultiGrid Framework for Multi-Modal Systems

The **MultiGrid** framework handles hybrid systems with discrete operational modes:

#### Core Concept
- **MultiGrid** = Directed graph of modes + Grid per mode
- Each mode has its own state space discretization
- Transitions between modes based on hybrid system dynamics
- Enables analysis of union-of-manifolds state spaces

#### Example Usage (Thermostat)
```python
from hybrid_dynamics.examples.thermostat import Thermostat
from hybrid_dynamics.src.multigrid import MultiGrid

thermostat = Thermostat()
multigrid = thermostat.create_multigrid(temp_subdivisions=1000)

# Compute cross-mode transitions
multi_boxmap = multigrid.compute_multi_boxmap(
    system=thermostat.system, 
    tau=0.5
)

# Analyze mode-specific behavior
intra_transitions = multi_boxmap.get_intra_mode_transitions(mode=0)
inter_transitions = multi_boxmap.get_inter_mode_transitions()
```

#### Applications
- Temperature control systems (heater on/off modes)
- Robot behavioral modes (patrol/charge/maintenance)
- Aircraft flight phases (takeoff/cruise/landing)
- Manufacturing processes with operational states

### Mathematical Context
The library targets **hybrid automata** and **hybrid dynamical systems** where:
- Continuous dynamics: `dx/dt = f(x,t)` within modes
- Guard conditions: `g(x) = 0` trigger discrete transitions  
- Reset maps: `x⁺ = r(x⁻)` define post-jump state
- Examples include mechanical impacts, switching controllers, biological networks

Understanding this mathematical foundation is essential for working effectively with the codebase.