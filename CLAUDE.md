# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Install for development
pip install -e .

# Install with all optional dependencies (parallel processing, JIT compilation, visualization)
pip install -e ".[all]"

# Run a demo
python demo/run_bouncing_ball.py
```

## Development Commands

### Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"  # Includes numba (JIT compilation), joblib (parallel processing), and pygraphviz (visualization)

# Or install specific optional dependencies
pip install numba           # For JIT compilation and better performance
pip install joblib          # For parallel box map computation
pip install pygraphviz      # For Morse graph visualization
```

### Running Demos
```bash
# Individual demo scripts
python demo/run_bouncing_ball.py
python demo/run_bipedal.py
python demo/run_thermostat.py
python demo/run_rimless_wheel.py
python demo/run_unstableperiodic.py
```

### Testing
The `tests/` directory is available for adding your own tests as needed.

## Architecture Overview

### Core Design Pattern
This is a **hybrid dynamical systems** library built around three fundamental mathematical concepts:

1. **Hybrid Time Domain**: (t,j) ∈ ℝ≥0 × ℕ where t is continuous time and j is discrete jump count
2. **Event-Driven Dynamics**: Continuous flow interrupted by discrete jumps when guard conditions are met
3. **Spatial Cubification**: Convert trajectory data into grid-aligned box representations for analysis

### Module Hierarchy and Data Flow

```
User → HybridSystem → HybridTrajectory → HybridBoxMap → NetworkX → Morse Graph
            ↓              ↓                    ↓            ↓          ↓
    scipy.integrate  Dense Output      Grid Analysis   Graph Theory  Recurrent Sets
```

### Source Module Organization (`hybrid_dynamics/src/`)

All core functionality is in the `src/` directory:

- **Core Systems**: `hybrid_system.py`, `hybrid_trajectory.py`, `hybrid_time.py`
- **Grid Analysis**: `grid.py`, `box.py`, `hybrid_boxmap.py`, `cubifier.py`
- **Graph Analysis**: `morse_graph.py`, `roa_utils.py` (regions of attraction)
- **Utilities**: `evaluation.py` (parallel processing), `data_utils.py`, `demo_utils.py`
- **Visualization**: `plot_utils.py` (HybridPlotter class and functions)
- **Configuration**: `config.py` (global configuration management)
- **Multi-Modal**: `multigrid.py` (systems with discrete modes)

### Examples Module Pattern (`hybrid_dynamics/examples/`)

Each example follows this consistent pattern:
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

## Critical Implementation Patterns

### Version Information
**Note**: Package version is 0.1.0 in pyproject.toml but 0.2.0 in __init__.py. Use __init__.py as authoritative.

### Parallel Processing Setup

For parallel box map computation, systems must be picklable. Use factory functions:

```python
# Factory function at MODULE LEVEL (required for pickling)
def create_rimless_wheel_system(alpha=0.4, gamma=0.2, max_jumps=50):
    wheel = RimlessWheel(alpha=alpha, gamma=gamma, max_jumps=max_jumps)
    return wheel.system

# Usage in parallel computation
box_map = HybridBoxMap.compute(
    grid=grid,
    system=wheel.system,
    tau=tau,
    parallel=True,
    system_factory=create_rimless_wheel_system,
    system_args=(wheel.alpha, wheel.gamma, wheel.max_jumps)
)
```

### Caching and Configuration Hashing

Box maps are cached with MD5 hashes of configuration:

```python
# Create hash including ALL parameters affecting computation
config_hash = create_config_hash(
    params_dict={"alpha": 0.4, "gamma": 0.2, "bloat_factor": 0.12},
    grid_bounds=domain_bounds,
    grid_subdivisions=[100, 100],
    tau=0.5
)

# Load from cache if valid
box_map = load_box_map_from_cache(grid, system, tau, cache_file, config_hash)

# Save after computation
if box_map is None:
    box_map = HybridBoxMap.compute(...)
    save_box_map_with_config(box_map, config_hash, config_details, cache_file)
```

**IMPORTANT**: Include `bloat_factor` in configuration hash to ensure cache validity.

### Run Directory Management

Demos use systematic run directory naming:
```
figures/rimless_wheel/run_tau_050_subdiv_100_100_bloat_12_001/
                      └─ parameters encoded in directory name ─┘
```

Use `create_next_run_dir()` to automatically increment run numbers.

### Configuration Management

Global `config` object provides centralized defaults:

```python
from hybrid_dynamics import config

# Access/modify configuration
config.grid.bloat_factor = 0.15
config.simulation.default_max_jumps = 100
config.set_output_dir("./my_results")

# Bulk updates
config.update_from_dict({
    'grid': {'bloat_factor': 0.2},
    'simulation': {'default_time_horizon': 10.0}
})
```

### Progress Tracking

Use the built-in progress callback for long computations:

```python
progress_callback = create_progress_callback(update_frequency=1000)
# Shows: Progress: 1234/5000 (24.7%) [■■■■■□□□□□□□□□□□□□□□] ETA: 2m 15s
```

## HybridBoxMap Computational Workflow

```python
# 1. Create grid and compute box map
grid = Grid(bounds, subdivisions)
box_map = HybridBoxMap.compute(
    grid=grid, 
    system=hybrid_system, 
    tau=time_horizon,
    sampling_mode='corners',  # or 'center'
    bloat_factor=0.1,
    parallel=False,  # Set True with factory function
    enclosure=False  # Set True for rectangular enclosures (corners mode only)
)

# 2. Convert to NetworkX for graph analysis
graph = box_map.to_networkx()

# 3. Analyze recurrent dynamics
morse_graph, morse_sets = create_morse_graph(graph)

# 4. Compute regions of attraction
roa_dict = compute_roa(graph, morse_sets)
```

### Enclosure Mode (New Feature)

When `enclosure=True` with `sampling_mode='corners'`, the box map computation uses rectangular enclosures:

```python
# Compute with rectangular enclosures
box_map = HybridBoxMap.compute(
    grid=grid,
    system=system,
    tau=tau,
    sampling_mode='corners',  # Required for enclosure
    enclosure=True,           # Enable enclosure mode
    bloat_factor=0.1
)
```

**How it works:**
- For each source box, if all corners map with the same number of jumps, computes the bounding box of all destination corners
- Fills in all grid boxes within this bounding box (plus bloat factor)
- Creates a more conservative over-approximation of the reachable set
- Falls back to per-point bloating if corners have different jump counts

**When to use:**
- When you need guaranteed enclosures of the reachable set
- For rigorous analysis where missing transitions is unacceptable
- When the dynamics preserve convexity locally (enclosure won't overshoot too much)

### Key Computational Optimizations

1. **Unique Points**: Extracts unique points from grid (corners are shared between boxes)
2. **Single Evaluation**: Each unique point evaluated only once
3. **Mapping Back**: Results mapped to all boxes containing that point
4. **Out-of-Bounds Tolerance**: Points outside domain by more than `config.grid.out_of_bounds_tolerance` are discarded

## Import Structure

All functionality is in the `src/` directory. Correct import patterns:

```python
# From within hybrid_dynamics package
from .src.box import Box, SquareBox
from .src.grid import Grid
from .src.hybrid_boxmap import HybridBoxMap
from .src.morse_graph import create_morse_graph
from .src.multigrid import MultiGrid, MultiGridBoxMap

# From outside the package
from hybrid_dynamics import HybridSystem, Grid, HybridBoxMap
```

## File Modification Guidelines

### DO NOT MODIFY
- `hybrid_dynamics/src/hybrid_system.py`
- `hybrid_dynamics/src/hybrid_time.py` 
- `hybrid_dynamics/src/hybrid_trajectory.py`
- Files in `hybrid_dynamics/examples/` directory

### Key Extension Points
- Add new systems in `examples/` following existing patterns
- Extend `src/evaluation.py` for new grid analysis functions
- Add visualization methods to `HybridPlotter` in `src/plot_utils.py`
- Add new graph algorithms to `src/morse_graph.py`
- Create new test modules in `tests/` directory

### Creating New Example Systems

To add a new hybrid system to `examples/`:

```python
# examples/my_system.py
from typing import Tuple, Optional
import numpy as np
from hybrid_dynamics import HybridSystem

class MySystem:
    def __init__(self, param1: float = 1.0, max_jumps: int = 50):
        self.param1 = param1
        self.max_jumps = max_jumps
        self.domain_bounds = [(0, 10), (-5, 5)]  # Define state space bounds
        self.system = self._create_system()
    
    def _create_system(self) -> HybridSystem:
        def ode(t, state):
            # Continuous dynamics
            x, v = state
            return [v, -self.param1 * x]
        
        def event_function(t, state):
            # Guard condition (zero when event occurs)
            return state[0] - 5.0
        
        def reset_map(state):
            # Discrete jump mapping
            x, v = state
            return [0, -0.8 * v]
        
        return HybridSystem(
            ode=ode,
            event_function=event_function,
            reset_map=reset_map,
            domain_bounds=self.domain_bounds,
            max_jumps=self.max_jumps
        )

# For parallel processing, add factory at module level
def create_my_system(param1=1.0, max_jumps=50):
    return MySystem(param1, max_jumps).system
```

## MultiGrid Framework for Multi-Modal Systems

For systems with discrete operational modes:

```python
from hybrid_dynamics.examples.thermostat import Thermostat
from hybrid_dynamics.src.multigrid import MultiGrid

thermostat = Thermostat()
multigrid = thermostat.create_multigrid(temp_subdivisions=1000)

# Compute cross-mode transitions
multi_boxmap = multigrid.compute_multi_boxmap(
    system=thermostat.system, 
    tau=0.5,
    use_interval_arithmetic=True  # More accurate mode transitions
)

# Analyze mode-specific behavior
intra_transitions = multi_boxmap.get_intra_mode_transitions(mode=0)
inter_transitions = multi_boxmap.get_inter_mode_transitions()
```

## Data Persistence

The library provides flexible data storage:

```python
# GridEvaluationResult for general data
result = GridEvaluationResult(grid, data_dict)
result.save("output.json")  # Human-readable
result.save("output.npz")   # Efficient binary

# Automatic format detection on load
loaded = GridEvaluationResult.load("output.json")
```

## Mathematical Context

The library implements analysis for **hybrid automata** where:
- Continuous dynamics: `dx/dt = f(x,t)` within modes
- Guard conditions: `g(x) = 0` trigger discrete transitions  
- Reset maps: `x⁺ = r(x⁻)` define post-jump state
- Hybrid time: Trajectories evolve in (t,j) where t is continuous time and j counts jumps

Understanding this foundation is essential for effective use of the library.

## Common Pitfalls and Solutions

### 1. Parallel Processing Failures
**Problem**: `Can't pickle local object` error when using `parallel=True`
**Solution**: Define factory function at module level, not inside class/function

### 2. Cache Invalidation
**Problem**: Changes not reflected after modifying parameters
**Solution**: Include ALL parameters (especially `bloat_factor`) in configuration hash

### 3. Test Execution
**Problem**: Tests not running properly
**Solution**: Tests are executable modules - run directly with `python tests/test_*.py`

### 4. Import Errors
**Problem**: `ModuleNotFoundError` for hybrid_dynamics modules
**Solution**: Install package in development mode: `pip install -e .`

### 5. Visualization Issues
**Problem**: Morse graph not rendering properly
**Solution**: Install pygraphviz system package (e.g., `brew install graphviz` on macOS)

## Performance Tips

1. **Parallel Computation**: Use `parallel=True` with proper factory functions for large grids
2. **Caching**: Leverage built-in caching for expensive box map computations
3. **Grid Resolution**: Start with coarse grids (e.g., [20, 20]) for initial exploration
4. **Progress Tracking**: Use `create_progress_callback()` for long-running computations
5. **Sampling Mode**: Use `'center'` sampling for faster computation, `'corners'` for accuracy