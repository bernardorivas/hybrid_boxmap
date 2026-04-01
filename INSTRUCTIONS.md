# INSTRUCTIONS.md - Hybrid Dynamics Library

## Overview

This library provides computational tools for analyzing hybrid dynamical systems via outer approximations. The core pipeline is:

1. Define a hybrid system (ODE + guard + reset map)
2. Discretize the state space into a grid
3. Compute a box map (combinatorial enclosure of the flow map)
4. Extract a directed graph and compute Morse decomposition (strongly connected components)
5. Compute regions of attraction via backward reachability

## Mathematical Setting

A hybrid system $H = (f, g, r)$ consists of:
- **Continuous dynamics**: $\dot{x} = f(x, t)$ integrated via `scipy.integrate.solve_ivp`
- **Guard condition**: $g(t, x) = 0$ triggers a discrete jump (terminal event)
- **Reset map**: $x^+ = r(x^-)$ applied at each jump

Trajectories evolve in hybrid time $(t, j) \in \mathbb{R}_{\geq 0} \times \mathbb{N}$, where $t$ is continuous time and $j$ is the jump count.

## Box Map Computation

The box map $F: \mathcal{B} \rightrightarrows \mathcal{B}$ is a set-valued map on grid boxes. For each box $B$:

1. **Sample** points in $B$ (corners, center, or subdivision)
2. **Flow** each sample for time $\tau$ using `HybridSystem.simulate`
3. **Bloat** the bounding box of destination points by a factor relative to grid cell size
4. **Fill** all grid boxes intersecting the bloated bounding box

### Sampling Modes
- `'corners'`: All $2^n$ vertices of each box (shared between adjacent boxes, so unique-point optimization applies)
- `'center'`: Single center point per box
- `'subdivision'`: Subdivide each box into $2^k$ sub-boxes per dimension, sample their corners

### Enclosure Mode
When `enclosure=True` with `sampling_mode='corners'`:
- If all corners of a source box have the same jump count, compute the axis-aligned bounding box of all destination corners
- Fill all grid boxes within this bounding box (plus bloat)
- Falls back to per-point bloating when corners have mixed jump counts

### Jump Time Penalty
When `jump_time_penalty=True`, each discrete jump consumes $\varepsilon$ time from the total horizon $\tau$. This prevents trajectories with many jumps from being flowed for an artificially long continuous time.

### Boundary Bridging
When `use_boundary_bridging=True` (default), boxes whose corners have mixed jump counts use a stratified enclosure that bridges the jump boundary more accurately.

### Cylindrical Box Map
`HybridBoxMap.compute_cylindrical` handles 4D systems $(x, y, \dot{x}, \dot{y})$ where the first two coordinates are constrained to a disk $x^2 + y^2 \leq r^2$. Samples in cylindrical coordinates and skips boxes outside the constraint region.

## Morse Decomposition

Given the box map graph $G$:
1. Compute strongly connected components (SCCs)
2. Filter to non-trivial SCCs (size > 1, or size 1 with self-loop) -- these are the Morse sets
3. Build the condensation graph restricted to non-trivial SCCs
4. Take the transitive reduction to get the Hasse diagram (partial order on Morse sets)

## Regions of Attraction

For each Morse set $M_i$, compute the backward reachable set in the box map graph: all boxes from which there exists a path to some box in $M_i$.

## MultiGrid Framework

For systems with discrete operational modes (e.g., thermostat on/off), `MultiGrid` maintains a separate grid per mode. `MultiGridBoxMap` stores transitions as `(mode, box_index)` pairs, capturing both intra-mode and inter-mode transitions.

## Module Map

```
hybrid_dynamics/
  __init__.py          # Public API (version 0.2.0)
  src/
    hybrid_system.py     # HybridSystem: ODE + event + reset
    hybrid_trajectory.py # HybridTrajectory, TrajectorySegment
    hybrid_time.py       # HybridTime, HybridTimeInterval
    grid.py              # Grid: rectangular domain discretization
    box.py               # Box, SquareBox geometric primitives
    hybrid_boxmap.py     # HybridBoxMap: box map computation (sample-and-bloat)
    morse_graph.py       # create_morse_graph: SCC-based Morse decomposition
    roa_utils.py         # compute_roa, analyze_roa_coverage
    multigrid.py         # MultiGrid, MultiGridBoxMap (multi-mode systems)
    cubifier.py          # DatasetCubifier: trajectory -> box representation
    evaluation.py        # evaluate_grid, parallel point evaluation
    config.py            # Global configuration (bloat, tolerances, logging)
    plot_utils.py        # HybridPlotter, Morse/ROA visualization
    demo_utils.py        # Run directory management, caching
    data_utils.py        # GridEvaluationResult: save/load (.json, .npz)
    grid_utils.py        # Grid utility functions
    print_utils.py       # Verbose printing
    timing_utils.py      # Performance profiling
    trajectory_utils.py  # Trajectory helpers
  examples/
    bouncing_ball.py     # Ball with coefficient of restitution
    rimless_wheel.py     # Passive walking on slope
    bipedal.py           # 4D bipedal walker (cylindrical constraint)
    thermostat.py        # On/off thermostat (multi-mode)
    unstableperiodic.py  # System with unstable periodic orbit
```

## Example Systems

Each example follows the pattern:
```python
class SystemName:
    def __init__(self, parameters):
        self.system = self._create_system()

    def _create_system(self) -> HybridSystem:
        # Closures capture self for parameter access
        ...
        return HybridSystem(ode, event_function, reset, ...)
```

Current examples: bouncing ball, rimless wheel, bipedal walker, thermostat, unstable periodic orbit.

## Typical Workflow

```python
from hybrid_dynamics import HybridSystem, Grid, HybridBoxMap, create_morse_graph
from hybrid_dynamics.src.roa_utils import compute_roa

# 1. Define system (or use an example)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
wheel = RimlessWheel(alpha=0.4, gamma=0.2)

# 2. Create grid
grid = Grid(bounds=[[-1, 1], [-2, 2]], subdivisions=[100, 100])

# 3. Compute box map
box_map = HybridBoxMap.compute(
    grid=grid, system=wheel.system, tau=0.5,
    sampling_mode='corners', bloat_factor=0.1,
    enclosure=True
)

# 4. Morse decomposition
graph = box_map.to_networkx()
hasse, morse_sets = create_morse_graph(graph)

# 5. Regions of attraction
roa_dict = compute_roa(graph, morse_sets)
```

## Caching

Box maps are cached with MD5 hashes of the full configuration (system parameters, grid bounds, subdivisions, tau, bloat factor). Use `create_config_hash`, `load_box_map_from_cache`, `save_box_map_with_config` from `demo_utils`.

## Parallel Processing

For parallel box map computation, the system must be picklable. Define a factory function at **module level**:

```python
def create_rimless_wheel_system(alpha=0.4, gamma=0.2, max_jumps=50):
    return RimlessWheel(alpha=alpha, gamma=gamma, max_jumps=max_jumps).system

box_map = HybridBoxMap.compute(
    ..., parallel=True,
    system_factory=create_rimless_wheel_system,
    system_args=(0.4, 0.2, 50)
)
```
