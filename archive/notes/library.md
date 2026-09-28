# Hybrid Dynamics library

This is the earlier README of the repository. It describes the general library
(hybrid trajectories, box maps, Morse decompositions) and the tagged-chart
(Atlas) pipeline. Most of the material it refers to is now in
[`archive/`](../README.md). For the computations of the paper, see the
[README](../README.md).

A Python library for analyzing hybrid dynamical systems: bouncing balls, walking robots, thermostat, etc.

## Key features

- Simulate hybrid trajectories
- Build combinatorial models (boxmap)
- Compute SCC/Morse decompositions and region-of-attraction diagnostics
- Visualize: phase portraits, time series, box maps
- Built-in examples of some hybrid systems

## Installation

Hybrid Dynamics requires Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The reusable library is silent by default. Applications and command-line
runners can opt into operational logs without changing result data:

```python
from hybrid_dynamics import configure_logging

configure_logging(level="INFO")
# Or: configure_logging(level="DEBUG", log_file="run.log")
```

Batch failures are summarized by category instead of producing one message per
sample. Machine-readable runners use either one final JSON document or a JSON
event stream; incidental library diagnostics use logging rather than being
mixed into that protocol.

The full maintained suite exercises the Atlas extension in the project CMGDB
fork. Install the development tools and the exact tested CMGDB revision, then
run the following. The pinned CMGDB install builds a C++ extension; its
[installation notes](https://github.com/bernardorivas/CMGDB#installation)
list the required compiler and native libraries.

```bash
python -m pip install -e ".[dev]"
python -m pip install --force-reinstall --no-deps --no-cache-dir \
  "git+https://github.com/bernardorivas/CMGDB.git@4150b75304a9b70e5656d46838353a18041a8beb"
python -m pytest -q
```

Research scripts that print reports or create exploratory figures live under
`experiments/`; they are not collected as automated tests.

The simulation package installs from this repository alone. The fixed-time
Atlas workflows additionally require the project
[CMGDB fork](https://github.com/bernardorivas/CMGDB), which provides
`AtlasModel` and the tagged multi-rectangle box-map interface.

## Package structure

The computation is organized in three layers. Each layer uses only the ones
below it, so ordinary CMGDB can be used without any hybrid code
(`import CMGDB` imports nothing from `hybrid_dynamics`).

1. **CMGDB core.** Cubical grids, box maps, the map graph, strongly connected
   components, Morse graphs, and Conley indices, as in upstream CMGDB.
2. **Generic extensions in the project fork of CMGDB** (commit `4150b75`).
   `AtlasModel` (a grid on a finite disjoint union of rectangular charts with
   tagged-union box maps), CSR map-graph checkpoints, and
   `ComputeRelativeHomologyShiftClass` (the shift class of an explicit relative
   chain map over `GF(5)`). None of these refers to guards, resets, or
   suspensions.
3. **`hybrid_dynamics`** (this package): hybrid systems, the unit-handle
   suspension semiflow, and the constructions below.

| Construction | Modules | Relation to the manuscript |
|---|---|---|
| Paper suspension grid `Xi_n` | `src/suspension_grid.py`, `src/suspension_grid_relation.py`, `src/suspension_grid_conley.py`, `src/suspension_grid_plot.py`, `examples/paper_examples.py`, `demo/run_paper_examples.py` | Implements `def:suspension-grid` with the cofiltration `Xi_n = Xi_{n-1} ^ Xi_n(X_n)`, `d_n` and the base readout of `prop:suspension-grid` and `prop:finite-grid-preimage`, and the sampled map of Section "Examples". Each elementary piece is sampled at its four vertices by default (`eval_mode="corners"`, as in CMGDB), with `center`, `random`, and the earlier `tensor` rule as options; the image is the set of atoms containing the endpoints, padded by one atom. SCCs use `scipy.sparse.csgraph`; CMGDB is used only for the final shift class. See [PAPER_GRID.md](PAPER_GRID.md). |
| Atlas tagged-chart pipeline | `src/cmgdb_suspension_boxmap.py`, `examples/*_atlas.py` | Runs `CMGDB.AtlasModel` on a base chart and a handle chart that are refined separately. The cells are not the atoms of `Xi_n`: collars are not merged into base elements and the reset preimages do not cut the handle. Produced the earlier manuscript figures. |
| Explicit fixed-time grid | `src/fixed_time_suspension_grid.py`, `src/sampled_suspension.py`, `src/implicit_phase_scc.py` | Base cells plus handle phase slabs over guard cells; pointwise unit-handle clock reference. |
| Finite complexes and index front end | `src/suspension_complex.py`, `src/atlas_conley.py` | Quotient nerve of actual rectangles, acyclic carriers, chain selectors; shared by the Atlas and the paper index computations. |

## Repository layout

- `hybrid_dynamics/src/` — reusable simulation, grid, suspension, Atlas,
  finite-relation, plotting, logging, and JSON I/O components
- `hybrid_dynamics/examples/` — importable physical-system definitions and
  reproducible workflow builders
- `hybrid_dynamics/tests/` — tests for the suspension, Atlas, Conley-audit,
  and physical-example layers
- `hybrid_dynamics/*.md` — per-model and per-workflow scientific status reports
- `test/` — logging and compatibility tests for the legacy suspension and SCC
  APIs
- `demo/` — command-line entry points for reproducible computations and
  figure generation
- `experiments/` — exploratory or diagnostic scripts that are run manually
- `data/` — compact acceptance records and provenance needed to review a
  result; large regenerable relation checkpoints remain local
- `figures/` — retained reference output from the classic examples
- `docs/archive/` — historical implementation notes retained for context

The package-level and top-level scientific status documents describe what each
recorded result does and does not certify. Start with
[`SUSPENSION_COMPUTATION.md`](../SUSPENSION_COMPUTATION.md) for the current
fixed-time computation boundary.

Evidence and status are organized by workflow:

- bouncing ball and rimless wheel —
  [`BOUNCING_BALL_ATLAS.md`](BOUNCING_BALL_ATLAS.md) and
  [`RIMLESS_WHEEL_ATLAS.md`](RIMLESS_WHEEL_ATLAS.md)
- Garcia walker Atlas, local, tube, and guard-aligned investigations —
  [`GARCIA_PASSIVE_WALKER_ATLAS.md`](GARCIA_PASSIVE_WALKER_ATLAS.md)
  and [`WALKER_LOCAL_INDEX.md`](WALKER_LOCAL_INDEX.md)
- adaptive spiking-neuron relation, provenance, and finite-relation index
  audit —
  [`SPIKING_NEURON_ATLAS.md`](SPIKING_NEURON_ATLAS.md)
- bouncing-ball and rimless-wheel finite-relation index outputs —
  [`PHYSICAL_CONLEY_RESULTS.md`](PHYSICAL_CONLEY_RESULTS.md)

The implementation, tests, and fingerprinted records under `data/` are the
computational source of truth; manuscripts provide background and motivation.
Current index outputs are finite-relation results, not certified Conley indices
for the underlying continuous systems.

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
trajectory = system.simulate([1.0, 0.0], time_span=(0, 2))
```

## Examples

Library contains examples of

```bash
python demo/run_bouncing_ball.py    # Bouncing ball with damping
python demo/run_rimless_wheel.py    # Passive walking
python demo/run_thermostat.py       # On/off temperature control
python demo/run_bipedal.py          # Two-legged walker (3D LIP)
```

The classic demos create visualizations in `figures/`. Newer audit runners
write explicitly named records under `data/`, emit JSON to standard output, or
accept an output path; see each runner's `--help` text.

### Fixed-time suspension examples

The primary computation is a CMGDB computation of one fixed-time suspension
map.  `CMGDB.AtlasModel` owns the grid, map graph, SCCs, and Morse order.
`CMGDBSuspensionBoxMap` changes only the box-map callback: instead of one
Euclidean rectangle, it returns a finite union of rectangles tagged by the
base chart or the intrinsic guard-coordinate--handle-phase chart.  CMGDB
covers each piece separately, so a reset never creates a fictitious rectangle
between pre- and post-reset states.

The callback also has a disabled-by-default
`single_handle_bridge=True` mode for one specific terminal sampling ambiguity.
If adjacent source tensor nodes land in base stages `2*j` and `2*j+2`, a
bounded deterministic bisection searches their source segment for stage
`2*j+1`.  A successful witness contributes a tagged full-handle carrier and
both quotient-face representatives to that source's terminal image; it does
not add intermediate-time graph arrows and it does not convexify the two base
branches.  Gaps larger than two, non-base gaps, failed probes, and exhausted
searches remain unresolved hard gates.  Every attempt, probe, carrier, and
assumption is exposed in `SourceBoxMapDiagnostic`.  Like the rest of the
sample-and-bloat callback, this is a nonrigorous outer-cover assumption rather
than a whole-cell enclosure certificate.  Bridge-bearing source records are
also retained separately in aggregate diagnostics even after the ordinary
diagnostics window is full, so no synthesized carrier loses its probe
provenance.

The rimless wheel is the first physical integration of this path.  Run its
recorded acceptance configuration with:

```bash
python demo/run_rimless_wheel_atlas.py \
  --depth 12 --tau 2 --samples-per-axis 3 --padding-cells 1 \
  --require-local-window-gates
```

The archived depth-12 nonempty local-window relation has two Morse nodes and
the edge `M(1) -> M(0)`; the walking candidate and saddle are distinct. Its
walking-reference endpoint coverage, nonempty-image quotient connectivity,
walking-support, and event-stage gates pass. The same run has 1,729 empty
source images and 33,290 failed sample evaluations across repeated callback
passes under its domain-path rule, predominantly consistent with exits from
the chosen chart window. It is therefore not yet a compact global self-map or
a global attractor-lattice computation. A justified cemetery/forward-complete
compactification or isolating restriction is still needed. The finite gates
are not a whole-cell enclosure proof, and the run computes no Conley index. See
[RIMLESS_WHEEL_ATLAS.md](RIMLESS_WHEEL_ATLAS.md) and the
[acceptance record](../../data/rimless_wheel_atlas/acceptance_tau200_depth10_depth12.json).

The bouncing ball now has an Atlas integration with an independent closed-form
ballistic audit. At `tau=1.5`, depths 8 and 10 both produce one connected
physical Morse node `M(0)`, with zero misses among 1,236 shrinking-bounce and
rest-fiber probes, zero disconnected nonempty image values, and zero skipped
event stages. The predeclared `tau=2.0` screen instead produces four and two
nodes and is retained as a failed single-node screen. Domain exits remain
explicit, so this is local-window evidence, not a global self-map result. See
[BOUNCING_BALL_ATLAS.md](BOUNCING_BALL_ATLAS.md) and the
[refinement record](../../data/bouncing_ball_atlas/acceptance_tau150_tau200_depth8_depth10.json).

The Garcia passive walker has a deliberately coarse, four-dimensional Atlas
plumbing run and a native active-subgrid diagnostic:

```bash
python demo/run_garcia_passive_walker_atlas.py \
  --depth 4 --tau 0.5 --require-plumbing-gates
```

Its handle uses three intrinsic guard coordinates plus phase and enforces
`phi=2*theta` and transverse heel strike throughout the chart.  The depth-4
run checks the stored period-two gait, quotient connectivity, skipped event
stages, and explicit failures. `AtlasModel.set_active_subgrid` now constructs
selected tagged dyadic cells directly, and the walker adapter uses a
positive-radius union of full four-dimensional cells around both stored
strides and both reset-handle traversals. The bounded depth-8/depth-12 screen
does not close under the sampled relation and has disconnected image covers,
so it is retained as a rejected diagnostic rather than a resolved walker
Morse result. See
[GARCIA_PASSIVE_WALKER_ATLAS.md](GARCIA_PASSIVE_WALKER_ATLAS.md).
The coarse output and active-screen counts are archived in
[acceptance_tau050_depth4.json](../../data/garcia_passive_walker_atlas/acceptance_tau050_depth4.json)
and
[active_subgrid_screen_tau050.json](../../data/garcia_passive_walker_atlas/active_subgrid_screen_tau050.json).
None of these physical Atlas modules currently certifies a Conley index of the
underlying continuous fixed-time suspension map.
The generic topology/carrier front end now builds a verified quotient-aware
nerve from the actual 2D ball/wheel Atlas rectangles, evaluates every cell and
face, and constructs a subordinate chain selector only after acyclicity and
pair-preservation checks. Finite-relation shift classes can be computed and
are explicitly scoped as non-certified for the continuous system. The stronger
continuous-system endpoint remains locked unless the physical callback and
index-pair obligations are separately certified. See
[ATLAS_CONLEY.md](ATLAS_CONLEY.md).

## Mathematical background

### Hybrid time
Hybrid trajectories evolve in "hybrid time" (t,j) where:
- t = continuous time (when the system flows)
- j = jump counter (increments at discrete transitions)

### Hybrid system
A hybrid system consists of
1. **Flow**: `ẋ = f(x,t)`
2. **Guard**: the points of `g(x) = 0` at which the flow crosses (or touches)
   the surface in the event direction `d`, that is `d · dg/dt ≥ 0`
3. **Reset**: `x⁺ = r(x⁻)`

A trajectory starts with a jump only from the guard (or from the side past the
event surface); a point of `g = 0` at which the flow moves against the event
direction, such as the rimless wheel's `(α + γ, ω)` with `ω < 0`, flows. See
`HybridSystem.on_guard` and `HybridSystem.jumps_at_start`; an explicit
`guard_predicate` replaces this rule.

### CMGDB box map / combinatorial analysis

```python
import CMGDB
from hybrid_dynamics import (
    CMGDBSuspensionBoxMap,
    SuspensionAtlasCharts,
    build_cmgdb_atlas_model,
)
from hybrid_dynamics.examples import BouncingBall

system = BouncingBall().system
charts = SuspensionAtlasCharts(
    base_bounds=((0.0, 2.0), (-5.0, 5.0)),
    guard_bounds=((-5.0, 0.0),),
    guard_coordinates=lambda x: [x[1]],
    guard_embedding=lambda u: [0.0, u[0]],
)
box_map = CMGDBSuspensionBoxMap(system, charts, t_star=1.0)
model = build_cmgdb_atlas_model(box_map, depth=10)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
```

The callback evaluates exactly `t_star` units of suspension time and returns
endpoint boxes.  States merely visited while flowing or crossing a reset are
not edges of the fixed-time graph.  If an endpoint lies inside a reset handle,
its handle cell is a legitimate target.  When an image reaches an identified
seam, the callback supplies both tagged face representatives so that the map
relation respects the quotient attachment.

### Pointwise clock reference and legacy prototypes

The attractor correspondence uses total execution time `T = t + j`, with one
unit of suspension time for every reset.  The pointwise API is useful for
testing this clock and the Atlas callback:

```python
from hybrid_dynamics import (
    Grid,
    HandleSuspensionSample,
    locate_augmented_cells,
    simulate_suspension_endpoint,
)
from hybrid_dynamics.examples import BouncingBall

system = BouncingBall().system
grid = Grid(bounds=system.domain_bounds, subdivisions=[20, 20])
endpoint = simulate_suspension_endpoint(system, [1.0, 0.0], total_time=0.5)
cells = locate_augmented_cells(endpoint, grid, handle_slabs=8)

if isinstance(endpoint, HandleSuspensionSample):
    print(endpoint.phase)  # the exact sample ended during the reset handle
```

`sampled_suspension.py` is a pointwise reference, not a graph-construction
engine.  `fixed_time_suspension_grid.py` and `implicit_phase_scc.py` are legacy
diagnostic/prototype modules.  In particular, expanding a sampled reset
itinerary as `base -> phase -> base` inserts intermediate-time transitions;
those transitions are not edges of the time-`t_star` map.  The graph utilities
remain valid for the finite graphs they are given, but graph equality there is
not evidence that the graph is an outer approximation of the fixed-time map.
See [SUSPENSION_COMPUTATION.md](../SUSPENSION_COMPUTATION.md).

The topology-bearing reset quotient and generalized CMGDB payload are
described in
[hybrid_dynamics/SUSPENSION_COMPLEX.md](SUSPENSION_COMPLEX.md).

The legacy SCC reconstruction accepts a graph on base cells plus implicit
phase descriptors:

```python
from hybrid_dynamics import PhasePathDescriptor, reconstruct_from_base_descriptors

jump = PhasePathDescriptor("guard-0", "a", "b", (0,))
result = reconstruct_from_base_descriptors(["a", "b"], [jump])

# This is the SCC decomposition of this explicitly expanded graph.
full_sccs = result.components
condensation = result.condensation
```

It must not be interpreted as a fixed-time suspension relation merely because
the descriptor came from a reset itinerary.

## Main classes

- `HybridSystem` – Define and simulate hybrid systems
- `Grid` – Discretize state space into boxes  (`MultiGrid` available)
- `CMGDBSuspensionBoxMap` – Return tagged fixed-time endpoint boxes to
  `CMGDB.AtlasModel`
- `SuspensionAtlasCharts` – Specify physical base and intrinsic handle charts
- `PlotHybridMorseSets` – Plot native CMGDB Atlas Morse boxes and Morse order;
  the intrinsic handle panel is optional and off by default
- `AtlasMorsePlotData` – Cache exact chart-tagged Morse boxes as JSON so plots
  do not require an expensive dynamical recomputation
- `HybridBoxMap` – Older standalone sampled-grid prototype
- `HybridPlotter` – Visualize trajectories and box maps

## Future directions

- Whole-cell event/flow enclosures that certify the Atlas box-map outer
  relation
- Certified relative index pairs for the accepted ball/wheel fixed-time
  relations
- A scientifically valid active domain for the four-dimensional passive
  walker that is closed under the certified relation
- Automatic index-pair and carried-chain-map construction on the physical
  reset-glued suspension complex

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

MIT License – see [LICENSE](../../LICENSE) file.
