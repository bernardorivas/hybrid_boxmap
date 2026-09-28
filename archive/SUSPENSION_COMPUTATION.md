# CMGDB box maps for a fixed-time hybrid suspension

The suspension grid `Xi_n` of the manuscript (Definition `def:suspension-grid`)
is implemented separately; see [PAPER_GRID.md](notes/PAPER_GRID.md). This document
describes the Atlas tagged-chart pipeline, whose base and handle charts are
refined separately and therefore do not form `Xi_n`.

## Goal

The computational object is the continuous fixed-time map

```text
f_tau = Phi_H^tau
```

of the hybrid suspension semiflow. The primary implementation uses CMGDB's
ordinary grid, directed-graph, SCC, and Morse-order machinery. The
hybrid-specific work is confined to constructing the box map passed to CMGDB.

The suspension is represented by two kinds of tagged chart:

- a base chart containing the physical state variables; and
- an intrinsic handle chart containing guard coordinates and the unit reset
  phase `s in [0,1]`.

The handle chart is generated from the guard/reset data on the base space. It
is not an ambient Euclidean embedding of the full quotient suspension.

## The graph is the graph of one fixed-time map

For each source cell `xi`, a valid multivalued box map returns cells covering
the endpoint set

```text
f_tau(|xi|).
```

An edge `xi -> eta` means that `eta` belongs to that fixed-time image cover.
It does not mean that a trajectory visits `eta` at some intermediate time.

This distinction is essential at resets. Suppose a trajectory starting in a
base cell flows to a guard, traverses a handle, resets, and then reaches a base
endpoint at time `tau`. The fixed-time relation contains the source-to-endpoint
edge. It does not replace that edge by an itinerary path

```text
base source -> intermediate handle -> post-reset base.
```

Those intermediate vertices occur at different times and therefore are not
edges of `f_tau`. A handle cell is a genuine target only when the time-`tau`
endpoint itself lies inside the handle. Similarly, the two tagged
representatives of a quotient seam are included to represent one endpoint
neighborhood at the identification; they are not extra time steps.

## Primary architecture

The implementation path is

```text
hybrid flow + guard + reset + tau
                 |
                 v
CMGDBSuspensionBoxMap(source chart, source rectangle)
                 |
                 v
finite union of tagged target rectangles
                 |
                 v
CMGDB.AtlasModel
                 |
                 v
CMGDB map graph -> SCCs -> Morse graph/order
```

`CMGDB.AtlasModel` is a grid on a finite disjoint union of rectangular charts.
Its callback has the form

```python
(source_chart_id, source_bounds) -> [
    (target_chart_id, target_bounds),
    ...,
]
```

CMGDB covers every returned rectangle separately. It never forms one
Euclidean hull across target pieces or charts. This is the needed change to
the ordinary box-map interface: CMGDB's graph algorithms themselves do not
need a hybrid replacement.

`Atlas` does not automatically glue chart faces. The suspension callback
therefore emits both tagged representatives at the guard/phase-zero seam and
at the phase-one/reset seam when they belong to the target cover. Quotient
incidence used for geometric diagnostics and the eventual cellular complex is
supplied separately.

In this precise sense, the method is generated from the base grid: the handle
coordinates and seam data come from guard cells and the reset map. The finite
graph nevertheless has real handle-chart cells. A graph whose vertices are
literally only physical base boxes cannot represent fixed-time endpoints that
lie inside a handle.

## Current sampled callback

`hybrid_dynamics.src.cmgdb_suspension_boxmap` provides:

- `SuspensionAtlasCharts`, which specifies the base chart, intrinsic guard
  coordinates, guard embedding, and handle chart;
- `CMGDBSuspensionBoxMap`, the tagged-union fixed-time callback; and
- `build_cmgdb_atlas_model`, which installs that callback in the local CMGDB
  fork.

For every source box, the callback currently:

1. samples a tensor grid containing corners and interior points;
2. evolves each point for exactly `tau` units of suspension time, using
   `T = flow time + completed reset handles`;
3. labels each endpoint by target chart and event stage;
4. hulls endpoints only within one `(chart, stage)` stratum;
5. detects adjacent sample nodes whose stages meet at a suspension seam and
   adds both tagged seam representatives;
6. records a skipped stage as unresolved instead of repairing it with a
   fictitious cross-reset rectangle; and
7. applies the requested chartwise CMGDB-style cell padding.

The resulting callable is passed directly to `CMGDB.ComputeMorseGraph` through
`AtlasModel`.

```python
import CMGDB
from hybrid_dynamics import (
    CMGDBSuspensionBoxMap,
    SuspensionAtlasCharts,
    build_cmgdb_atlas_model,
)

charts = SuspensionAtlasCharts(
    base_bounds=base_bounds,
    guard_bounds=guard_bounds,
    guard_coordinates=guard_coordinates,
    guard_embedding=guard_embedding,
)
box_map = CMGDBSuspensionBoxMap(system, charts, t_star=tau)
model = build_cmgdb_atlas_model(box_map, depth=depth)
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
```

## What the sampled callback establishes

The SCCs and Morse order are exact properties of the finite relation returned
to CMGDB. The callback also records falsification diagnostics, including:

- target strata observed only at interior source samples;
- failed sample evaluations and empty source images;
- adjacent sample nodes that skip an event stage;
- dense independent endpoint probes against the recorded relation; and
- connectedness of a returned image under closed-cell incidence after the
  suspension seam identifications.

These checks are useful because an exact image of a connected source under the
continuous suspension map is connected. They can expose a missed event stratum
or a detached sampled branch. Quotient connectedness is not, by itself, an
outer-enclosure proof, and spatial connectedness is not a universal property
of every SCC support.

The current tensor-sample-and-bloat callback does not certify

```text
f_tau(|xi|) subset interior(|F(xi)|)
```

for a whole source cell. Corners plus interior samples can still miss a
nonlinear extremum or a thin event-time stratum. A theorem-level computation
needs interval, validated, or otherwise justified whole-cell flow/event/reset
enclosures. For a Conley index, the quotient-cell carrier must additionally be
face compatible and acyclic over the chosen coefficient field.

## Rimless-wheel acceptance record

The first physical Atlas run is documented in
[`hybrid_dynamics/RIMLESS_WHEEL_ATLAS.md`](notes/RIMLESS_WHEEL_ATLAS.md).
At depth 12 and `tau=2`, its nonempty local-window relation has two Morse nodes
and the edge `M(1) -> M(0)`. Independent walking-reference endpoints have no
misses, every nonempty returned image passes the quotient-connectedness check,
the walking candidate has connected base support, and no sampled adjacency
skips an event stage.

The same record also contains 1,729 empty source images and 33,290 failed sample
evaluations accumulated across CMGDB callback passes under
`require_domain_path=True`. These are predominantly consistent with exits
from the selected local chart window, but that interpretation is not a proof.
Consequently, the two-node result is not yet a compact global self-map and is
not yet a global attractor-lattice computation. A justified exit treatment,
such as a cemetery/forward-complete compactification or a proved isolating
restriction, remains necessary. No Conley index is computed by this run.

The complete recorded settings and counts are in
[`data/rimless_wheel_atlas/acceptance_tau200_depth10_depth12.json`](../data/rimless_wheel_atlas/acceptance_tau200_depth10_depth12.json).

## Legacy diagnostic and prototype modules

Several older modules remain useful for testing, but they are not the primary
fixed-time computation:

- `sampled_suspension.py` is the pointwise reference for the execution clock
  and for endpoints in the base or handle;
- `fixed_time_suspension_grid.py` constructs an exploratory standalone sampled
  relation from base/phase labels;
- `implicit_phase_scc.py` exactly reconstructs SCCs of a finite graph described
  by base vertices and phase gadgets; and
- `physical_suspension_examples.py` exercises algebraic relative-chain
  endpoints on hand-built local suspension skeletons.

The graph theorem behind `implicit_phase_scc.py` is valid for the expanded
finite graph it is given. The earlier use of reset itineraries as graph paths
is not a construction of the time-`tau` relation: it promotes states visited at
intermediate times to edges. Agreement between an explicitly materialized
itinerary graph and its implicit descriptor expansion proves only bookkeeping
agreement for that graph.

Likewise, the `x-1` outputs on hand-built periodic suspension skeletons are
algebraic integration regressions. They are not Conley indices of the sampled
physical Atlas relations.

## Conley-index boundary

`ComputeMorseGraph` supports `AtlasModel`. The legacy cubical CHOMP entry point
does not encode the reset quotient and therefore is not called directly on an
Atlas result.

A suspension Conley-index computation still requires:

1. a whole-cell outer relation for the physical fixed-time map;
2. a closed index pair derived from the selected recurrent component;
3. the reset-glued quotient cell complex;
4. a face-compatible acyclic carrier and carried chain map; and
5. only then, the existing CMGDB relative-homology and Frobenius/shift-class
   endpoint.

The generic 2D Atlas front end now supplies items 3 and 4 using the verified
nerve of the actual reset-glued ball/wheel rectangles, facewise raw covers,
acyclic-carrier checks, and an inductively constructed subordinate chain map.
It permits the explicitly scoped finite-relation CMGDB shift-class endpoint
after those algebraic gates pass. It refuses only the stronger claim that this
is the Conley index of the continuous fixed-time suspension map until items 1
and 2 are separately certified. See `hybrid_dynamics/ATLAS_CONLEY.md`.

There is no return map, second mapping torus, Leray reduction, or alternative
discrete-index formalism in this pipeline. The dynamical map throughout is the
fixed-time suspension map `f_tau`.

## Remaining end-to-end work

- replace sampled image construction with a justified whole-cell enclosure;
- specify and implement the exit/compactification semantics needed for a
  global attractor lattice;
- construct a declared four-dimensional passive-walker active domain that is
  closed under the certified relation. The native active-subgrid API,
  physical-coordinate depth-16/depth-20 relations, and guard-aligned
  `q=phi-2 theta` depth-16/depth-20 relations now exist. The aligned grid
  improves the stored-gait local diagnostics, but the complete unpruned
  reference component still meets open boundaries, lacks represented support,
  and has disconnected images. Both aligned runs are therefore persisted as
  negative diagnostics, with no accepted walker Morse result or index;
- derive physical index pairs and quotient-complex chain maps before reporting
  Conley indices.

Chart-aware plotting is implemented by `PlotHybridMorseSets`: it consumes
`MorseGraph.morse_set_chart_boxes` directly and draws the CMGDB Morse order
without recomputing SCCs. The accepted ball and wheel diagnostic caches and
figures are generated by `demo/generate_atlas_morse_diagnostics.py`. They are
still local-window sampled diagnostics, not evidence of the missing whole-cell
outer-enclosure or global-self-map hypotheses.
