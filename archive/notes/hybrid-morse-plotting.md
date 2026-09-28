# Plotting hybrid Morse sets

## Native CMGDB Atlas path

`PlotHybridMorseSets` now accepts an Atlas-backed native CMGDB `MorseGraph`
(together with its `SuspensionAtlasCharts`) or a cached `AtlasMorsePlotData`.
For each CMGDB Morse vertex it reads the tagged rectangles from
`MorseGraph.morse_set_chart_boxes`; it does not rebuild SCCs from a Python
relation. Base boxes are projected in physical coordinates, the optional
handle panel uses intrinsic guard-coordinate--phase coordinates, and the graph
panel draws the CMGDB Morse order. The handle panel is off by default.

```python
from hybrid_dynamics import PlotHybridMorseSets

plot = PlotHybridMorseSets(
    morse_graph,
    atlas_charts=setup.charts,
    axis_labels=(r"$\theta$", r"$\dot\theta$"),
)
```

For expensive runs, `extract_atlas_morse_plot_data` and
`save_atlas_morse_plot_data` store the exact graph, chart tags, and rectangles
as JSON. Loading that cache regenerates a figure without reevaluating the
hybrid dynamics. `demo/generate_atlas_morse_diagnostics.py` implements this
workflow for the accepted rimless-wheel (`tau=2`, depth 12) and bouncing-ball
(`tau=1.5`, depth 10) configurations, checks the fresh node/edge/box counts
against their acceptance reports, and writes new files under
`paper/figures/atlas-diagnostics`.

The Atlas plotting path deliberately rejects legacy transient overlays,
schematic reset arrows, cemetery markers, and arbitrary Conley-index
annotations. It accepts a second, narrowly typed input loaded by
`load_atlas_finite_relation_index_annotations`: a persisted physical audit in
which the top-cell pair, actual-Atlas-box reset-quotient nerve, relative
cellular pair, acyclic carrier, selected chain map, and finite algebra have all
passed their recorded checks. The loader also matches model, depth, fixed
time, and Morse-node ids to the plot cache and requires continuous-system
certification and analytic-label fallback to remain false. Thus a plotted
tuple is a shift class for the finite sampled relation over `GF(5)`, not an
index certified for the continuous fixed-time suspension map.

The visual convention is intentionally close to CMGDB's usual output: Morse
node ids select the same cyclic CMGDB color palette, only recurrent boxes are
colored, and the directed graph places repellers above attractors. The Atlas
adapter draws the exact rectangular patches instead of converting them to
untagged square scatter markers, because base and handle charts may have
different coordinates (and, in the general Atlas API, different dimensions).
By default the graph labels use only `M(i)`. The accepted physical diagnostic
runner adds a second line only after loading the corresponding persisted
finite-relation audit. Every such figure carries a footer stating that the
tuple is a finite sampled-relation Conley index over `GF(5)` and that
continuous-system certification has not been established.

Figures produced by `demo/generate_hybrid_morse_figures.py` remain exploratory
visualizations of the older standalone-grid sampled relation. They are not
figures of the current CMGDB Atlas path and should not be used as evidence for
the physical Morse graph, a global attractor lattice, or a Conley index.

## Legacy plotting API

For inspection of a legacy standalone-grid result, the existing API uses the
same deterministic component/color assignment in its state-space, handle, and
graph panels:

```python
from hybrid_dynamics import PlotHybridMorseSets, save_hybrid_morse_figure
from hybrid_dynamics.examples.physical_suspension_grid import (
    build_rimless_wheel_suspension_grid,
)

result = build_rimless_wheel_suspension_grid(
    subdivisions=(201, 201),
    handle_slabs=16,
    padding_cells=1.0,
)
plot = PlotHybridMorseSets(
    result,
    proj_dims=(0, 1),
    base_view="domain",
    show_handles=False,
    show_morse_graph=True,
    show_cemetery=False,
)
save_hybrid_morse_figure(plot, "rimless_hybrid_morse")
```

The legacy layout has three geometrically distinct parts:

1. The base panels show projected recurrent base boxes. Sampled transient
   support can optionally be shown in gray with `show_transient=True`; it is
   never labeled as a Morse set.
2. The handle panel projects the actual globally identified guard-prism cells
   in guard-coordinate--phase coordinates, with `s in [0,1]`. It does not
   interpolate a reset through the base state space. Legacy results without a
   guard grid fall back to schematic phase lanes.
3. The graph panel is the transitive reduction of reachability between the
   displayed recurrent SCCs, computed through the full relation (including
   transient SCCs). Its node colors are the color key for the geometric
   panels. A cemetery SCC is categorical and is hidden by default.

For four-dimensional systems the default projections are `(0, 1)` and
`(2, 3)`.  Pass several pairs through `proj_dims`, for example
`((0, 1), (0, 2))`, to choose another paper layout.  The plotting function
does not call `show()` and returns its figure and axes for further editing.

Presentation defaults omit titles, panel titles, legends, grids, and schematic
reset arrows. Optional Conley-index text is displayed only when genuine
annotations are supplied through `conley_indices`; no index is inferred from
an SCC. Each
annotated graph vertex is an annotation-sized CMGDB-style ellipse with two
lines: `M(i)` and the degreewise tuple `(p0, p1, ...)`.  For example, a caller
that has already computed and validated the index data can pass
`conley_indices={0: ("x-1", "0")}`.  Omitting the argument leaves the second
line absent rather than synthesizing an index from the SCC or its cell count.

Set `show_handles=False` when the reset-cylinder geometry is already explained
elsewhere and the desired figure is the usual base-space Morse-set projection
beside its Conley--Morse graph.  This removes only the presentation panel; it
does not remove phase cells from the relation, SCC computation, or Morse order.

The plot reports the recurrent SCCs of the supplied finite relation. The
legacy full-grid example builder evaluates every active base cell, unlike the
older one-orbit smoke pipeline. Its corner-sampled padded image cover does not
establish a whole-cell outer relation, refinement persistence, an index pair,
or a physical Conley-index claim. Its exploratory figures and JSON diagnostics
are generated by:

```console
python demo/generate_hybrid_morse_figures.py
```

For the current rimless-wheel Atlas result, the authoritative geometric output
remains the CMGDB graph and chart-aware box lists recorded through
`morse_set_chart_boxes`; see
[`RIMLESS_WHEEL_ATLAS.md`](RIMLESS_WHEEL_ATLAS.md). The depth-12 two-node
local-window result still has source-cell exits and does not support a global
attractor-lattice conclusion or a continuous-system Conley-index claim. Its
persisted quotient-nerve/carrier audit does support the displayed finite
sampled-relation shift classes over `GF(5)`. Publication use must keep those
two scopes separate.
