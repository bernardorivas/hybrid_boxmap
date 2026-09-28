# Rimless wheel through the CMGDB suspension box map

The physical wheel computation now uses CMGDB for the grid, directed graph,
SCCs, and Morse order.  The hybrid-specific change is confined to the box-map
callback supplied to `CMGDB.AtlasModel`.

The Atlas has two two-dimensional charts:

- base: `(theta, omega)` on `[-0.2,0.6] x [-0.5,1]`;
- reset handle: `(omega_guard, s)` on `[0,1] x [0,1]`.

Only nonnegative guard velocity is included because the wheel event detects
increasing crossings of `theta=0.6`.  The callback samples each source cell on
a tensor grid, keeps different event stages in different tagged rectangles,
and inserts both chart representatives at the guard/phase-zero and
reset/phase-one seams.  CMGDB covers the returned finite tagged union directly;
it never takes one Euclidean hull across a reset.

Run the recorded local-window comparison with:

```bash
cd code
PYTHONPATH=. .venv/bin/python demo/run_rimless_wheel_atlas.py \
  --depth 12 --tau 2 --samples-per-axis 3 --padding-cells 1 \
  --require-local-window-gates
```

At depth 12 (64 by 64 cells in each chart), the nonempty relation in the
selected chart window has exactly two Morse nodes and the edge `1 -> 0`. Node
0 is the walking-cycle candidate and node 1 contains the saddle. The full
settings and the depth-10 comparison are stored in
`data/rimless_wheel_atlas/acceptance_tau200_depth10_depth12.json`.

This is a local-window computation on the nonempty part of the recorded
relation.  Of the 8,192 depth-12 source cells, 6,463 have nonempty images and
1,729 have empty images.  Across CMGDB's two callback passes, 33,290 of 147,456
tensor-point evaluations fail because the sampled trajectory or encoded target
does not remain in the declared local charts.  Those failures are not silently
reclassified as covered points.  They are also distinct from the analytic gait
audit: all 363 gait endpoint probes evaluate successfully and are covered.

The acceptance run checks all of the following without deleting or relabeling
Morse nodes:

- independently sampled endpoints along the analytic walking suspension cycle
  are contained in the recorded image of every incident source cell;
- every one of the 6,463 nonempty box-map values is connected in closed-cell
  incidence after
  applying both suspension seam identifications;
- the base-chart support of the walking Morse set is connected;
- the analytic saddle and walking cycle occur in distinct Morse nodes with the
  direct saddle-to-gait order; and
- no adjacent tensor samples skip an event stage.

These are falsification and reproducibility gates.  They do not certify the
whole-cell outer-enclosure condition, and this Atlas run does not compute or
display a Conley index.  Empty source images are excluded from the connected-
image audit, so the two-node result is not yet a global attractor-lattice
computation on the displayed rectangle.  A global interpretation requires
either a forward-complete compactification/cemetery construction that records
all exits or a larger compact, forward-invariant state space `X` with a valid
outer relation on every source cell.

## Chart-aware diagnostic figure

`demo/generate_atlas_morse_diagnostics.py rimless-wheel` extracts the actual
tagged rectangles through `MorseGraph.morse_set_chart_boxes`, verifies its
nodes, edge, and per-chart counts against the acceptance JSON, checks that the
closed base boxes of `M(0)` form one connected union, and then draws the base
support beside the CMGDB Morse graph. It stores a reusable JSON plotting cache
under `data/rimless_wheel_atlas` and writes new, non-manuscript diagnostics
under `paper/figures/atlas-diagnostics`. It adds no Conley label.
