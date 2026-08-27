# Bouncing-ball Atlas computation

The bouncing-ball example now uses the same native CMGDB path as the rimless
wheel. CMGDB constructs the grid, directed graph, SCCs, and Morse graph. The
only hybrid-specific part is the callback supplied to `CMGDB.AtlasModel`.

## Charts and box map

The Atlas has two two-dimensional charts:

- base: `(h, v)` on `[0,2] x [-5,5]`;
- reset handle: `(v_guard, s)` on `[-5,0] x [0,1]`.

Only `v_guard <= 0` is included because the impact event is the outgoing
(falling) part of `h=0`. The lower handle face is glued to `(0,v_guard)` and
the upper face to the reset state `(0,-0.8 v_guard)`. The callback uses a 3x3
tensor sample in each source box, one target-cell padding width, event-stage
separation, and both representatives of every detected quotient seam. A
trajectory is rejected if any recorded flow or reset state leaves the declared
base rectangle.

This is still a CMGDB-style sample-and-bloat relation. It is not a proof that
every whole-cell image is enclosed.

## Independent reference gate

The acceptance test does not reuse the callback's sampled event-driven flow
evaluator for its endpoint values. It evaluates

```text
h(t) = h0 + v0 t - g t^2/2,    v(t) = v0 - g t
```

and the positive ballistic impact root in closed form, inserting one unit of
suspension time per impact. The probe family contains rebound arcs, the two
identified reset faces, the exact rest fiber, and pre-impact speeds spaced
from `1e-6` through `5`. Thus the test approaches the Zeno state without asking
an instantaneous-reset evaluator to execute infinitely many impacts: at finite
suspension time only finitely many unit handles can be crossed.

## Recorded refinement screen

Both `tau=1.5` and `tau=2.0` were screened in advance at Atlas depths 8 and 10
(16x16 and 32x32 boxes per chart). The complete record is in
[`acceptance_tau150_tau200_depth8_depth10.json`](../data/bouncing_ball_atlas/acceptance_tau150_tau200_depth8_depth10.json).
The noninteger default also avoids making the fixed-time map the identity on
the period-one rest fiber, as `tau=2.0` does after two complete unit handles.
This observation motivates the default but is not used to explain away the
recorded `tau=2.0` nodes.

| tau | depth | Morse nodes | analytic misses | disconnected nonempty images | skipped stages | result |
|---:|---:|---:|---:|---:|---:|:---|
| 1.5 | 8 | 1 | 0 / 1236 | 0 / 384 | 0 | local gates pass |
| 1.5 | 10 | 1 | 0 / 1236 | 0 / 1494 | 0 | local gates pass |
| 2.0 | 8 | 4 | 0 / 1236 | 0 / 384 | 0 | single-node gate fails |
| 2.0 | 10 | 2 | 0 / 1236 | 0 / 1494 | 0 | single-node gate fails |

The `tau=1.5` calculation therefore gives the stable intended result: one
physical Morse node `M(0)`. Its reset-glued support and its base support are
connected at both depths. The `tau=2.0` outputs are retained as negative
refinement evidence; they were not deleted or relabeled to obtain the desired
graph. No Conley label is attached to any node.

Run the full screen with:

```bash
PYTHONPATH=. .venv/bin/python demo/run_bouncing_ball_atlas.py \
  --depth 8 --depth 10 --tau 1.5 --tau 2.0 \
  --samples-per-axis 3 --padding-cells 1 \
  --output data/bouncing_ball_atlas/acceptance_tau150_tau200_depth8_depth10.json
```

To make the accepted `tau=1.5` refinement pair an executable gate, add
`--require-local-window-gates --require-stable-two-depths` and omit `--tau 2.0`.

## Scope of the result

At depths 8 and 10, respectively, 128 of 512 and 554 of 2048 relation sources
have empty images. Across CMGDB's repeated callback passes, 2,644 of 9,216 and
10,632 of 36,864 tensor evaluations fail; every retained reason is a path exit
from the rectangular state-space window. Those failures are disclosed rather
than replaced by an absorbing state or an invented image.

Consequently this is a successful computation on the nonempty local-window
relation, not yet a compact global self-map or global attractor-lattice
calculation. The finite probes and connectivity tests are falsification gates,
not a whole-cell enclosure proof. A Conley index additionally requires a
verified carrier on the reset-glued suspension cell complex and is not computed
here.

## Chart-aware diagnostic figure

`demo/generate_atlas_morse_diagnostics.py bouncing-ball` extracts the actual
tagged rectangles through `MorseGraph.morse_set_chart_boxes`, verifies its
single node and per-chart counts against the acceptance JSON, and draws the
base support beside the CMGDB Morse graph. It stores a reusable JSON plotting
cache under `data/bouncing_ball_atlas` and writes new, non-manuscript
diagnostics under `paper/figures/atlas-diagnostics`. It adds no Conley label.
