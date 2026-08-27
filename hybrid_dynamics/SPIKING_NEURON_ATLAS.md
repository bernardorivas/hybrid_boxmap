# Spiking-neuron CMGDB Atlas benchmark

## Scope

This benchmark implements the quadratic integrate-and-fire example in
`literature-review/arXiv-2606.18501v1/examples/neuron.tex`.  The formal and
computational hybrid system is restricted to the compact L-shaped set

\[
N=([-80,-40]\times[-300,600])\cup([-40,35]\times[-300,160]),
\]

with active guard \(G\cap N=\{35\}\times[-300,160]\) and reset
\((35,u)\mapsto(-50,u+100)\).  This restriction repairs two problems with the
paper's preliminary global description: the half-plane \(v\le 35\) is not
compact, and the vector field is not transverse to the entire unbounded guard.

The exact checks recorded in every run are:

- \(\dot u\ge3.3\) on the bottom boundary;
- \(\dot u\le-16.8\) on the upper boundary of the left arm;
- \(\dot u\le-6\) on the upper boundary of the lower-right arm;
- \(\dot v\ge0.3\) on \(v=-80\);
- \(\dot v\le-0.9\) on the vertical notch;
- \(48.975\le\dot v\le53.575\) on the active guard;
- \(r(G\cap N)=\{-50\}\times[-200,260]\subset\operatorname{Int}N\);
- the equilibrium equation has discriminant \(-52\), so there is no
  continuous equilibrium; and
- the post-reset dwell time is at least \(85/53.575>1.5865\), excluding a
  reset cascade or Zeno behavior in \(N\).

## Cubical Atlas

The dyadic address charts are

- base ambient chart \([-120,200]\times[-400,880]\), with only the full cells
  contained in \(N\) active; and
- handle ambient chart \([-400,880]\times[0,1]\), with only
  \(u_{guard}\in[-300,160]\) active.

At per-axis depth 6 the mesh widths are \(\Delta v=5\) and \(\Delta u=20\).
Consequently \(v=-80,-50,-40,35\), all relevant horizontal boundaries, and
the translation \(u\mapsto u+100\) are exact cubical faces/maps.  The active
family sizes are:

| CMGDB total depth | Per-axis depth | Base cells | Handle cells | Total |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 6 | 705 | 1,472 | 2,177 |
| 14 | 7 | 2,820 | 5,888 | 8,708 |
| 16 | 8 | 11,280 | 23,552 | 34,832 |

Every padded base target rectangle is intersected separately with the left
arm and lower-right arm before CMGDB covers it.  A target is never convexified
across the missing upper-right notch.  The callback also emits both base and
handle representatives whenever a sampled image meets a quotient seam.

The reset line \(v=-50\) is an interior cubical seam.  The 2D quotient-nerve
adapter must therefore be called with
`interior_seam_subcomplexes=("reset",)`.  Its constructor, rather than the
caller, verifies that no base cell straddles the seam, that the complete top
handle face is covered, and that both incident base cofaces are present.

## Frozen protocol

The protocol was fixed before inspecting any Morse-node count:

- suspension time \(t_*=0.5\);
- one-cell target padding;
- integrator `max_step=0.02`;
- 3-by-3 tensor samples at total depths 12, 14, and 16; and
- a separate 5-by-5 tensor-sampling sensitivity run at total depth 12.

All four stages are reported.  A stage is not selected or rejected by its SCC
count.  Its finite gates require:

- the independently computed reference cycle to lie in recurrent component(s)
  and every independent reference endpoint to be covered;
- every represented nonempty image to be connected in the reset quotient;
- every reported Morse support to be quotient-connected;
- no failed, empty, missing, unresolved, or nonglued-open source in a recurrent
  component containing the reference cycle; and
- separation of those reference component(s) from every nonglued boundary.

The independent reference calculation gives approximately

\[
u^-=-36.92166999,\qquad u^+=63.07833001,
\]

with continuous flight time \(147.85450497\) and suspension period
\(148.85450497\).  These points are only falsification probes.  They neither
seed the active family nor select a Morse component.

### Near-identity protocol result

The complete \(t_*=0.5\) protocol is retained as a negative plumbing study:

| Stage | Morse nodes | Reference-node cells | Nonglued boundary cells in reference node | Result |
| --- | ---: | ---: | ---: | --- |
| depth 12, 3x3 | 1 | 2,177 | 217 | reject |
| depth 14, 3x3 | 1 | 8,708 | 436 | reject |
| depth 16, 3x3 | 964 | 32,866 | 716 | reject |
| depth 12, 5x5 | 1 | 2,177 | 217 | reject |

All four runs have zero failed, empty, missing, unresolved, or disconnected
images, and the complete reference cycle is covered.  Their failure is
instead structural: one-cell padding at a near-identity clock produces a very
large recurrent support that reaches the nonglued boundary.  The depth-16
node count does not rescue the calculation; the node containing the reference
cycle is still the 32,866-cell boundary-touching component.  No index pair or
Conley index is reported from this protocol.

### Separately versioned scientific-clock amendment

The near-identity result is not overwritten or reinterpreted.  A distinct
protocol, fixed from the independent suspension period before inspecting its
Morse counts, runs all four combinations

\[
(t_*,\text{depth})\in\{(20,12),(20,14),(40,12),(40,14)\}.
\]

Sampling, padding, address charts, and `max_step=0.02` remain unchanged.  The
jump caps are 10 for \(t_*=20\) and 20 for \(t_*=40\).  The exact dwell bound
gives the conservative requirements
\(\lceil20/(1+1.58656)\rceil=8\) and
\(\lceil40/(1+1.58656)\rceil=16\), so these caps cannot truncate a valid
trajectory in \(N\).  Zero jump-limit failures and zero unresolved stage
changes are explicit gates.

Depth 16 was excluded from this scientific protocol at cost preflight, before
the scientific-clock results: it has 34,832 sources and 140,570 unique 3x3
endpoints, each requiring 20 or 40 time units at `max_step=0.02`.  Depths 12
and 14 already provide the predeclared spatial refinement check.

The scientific protocol is run with:

```console
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_scientific.py --workers 8
```

All four complete uniform scientific-clock relations are negative:

| Clock, depth | Relation edges | Morse nodes | Reference-node cells | Disconnected images | Unresolved sources | Reference-node problems | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20, 12 | 53,537 | 1 | 563 | 9 | 27 | 10 | reject |
| 20, 14 | 226,454 | 1 | 1,274 | 6 | 31 | 8 | reject |
| 40, 12 | 54,727 | 1 | 299 | 76 | 106 | 10 | reject |
| 40, 14 | 268,632 | 1 | 950 | 60 | 93 | 10 | reject |

Every row has zero failed, missing, nonglued-open, and jump-limit failures;
the independently sampled reference cycle is contained in node 0 and is
separated from the nonglued boundary.  The rejection is nevertheless
mandatory: unresolved stage changes occur globally, and quotient-disconnected
source images occur in every stage.  Some of these defects lie in the
reference recurrent node.  Thus no finite Conley index is attached to any
uniform scientific-clock relation.

### Relation-bound adaptive amendment

After the uniform results, a deterministic defect-refinement rule was fixed
without consulting any SCC count.  For each depth-14 clock relation it selects
every source with an unresolved stage or quotient-disconnected image, adds one
closed same-chart cell ring, adds all positive-length guard/reset face
cofaces, and replaces precisely those depth-7 leaves by their four depth-8
children.  All remaining leaves are retained.  The authenticated no-dynamics
preflights give:

| Clock | Witness union | Refined parents | Mixed leaves | Unique 3x3 endpoints | Linear timing projection | Cost gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 31 | 122 | 9,074 | 37,030 | 136.4 s | pass |
| 40 | 93 | 309 | 9,635 | 39,444 | 301.3 s | pass |

For (t_*=20), all 31 witnesses are base-chart unresolved (0\to2)
stage changes; the six disconnected-image sources are a subset.  The mixed
family fingerprint is
`45917b973ccf7e10eb891ad27276529377db5f9de961dedbb62468936d445706`
and the tensor-point fingerprint is
`f8c7344cf9403fad8a7fa91fc92340daf0fb5732ed604ffe008c7ea5dae2d774`.
The primary adaptive run is authorized only with these exact values and the
unchanged 50-million-edge and 1-GiB native/CSR caps.

For (t_*=40), the 93 witnesses comprise 65 base and 28 handle cells; the
unresolved edge types are (0\to2) (40), (2\to4) (26), and (4\to6)
(27).  Its adaptive dynamics run is conditional: it is performed only if the
primary (t_*=20) adaptive relation passes every gate.  A failed primary run
leaves (t_*=40) as a uniform negative plus its persisted no-dynamics
preflight, preventing a post-hoc search for an attractive Morse count.

The authorized (t_*=20) adaptive primary run retained the frozen 122-parent
selection exactly.  Its complete unpruned relation has 9,074 vertices and
274,029 arrows.  The single recurrent node contains the complete reference
cycle and 1,113 mixed leaves, is quotient-connected, and is separated from
the nonglued boundary.  Independent endpoint coverage and the full 9,074-row
mixed sparse/geometric incidence parity audit pass.  Nevertheless, the run is
rejected: it has 8 quotient-disconnected source images, 43 unresolved sources
globally, and 13 problem sources in the reference recurrent node.  All
failed integration, jump-limit, missing-cover, nonglued-open, and empty-source
counts are zero.

The finer leaves localize but do not remove the sampled stage ambiguity.  All
43 adaptive unresolved sources are depth-8 base leaves with the same
(0\to2) stage gap.  Their total base-chart area is 268.75, down from 775 for
the 31 uniform depth-7 sources; within the recurrent node it falls from 200
to 81.25.  The gate is qualitative, however, and requires zero such sources.
In accordance with the frozen conditional rule, the (t_*=40) adaptive
dynamics stage was not launched.  No finite-relation Conley index is reported.

### Terminal-carrier and finite-index amendment

The rejected adaptive primary above is retained unchanged.  Its remaining
defect is a terminal codimension-one `0 -> 2` stage transition: a source sample
lands on the far face of exactly one reset handle, so source-box refinement
alone need not remove the skipped terminal stratum.  A new, opt-in protocol
therefore adds only the complete terminal carrier for that one handle together
with its guard and reset faces.  It adds no intermediate time-map edge, is
limited to an exact stage gap of two, and records every bisection probe and
emitted tagged bound in the complete source provenance.  This is an explicit
sampled modeling assumption; it is not an interval-rigorous enclosure.

With this terminal carrier enabled, the frozen 3x3 mixed-family relation has
9,074 vertices and 277,834 arrows.  There are 43 raw unresolved sources and
79 carrier attempts; all 79 are synthesized, all 118 probes succeed, and the
residual unresolved count is zero.  Every image and the recurrent support are
connected in the reset quotient, and all failed, empty, missing,
nonglued-open, jump-limit, boundary, and reference-component problem counts
are zero.  The unique recurrent Morse set containing the complete reference
cycle has 1,113 cells (345 base and 768 handle), and its standard finite pair
is

\[
S=M_0,\qquad F(S)=S,\qquad X=S\cup F(S)=S,\qquad A=F(S)\setminus S=\varnothing.
\]

The actual-rectangle quotient nerve uses the guard boundary seam and the
constructor-verified two-sided interior reset seam at \(v=-50\).  It has cell
counts

\[
(1113,3934,3789,994,31,5)
\]

in dimensions zero through five.  All 9,866 nonempty finite intersections are
contractible.  The actual-relation carrier passes face nesting and acyclicity
on every value; it preserves the pair, the selected chain map is subordinate,
and both \(d^2=0\) and \(dF=Fd\) validate.  The independently computed finite
relation over \(\mathbb F_5\) has

\[
\dim H_*=(1,1,0,0,0,0),\qquad F_{*0}=F_{*1}=[1],
\]

and shift class

\[
(x-1,x-1,0,0,0,0).
\]

No expected polynomial or analytic orbit label is used as a gate or fallback.
The exact sparse boundary and chain-map payload is persisted and strictly
reloaded through CMGDB before publication.

The separately predeclared 5x5 sensitivity uses the exact same 9,074-cell
family, \(t_*=20\), padding, step size, jump cap, and terminal-carrier rule.
Its no-dynamics preflight fixes 226,850 logical and 146,650 unique endpoint
nodes before launch.  The complete sensitivity relation has 282,448 arrows;
26 raw unresolved sources produce 55 successful carrier attempts and 64
successful probes, again leaving zero residual or recurrent blocker.  Its
standard pair, quotient nerve, carrier, homology, induced maps, and shift
class are independently recomputed and agree exactly with the 3x3 result.
The two complete relations themselves are not identical: 68 source rows
change, with 4,831 arrows added and 217 removed.  Thus the agreement is a
genuine fixed sampling-sensitivity result rather than a copied label.  The
authenticated comparison fingerprint is
`dee78586b9e23b3ed4e5bd1a4e89acf7915f6ce4c697c3360033522b60d00226`.

These statements concern the validated sampled finite relation only.  The
whole-cell box map is not a numerically rigorous outer approximation, the
terminal carrier remains an explicit assumption, and no continuous-system
Conley index is certified by this computation.  The analytic argument in the
paper is logically separate.

## Reproducibility and claims

Run the frozen protocol from the `code` directory:

```console
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_atlas.py --workers 8
```

Reproduce the versioned scientific/adaptive result and finite index with:

```console
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_scientific.py --workers 8
PYTHONPATH=. .venv/bin/python demo/preflight_spiking_neuron_adaptive.py
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_adaptive.py --authorize-primary-t20 --workers 8
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_adaptive_bridge.py --authorize-reviewed-terminal-bridge --workers 8
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_conley.py
PYTHONPATH=. .venv/bin/python demo/preflight_spiking_neuron_samples5.py
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_samples5_sensitivity.py --authorize-predeclared-sensitivity --workers 8
PYTHONPATH=. .venv/bin/python demo/run_spiking_neuron_conley.py --stage-dir data/spiking_neuron_atlas/scientific_clock_v1/adaptive_terminal_bridge_samples5_v1/t20
PYTHONPATH=. .venv/bin/python demo/compare_spiking_neuron_samples5.py
PYTHONPATH=. .venv/bin/python demo/plot_spiking_neuron_finite_relation.py
```

The accepted primary artifacts are under
`data/spiking_neuron_atlas/scientific_clock_v1/adaptive_terminal_bridge_v1/t20/`;
the sampling sensitivity and exact comparison are under
`adaptive_terminal_bridge_samples5_v1/`.  Each Conley directory contains a
self-fingerprinted summary and an exact compressed sparse boundary/chain-map
checkpoint bound to the authoritative relation CSR and complete source
provenance.  The vector diagnostic, PNG preview, and strict figure manifest
are written under `output/pdf/`.

Each stage writes a fingerprinted memory-mapped CSR relation, complete
per-source compressed diagnostic provenance with an authenticated trailer
bound to the CSR fingerprint, a JSON summary, and a chart-aware plot under
`data/spiking_neuron_atlas/`.  The protocol summary is
`data/spiking_neuron_atlas/protocol_summary.json`.

This is intentionally a sampled finite relation.  It does not claim a
numerically rigorous outer enclosure.  No analytic Conley label is inserted
as a substitute for a finite computation; a finite-relation Conley index may
only be computed after the stated recurrence, openness, connectivity, and
relative-pair gates pass.
