# Garcia passive walker through the CMGDB suspension box map

This module integrates the four-dimensional Garcia passive walker with
`CMGDB.AtlasModel`. CMGDB owns the grid, directed graph, SCC decomposition, and
Morse order. The hybrid-specific part is the tagged finite-union box-map
callback. Two modes are available: the deliberately coarse full-chart
plumbing run and a native active-subgrid diagnostic.

The two four-dimensional charts are:

- base: `(theta, theta_dot, phi, phi_dot)` on the existing
  `GarciaPassiveWalker` domain; and
- reset handle: `(theta_guard, theta_dot_guard, rho, s)`.

The handle coordinate `rho in [0,1]` parametrizes the admissible `phi_dot`
interval at fixed `theta_dot`. Its embedding is

```text
phi = 2 theta
phi_dot = L(theta_dot) + rho (phi_dot_max - L(theta_dot))
L(theta_dot) = max(phi_dot_min, 2 theta_dot + eta).
```

Thus every point in the handle chart lies on the geometric heel-strike guard
and satisfies `phi_dot - 2 theta_dot >= eta`. Both impacts of the stored
period-two gait lie in this chart. The phase coordinate `s` is the unit reset
handle coordinate; the callback supplies both tagged representatives at
`s=0` and `s=1` when a target reaches a seam.

Run the coarse acceptance configuration with:

```bash
cd code
PYTHONPATH=. .venv/bin/python demo/run_garcia_passive_walker_atlas.py \
  --depth 4 --tau 0.5 --samples-per-axis 3 --padding-cells 1 \
  --require-plumbing-gates
```

At depth 4, every coordinate is bisected once. The two charts therefore have
16 cells each, for 32 Atlas cells total. This is intentionally a plumbing
regression, not a scientifically resolved Morse computation. The recorded
run has one coarse Morse node. It makes no claim that the desired gait SCC or
Morse order has been recovered. The complete machine-readable record is
`data/garcia_passive_walker_atlas/acceptance_tau050_depth4.json`.

The acceptance object reports, without suppressing failures:

- coverage of fixed-time endpoints sampled on both continuous strides and
  both reset handles of the stored period-two gait;
- quotient connectedness of every nonempty recorded image;
- adjacent tensor samples that skip an event stage;
- empty grid-source images;
- every failed callback sample, separated into domain exits, target-chart
  failures, and other failures; and
- how the stored gait source cells meet the computed Morse nodes.

The depth-4 run passes the finite endpoint, quotient-connectivity, and
event-stage plumbing gates. It also has many sampled trajectories leaving the
declared rectangular state-space window. Those points are retained as failed
samples; they are not turned into edges or silently relabeled as successful
images. Consequently the run is not a complete sampled self-map, much less a
whole-cell outer enclosure or a global attractor-lattice computation.

## Native active-subgrid diagnostic

`AtlasModel.set_active_subgrid` now constructs selected tagged dyadic cells
directly, without first materializing the surrounding rectangular grid. It
preserves both chart identifiers and retains omitted charts as registered,
empty charts. A target cover is always taken against the declared active
family. Thus a nonempty target piece with empty active cover is recorded as an
active-boundary exit; it is not redirected to an implicit cemetery vertex.

The Garcia family is not an orbit-only graph. Its seeds are the cells met by
the two stored continuous stride arcs and the two reset-handle traversals.
For a positive integer radius `r`, the active domain is the union of every
full, closed, four-dimensional, same-chart dyadic cell whose index is within
Chebyshev distance `r` of a seed. Radius zero is rejected by the builder.

For example, the first resolved diagnostic can be reproduced with:

```bash
cd code
PYTHONPATH=. .venv/bin/python demo/run_garcia_passive_walker_atlas.py \
  --depth 8 --tau 0.5 --samples-per-axis 3 --padding-cells 1 \
  --active-stencil-radius 1
```

We screened radius 1 and 2 at total CMGDB depths 8 and 12. Because the walker
has four coordinates per chart, these correspond to per-axis dyadic depths 2
and 3. The following are diagnostic outcomes, not accepted Morse results:

| depth | radius | active cells (base + handle) | disconnected images | empty graph images | active-boundary sources | missing target cells | runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 274 (210 + 64) | 1 | 0 | 274 | 19,036 | 29.04 s |
| 12 | 1 | 1,182 (846 + 336) | 3 | 27 | 1,177 | 289,510 | 117.21 s |
| 8 | 2 | 436 (256 + 180) | 1 | 5 | 431 | 8,337 | 35.98 s |
| 12 | 2 | 3,349 (2,581 + 768) | 11 | 342 | 3,048 | 439,526 | 329.93 s |

All four positive-padding cases cover the stored reference endpoints and have
no unresolved sampled event-stage edge, but none is closed under the computed
relation and none passes image connectedness. Increasing the radius admits
many cells whose sampled trajectories leave the declared ambient chart; it
does not repair the result. A depth-8, radius-2, zero-padding sensitivity run
reduced some exit counts but increased disconnected images from 1 to 28, so it
is also rejected rather than used to tune away failures.

The complete counts, including explicit-empty sources, wholly exiting
sources, pieces crossing the ambient chart boundary, and failed endpoint
samples across CMGDB's two map passes, are recorded in
`data/garcia_passive_walker_atlas/active_subgrid_screen_tau050.json`.

## Exit-aware local refinement

`garcia_passive_walker_local.py` replaces the one-shot stencil screen with a
resumable local calculation.  The first pass uses a complete coarse cover of
both charts.  It identifies the recurrent component met by the two stored
strides and reset handles and forms the ordinary finite-relation candidate

```text
S = reference-carrying recurrent component
X = S union F(S)
A = X minus S.
```

The coarse `X` can be far too large to decide where resolution is useful.  A
separate locator graph therefore retains only evaluated, nonempty,
quotient-connected source values with no recorded open exit.  Its recurrent
component with the most stored-gait cell hits is unioned with *all* stored-gait
source cells.  This pruning is used only to place the adaptive mesh; it is
explicitly not an invariant-set or acceptance computation.

For the next requested depth, the locator seed, a fixed physical collar, and
all of their quotient-seam neighbors are replaced by dyadic descendants.  The
coarse exterior is removed, giving a declared local open window `N`; images
leaving `N` are recorded as exits, not cemetery edges.  Only after evaluating
this finer source family is the standard finite pair audited:

```text
N_pair = S union F(S),   L = N_pair minus S.
```

A later round may insert only represented in-domain image cells missing from
the active window for sources in `N_pair minus L`.  Images and failures from
`L` remain exits.  They are not chased toward the ambient chart boundary.
Saturation ends only at a fixed point or an explicit cost, round, or terminal
pair failure.  This avoids both refining a huge coarse padded `F(X)` and
silently converting a local open computation into a global-domain one.

The primary collar is `0.0625` of each chart-axis span.  It is a genuine fixed
physical neighborhood: one uniform cell at per-axis depth 4 and two at depth
5.  The wider `0.125` collar is retained as a disclosed preflight sensitivity,
but is not launched blindly because its four-dimensional cell count can grow
by hundreds of thousands before any dynamics are evaluated.

The event path is hardened for this computation.  Its root function is
continuous, and a reset is accepted only at a state satisfying the declared
contact and transversality guard.  A root caused by switching between the
constraint branches is rejected as an inadmissible heel strike and recorded
as a failed/open-exit sample.  It cannot create a spurious reset edge.

Each persisted `garcia-walker-open-exit-atlas-relation-v1` record contains the
complete directed relation and, for every source cell:

- tagged dyadic coordinates, mixed depth, and physical bounds;
- its complete active target list and the raw tagged target rectangles;
- all callback failure reasons and explicit-empty status;
- in-domain target cells absent from the active family; and
- ambient-boundary and wholly-outside image counts.

The payload also stores the additional base/handle leaf contacts made by the
guard and reset identifications as `quotient_neighbor_pairs`.  Consequently a
consumer does not have to guess cross-chart seam adjacency from rectangular
bounds.

Run the full depth-12 discovery pass with:

```bash
cd code
PYTHONPATH=. .venv/bin/python demo/run_garcia_passive_walker_local.py \
  --axis-depths 3 --tau 0.5 --samples-per-axis 3 --padding-cells 1
```

Then refine the persisted relation without recomputing depth 12:

```bash
PYTHONPATH=. .venv/bin/python demo/run_garcia_passive_walker_local.py \
  --resume-from data/garcia_passive_walker_atlas/local_relation_tau050_depth12.json.gz \
  --axis-depths 4 --tau 0.5 --samples-per-axis 3 --padding-cells 1 \
  --precompute-workers 8 --max-relation-edges 50000000 \
  --max-spatial-adjacencies 20000000 --max-connectivity-memory-gib 1 \
  --stop-after-initial-audit
```

The explicit stop makes depth 16 a reviewable milestone: it writes the full
relation and the exit-aware `(N,L)` audit before any support-expansion run is
allowed.  A subsequent depth or saturation run is launched only after that
certificate has been inspected.

The runner prints and enforces explicit cell and callback-evaluation caps
before each pass.  After CMGDB has completed the MapGraph and all cell/raw
callback provenance is available, it atomically writes a separate
`garcia-walker-open-exit-raw-relation-checkpoint-v1` JSON-lines file before
running connectivity, reference, Morse-selection, or candidate audits.  Its
header and trailer both say that no scientific result has been accepted.  A
complete trailer, raw-content fingerprint, file and directory sync, and
same-directory atomic rename make it strictly resumable with
`--resume-raw-checkpoint`.  This is an audit-stage checkpoint: it prevents a
postprocessing failure from losing the MapGraph, but it does not resume an ODE
callback traversal interrupted before CMGDB returned.
Resuming skips the expensive box-map/MapGraph ODE evaluation.  It still runs
the small stored-gait reference simulation and endpoint audit used to select
and check the reference-carrying component.
The runner includes a short hash of the complete active family and physical
box-map configuration in each per-round checkpoint filename, so a different
collar, padding, sample count, or colliding rounded `tau` tag cannot overwrite
an incompatible valid checkpoint.

Raw checkpoints use independent physical-model, box-map, and raw-relation
revisions.  Their immutable hash covers the physical configuration, every
cell image, every raw target piece and exit record, and all quotient pairs; it
does not include later candidate/connectivity/reference decisions.  The
strict loader reconstructs the Atlas geometry and raw piece covers, recomputes
missing support, ambient exits, quotient pairs, SCCs, canonical Morse labels,
and Morse reachability, and checks all derivable execution totals.  Recorded
sample-failure reasons are internally counted and hash-bound but cannot be
physically re-evaluated without repeating the ODE calculation.  The completed
depth-12 v3 relation has been migrated without ODE evaluation to
`data/garcia_passive_walker_atlas/raw_relation_checkpoint_tau050_depth12.jsonl.gz`;
the older depth-8 fixture lacks sufficient raw provenance and is deliberately
not migratable.

The deep runner enforces separate conservative caps for the raw relation plus
source provenance, for relation-edge count, and for sparse spatial adjacency.
After a fresh atomic checkpoint is safely written, it releases the endpoint
and whole-source evaluation caches and replaces the native MapGraph handle by
the already stored relation before postprocessing.  Consequently, a
checkpoint-producing run cannot donate those caches to a later refinement;
each accepted family is evaluated freshly (parallel endpoint precomputation is
still available within that family).

Image connectivity no longer scans every pair of cells or retains an
unbounded nonlinear-incidence cache.  Uniform grids use the 80 possible
four-dimensional dyadic neighbor offsets.  General mixed antichains use a
16-ary dyadic prefix trie whose boundary-bit constraints enumerate exactly the
touching finer leaves.  The stored guard/reset quotient pairs are then added
as cross-chart edges.  At depth 12 this sparse audit reproduces the previous
ten disconnected images and their exact component partitions while replacing
394,206,377 possible pair tests by a 234,946-edge spatial graph.  Explicit raw
relation-edge and sparse-adjacency caps stop postprocessing only after the raw
checkpoint has been preserved.

Depth 12 is discovery only.  A local gait result requires
the same reference-carrying component and its isolation/coverage gates to
persist through at least total depths 16 and 20.  The reports retain
`whole_cell_outer_enclosure_certified=false`: sample-and-padding relations are
CMGDB-style finite computations, not interval-certified images of every point
of a source cell.

The callback is deterministic, so its tagged rectangle value is evaluated
once and replayed when CMGDB requests the same source during a second graph
traversal.  Physical endpoints at closed tensor nodes shared by adjacent
dyadic boxes are likewise memoized.  Optionally, local worker processes
precompute those exact endpoint values in native Atlas source order; they do
not perform hulling, padding, graph construction, or scientific pruning.
Serial, 2-, 4-, and 8-worker depth-8 runs have identical scientific payloads.
Cross-family endpoint reuse is deliberately forbidden: adjacent grids can
choose source representatives differing at the last floating-point bit, so a
refined active family is always evaluated fresh.  The report records process
count, execution mode, parity provenance, logical tensor samples, unique
endpoint integrations, and cache reuses.

Resume artifacts carry a versioned content fingerprint over the physical
parameters, chart bounds, complete cells/relation, quotient pairs, Morse and
reference data, candidate pair, and locator.  Before refinement, the loader
recomputes raw-piece covers, missing-target and open-exit flags, quotient
incidence, recurrent SCCs, and `S/X/A`; a self-consistent but altered derived
field is rejected.

### Current discovery outcome

The hardened full depth-12 computation at `tau=0.5`, three samples per axis,
and one-cell padding completed in 319.6 seconds.  The 66 stored-gait endpoint
probes have no misses or evaluation failures, and no tensor edge skips an
event stage.  Nevertheless, the reference-carrying SCC is not a local gait
set:

| quantity | count |
|---|---:|
| ambient cells | 8,192 |
| `S` | 7,056 |
| `X = S union F(S)` | 8,156 (99.56% of ambient) |
| `A = X minus S` | 1,100 |
| `S` sources with a sampled open exit | 5,611 |
| `S` sources with a failed sample | 2,616 |
| `S` sources on a nonglued ambient boundary | 4,186 |
| quotient-disconnected nonempty `S` images | 10 |

All ten disconnected values come from source tensors with 60--78 failed
samples out of 81.  Their surviving base and handle pieces have no intervening
seam carrier.  Joining them by an artificial hull would invent dynamics, so
the computation preserves the two components and rejects the candidate.

After the full attachment-carrier seam audit, the refinement-only safe graph
has 1,445 eligible cells and one selected recurrent component of 953 cells,
meeting 7 of the 41 distinct reference source cells.  Unioning all reference
cells gives a 987-cell seed.  The stage-zero depth-16 preflight records both
collars: `0.0625` produces an 82,944-cell open fine window (31,648 base and
51,296 handle cells), while `0.125` produces 120,032 cells.  Neither count
contains the inflated coarse `F(seed)`; that support is added only after the
fine relation has been evaluated.

The complete relation is
`data/garcia_passive_walker_atlas/local_relation_tau050_depth12.json.gz`; the
cheap lower/upper refinement bounds are
`data/garcia_passive_walker_atlas/local_refinement_preflight_tau050.json`.

The implementation blocker is therefore no longer the absence of an active
Atlas API.  For the CMGDB-style finite computation, the remaining scientific
test is whether the sample-and-bloat relation on the declared local window has
a stable, connected reference SCC and a valid exit-aware pair through the
refinement levels.  A separate whole-cell outer-enclosure theorem would be
needed only to certify the continuous fixed-time map; the reports keep that
stronger flag false.  No SCC or Morse-node count from the coarse screen is
promoted as a physical result.

## Guard-aligned local-window falsification

The follow-up calculation aligns the base grid with the heel-strike guard.
Starting from physical coordinates `(theta, omega, phi, phi_dot)`, set

```text
q  = phi - 2 theta,
nu = phi_dot - 2 omega.
```

The inverse is `phi=q+2 theta`, `phi_dot=nu+2 omega`, so this is an
invertible linear coordinate change. In the aligned state
`z=(theta,omega,q,nu)`, the stance equations are

```text
theta_dot = omega,
omega_dot = sin(theta-gamma),
q_dot     = nu,
nu_dot    = -sin(theta-gamma)
            + omega^2 sin(q+2 theta)
            - cos(theta-gamma) sin(q+2 theta).
```

The outgoing guard is the interior hyperplane
`q=0`, `theta<=-delta`, `nu>=eta`. On that guard, with
`c=cos(2 theta)`, physical reset followed by the coordinate change is exactly

```text
(theta+, omega+, q+, nu+) = (-theta, c omega, 0, -c(1+c) omega).
```

Thus the guard and reset patches are disjoint subsets of the same cubical
hyperplane `q=0`, separated by the sign of `theta`; no rectangular hull is
needed to locate the attachment itself. The base-chart hull is

```text
theta in [-0.32, 0.32],  omega in [-0.35, 0.05],
q     in [-1.29, 1.29],  nu    in [-0.65, 0.85],
```

and the handle chart is `(theta,omega,nu,s)` on
`[-0.32,-0.05] x [-0.35,0.05] x [0.1,0.85] x [0,1]`.
The flagship parameters remain `gamma=0.0172`, `delta=0.05`, `eta=0.1`,
`t_star=0.5`, three samples per source axis, and one-cell padding.

The local family was frozen before either aligned relation was computed. It
contains a normalized-radius `0.0625` tube around the complete dense
two-stride base-and-handle trace, plus one-ring neighborhoods of the fourteen
unsafe physical-coordinate reference boxes and their persisted missing-image
corridors. Every selected handle face contributes its complete base
attachment carrier once; reverse base-to-handle cofaces are retained as an
explicit open boundary rather than recursively importing the full guard.
The depth-20 run is the predeclared refinement/falsification check of exactly
this construction. No radius, time, padding, clock, or source-selection rule
was changed after seeing the depth-16 graph.

The complete unpruned finite relations give the following comparison.
`Open`, `missing`, `failed`, and `disconnected` count source cells in the
reference-carrying recurrent component `S`; overlapping columns must not be
added. Every row is a rejected diagnostic, not an accepted Morse result.

| coordinates/run | active cells | relation edges | `S` | open | missing | failed | disconnected |
|---|---:|---:|---:|---:|---:|---:|---:|
| physical, depth 16 local window | 82,944 | 25,928,612 | 70,691 | 50,786 | 40,378 | 8,943 | 19 |
| physical, depth 20 fixed tube | 29,906 | 7,419,811 | 27,353 | 24,572 | 24,564 | 1,173 | 1,393 |
| guard-aligned, depth 16 | 9,255 | 2,137,413 | 8,822 | 8,759 | 8,726 | 855 | 93 |
| guard-aligned, depth 20 | 61,237 | 16,263,818 | 50,084 | 45,726 | 45,511 | 2,017 | 78 |

The physical depth-16 component also meets 18,007 nonglued ambient-boundary
cells and requires 36,656 additional represented support cells. The physical
depth-20 tube exhausts its predeclared 50,000-cell support cap. A separate
strict-source diagnostic on that same persisted relation leaves only 2,783 of
29,906 source rows and has no recurrent SCC; row deletion is not an outer
relation and is therefore not reported as a Morse computation.

The aligned depth-16 component meets 6,794 closed same-chart one-ring
boundary cells, 594 reverse-seam boundary cells, and 1,310 ambient-boundary
cells. At depth 20 the corresponding counts are 28,424, 1,684, and 1,690,
and support closure has a certified lower bound of 80,001 active cells,
exceeding the frozen 80,000-cell cap. Both aligned relations recover all 52
stored gait labels in one recurrent SCC and have zero independent reference
endpoint misses or evaluation failures. This is necessary but not sufficient
for isolation.

Guard alignment and refinement do improve the reference-local diagnostics:
from aligned depth 16 to depth 20, reference-source open values fall from
62 of 64 cells to 14 of 69, missing support from 57 to 8, failed sources from
4 to 2, and ambient exits from 10 to 6. The global candidate does not share
that improvement: at depth 20 more than 90 percent of `S` still has an open
or missing value, and `S` intersects every declared kind of local open
boundary. Hence `q=0` grid alignment improved the reference-local diagnostics
but did not produce an isolating component.

The authoritative aligned artifacts and semantic SHA-256 fingerprints are:

```text
depth 16 geometry  4bcefbb0f403e6794e93c2db9faf4b1c3e717ae0f49aba084a1ddfa1dfffecdf
depth 16 bundle    021acb026eb49dbe84d21a8119bb8579bcb1ad34decbe280057c97a4e4c1c52b
depth 16 relation  d076a83a67fce6bca4eb9097de39f76b1f4c04f173ef049f2a91790379ef92a5
depth 16 audit     d19ee7ebc8f0b5b637ae0d90609b5cc2b969213283ee3e2089ed3ac2d38b997a
depth 20 geometry  de4cfd376130e081f2cea889866c39bfdfa0dbc1c4b576feb58b226eefc59d4e
depth 20 bundle    a221c24e8771d8b427ec8dc1c61ce5d7548d11531ecfdd391dff3613c8682157
depth 20 relation  1f69b65aa7691411e3b8a05964b1668f84ffe9942daf67e226fa2a52fe43ac9a
depth 20 audit     14cfa643f61053d02269e8aa286190676c82723a25069098bb8f6ab62aa69b96
```

They live under `data/garcia_passive_walker_guard_aligned/`. Both strict
loaders validate the coordinate system, physical and box-map configuration,
geometry, provenance, Morse snapshot, and mmap CSR relation without rerunning
source-box dynamics. Since the isolation, support, connected-image, and
refinement-persistence gates do not pass, no reset-glued relative complex is
built for these candidates and no finite-relation or continuous-system Conley
index is reported.

No Conley index is computed here. That still requires a valid whole-cell
relation, index pair, reset-glued quotient cell complex, and compatible
carrier/chain map for the fixed-time suspension map.
