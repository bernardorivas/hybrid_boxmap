# Exit-aware local index pairs for the walker

`examples/walker_local_index.py` constructs a local pair from a **persisted**
CMGDB Atlas relation. It is intended for the deeper adaptive walker runs, where
discarding every source with an image piece outside the active window would
destroy the gait SCC.

This layer does not change the box map, recompute SCCs, or replace the
fixed-time suspension map by a return map. Its result is a certificate about
the stored finite relation. Continuous-system certification remains false
unless a whole-cell outer enclosure and a continuous index-pair comparison
theorem are supplied separately.

## Construction

Let `S` be the recorded gait Morse SCC. For a requested number of collar
layers, the code constructs `N` by closed-rectangle contact in each Atlas
chart. Reset-quotient contacts between different charts are never guessed;
they must be supplied as explicit `quotient_neighbor_pairs`.

The rectangles can have arbitrary side lengths. The contact search and the
certificate make no uniform-grid or single-depth assumption, so one collar can
contain adaptive leaves from depth 16 through depth 20. An optional
`allowed_cells` set restricts the construction to a persisted adaptive family.

For this fixed `N`, the initial exit set contains:

1. every source in `N` whose complete per-source flag says that a represented
   image has an open exit from the ambient Atlas window;
2. every source in `N` with a represented relation edge to a target outside
   `N`; and
3. conservatively, every unevaluated source or source missing an exit flag.

It then applies the monotone closure

```text
L_next = L union (F(L) intersection N)
```

until stable. Because the relation is finite, this produces the least closed
exit set containing the seeds. The audit independently checks

```text
F(N \ L) subset N
F(L) intersection N subset L
S intersection L = empty
```

and requires `S` to be the only recurrent component in `N \ L`. If exit
closure reaches the gait, that cell list is persisted as the concrete failure
witness; the code does not delete the offending edges.

## Input and runner

The runner always consumes a persisted relation. A Garcia walker cache can be
audited directly because it embeds its gait Morse node, source exit flags, and
reset-quotient neighbor pairs:

```console
../.venv-cmgdb/bin/python demo/run_walker_local_index_audit.py \
  data/garcia_passive_walker_atlas/walker_depth20_relation.json.gz \
  --collar-layers 2 \
  --output data/garcia_passive_walker_atlas/walker_depth20_local_index.json
```

Without `--collar-layers`, this relation-only path uses one contact layer. An
optional second file can override that embedded choice or supply data absent
from the generic relation schema:

- either a `physical-conley-atlas-relation-v1` gzip JSON cache or a
  `garcia-walker-open-exit-atlas-relation-v1` cache; and
- an `exit-aware-local-index-input-v1` JSON specification.

A minimal specification has this form:

```json
{
  "schema": "exit-aware-local-index-input-v1",
  "morse_node": 0,
  "collar_layers": 2,
  "allowed_cells": [100, 101, 102, 205],
  "source_exit_flags": [],
  "quotient_neighbor_pairs": [[102, 205]],
  "metadata": {
    "expected_relation": {
      "model": "garcia_passive_walker_fixed_time_suspension_atlas",
      "depth": 20,
      "t_star": 0.5
    }
  }
}
```

The Garcia walker schema embeds `open_exit.has_open_exit` for every source, so
the external flag list is normally empty. For the generic physical schema, the
specification must provide the flags explicitly. In either path the merged
flags must cover every source that lands in `N`; providing flags for every
allowed source is easiest. External flags are cross-checked against embedded
flags and a contradiction is fatal. A missing flag is treated conservatively
as an exit and is also a certificate blocker. The optional expected-relation
values keep a depth-20 exit file from being applied to a depth-16 relation by
mistake.

Run with an explicit specification:

```console
../.venv-cmgdb/bin/python demo/run_walker_local_index_audit.py \
  data/garcia_passive_walker_atlas/walker_depth20_relation.json.gz \
  data/garcia_passive_walker_atlas/walker_depth20_local_index_input.json \
  --output data/garcia_passive_walker_atlas/walker_depth20_local_index.json
```

The output is written whether the candidate passes or fails. Exit status `0`
means that the finite-relation pair gates passed; exit status `2` means that a
failure certificate was written. The output records the exact `S`, `N`, and
`L` cells, collar shells, exit edges, every closure round, recurrence checks,
and a SHA-256 digest of the source relation.

## What still has to come from the adaptive relation builder

For each depth-16-to-20 candidate, the relation builder must persist:

- every top-cell rectangle and complete represented adjacency;
- the accepted gait SCC id;
- one boolean open-exit flag for every candidate source, with missing targets
  retained as exits rather than silently clipped; and
- all base/handle cell pairs whose closures meet under the reset quotient.

The local-pair layer can then decide, with a reproducible pass/fail witness,
whether a chosen collar separates the gait from those exits. It cannot repair
an incomplete relation or establish the missing whole-cell enclosure.

## Pinned depth-20 orbit-validation tube

The flagship depth-20 path is local and separate from the failure-pruned
discovery refinement above. It uses the stored Garcia period-two seed to
integrate two complete strides and both intrinsic handle traversals.
Independent closed normalized-coordinate tubes are formed in the base and
handle charts with radius `0.0625`. Attachment closure is nonpercolating: each
selected handle face at `s=0` or `s=1` adds its complete analytic base carrier
once, and newly added base cells do not import remote handle cofaces.

The pinned geometry contains 29,906 depth-20 top cells (21,906 base and 8,000
handle). The one-pass attachment adds 60 base cells and retains 2,921 active
quotient incidences. The 28,938 omitted reverse incidences are an explicit
open seam boundary. Acceptance requires the recurrent set `S`, but not the
exit set `L=A`, to avoid both the 22,924-cell closed same-chart one-ring
boundary and the 1,237 active base cells incident to an omitted reverse seam
coface. Every failure, unresolved stage, open value, empty value, disconnected
represented image, or missing support in `S` blocks acceptance. In `L=A`, an
explicit state-space exit and a disconnected or empty represented image may
remain an exit only when `F(A) intersection N` is contained in `A`;
unresolved stages and failures not classified as explicit domain exits still
block.

The geometry command is safe by default and stops before evaluating source
boxes:

```console
.venv/bin/python demo/run_garcia_passive_walker_tube.py
```

The expensive relation requires the explicit `--compute` flag. Its pinned
configuration is `tau=0.5`, 3 samples per axis, one-cell bloat, maximum ODE
step `0.02`, and at most 20 jumps. The runner rejects changes to those values
instead of allowing differently configured relations to collide under the
same filename.

```console
.venv/bin/python demo/run_garcia_passive_walker_tube.py --compute
```

CMGDB's native MapGraph is capped before construction and immediately written
as an mmap-ready CSR relation. The authoritative checkpoint is one atomic
directory containing `relation.csr`, `provenance.jsonl.gz`, and a
fingerprint-bound `manifest.json`; per-source JSON retains target rectangles,
sample failures, missing support, and open-exit data but never adjacency
lists. Only a fully fsynced temporary directory is renamed into the final
bundle. Strict resume validates every file, configuration, geometry record,
Morse snapshot, and CSR checksum before rerunning the derived audits:

```console
.venv/bin/python demo/run_garcia_passive_walker_tube.py \
  --resume-provenance data/garcia_passive_walker_atlas/TUBE_BUNDLE_NAME
```

The small derived audit is written beside the bundle. It contains a canonical
six-digest `relation_bundle` reference and its own canonical SHA-256, while the
large adjacency remains a lazy read-only CSR mapping. Loading the derived
audit requires the expected physical/run configuration and fresh vertex,
edge, payload, and artifact-size caps supplied by the caller; it never
reconstructs per-cell JSON image lists.

Before enabling the depth-20 run, the same atomic-CSR path was exercised on a
complete depth-8 analog: 512 cells, 69,605 directed relation edges, 10,238
spatial adjacencies, one native Morse node, and a 282,524-byte CSR payload.
Fresh and strict-resumed candidate, locator, connectivity, Morse graph, and
every CSR row agreed exactly. This is an implementation smoke test, not a
scientific walker result.

This remains a CMGDB-style sampled-and-bloated finite relation. Neither the
bundle nor a passing local-pair audit certifies a whole-cell outer enclosure
or a continuous-system Conley index.

## Strict-source restriction diagnostic

The persisted depth-20 relation can also be audited without rerunning either
the ODE or the box map. The diagnostic below replaces a source row by the
empty set whenever its stored provenance records any sample failure,
unresolved stage, ambient/open exit, missing active target, or
quotient-disconnected represented image:

```console
.venv/bin/python demo/run_garcia_passive_walker_strict_restriction.py
```

The authoritative mmap CSR and every original provenance record remain
unchanged. The output stores the complete row mask with overlapping reasons,
then recomputes SCCs, the Morse reachability order, the fixed-reference
candidate, and the standard `S`, `X=S union F(S)`, `A=X-S` pair for the masked
finite relation. This is intentionally diagnostic: emptying a row with an
uncertain image is not a valid outer enclosure and cannot establish a
continuous-system or finite-relation Conley index.

## Guard-aligned follow-up

The completed follow-up uses the invertible base coordinates
`q=phi-2 theta` and `nu=phi_dot-2 omega`, so the guard is the interior cubical
hyperplane `q=0`. On the guard the exact transformed reset is

```text
(theta,omega,q,nu) -> (-theta, cos(2 theta) omega, 0,
                        -cos(2 theta)(1+cos(2 theta)) omega).
```

Frozen depth-16 and depth-20 local families were evaluated as complete,
unpruned CMGDB relations. They contain 9,255 cells/2,137,413 edges and
61,237 cells/16,263,818 edges, respectively. Both recover all 52 stored gait
labels in one recurrent SCC, but both fail the open-value, missing-support,
connected-image, and local-boundary isolation gates. At depth 20 the
50,084-cell candidate contains 45,726 open sources, 45,511 sources with
missing support, 2,017 failed sources, and 78 disconnected represented
images; its support closure exceeds the predeclared 80,000-cell cap.

Refinement improves the reference-local counts substantially (open reference
sources fall from 62/64 to 14/69), but it does not isolate the global
component. The exact relations and self-hashed negative audits are under
`data/garcia_passive_walker_guard_aligned/`. No walker Morse plot or Conley
index is accepted from them.
