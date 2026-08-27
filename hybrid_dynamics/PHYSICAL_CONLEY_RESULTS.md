# Physical Atlas finite-relation Conley results

`demo/run_physical_conley.py` computes the CMGDB shift-equivalence class of
the accepted **finite Atlas relations** for the bouncing ball and rimless
wheel.  It does not paste labels from an analytic orbit model.

For each Morse set `S`, the runner first constructs CMGDB's standard top-cell
pair

```text
X = S union F(S),       A = F(S) minus S.
```

The construction is recorded before cellular closure.  The actual closed
base- and handle-chart boxes in `X` then form a nerve in the reset quotient.
Every admitted finite intersection is checked to be contractible.  The
induced pair is `P1=N(X), P0=N(A)`.

The relation carrier is face nested.  On ordinary vertices it uses the
stored `MapGraph` value.  On an exit vertex in `A`, it uses the connected
component of the induced `A` nerve; this is a pair-preserving extension only,
because all of `P0` is zero in the relative chain complex.  Original
`F(A)\X` edges remain in the relation cache as exits.  The code verifies every
carrier value is acyclic over `GF(5)`, constructs a subordinate cellular chain
map, and independently checks pair preservation and `dF=Fd` before calling
`CMGDB.ComputeRelativeHomologyShiftClass`.

## Current results

| candidate | `(S,X,A)` top boxes | relative basis | finite-relation shift class |
|---|---:|---:|---|
| bouncing-ball `M(0)` | `(674,674,0)` | `(674,2527,2456,603)` | `(x-1,x-1,0,0)` |
| rimless-wheel gait `M(0)` | `(1174,1174,0)` | `(1174,4280,4123,1017)` | `(x-1,x-1,0,0)` |
| rimless-wheel saddle `M(1)` | `(34,352,318)` | `(34,176,192,51)` | `(0,x-1,0,0)` |

The bouncing-ball nerve includes the bottom-to-top rest-seam intersection and
its cofaces.  That deliberate quotient correction contributes one edge, two
triangles, and one tetrahedron compared with the earlier prototype count; it
does not change the shift class.

The wheel saddle cache retains exits from 188 `A` sources.  Its callback also
records 1280 failed samples in 153 `A` sources, all because a sampled
trajectory leaves the declared local state-space window.  There are no such
failures in the saddle `S`, none in the gait `X`, and no unresolved event-stage
edges.  These failures do not alter the already-stored finite directed
relation, but they are a blocker for interpreting it as a certified enclosure
of the continuous map.

## Certification boundary

All three entries above are finite-reset-quotient relation results.  None is
currently a certified Conley index of the continuous fixed-time suspension
map.  Two independent obligations remain explicit in every report:

1. the sampled-and-bloated callback is not a proved whole-cell outer
   enclosure; and
2. no external theorem has certified the finite pair as a continuous-map
   index pair.

The result JSON therefore sets
`continuous_system_conley_index_certified=false`.  The finite result remains
available under `finite_relation_conley_index` and
`finite_relation_shift_class`.

## Reproduction and caches

From the `code` directory:

```bash
PYTHONPATH=. .venv/bin/python demo/run_physical_conley.py \
  --model all --require-finite-relation-index
```

The ball uses the full cached adjacency.  The wheel runner reconstructs only
the sources required by the two pairs and marks every other source as
unevaluated, so an absent targeted value cannot be mistaken for an empty map
value.

- `data/physical_conley/ball_tau150_depth10_relation.json.gz`
- `data/physical_conley/wheel_tau200_depth12_targeted_relation.json.gz`
- `data/physical_conley/physical_conley_audit_ball.json`
- `data/physical_conley/physical_conley_audit_wheel.json`

Each relation cache preserves chart ids, physical bounds, adjacency, Morse
membership, evaluation flags, callback failures, and failure reasons.  No
Conley label is attached unless the finite pair/carrier/chain gates pass.
