# Reset-glued suspension complex adapter

## What this layer represents

`src/suspension_complex.py` retains the cellular topology needed for a Conley
index computation of one fixed-time suspension map

```text
f = Phi_H^t_star.
```

It is generated from a base cubical complex and cellular guard/reset data.  It
does not construct a second geometric grid in quotient coordinates.

For every reset handle, the builder introduces:

- one canonical base copy of every base cell;
- a globally registered prism cell `(handle_id, guard_cell, slab)`;
- an interior slice cell at every non-endpoint phase level;
- the bottom attachment to the guard cell; and
- the top attachment prescribed by the reset cellular chain map.

The resulting oriented incidence satisfies `d^2 = 0` over the integers.  Phase
cells are interned by a `PhaseCellRegistry`, so two image evaluations referring
to the same handle, guard cell, and phase slab refer to the same cell.

## Minimal construction

The following interval/reset example produces a cellular circle.

```python
from hybrid_dynamics import (
    CellularChainMap,
    CellularResetMap,
    CubicalCell,
    CubicalGridComplex,
    RelativeCellPair,
    ResetHandle,
    SuspensionCellComplex,
)

base = CubicalGridComplex((1,))
left = CubicalCell((0,), (False,))
right = CubicalCell((1,), (False,))

reset = CellularResetMap.from_cell_map(
    base,
    guard_cells={right},
    cell_map={right: left},
)
suspension = SuspensionCellComplex.from_base(
    base,
    [ResetHandle("impact", reset, slabs=2)],
)

assert suspension.betti_numbers(modulus=5) == (1, 1)
```

`CellularResetMap` checks that the guard is a subcomplex and that the supplied
integer chains commute with the cellular boundary.  A cellwise reset is the
simple case; chain-valued images permit a reset represented after compatible
subdivision.

## Independent guard and nonlinear seams

For a four-dimensional base with a three-dimensional guard, constructing the
nerve of all closed base/handle boxes is usually prohibitive: as many as 16
closed four-boxes meet at one cubical vertex, yielding up to `2^16 - 1` local
nerve simplices. `DoubleMappingCylinderComplex` therefore uses an independent
guard complex `G` and two attachment maps

```text
i_#, r_# : C_*(G; Z) -> C_*(X; Z).
```

For a guard cell `g` of dimension `d`, its prism boundary is

```text
d(g x I) = (d g) x I + (-1)^d (r_# g - i_# g).
```

`CellularAttachmentMap` verifies both attachments over the integers before
the cylinder is built. The ordinary `FiniteCellComplex` constructor then
rechecks `d^2 = 0` on the complete glued complex.

`CrossComplexAcyclicCarrier(G, X, ...)` constructs a subordinate map from a
face-compatible carrier covering a nonlinear guard or reset. It validates
every carrier value as acyclic over the selected field, checks face nesting,
and solves `dF = Fd` degree by degree. Its field-valued result is never silently
used as a CW attachment. `construct_integral_attachment_map()` accepts the
canonical lift only if it independently satisfies the integral chain-map and
vertex-augmentation equations; otherwise it stops and requires an explicit
integral attachment or a certified integral selector.

The synthetic regression uses a 4D cubical base, an independent 3D cubical
guard, two phase slabs, nonidentical bottom/top attachments, and a nonidentity
relative fixed-time carrier. It checks the integral prism signs, `d^2 = 0`,
carrier acyclicity and nesting, pair preservation, subordination, `dF = Fd`,
and CMGDB payload construction.

`audit_cubical_hyperplane_attachment` is the opt-in geometric hard gate for an
attachment supported on a cubical coordinate hyperplane. It is used only
after `CrossComplexAcyclicCarrier` has established face nesting and
acyclicity and `CellularAttachmentMap` has established the integral chain
equations. The audit additionally requires nonempty source-cell images,
support entirely in the declared hyperplane, unit top-dimensional attachment
coefficients, and retention of every ambient top-cube coface incident to a
used facet. An interior facet therefore requires both sides. Missing cofaces
are rejected; the audit never expands the selected base family.

Two independently audited attachments may occupy disjoint patches of one
interior hyperplane. `DoubleMappingCylinderComplex` preserves them as the two
ends of a handle without making top-cell adjacency transitive along the
hyperplane. The 4D guard-aligned regression checks exactly this case: guard
and reset use separated patches of the same interior `q=0` subcomplex, the
relative-pair registry closes under faces but not top cofaces, and the gap
between the patches remains outside the selected local pair. A separate 2D
regression verifies the analogous interior-reset nerve and rejects one-sided
local seam support.

`SparseCubicalGridComplex` constructs the cellular closure of selected top
cubes in one common tensor grid, so a local 4D candidate does not allocate the
entire ambient grid. It intentionally rejects mixed resolutions within the
selected candidate; hanging-depth faces require a separate conforming
adaptive complex.

## Fixed-time relation, carrier, and chain map

These are intentionally distinct objects.

`FixedTimeCellRelation` stores the finite multivalued relation used for SCC and
Morse-graph calculations.  Its integer relabeling is available through
`to_integer_relation()` and is JSON serializable.

`FixedTimeCarrier` stores a cellular carrier on every cell.  It closes each
supplied image under faces, checks the face-nesting condition

```text
face e <= cell c  implies  Carrier(e) subset Carrier(c),
```

and, by default, computes cellular homology to verify that every value is
acyclic over `GF(5)`.

`CellularChainMap` is a chosen chain approximation.  It independently checks

```text
d F = F d
```

over the selected prime field. `carrier.require_carries(chain_map)` checks that
its support is subordinate to the carrier. When every carrier value has been
validated acyclic, `carrier.construct_chain_map(pair=...)` performs the
acyclic-carrier induction degree by degree and independently rechecks the
result. It does not choose a map from SCC adjacency alone.

## Relative pairs and CMGDB

`RelativeCellPair(complex, P1, P0)` requires both sets to be subcomplexes and
`P0 <= P1`.  `RelativeCellPair.generated(...)` takes cellular closures of
generator sets as a convenience.

For a chain map selected from a validated face-compatible carrier and
preserving both members of a generic pair:

```python
pair = RelativeCellPair(suspension, suspension.cells)
# `carrier` comes from the fixed-time map's verified cell images.
chain_map = carrier.construct_chain_map(pair=pair)
payload = chain_map.to_cmgdb_payload(pair)

cell_counts, boundary_entries, chain_map_entries = payload.as_compute_args()
result = CMGDB.ComputeRelativeHomologyShiftClass(
    cell_counts,
    boundary_entries,
    chain_map_entries,
)
```

The sparse layout exactly matches the generalized fork API:

- `cell_counts[d]` is the number of relative `d`-cells, including zeros for
  empty intermediate degrees;
- `boundary_entries[d]` contains `(row, column, coefficient)` triples for
  `C_d -> C_(d-1)`, with `boundary_entries[0] == []`; and
- `chain_map_entries[d]` contains square degree-`d` triples.

All coefficients are reduced to `1, ..., 4` modulo `5`; zero entries are
omitted and duplicate coordinates are rejected.  `payload.to_json()` retains
the ordered cell bases as string provenance.

The older six-argument `CMGDB.ComputeConleyIndex` assumes a rectangular
cubical grid with only coordinate-periodic identifications.  An arbitrary
reset attachment cannot be encoded by its `sizes` and `periodic` arguments.
Use the generalized relative-chain endpoint for the reset-glued complex.

For small actual two-chart CMGDB covers, including non-grid-aligned affine
2D resets, `AtlasQuotientNerveComplex2D` remains available. For a dense 4D
Atlas, use the independent-guard mapping cylinder and the explicit Atlas-id
registry documented in [ATLAS_CONLEY.md](ATLAS_CONLEY.md). Neither path
substitutes an identity chain map for a physical fixed-time map.

## Compatibility with sampled suspension tokens

Existing sampling code uses `BaseCell(box_index)` and
`PhaseCell(guard_box_index, slab)`.  Convert such a relation with
`SampledSuspensionCellAdapter`:

```python
adapter = SampledSuspensionCellAdapter(
    suspension,
    base,
    handle_id="impact",
    guard_cell_by_box={0: right},
)
cell_relation = adapter.relation_from_tokens(tokens, images)
```

The explicit `guard_cell_by_box` map is necessary because the sampled phase
token names a guard-intersecting full-dimensional box, whereas the cellular
prism is built over an actual guard-trace cell.  The compatibility adapter is
only valid when that correspondence is single-valued.  A box containing
several independently meshed guard cells must use `GuardPrismCell` labels
directly.

## What is validated, and what is not

The implementation validates:

- unique handle identifiers and globally canonical phase cell identities;
- guard-subcomplex closure;
- reset chain-map compatibility;
- independent-guard bottom/reset attachment compatibility over the integers;
- complete, two-sided base coface support for explicitly audited interior
  cubical-hyperplane attachments, without implicit support expansion;
- oriented reset-gluing incidence and `d^2 = 0`;
- relative-pair closure;
- carrier nesting and field-acyclicity when requested;
- `dF = Fd`, pair preservation, and the exact CMGDB sparse payload shape;
- inductive chain selection subordinate to a validated acyclic carrier;
- cross-complex carrier selection and hard-gated integral lifting; and
- the CMGDB bridge on the reset-circle smoke test.

The implementation does not establish:

- that a sampled image cover contains the full fixed-time image;
- that a smooth guard/reset has been approximated by the supplied cellular
  reset map with the required error control;
- that a proposed pair is isolating or is a valid Conley index pair;
- that a carrier represents the continuous quotient time map without a
  separate carrier/cellular-approximation argument; or
- the collared hybrid-to-suspension index-pair comparison required by the
  paper.

The cell incidence records the cellular chain complex induced by the reset.
For a chain-valued reset, incidence alone does not reconstruct the full
attaching maps or certify a geometric CW realization.  The caller must supply
that realization theorem when more than cellular homology is claimed.
