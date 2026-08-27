# Physical Atlas Conley front end

`src/atlas_conley.py` supplies the topology and carrier construction that must
precede `CMGDB.ComputeRelativeHomologyShiftClass`. It does not assign a Conley
label from a Morse node or from an orbit skeleton.

## Actual reset-quotient complex

For the two-dimensional ball and rimless-wheel models, every vertex of
`AtlasQuotientNerveComplex2D` is an actual indexed closed rectangle from the
CMGDB base or handle chart. The guard and reset identifications are supplied
as injective affine boundary embeddings. Intersections are evaluated in the
reset quotient, not in the disjoint chart union.

The reset image generally does not align with the base cubical grid. Rather
than manufacture a cubical reset map, the adapter constructs the simplicial
nerve of the actual closed-cell cover. Every nonempty finite intersection used
by the nerve must pass a contractibility check. A disconnected guard/reset
intersection, an extra handle self-identification, or an overlap beyond the
declared enumeration bound raises `AtlasGoodCoverError`.

An interior seam also fails by default. The explicit opt-in
`interior_seam_subcomplexes=("guard",)` or `("reset",)` is a request for a
constructor-generated certificate, not a bypass. It succeeds only when no
base top cell straddles the seam, the embedded selected handle endpoint faces
are completely covered by base faces on the seam, and both incident base top
cofaces are retained along every used seam segment. The same exhaustive
finite-intersection and contractibility checks then still apply. Thus a
one-sided local boundary cannot be silently relabeled as an interior quotient
attachment. Under this verified good-cover condition, the nerve models the
physical reset-glued union. Its oriented simplicial boundary is checked over
the integers, including `d^2 = 0`.

The model-specific affine data are exposed by:

- `bouncing_ball_atlas_reset_gluing`, with
  `(v,0) ~ (0,v)` and `(v,1) ~ (0,-c v)`; and
- `rimless_wheel_atlas_reset_gluing`, with
  `(omega,0) ~ (alpha+gamma,omega)` and
  `(omega,1) ~ (gamma-alpha,cos(2 alpha) omega)`.

The actual fixed-depth depth-4 Atlas covers for both models pass the current
finite-intersection/good-cover construction. This removes reset-grid
nonalignment as the immediate topology blocker for these two charts. It does
not validate their sampled dynamical images.

## Scalable four-dimensional mapping cylinder

A direct closed-box nerve is not the default for the passive walker. In four
dimensions, up to 16 grid boxes meet at a vertex, so clique-sized local nerve
families can contain `2^16 - 1` simplices. Moreover, the current walker
incidence routine answers only pairwise intersection questions; pairwise
overlap does not certify a common multiway quotient intersection.

The scalable generic path instead uses:

- a 4D cubical base complex;
- an independent 3D guard complex;
- verified integral chain maps for the nonlinear guard inclusion and reset;
- `DoubleMappingCylinderComplex` to add the phase direction; and
- an explicit `AtlasMappingCylinderRegistry` associating every retained
  CMGDB Atlas id with its base top cell or guard-phase prism.

`AtlasMappingCylinderRelativePair` takes `P1` and `P0` as actual Atlas ids and
then forms their cellular closures. `prepare_atlas_mapping_cylinder_conley`
requires a carrier value for every cell of that relative complex, including
all faces. It checks that every non-exit Atlas top-cell value contains the
actual MapGraph targets, then applies the same acyclicity, face-nesting, pair,
subordination, chain-selection, and `dF = Fd` gates used by the 2D path.

The attachment maps may be selected from `CrossComplexAcyclicCarrier` values
covering the nonlinear seam. A field-valued selector is accepted as an
integral attachment only when the lifted coefficients independently satisfy
the integral chain equations. There is no implicit identity attachment and no
pairwise-clique completion.

For a guard-aligned cubical chart,
`audit_cubical_hyperplane_attachment` adds the geometric hard gate before the
mapping cylinder is built. It requires every attachment image to lie in one
declared cubical hyperplane, every handle-face cell to have a nonempty image,
and every used hyperplane facet to retain all of its ambient incident base top
cofaces (two at an interior hyperplane). It does not add missing cofaces.
Guard and reset may attach to two disjoint patches of the same interior
hyperplane and are audited independently; their supports are not completed
transitively across that hyperplane. The explicit Atlas registry later closes
selected `P1` and `P0` generators under faces only, so it cannot pull remote
registered top boxes into the pair.

The Garcia adapter builds the cylinder in a sparse ambient base containing
both the base cells registered in `X` and any additional cubical cells needed
to support the guard/reset attachment carriers. These auxiliary cells receive
no invented Atlas ids and are not added to `X`: the relative pair is still the
closure of exactly the registered `X` top cells. Consequently an interior
handle slab does not falsely require its remote seam-support boxes to belong
to `P1`, while a selected endpoint slab acquires the appropriate attachment
faces through ordinary cellular closure. The adapter records candidate,
attachment-support, and auxiliary coordinate families separately.

The generic 4D synthetic regression is complete. The model-specific adapter in
`examples/garcia_passive_walker_conley.py` now consumes persisted
`garcia-walker-open-exit-atlas-relation-v1` artifacts. It uses the declared
linear guard equation `phi = 2 theta`, the nonlinear intrinsic `phi_dot`
parametrization, and the implemented Garcia collision reset. For every guard
cell and every face, it tensors samples, forms a padded axis-aligned target
hull, clips only to the analytically valid ambient base chart, and requires the
covered base subcomplex to be explicit and acyclic. Both subordinate
attachment selectors must then pass the exact integral chain-map gate.

This is a disclosed nonrigorous assumption, not an interval proof: the sampled
and bloated hull is assumed to contain the full nonlinear image. The returned
continuous-system certification flag therefore remains false.

For the fixed-time relation, the adapter takes `P1 = X` and `P0 = A`. For a
lower-dimensional source cell it intersects the closed target subcomplexes of
all incident Atlas top cells. This is face nested, and it contains the face
image conditional on every stored closed-box relation being a valid outer
cover. Empty or nonacyclic intersections stop the construction. Exit top cells
are extended through their connected `A` component before this intersection;
this is an algebraic pair-preserving extension only, since all of `A` is zero
in the relative quotient, and is not described as the physical image of an
exit face. The raw relation, including out-of-`X` targets, remains unchanged
in the returned preparation metadata and in the source artifact. The generic
pair gate permits such exits from `P0` but still rejects
`F(P0) intersection P1` outside `P0`.

Accordingly, the open local-pair convention treats missing or outside support
on `A = P0` as exit provenance, not as a failed pair by itself. It is allowed
exactly when the retained relation satisfies
`F(A) intersection X subset A`. On `S = X minus A`, evaluation, image support,
and open-exit gates remain strict: any missing in-domain support, failed
sample, unresolved stage, empty image, or ambient exit blocks preparation.

The loader and public preparation endpoint do not trust a single persisted
acceptance flag. They recompute `X = S union F(S)`, `A = F(S) minus S`, the
recurrent SCC/Morse-set identity, the second pair condition, strict boolean
evaluation/exit provenance, unresolved-stage sources in `S`, and reference
miss/failure counts. The public endpoint has no gate-bypass option.

Mixed-depth ambient artifacts are accepted when candidate `X` has one common
axis depth. `SparseCubicalGridComplex` materializes only the selected cube
closures. A genuinely mixed-depth `X` hard-fails because a conforming adaptive
complex, rather than a silent global finest-grid expansion, is required.

The persisted depth-8 artifact passes the attachment topology layer with
6,561 base cells, 729 guard cells, 11,664 cylinder cells, and 1,458 verified
guard/reset carrier values at three samples per varying axis and one-cell
padding. It is not promoted to a finite Conley result: its discovery/exit gate
is false. A separate internal diagnostic of the downstream construction also
finds that the coface-intersection fixed-time carrier is empty on a shared base
vertex. Thus the current blocker is the candidate relation/refinement (or
additional face-level enclosure data), not reset-quotient incidence or chain
algebra.

The older pairwise/numerical `GarciaWalkerQuotientIncidence` remains useful for
connectivity diagnostics, but it is not used as a multiway nerve certificate.

## Relative pair and carrier

`AtlasRelativeIndexPair2D` forms the induced nerve pair from explicit `P1` and
`P0` families of actual Atlas cell indices. It checks cellular closure and can
check that a supplied top-cell relation preserves both members. The flag
`index_pair_certified` is never inferred from an SCC; it records a separate
caller-supplied index-pair proof and requires provenance text.

`prepare_atlas_physical_conley_2d` then:

1. evaluates the tagged physical box map on every source simplex, including
   all lower-dimensional faces and every quotient representative;
2. covers every returned rectangle with the same CMGDB Atlas;
3. optionally checks that vertex results exactly reproduce the stored
   MapGraph adjacency;
4. defines the carrier of a simplex as the closure of the union of the raw
   image subcomplexes of all its faces;
5. checks face nesting, `GF(5)` acyclicity, and preservation of `P1` and `P0`;
6. constructs a subordinate chain selector by the acyclic-carrier induction;
   and
7. independently checks carrier subordination, `dF = Fd`, pair preservation,
   and the sparse relative-chain payload.

The selector is solved degree by degree. Vertices are sent to vertices in
their carrier values. In degree `d > 0`, a finite linear system inside the
carrier value solves

```text
d F(sigma) = F(d sigma).
```

No identity map is inserted as a surrogate for the physical fixed-time map.

For an already materialized CMGDB MapGraph,
`prepare_atlas_relation_conley_2d` provides the corresponding efficient
construction. The carrier of a simplex is the induced target nerve on the
union of the actual MapGraph images of its vertices. For a relative exit set
`P0`, each exit vertex can instead use its entire connected component in the
induced `P0` nerve. This is a pair-preserving extension on cells that vanish in
`C(P1)/C(P0)`, not an identity-map assumption. Every such component and every
resulting carrier value must still pass the same acyclicity, subordination,
pair-preservation, and chain-map checks.

## CMGDB boundary

`compute_finite_relation_shift_class()` calls the CMGDB explicit-chain bridge
after all finite nerve, carrier, pair-preservation, subordination, and chain-map
gates pass. Its returned metadata states
`result_scope="finite_reset_quotient_relation"` and
`continuous_system_conley_index_certified=false`. This is the appropriate
CMGDB-style combinatorial result even when the box map is a nonrigorous outer
approximation.

The stronger `compute_cmgdb_shift_class()` endpoint is reserved for a claim
about the continuous fixed-time suspension map. It additionally requires:

- the box map is separately certified as a whole-cell outer enclosure; and
- the relative pair is separately certified as an index pair for the same
  fixed-time suspension map.

The current ball and wheel callbacks set the first continuous certificate to
false. Their finite-relation shift classes may still be reported with the
scope above, but they are not relabeled as certified continuous-system Conley
indices. Remaining work for that stronger claim is validated flow/event
enclosure plus a continuous index-pair justification; the reset-glued
finite-complex and chain-selection layers are no longer represented by a
hand-built circle.
