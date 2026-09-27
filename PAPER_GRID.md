# The manuscript's suspension grid in code

This note maps the statements of Section "Finite Realization and the Conley
Index" and Section "Examples" of `paper/b_main.tex` to the implementing
functions and to the tests that check them. The implementation covers a
two-dimensional base with a one-dimensional guard.

| Manuscript | Implementation | Test (`hybrid_dynamics/tests/test_suspension_grid.py`) |
|---|---|---|
| Phase grids `I_{n,k}`, `a_n = 2^{-n-2}` | `SuspensionGrid.n_phase = 2**(n+2)`, `SuspensionGrid.a_n` | `test_grid_axioms_and_d_n` |
| Contracting cofiltration `(X_n)` of base grids | `DyadicBaseWindow` (level `j` has `2**(j + level_offset)` cells per axis; cells of `X_j` are traces on `R`) | `test_grid_axioms_and_d_n`, `test_refinement_and_d_commute` |
| `def:suspension-grid` (`E_n(mu)`, `K_n(mu)`, `Q_n(mu,k)`, `Xi_n(X)`) | `build_suspension_grid`: generator signatures of elementary pieces, `GeneratorKey` | `test_level_zero_ball_grid_matches_the_manuscript_figure`, `test_neuron_reset_on_a_cell_face_gives_shared_collar_atoms` |
| `fig:suspension-grid` (`Xi_0` with three atoms) | `build_suspension_grid(..., 0)` for the ball | `test_level_zero_ball_grid_matches_the_manuscript_figure` |
| `Xi_n = Xi_{n-1} ^ Xi_n(X_n)` (generators `K_j`, `Q_j`, `j <= n`) | signatures range over all levels `j <= n` | `test_coarse_top_collars_split_fine_middle_cells`, `test_refinement_and_d_commute` |
| `rem:guard-grids` (guard partition) | `SuspensionGrid.u_edges`: common refinement of `abs(mu) cap G` and `r^{-1}(abs(mu))` | `test_curved_interior_guard_is_supported` |
| `prop:suspension-grid` (grid, `d_n` injective, `iota^{-1}(abs(d_n(mu))) = abs(mu)`, `iota(Int abs(mu)) subset Int abs(d_n(mu))`) | `SuspensionGrid.d`, `check_suspension_grid` | `test_grid_axioms_and_d_n` |
| `prop:finite-grid-preimage` (`rho_n(abs(U)) = abs(d_n^{-1}(U))`, Boolean homomorphism) | `SuspensionGrid.base_readout` | `test_base_readout_is_a_boolean_homomorphism` |
| `q_mn o d_m = d_n o p_mn` | `SuspensionGrid.refinement_map`, `SuspensionGrid.piece_parent` | `test_refinement_and_d_commute`, `test_curved_guard_refinement` |
| Suspension semiflow `Phi` (unit handle) | `SuspensionFlow`, `SuspensionPath` | `test_flow_agrees_with_sampled_suspension`, `test_zeno_point_circulates_through_the_handle`, `test_flow_does_not_reset_off_guard_event_points` |
| `def:suspension-multivalued-map` with the sampling rule of Section "Examples" | `compute_suspension_grid_relation` (`eval_mode`, default `corners`), `piece_evaluation_offsets` | `test_translation_cylinder_relation_morse_graph_probes_and_index`, `test_ball_relation_is_one_morse_node_at_level_three`, `test_default_sampling_is_the_corners_of_every_piece`, `test_evaluation_offsets_mirror_cmgdb`, `test_center_forces_padding_and_random_is_deterministic` |
| Window `D = Sigma(R)`, exits discarded | `SuspensionGrid.locate_base_points`, `SuspensionGrid.locate_handle_points` | `test_identifications_contribute_both_sides` |
| Morse sets, order, base readout (`MG(F_n)`, `d_n^{-1}(M)`) | `compute_suspension_morse_graph`, `SuspensionMorseGraph.base_readouts`, `SuspensionGridRelation.forward_closure` | `test_ball_relation_is_one_morse_node_at_level_three` |
| Finite-relation index labels (Section "Examples") | `compute_suspension_grid_conley_index` (pair `(S cup F(S), F(S) minus S)`, nerve of elementary pieces, `atlas_conley`, CMGDB shift class over `GF(5)`) | `test_translation_cylinder_relation_morse_graph_probes_and_index`, `test_seam_embeddings_of_the_examples` |
| Examples: ball, wheel, neuron | `examples/paper_grid_examples.py`, `demo/run_paper_grid_examples.py` | `test_example_reset_specifications_match_the_systems` |

## Construction

Every generator is a finite union of elementary pieces:

- base pieces, the finest base cells of level `n` contained in `R`;
- handle pieces `pi(J x I_{n,k})`, where the guard intervals `J` are cut at
  every parameter where `gamma` or `r o gamma` crosses a line of the finest
  base grid (these contain the breakpoints of every coarser level).

A base piece `mu` lies in `K_j(mu_j)` for its ancestors `mu_j`. A handle piece
`(J, k)` lies, for each level `j`, in `K_j(nu)` for the cells `nu` with
`gamma(J) subset abs(nu)` when `k >> (n-j) = 0` (bottom collar), in `K_j(nu)` for
the cells with `r(gamma(J)) subset abs(nu)` when `k >> (n-j) = 2^{j+2}-1` (top
collar), and otherwise in `Q_j(nu, k >> (n-j))` for the cells with
`gamma(J) subset abs(nu)`. The atoms of `Xi_n` are the classes of pieces with
the same signature. Degenerate parts of `E_j(mu)` (a cell meeting the guard or
the reset image in a point) contain no piece, which is the regularization.
Two consequences are visible in the tests: the coarse top collars
`[1 - a_j, 1]` split the level-`n` middle cells at the coarse breakpoints of
`r^{-1}`, and for the neuron, whose reset image `v = -50` is a face of the
cells of level `j >= 5`, the top collar over each guard interval is its own
atom (it lies in `K_n` of both adjacent cells).

The computation takes place on the window `D = Sigma(R)`: the generators are
intersected with `D`. This requires `r(G cap R) subset R`, which holds for all
three examples; `build_suspension_grid` raises
`UnsupportedSuspensionGridError` otherwise.

Closure incidence of pieces in `Sigma X` (`SuspensionGrid.piece_adjacency`)
includes the identifications: a phase-zero piece over `J` meets the base
cells meeting `gamma(J)`, a phase-one piece meets the base cells meeting
`r(gamma(J))`, and a phase-zero piece over `J` meets a phase-one piece over
`J'` when `gamma(J)` meets `r(gamma(J'))` (the Zeno point of the ball).

## The sampled map

`compute_suspension_grid_relation` samples each element of `Xi_n` on each
of its elementary pieces, follows each sample for `tau` units of suspension
time, locates the endpoint in every closed piece that contains it (including
both sides of an identification), and discards targets outside `D`. The
implemented `epsilon_n` is one-cell padding: `F_n(xi)` consists of the atoms
whose closure meets the closure of an atom containing a sampled endpoint of
`xi`.

The sample points are set by `eval_mode`, whose values `corners`, `center`,
and `random` have the names and meaning of `eval_mode` in
`CMGDB.PrecomputedBoxMap` (`piece_evaluation_offsets` returns the offsets of
`CMGDB.PrecomputedBoxMap.evaluation_offsets` for dimension two). A sample is
a relative position `k / m` in the chart of a piece: `(x, y)` for a base cell
and `(u, s)` for a handle piece `pi(J x I_{n,k})`, with `u` the guard
coordinate on `J` and `s` the phase on `I_{n,k}`.

- `corners` (the default): the four vertices of every elementary piece. For
  a base piece these are the corners of a finest base cell in `R`; for a
  handle piece with `J = [u_0, u_1]` and `I_{n,k} = [k a_n, (k+1) a_n]` they
  are `pi(gamma(u_i), s)` with `u_i` in `{u_0, u_1}` and `s` in
  `{k a_n, (k+1) a_n}`. Vertices shared by several pieces are evaluated once:
  the vertices of the base cells and the `(#guard intervals + 1) x (2^{n+2} +
  1)` vertices of the handle. The phase-zero and phase-one vertices are the
  base points `gamma(u_i)` and `r(gamma(u_i))`, and a base vertex on
  `G cap R` starts on its handle.
- `center`: the midpoint of every piece. One sample gives one target, so
  padding is forced on, as in CMGDB.
- `random`: `num_pts` offsets `k / 2^sample_depth`, with `k` drawn once from
  `{0, ..., 2^sample_depth}^2` by `numpy.random.default_rng(seed)`, the same
  for every piece (CMGDB defaults `num_pts=10`, `sample_depth=4`, `seed=0`).
- `tensor` (explicit legacy option): the `samples_per_axis x
  samples_per_axis` tensor array (corners, edge midpoints, and center for the
  default 3) that the runs recorded below used. `tensor` with
  `samples_per_axis=2` equals `corners`.

The image rule differs from CMGDB's box map, which pads the rectangular hull
of the sampled images: here the image is the union of the atoms containing
the sampled endpoints, with no hull, padded by one atom.

Handle samples are evaluated with one path per guard coordinate `u`: the
path starting at `pi(gamma(u), 0)` is followed for `tau + 1` units, and
`f_tau(pi(gamma(u), s))` is its value at time `tau + s`.

`SuspensionFlow` integrates the flow with the system's vector field, event,
tolerances, and reset, as `HybridSystem.simulate` does. Whether a base point
starts on its handle is decided by the explicit guard `G cap R`, so a point
of the event surface outside `G` (the wheel's `theta = alpha + gamma` with
`omega < 0`) flows. `HybridSystem.simulate` makes the same decision from the
event function and its direction (`HybridSystem.jumps_at_start`). Before
this rule the simulator reset every initial state with a nonnegative event
value, including those wheel states.

`exit_policy="endpoint"` (the default) discards endpoints outside `D`. The
option `exit_policy="path"` also discards endpoints whose trajectory left `D`
and returned; the runner reports the Morse graph under both.

`gap_refinement_depth > 0` is an opt-in rule that is not in the manuscript:
a sample-lattice edge whose endpoints lie in atoms with disjoint closures is
bisected, and the new endpoints are added to the image of every piece
containing that edge. With `corners` the lattice edges are the four edges of
each piece (an edge shared by two pieces serves both), so the refinement
follows the image of the boundary of each piece; with `tensor` they are the
edges of the tensor array. `center` and `random` have no lattice edges, and
the combination raises `ValueError`. The rule is used only to report what the
labels would be if the sampling were refined; its outputs carry the suffix
`-gap-refined`.

## Index labels

For a Morse set `S`, `compute_suspension_grid_conley_index` forms
`X = S cup F(S)` and `A = F(S) minus S`, takes the elementary pieces of their
atoms as actual closed rectangles in the base chart and in the handle chart
`(u, s)`, and uses `AtlasQuotientNerveComplex2D` (seams `(u,0) ~ gamma(u)`,
`(u,1) ~ r(gamma(u))`), `prepare_atlas_relation_conley_2d`, and
`CMGDB.ComputeRelativeHomologyShiftClass`. A piece of the atom `xi` is sent to
every piece of `F(xi)`. Any failed gate is returned as a blocker, never
replaced by a label. The results are finite-relation shift classes over
`GF(5)`; they are not certified indices of the continuous map.

A seam whose line lies outside the extent of the base pieces of `X` meets no
base piece of `X`. This is the case of a Morse set with handle pieces but no
base cell at the guard, for example the neuron, whose bottom collars share
an atom with the base cell at `v = 35`, so that such an `X` has no phase-zero
handle face either. The quotient nerve accepts such a seam: it contributes no
base/handle intersection, and handle/handle identifications through it are
audited like all other intersections (`test_neuron_index_without_a_base_cell_at_the_guard`,
`test_seam_outside_the_base_window_attaches_no_base_cell`,
`test_seams_outside_the_base_window_still_glue_handle_cells`). Earlier runs
reported these Morse sets as blocked with `AtlasGoodCoverError: guard seam is
not on the boundary of the base Atlas window`.

## Reproduction

From `code/`:

```bash
.venv/bin/python demo/run_paper_grid_examples.py --workers 12
.venv/bin/python demo/run_paper_grid_examples.py --workers 12 --gap-refinement-depth 12
.venv/bin/python demo/run_paper_grid_examples.py --workers 12 --eval-mode tensor --samples-per-axis 3
```

The runner options `--eval-mode`, `--num-pts`, `--sample-depth`, `--seed`,
`--samples-per-axis`, and `--gap-refinement-depth` select the sampling; the
options in force, the offsets, and the command line are recorded under
`image_rule` in the JSON summary. Outputs (PDF, PNG, JSON) are written to
`figures/paper_grid/`; every non-default choice adds a suffix to the file
names (`-center`, `-random10d4s0`, `-tensor3`, `-gap-refined`).

## Recorded runs

All runs below used code commit `350c93e` (clean tree), the `3 x 3` tensor
rule that was then the default (now `--eval-mode tensor --samples-per-axis 3`),
12 worker processes,
and the tolerances of the example classes (`rtol=1e-10`, `atol=1e-12`,
`max_step=0.02`). Level `n` has `2^n` base cells per axis of the ambient
rectangle and phase width `a_n = 2^{-n-2}`. The JSON file next to each figure
in `figures/paper_grid/` holds every count quoted here; these files predate the
corner default and carry no sampling suffix in their names.

### Tensor rule (`3 x 3` samples, one-atom padding)

| | Ball | Wheel | Neuron |
|---|---|---|---|
| `tau`, level `n`, `a_n` | 1.5, 5, 1/128 | 2, 6, 1/256 | 20, 8, 1/1024 |
| base cells (size) | 1,024 (0.0625 x 0.3125) | 4,096 (0.0125 x 0.0234) | 11,280 (1.25 x 5) |
| guard intervals x phases | 25 x 128 | 73 x 256 | 92 x 1,024 |
| pieces / atoms of `Xi_n` | 4,224 / 3,043 | 22,784 / 15,095 | 105,488 / 105,396 |
| edges of `F_n` | 82,093 | 347,623 | 1,506,348 |
| Morse graph | `M(0)` | `M(1) -> M(0)` | `M(0)` |
| atoms (base readout cells) | 1,396 (97) | gait 2,395 (522), saddle 10 (10) | 443 (382) |
| discarded exit endpoints (source atoms) | 1,359 (352) | 16,665 (2,925) | 0 (0) |
| atoms with a disconnected image | 150 | 1,560 | 6,064 |
| missed endpoint probes | 0 of 2,774 | 168 of 3,209 | 49 of 3,890 |
| finite-relation label | blocked | saddle `(0, x-1, 0, 0)`; gait blocked | blocked |
| run time | 14 s | 23 s | 313 s |

The blocked labels fail the carrier acyclicity gate: the recorded image of
some atom of `S` is disconnected. The images of base cells that land on a
handle are strips spanning many phase cells of width `a_n`, and three samples
per axis with one-atom padding leave gaps in them. For the neuron the Morse
set meets only 18 of the 1,024 phase intervals and has 10 components in
`Sigma X`. With `exit_policy="path"` the ball and wheel have the same Morse
graphs and base readouts (77,836 and 341,233 edges); the neuron has no exits.

A same-rule run of the neuron at level 7 also gives one Morse node (250
atoms, 5 components); its label was blocked because `X` contains no base cell
at the guard, which the quotient-nerve adapter then rejected (see Index
labels).

### Tensor rule with opt-in gap refinement

| | Ball, `n = 5` | Wheel, `n = 6` | Neuron, `n = 6` | Neuron, `n = 7` |
|---|---|---|---|---|
| inserted samples (unresolved gaps) | 16,581 (0) | 94,040 (0) | 165,043 (0) | 654,818 (0) |
| edges of `F_n` | 83,400 | 398,520 | 338,598 | 1,343,353 |
| Morse graph | `M(0)` | `M(1) -> M(0)` | `M(1) -> M(0)` | `M(0)` |
| atoms (base readout cells) | 1,398 (97) | 2,426 (522), 10 (10) | 1,674 (140), 1 (1) | 3,347 (274) |
| disconnected images, missed probes | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| finite-relation labels | `(x-1, x-1, 0, 0)` | gait `(x-1, x-1, 0, 0)`, saddle `(0, x-1, 0, 0)` | both carriers not acyclic | `(x-1, x-1, 0, 0, 0, 0)` |
| run time | 86 s | 167 s | 258 s | 596 s |

At neuron level 8 the refinement grew to 286,394 inserted samples at depth
6, roughly doubling per round, and was stopped. At level 6 the one-atom node
`M(1)` is spurious and the carrier values are connected but not acyclic.
