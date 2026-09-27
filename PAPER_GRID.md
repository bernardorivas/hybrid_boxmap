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
| `def:suspension-multivalued-map` with the sampling rule of Section "Examples" | `compute_suspension_grid_relation` | `test_translation_cylinder_relation_morse_graph_probes_and_index`, `test_ball_relation_is_one_morse_node_at_level_three` |
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

`compute_suspension_grid_relation` implements the sentence of Section
"Examples": each element of `Xi_n` is sampled by a `3 x 3` tensor array
(corners, edge midpoints, center) on each of its elementary pieces; each
sample is followed for `tau` units of suspension time; the endpoint is
located in every closed piece that contains it, including both sides of an
identification; targets outside `D` are discarded. The implemented
`epsilon_n` is one-cell padding: `F_n(xi)` consists of the atoms whose closure
meets the closure of an atom containing a sampled endpoint of `xi`.

Handle samples are evaluated with one path per guard coordinate `u`: the
path starting at `pi(gamma(u), 0)` is followed for `tau + 1` units, and
`f_tau(pi(gamma(u), s))` is its value at time `tau + s`.

`SuspensionFlow` integrates the flow with the system's vector field, event,
tolerances, and reset, as `HybridSystem.simulate` does, with one difference:
whether a base point starts on its handle is decided by the explicit guard
`G cap R`. The legacy simulator resets every initial state with a nonnegative
event value; for the rimless wheel this resets states on
`theta = alpha + gamma` with `omega < 0`, which are not guard points.

`exit_policy="endpoint"` (the default) discards endpoints outside `D`. The
option `exit_policy="path"` also discards endpoints whose trajectory left `D`
and returned; the runner reports the Morse graph under both.

`gap_refinement_depth > 0` is an opt-in rule that is not in the manuscript:
a sample-lattice edge whose endpoints lie in atoms with disjoint closures is
bisected, and the new endpoints are added to the image. It is used only to
report what the labels would be if the sampling rule were refined; its
outputs carry the suffix `-gap-refined`.

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

## Reproduction

From `code/`:

```bash
.venv/bin/python demo/run_paper_grid_examples.py --workers 12
.venv/bin/python demo/run_paper_grid_examples.py --workers 12 --gap-refinement-depth 12
```

Outputs (PDF, PNG, JSON) are written to `figures/paper_grid/`.

## Recorded runs

All runs below used code commit `350c93e` (clean tree), 12 worker processes,
and the tolerances of the example classes (`rtol=1e-10`, `atol=1e-12`,
`max_step=0.02`). Level `n` has `2^n` base cells per axis of the ambient
rectangle and phase width `a_n = 2^{-n-2}`. The JSON file next to each figure
in `figures/paper_grid/` holds every count quoted here.

### Rule of the manuscript (`3 x 3` samples, one-atom padding)

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
atoms, 5 components); its label is blocked because `X` contains no base cell
at the guard, which the quotient-nerve adapter rejects.

### Opt-in gap refinement (not the rule of the manuscript)

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
