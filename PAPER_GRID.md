# The manuscript's suspension grid in code

This note maps the statements of Section "Finite Realization and the Conley
Index" and Section "Examples" of `paper/b_main.tex` to the implementing
functions and to the tests that check them. The implementation covers a
two-dimensional base with a one-dimensional guard.

| Manuscript | Implementation | Test (`hybrid_dynamics/tests/test_suspension_grid.py`) |
|---|---|---|
| Phase grids `I_{n,k}`, `a_n = 2^{-n-2}` | `SuspensionGrid.n_phase = 2**(n+2)`, `SuspensionGrid.a_n` | `test_grid_axioms_and_d_n` |
| Contracting cofiltration `(X_n)` of base grids | `DyadicBaseWindow` (level `j` has `2**(j + level_offset)` cells per axis; cells of `X_j` are traces on `R`); runner option `--level-offset` | `test_grid_axioms_and_d_n`, `test_refinement_and_d_commute`, `test_atoms_are_the_classes_of_generator_signatures` |
| `def:suspension-grid` (`E_n(mu)`, `K_n(mu)`, `Q_n(mu,k)`, `Xi_n(X)`) | `build_suspension_grid`: generator signatures of elementary pieces, `GeneratorKey`, `reference_signatures` | `test_level_zero_ball_grid_matches_the_manuscript_figure`, `test_neuron_reset_on_a_cell_face_gives_shared_collar_atoms` |
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
| Parameter variants (the oscillator at `beta = 0.76`) | `PAPER_GRID_VARIANTS`, `paper_grid_problem` | `test_beta076_variant` (`test_impact_vdp_duffing.py`), `test_a_variant_has_its_own_output_names_and_replots` (`test_suspension_grid_plot.py`) |

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
the same signature. `build_suspension_grid` forms these classes from a short
key per piece (the finest cells meeting `gamma(J)` or `r(gamma(J))`, the
phase, and the top-collar levels of the phase; see `_signature_classes`)
instead of enumerating the generators; `reference_signatures` enumerates them
as the definition does, and the tests compare the two. Degenerate parts of `E_j(mu)` (a cell meeting the guard or
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
and returned; the runner reports the Morse graph under both. The runner
passes one `EndpointCache` to both computations, so the second one evaluates
no sample: endpoints are looked up by the exact coordinates of the sample
and are a deterministic function of the grid, the problem, and the sample,
and with gap refinement the gaps under `"path"` are among those under
`"endpoint"`.

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
`GF(5)`; they are not certified indices of the continuous map. The quotient
nerve has about ten simplices per elementary piece of `X` and is held in
memory; `max_pieces` (runner option `--index-max-pieces N`) skips a Morse set
whose `X` has more than `N` pieces and reports it as blocked with
`IndexSizeLimitError`. By default there is no limit.

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

## Figures

The runner writes two figure variants per run (`--figure-variants`, default
`all,nontrivial`; `all` only with `--no-conley`):

- `all` (`<stem>.pdf`, `<stem>.png`): every Morse node.
- `nontrivial` (`<stem>-nontrivial.pdf`, `.png`): the Morse nodes whose
  finite-relation index was computed and is trivial (every homology dimension
  zero) are hidden. The remaining nodes keep their numbers and colors, and
  the order drawn between them is reachability in the full Morse graph,
  through hidden nodes, transitively reduced.

In both variants a node whose index is blocked is kept, drawn with a dashed
outline, and labeled `blocked`.

Each figure has three panels in the same colors: the base chart (the base
readout `d_n^{-1}(M)` of each Morse set), the handle chart (its handle pieces
`pi(J x I_{n,k})`, guard coordinate against the phase `s` in `[0, 1]`, over
the whole guard interval), and the Morse graph. A chart shown whole is
widened by 2% of its span on each side, so cells on the boundary of the
window lie inside the axis lines. A Morse set too small to see in a chart
panel (its cells cover less than 16 of 200 x 200 panel bins) is drawn in a
zoom panel, labeled `A`, `B`, ... in a column next to the chart panel, when
its zoom window spans at most 1/8 of the panel in each direction; the window
of the zoom is outlined and labeled in the chart panel. A zoom panel is a
quarter to 0.4 of the chart panel wide, so every zoom magnifies both axes
at least about twice. Small sets closer than 6% of the panel share a zoom
when their shared window is within the same limit. There are at most three
zooms per panel, with windows that do not overlap; the groups with the most
sets are chosen first, and a small set left out is drawn in a zoom whose
window contains it, if there is one. A zoom draws the sets it is for opaque
and the other sets in its window faded. It magnifies the two axes of the
panel by factors whose ratio is at most 3, and its tick labels give the
window. Every cell is drawn at its true extent,
with no symbol. The cells of a set too small to see in a chart panel (the
area test above) are also outlined in the color of the set by a line 0.5 pt
wide, in the panel and in its zooms, so that cells a fraction of a point
wide are seen; this is how a small set spread too far for one zoom is seen
in the panel. Larger sets are not outlined. This is
`draw_paper_grid_figure` in `hybrid_dynamics/examples/paper_grid_figures.py`,
with the options `frame_margin`, `handle_view`, and `detail_zooms` of
`plot_atlas_hybrid_morse_sets`.

The JSON summary records each variant under
`figure_variants` (shown nodes, hidden nodes with the reason, blocked nodes,
the order drawn, the zooms with their windows and nodes, files) and stores
the atoms of every Morse set under `morse_graph.morse_set_atoms` (strings of
atom ranges, `a-b` for the closed range). Records written before cells were
outlined also list, under `marked_in_panel`, the sets whose cells were
marked by squares in a chart panel. `demo/replot_paper_grid.py` redraws both
variants from a summary: it rebuilds `Xi_n` from the example and level,
checks it against the recorded `grid`, and replaces `figures` and
`figure_variants` in the JSON (dropping `marked_in_panel`). Summaries
written before the atoms were stored (schema `paper-suspension-grid-run-v2`
and earlier) cannot be redrawn and have to be rerun. Selection and replot are
tested in `hybrid_dynamics/tests/test_suspension_grid_plot.py`.

## Reproduction

From `code/`:

```bash
.venv/bin/python demo/run_paper_grid_examples.py --workers 12
.venv/bin/python demo/run_paper_grid_examples.py --workers 12 --gap-refinement-depth 12
.venv/bin/python demo/run_paper_grid_examples.py --workers 12 --eval-mode tensor --samples-per-axis 3
.venv/bin/python demo/replot_paper_grid.py figures/paper_grid/paper-grid-spiking-neuron-*.json
```

The runner options `--eval-mode`, `--num-pts`, `--sample-depth`, `--seed`,
`--samples-per-axis`, and `--gap-refinement-depth` select the sampling; the
options in force, the offsets, and the command line are recorded under
`image_rule` in the JSON summary. Outputs (PDF, PNG, JSON) are written to
`figures/paper_grid/`; every non-default choice adds a suffix to the file
names (`-center`, `-random10d4s0`, `-tensor3`, `-gap-refined`), and a base
offset adds `-base<cells per axis>` after the level (see the next section).
The commands above use the manuscript's `tau`; the configurations for the
paper figures are in "Paper figures at small `tau`". A parameter variant
(`PAPER_GRID_VARIANTS`, at present `impact-vdp-duffing-beta076`) is run only
when named and has its name in the output names; see "The oscillator at
`beta = 0.76`". Recorded runs are sorted
into subfolders of `figures/paper_grid/`, listed in
`figures/paper_grid/README.md`.

## Base grids finer than the phase grid

The cofiltration of the manuscript allows any contracting sequence of base
grids, so the base grid of `Xi_n` can be finer than its phase grid.
`--level-offset EXAMPLE=K` builds level `n` with `2^(n+K)` base cells per
axis of the ambient rectangle (`X_j` has `2^(j+K)` cells per axis) and
`2^(n+2)` phase cells of width `a_n = 2^(-n-2)`; the generators `K_j`, `Q_j`
range over `j <= n` as before. A positive offset adds `-base<cells per axis>`
after the level in the output names
(`paper-grid-bouncing-ball-tau050-level6-base1024-corners.json`); offset `0`
keeps the earlier names. The JSON summary records `level_offset`,
`base_cells_per_axis`, and `phase_cells`, and `demo/replot_paper_grid.py`
rebuilds the grid with the recorded offset.

```bash
.venv/bin/python demo/run_paper_grid_examples.py bouncing-ball --level bouncing-ball=6 \
    --level-offset bouncing-ball=4 --tau bouncing-ball=0.5 --workers 3 --index-max-pieces 100000
```

Where the time goes:

- Base samples. With `corners` every vertex of the base lattice in `R` is
  evaluated once, with the integrator of the example (`solve_ivp`, RK45,
  `rtol=1e-10`, `atol=1e-12`, `max_step=0.02`). The cost is linear in the
  number of vertices, about 0.2 ms (ball, `tau = 0.5`) to 0.9 ms (neuron,
  `tau = 2`) of wall time per sample with `--workers 3` and four runs at once.
  This is the floor of a run: the endpoints, and so the relation, depend on
  the integration.
- Handle samples: one path per guard coordinate, seconds.
- Grid, point location, relation assembly, Morse graph, and image
  connectivity: seconds at 1024 cells per axis, under a minute at 2048.
- Gap refinement: the inserted samples cost what base samples cost.
- Index labels: the quotient nerve of `X = S cup F(S)` has about ten
  simplices per elementary piece and is held in memory with the carriers. A
  pair blocked by a non-acyclic carrier stops at the first failing cell; a
  computed index takes about two minutes for `10^4` pieces and ten minutes
  and 6 to 7 GB for `6 x 10^4` pieces. `--index-max-pieces` skips larger
  pairs.

The code is exercised against the earlier computations in
`hybrid_dynamics/tests/test_suspension_grid_scaling.py`: small runs of all
four examples (with offsets, gap refinement, and both exit policies)
reproduce the digests of their relations, statistics, Morse graphs, and index
records computed with commit `24ecdda`.

### Feasibility at phase level 6

Four runs at a time with `--workers 3` on 14 cores (48 GB), commits
`5a07656` to `575f603`, `--index-max-pieces 100000`, gap refinement depth 12
where marked. Wall time and the peak resident memory of the run (runner and
workers). Repeating 17 of these runs with `1eedf06` took between 0.7 and 6.7
times as long, depending on the load from other processes; most index times
below predate `575f603`, which shortened a computed index of 11,297 pieces
from 162 to 112 s. Index labels: nontrivial shift classes with the pieces of
`X`; the remaining Morse nodes are trivial, blocked by a non-acyclic carrier,
or over the piece limit.

| Example, `tau` | base cells | gap | wall (s) | peak (GB) | base samples | integration (s) | gap refinement (s, samples) | index (s) | Morse nodes | nontrivial labels (pieces of `X`) |
|---|---|---|---|---|---|---|---|---|---|---|
| ball, 0.5 | 256 | no | 27 | 0.9 | 66,049 | 15 | | 3 | 1: 1 blocked | |
| ball, 0.5 | 512 | no | 63 | 0.9 | 263,169 | 52 | | 3 | 1: 1 blocked | |
| ball, 0.5 | 1024 | no | 218 | 2.4 | 1,050,625 | 201 | | 3 | 1: 1 blocked | |
| ball, 0.5 | 2048 | no | 942 | 7.7 | 4,198,401 | 899 | | 3 | 1: 1 blocked | |
| ball, 0.5 | 256 | yes | 187 | 1.7 | 66,049 | 15 | 8 (40,436) | 158 | 2: 1 blocked | `(x-1, x-1, 0, 0)` (11,297) |
| ball, 0.5 | 512 | yes | 252 | 2.3 | 263,169 | 52 | 32 (180,232) | 160 | 2: 1 blocked | `(x-1, x-1, 0, 0)` (12,021) |
| ball, 0.5 | 1024 | yes | 490 | 3.0 | 1,050,625 | 203 | 142 (791,455) | 128 | 1 | `(x-1, x-1, 0, 0)` (13,270) |
| wheel, 1 | 256 | no | 112 | 1.3 | 66,049 | 24 | | 76 | 8: 4 trivial, 2 blocked | gait `(x-1, x-1, 0, 0)` (9,443); saddle `(0, x-1, 0, 0)` (88) |
| wheel, 1 | 512 | no | 208 | 1.6 | 263,169 | 93 | | 104 | 6: 4 trivial | gait (13,136); saddle (88) |
| wheel, 1 | 1024 | no | 543 | 2.5 | 1,050,625 | 370 | | 154 | 6: 4 trivial | gait (19,822); saddle (88) |
| wheel, 1 | 2048 | no | 1,562 | 9.2 | 4,198,401 | 1,506 | | 8 | 6: 4 trivial, 1 blocked (gait) | saddle (88) |
| wheel, 1 | 256 | yes | 136 | 1.3 | 66,049 | 23 | 29 (77,863) | 75 | 5: 2 trivial, 1 blocked | gait (9,531); saddle (97) |
| wheel, 1 | 512 | yes | 293 | 1.6 | 263,169 | 89 | 83 (225,537) | 106 | 4: 2 trivial | gait (13,271); saddle (97) |
| wheel, 1 | 1024 | yes | 863 | 3.6 | 1,050,625 | 392 | 346 (969,391) | 102 | 4: 2 trivial | gait (20,408); saddle (97) |
| neuron, 2 | 256 | no | 37 | 0.7 | 11,553 | 11 | | 8 | 143: 142 trivial, 1 blocked | |
| neuron, 2 | 512 | no | 63 | 1.0 | 45,665 | 39 | | 8 | 84: 83 trivial, 1 blocked | |
| neuron, 2 | 1024 | no | 185 | 1.8 | 181,569 | 157 | | 9 | 15: 14 trivial, 1 blocked | |
| neuron, 2 | 2048 | no | 679 | 2.8 | 724,097 | 641 | | 15 | 1: 1 blocked | |
| neuron, 2 | 256 | yes | 165 | 1.4 | 11,553 | 10 | 20 (26,247) | 120 | 143: 142 trivial | `(x-1, x-1, 0, 0, 0, 0)` (14,694) |
| neuron, 2 | 512 | yes | 234 | 1.7 | 45,665 | 37 | 44 (56,663) | 137 | 84: 83 trivial | `(x-1, x-1, 0, 0, 0, 0)` (20,053) |
| neuron, 2 | 1024 | yes | 429 | 3.0 | 181,569 | 163 | 117 (131,410) | 131 | 15: 14 trivial | `(x-1, x-1, 0, 0, 0, 0)` (28,826) |
| impact, 0.5 | 256 | no | 80 | 3.2 | 66,049 | 19 | | 46 | 1: 1 blocked | |
| impact, 0.5 | 512 | no | 809 | 6.2 | 263,169 | 71 | | 721 | 25: 6 trivial, 17 blocked | `(x-1, x-1, 0, 0)` (62,782); `(x-1, 0, 0, 0)` (10,579) |
| impact, 0.5 | 1024 | no | 382 | 2.6 | 1,050,625 | 288 | | 67 | 24: 6 trivial, 16 blocked, 1 over limit (100,895) | `(x-1, 0, 0, 0)` (5,529) |
| impact, 0.5 | 2048 | no | 1,267 | 9.3 | 4,198,401 | 1,141 | | 59 | 16: 9 trivial, 5 blocked, 1 over limit (180,052) | `(x-1, 0, 0, 0)` (5,539) |
| impact, 0.5 | 256 | yes | 197 | 3.7 | 66,049 | 18 | 6 (31,767) | 163 | 1: 1 blocked | |
| impact, 0.5 | 512 | yes | 903 | 6.2 | 263,169 | 69 | 17 (74,452) | 800 | 23: 4 trivial, 17 blocked | `(x-1, x-1, 0, 0)` (63,026); `(x-1, 0, 0, 0)` (10,580) |
| impact, 0.5 | 1024 | yes | 624 | 4.5 | 1,050,625 | 306 | 75 (269,605) | 212 | 24: 6 trivial, 15 blocked, 1 over limit | `(x-1, 0, 0, 0)` (5,529); `(x-1, x-1, 0, 0)` (15,231) |

Not run, estimated from the rows above (inserted samples at 0.8, 0.9, 0.6,
and 0.25 times the base samples, at the per-sample cost of the 2048 rows):
gap refinement at 2048 cells takes about 30 minutes for the ball, 50 for the
wheel, 20 for the neuron, and 25 to 40 for the impacting oscillator, with
peaks of about 4 GB for the neuron and 9 to 11 GB for the others. Since `1eedf06` the Morse graph no longer copies the
relation; on the 2048 ball grid with a synthetic map (72.7 million edges)
this lowered the peak from 9.8 to 7.6 GB. The 2048 rows predate it.

## Paper figures at small `tau`

### Criteria

The configurations for the figures of Section "Examples" follow three rules
of the author:

- `tau` is small, as in CMGDB practice: just large enough that the
  time-`tau` map is not close to the identity at the scale of the grid. The
  CMGDB ODE examples use `tau` from 0.1 to 1.
- Extra Morse nodes are acceptable, and a figure may show them. A spurious
  node with a nontrivial index would be incorrect. A blocked index is
  unknown, not trivial, so a blocked spurious node is not ruled out.
- Each run is drawn with every Morse node (`<stem>.pdf`, `.png`) and without
  the nodes whose computed index is trivial (`<stem>-nontrivial.pdf`,
  `.png`); see Figures.

### Recommended configurations

All four runs use corner sampling with gap refinement of depth 14,
`--workers 3`, and code `8e99989` on a clean tree. Each was chosen from the
sweep below and then checked independently: the grid was rebuilt, the Morse
sets decoded from the JSON, the invariant sets located in them, and the
order recomputed from the edges. The JSON summaries and both figure variants
are in `figures/paper_grid/`.

| Example | `tau` | level (phase cells) | base cells | Morse graph and labels | wall (s) | check |
|---|---|---|---|---|---|---|
| ball | 0.5 | 6 (256) | 1024 | 1 node: `Z~` `(x-1, x-1, 0, 0)` | 528 | passed |
| wheel | 0.5 | 7 (512) | 2048 | 17 nodes: saddle `(0, x-1, 0, 0)` -> gait `(x-1, x-1, 0, 0)`; 15 trivial | 1,797 | passed; this is the gap-refined twin of the sweep's pick |
| neuron | 5 | 7 (512) | 1024 | 1 node: cycle `(x-1, x-1, 0, 0, 0, 0)` | 1,232 | the sweep's pick, `tau = 1`, failed the `tau` rule |
| impact | 1 | 6 (256) | 1024 | 9 nodes: C `(x-1, x-1, 0, 0)`, F `(x-1, 0, 0, 0)`, Z `(x-1, x-1, 0, 0)`; S, U_Z, and a ring around F blocked; 3 trivial | 1,396 | passed |

Figures (`<stem>.png` shows every node, `<stem>-nontrivial.png` hides the
trivial ones; PDFs alongside), with `<stem>` in `figures/paper_grid/`:

- ball: `paper-grid-bouncing-ball-tau050-level6-base1024-corners-gap-refined`
- wheel: `paper-grid-rimless-wheel-tau050-level7-base2048-corners-gap-refined`
- neuron: `paper-grid-spiking-neuron-tau500-level7-base1024-corners-gap-refined`
- impact: `paper-grid-impact-vdp-duffing-tau100-level6-base1024-corners-gap-refined`

The wheel and neuron files were copied from the sweep. The figures of all
four were redrawn at `ff7ca77` with `demo/replot_paper_grid.py` in the layout
of Figures (base chart, zoom panels, handle chart, Morse graph); the figures
of the sweep, and of the neuron run at `tau = 1` below, have the earlier
layout of a base chart and a Morse graph. The folder also keeps
`paper-grid-spiking-neuron-tau100-level7-base1024-corners-gap-refined`, the
sweep's first pick for the neuron, a rerun that equals the sweep run except
for timings and paths.

To reproduce, from `code/` (add `--output-dir`, or the runner overwrites the
recorded files of the same name):

```bash
.venv/bin/python demo/run_paper_grid_examples.py bouncing-ball --level bouncing-ball=6 \
    --level-offset bouncing-ball=4 --tau bouncing-ball=0.5 --gap-refinement-depth 14 \
    --workers 3 --index-max-pieces 100000
.venv/bin/python demo/run_paper_grid_examples.py rimless-wheel --level rimless-wheel=7 \
    --level-offset rimless-wheel=4 --tau rimless-wheel=0.5 --gap-refinement-depth 14 \
    --workers 3 --index-max-pieces 100000
.venv/bin/python demo/run_paper_grid_examples.py spiking-neuron --level spiking-neuron=7 \
    --level-offset spiking-neuron=3 --tau spiking-neuron=5 --gap-refinement-depth 14 \
    --workers 3 --index-max-pieces 100000
.venv/bin/python demo/run_paper_grid_examples.py impact-vdp-duffing --level impact-vdp-duffing=6 \
    --level-offset impact-vdp-duffing=4 --tau impact-vdp-duffing=1 --gap-refinement-depth 14 \
    --workers 3 --index-max-pieces 150000
```

#### Ball

The only invariant set is `Z~`, the periodic orbit of the Zeno point through
the handle (`prop:ball-window`). In all 40 runs of the sweep one Morse node
contains the origin cell and all 257 sampled points of `Z~`. `tau = 0.5` is
the smallest tested `tau` with a run that has no other node (this grid, and
level 5 with 512 cells). Every gap-refined run at `tau = 0.25` and `0.375`
has 1 to 10 extra nodes, all blocked. `tau = 1` has no extra node on any
grid, but 1 is the period of `Z~`, so the time-1 map is the identity on the
attractor.

The pair has `A` empty and 13,270 pieces, homology dimensions `(1, 2, 0, 0)`,
and shift class `x-1` in degrees 0 and 1. The check reran the configuration
and matched every field except timings and paths, with byte-identical PNGs.
It confirmed `dim H_1 = 2` by counting the Euler characteristic of the pieces
with both seams glued, and attributes the second class to gaps of the handle
part over five phase intervals.

Caveats:

- The result is fragile. At `tau = 0.5` the five other gap-refined grids have
  one to three blocked extra nodes.
- The endpoint probe covers 150 random base cells and 20 guard intervals,
  none chosen near `Z~`.
- In the figure, the base readout (105 cells, `h` in `[0, 0.0039]`, `v` in
  `[-0.215, 0.303]`, on the edge `h = 0`) is a hairline in the base panel
  and is drawn in zoom A. The 13,165 handle pieces (99.2% of the pieces) lie
  over `v_G` in `[-0.44, 0]` and meet all 256 phase intervals. The two
  variants are identical, since no node is trivial.

#### Wheel

The invariant sets in `R = [-0.2, 0.6] x [-0.5, 1]` are the saddle at
`(0, 0)` and the gait, with speed 0.540 after impact, 0.502 at `theta = 0`,
and 0.775 before impact. The right unstable branch of the saddle reaches the
guard and converges to the gait, so saddle -> gait.

The sweep picked the corner run of this grid: 18 nodes, the same labels, and
saddle -> gait through trivial nodes. The check passed it, but in that run
3,574 atoms have a disconnected padded image, which cannot be an outer
approximation of a connected image. The gap-refined run of the same grid has
no such atom, the same nontrivial nodes and order, and one trivial node
fewer, so it is the one recommended here.

In the recommended run the gait node spans `theta` in `[-0.2, 0.6]` with
`thetadot` in `[0.492, 0.785]`, which contains the gait, and meets all 512
phase intervals. The saddle node is 72 base cells in
`[-0.0031, 0.0031] x [-0.0034, 0.0032]`, with no handle piece. The saddle
reaches the gait along `M(9) -> M(7) -> M(4) -> M(0)` and
`M(9) -> M(8) -> M(6) -> M(3) -> M(0)`, through trivial nodes only. The 15
trivial nodes are single base cells inside the bounding box of the saddle
node.

Why `tau = 0.5`: at `tau = 0.5` the grids with four base cells per phase
cell (1024 at level 6, 2048 at level 7) have no blocked node, with or without
gap refinement. At `tau = 0.25` every base grid finer than 256 cells has 2
to 7 blocked nodes, which gap refinement does not remove. At 256 cells the
corner runs are clean, which the sweep attributes to a wide gait enclosure
(`thetadot` from 0.32 to 0.91), and the gap-refined runs already have a
blocked node. The check calls this a judgment rather than a demonstrated
failure: the blocked counts are not monotone in the base grid (at
`tau = 0.5`: 1, 10 to 17, 0, and 0 or 1 for 256 to 2048 cells), and
`tau = 0.25` at level 7 with 2048 cells was not run.

Figure: the saddle node and the 15 trivial single cells around it are specks
in the base panel, where their window is outlined at the origin, and are
drawn in zoom A (window `[-0.005, 0.005] x [-0.0056, 0.0054]`). The
handle pieces of the gait node lie over `thetadot_G` in `[0.763, 0.788]`.
The run at `tau = 0.5`, level 6, 1024 base cells, gap-refined, gives the
same Morse graph and labels with a saddle set twice as wide (17 nodes, none
blocked, 732 s; in `sweep-rimless-wheel/`).

#### Neuron

The only invariant set is the spiking cycle: the guard return map has the
single fixed point `u* = -36.92`, the flow period is 147.85, and the cycle
lies in `v` in `[-56.11, 35]`, `u` in `[-36.92, 63.08]`; there is no
equilibrium. In every gap-refined run of the sweep one node, the unique sink,
contains all 22,002 sampled points of the cycle and its handle orbit and is
labeled `(x-1, x-1, 0, 0, 0, 0)`. Every other node has trivial index and is
a small set of base cells (at most 27 atoms, no handle piece) in the slow
region, `v` in `[-80, -26.25]`.

The sweep picked the smallest `tau`, `tau = 1`: 332 nodes, 331 of them
trivial. The check rejected it under the `tau` rule. It measured how far the
time-`tau` map moves points of the cycle, in cells of the 1024 grid: at
`tau = 1` the median is 0.52 cells, and the displacement is below one cell
for 74% of the cycle time; at `tau = 2` the median is 1.04 cells, below one
cell 48% of the time; at `tau = 5` it never falls below 1.33 cells (median
2.54). So `tau = 5` is the smallest tested value that satisfies the rule.
The sweep report had already named it as the alternative with a tight Morse
set and no extra node.

At `tau = 5`, level 7, 1024 base cells, the Morse graph is a single node
with 6,549 base cells in `v` in `[-58.75, 35]`, `u` in `[-48.75, 76.25]`, all
512 phase intervals, and `X` of 16,773 pieces. The 512-cell grid also gives
a single node. For the `tau = 1` run the check confirmed every recorded
field: the other 331 nodes all reach `M(0)`, and the `all` figure, with 332
overlapping labels, cannot be read.

#### Impacting oscillator

The runner locates five named sets (`impact_vdp_duffing_reference_sets`):
F, the attracting focus at `(-1, 0)`; S, the saddle at `(0, 0)`; Z, the
point `(0.8, 0)` on the wall with its handle orbit; C, the attracting impact
cycle; and U_Z, the repelling impact cycle around Z. The expected order is
U_Z -> Z, U_Z -> S, S -> C, S -> F.

No run labels all five sets: S and U_Z are never labeled. The recommended
run separates the five sets, labels C, F, and Z as expected, and has exactly
the expected order. Among the runs that do this, it has the fewest blocked
nodes (3), as do level 7 on the same grid (at a higher cost) and `tau = 2`
at 512 cells. The smaller `tau = 0.5` also labels C, F, and Z at 1024 cells
but leaves 12 or 13 spurious blocked nodes. The blocked nodes:

- U_Z (`M(8)`): its exit set `F(S) minus S` is two annuli, and
  `prepare_atlas_relation_conley_2d(use_exit_component_carrier=True)` sends
  each exit vertex to its whole exit component, which is then not acyclic.
  This is a limitation of the code, not of the sampling, and it blocks U_Z
  in every run of the sweep. The pair has relative homology of rank
  `(0, 1, 1)`, consistent with the expected `(0, x-1, x-1)`.
- S (`M(5)`): the Morse set (8,160 cells) is a thin set along the left
  unstable branch of S, which passes inside the stable branch at distance
  0.0143, about five base cells. One exit component is an annulus.
- `M(3)`: 16 cells in 9 components around F, a period-9 combinatorial cycle
  of the slow rotation of the focus. The check showed that its index is
  trivial although the code reports it blocked: each of the 9 components of
  `X` is acyclic and contains one acyclic exit component, so
  `H_*(X, A) = 0` over `GF(5)`.

The trivial nodes `M(4)`, `M(6)`, and `M(7)` are one or two cells on the
connections S -> C and U_Z -> S. The nontrivial figure hides them and draws
U_Z -> Z, U_Z -> S, S -> C, and S -> `M(3)` -> F. The check found the labels,
the order, and the locations in agreement with an independent computation of
C (multiplier 0.2448) and U_Z (multiplier 2.024).

Caveats:

- "No spurious nontrivial node" holds for the computed indices only. The
  runs at `tau = 0.25` and `0.5` that separate C and F leave 12 to 19
  spurious blocked nodes whose index is unknown.
- Figures. The 42 base cells of Z (`x` in `[0.7946, 0.8]`, on the wall)
  are drawn in zoom C of the `all` figure and zoom B of the `nontrivial`
  one; its 6,308 handle pieces lie over `v_G` in `[0, 0.081]`. The 16 cells
  of `M(3)` on the rim of F's node are drawn in zoom A. The trivial nodes
  `M(4)`, `M(6)`, and `M(7)` at the tip of S's node are in zoom B of the
  `all` figure. In the Morse graph of the `all`
  variant the labels of `M(2)` and `M(7)`, and of `M(1)` and `M(0)`,
  overlap.

### The oscillator at `beta = 0.76` (`impact-vdp-duffing-beta076`)

At `beta = 0.8` the left branches of the saddle's manifolds come within
0.0143 of each other (first turning points on `v = 0`, `x < -1`), and the
homoclinic loop around F occurs at `beta = 0.8138`. In the recommended run
above, the Morse set of S is then a loop of 8,160 cells around F. The
variant `impact-vdp-duffing-beta076` sets `beta = 0.76` and keeps every other
parameter, the window `R`, the guard, and the reset. The five sets and their
order persist for `beta` in `(0.7147, 0.8138)`. At 0.76 the turning points
are 0.0566 apart, which is 21 base cells of the 1024 grid and 42 of the 2048
grid. The last time an orbit from `Sigma(R)` is outside its interior is
3.75, against 4.58 at `beta = 0.8`, so `R` is unchanged.

The invariant sets at `beta = 0.76` (pre-impact speed `u`, suspension
period with the unit handle):

- F = `(-1, 0)`, a stable focus, eigenvalues `-0.12 +- 1.4091 i`.
- S = `(0, 0)`, a saddle, eigenvalues 1.4498 and -0.6898.
- Z = `(0.8, 0)`, the Zeno point; its basin is `u < 0.6570`.
- C, attracting: `u = 1.5098383`, multiplier 0.28870, period 5.9557, `x` in
  `[-1.6859, 0.8]`, `v` in `[-1.8769, 1.5098]`.
- U_Z, repelling: `u = 0.6569620`, multiplier 2.22912, period 4.1675, `x` in
  `[0.3761, 0.8]`, `v` in `[-0.4599, 0.6570]`.

The expected order is again U_Z -> Z, U_Z -> S, S -> C, S -> F. The labels
predicted are those of an attracting periodic orbit for C and Z,
`(x-1, 0, ...)` for F, `(0, x-1, 0, ...)` for S, and `(0, x-1, x-1, ...)`
for U_Z.

The variant. `PAPER_GRID_VARIANTS` in `examples/paper_grid_examples.py` maps
a variant name to its example and the factory arguments it replaces
(`impact-vdp-duffing-beta076`: `impact-vdp-duffing` with `beta = 0.76`), and
`paper_grid_problem(name, ...)` builds an example or a variant. The runner
takes a variant as a positional name and in `--level`, `--tau`, and
`--level-offset`, but runs it only when named: the default list is still the
four examples. The variant's name replaces the example's in the output
names, so its files never overwrite the runs at `beta = 0.8`. The JSON
summary records `variant_of`, `variant_overrides`, and `parameters.beta`.
`demo/replot_paper_grid.py` rebuilds variants, and the figures use the axes
of the example. The reference sets bracket the cycles by `u` in
`[1.4, 1.9]` (C) and `[0.55, 0.70]` (U_Z); the second bracket ends below
`(0.7109, 0.7434)`, where orbits after an impact go to F and the impact map
is undefined. Tests: `test_beta076_variant` in `test_impact_vdp_duffing.py`
and `test_a_variant_has_its_own_output_names_and_replots` in
`test_suspension_grid_plot.py`.

Runs. Code `a7ac462` on a clean tree, corner sampling, gap refinement of
depth 14, `--index-max-pieces 150000`, `--workers 12`, one run at a time on
14 cores. Peak memory is the largest resident set of one process
(`/usr/bin/time -l`). In both runs no endpoint probe is missed (0 of 3,515),
no padded image is disconnected, and the path exit policy gives the same
Morse sets and edges.

| `tau` | level (phase cells) | base cells | wall (s) | peak (GB) | base samples | gap samples | Morse nodes | labels | blocked | trivial |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 (256) | 1024 | 643 | 8.1 | 1,050,625 | 672,562 | 15 | C `(x-1, x-1, 0, 0)`, F `(x-1, 0, 0, 0)`, Z `(x-1, x-1, 0, 0)`, S `(0, x-1, 0, 0)` | 7: U_Z and 6 spurious | 4 |
| 0.5 | 7 (512) | 2048 | 780 | 11.9 | 4,198,401 | 1,037,721 | 22 | F `(x-1, 0, 0, 0)`, Z `(x-1, x-1, 0, 0)`, S `(0, x-1, 0, 0)` | 3: C (240,383 pieces, over the limit), U_Z, 1 spurious | 16 |

Stems in `figures/paper_grid/` (JSON, and `<stem>.pdf/.png`,
`<stem>-nontrivial.pdf/.png`):

- `paper-grid-impact-vdp-duffing-beta076-tau100-level6-base1024-corners-gap-refined`
- `paper-grid-impact-vdp-duffing-beta076-tau050-level7-base2048-corners-gap-refined`

The Morse nodes were identified from the recorded Morse sets on the rebuilt
grid: the runner locates points of the five sets in the pieces of each node
(`reference_set_identification`), and a separate check located the four
branches of the saddle's manifolds, computed the extents of the base cells
and handle pieces of each node, and tested whether its base cells enclose F.
Named nodes of the `tau = 1` run (the `tau = 0.5` run in parentheses):

| set | node | atoms | base cells and extent | handle pieces, `v_G` | label |
|---|---|---|---|---|---|
| C | `M(0)` (`M(0)`) | 57,007 (220,442) | 49,769 in `[-1.706, 0.8] x [-1.934, 1.564]` | 12,338 in `[1.438, 1.575]`, all phases | `(x-1, x-1, 0, 0)` (over the piece limit) |
| F | `M(1)` (`M(1)`) | 894 (3,868) | 894 in `[-1.048, -0.951] x [-0.070, 0.065]` | none | `(x-1, 0, 0, 0)` (same) |
| Z | `M(2)` (`M(2)`) | 3,491 (15,031) | 32 in `[0.7946, 0.8] x [-0.032, 0.035]` | 5,775 in `[0, 0.073]`, all phases | `(x-1, x-1, 0, 0)`, dimensions `(1, 2, 0, 0)` (same) |
| S | `M(10)` (`M(11)`) | 19 (58) | 19 in `[-0.0137, 0.0104] x [-0.0110, 0.0100]` (58 in `[-0.0110, 0.0104] x [-0.0089, 0.0100]`) | none | `(0, x-1, 0, 0)` (same) |
| U_Z | `M(14)` (`M(21)`) | 14,682 (60,770) | 9,901 in `[0.319, 0.8] x [-0.477, 0.690]` | 8,091 in `[0.602, 0.699]`, all phases | blocked (same) |

- S. The Morse set of S is now a small connected set of base cells at the
  origin, no farther than 0.013 (0.011 at 2048 cells) from it, with no
  handle piece. It does not enclose F, and the four branches of the
  saddle's manifolds start in it. Its label is computed in both runs and is
  the predicted `(0, x-1, 0, 0)`: `X` has 135 pieces (185) and is acyclic,
  and `A` has two acyclic components. At `beta = 0.8` the same
  configuration gave 8,160 cells in `[-1.440, 0.010] x [-0.981, 0.568]`,
  enclosing F, and a blocked label.
- U_Z. Blocked in both runs by a non-acyclic carrier, as in every run at
  `beta = 0.8`. `A` has two components, each an annulus
  (`H_*(A) = (2, 2)`), and the exit-component carrier sends an exit vertex
  to its whole component. `H_*(X, A; GF(5))`, computed separately on the
  same quotient nerve, is `(0, 1, 1, 0)` in both runs (`X` of 25,148 and
  89,084 pieces), consistent with the predicted `(0, x-1, x-1, 0)`.
- Z. The shift class is `x-1` in degrees 0 and 1 although `dim H_1 = 2`
  (it was 1 at `beta = 0.8`); the ball's label has the same form.
- C at 2048 cells. `X` has 240,383 pieces and was not attempted. If time
  and memory grow linearly in the pieces (62,107 pieces took 323 s in the
  `tau = 1` run; 6 to 7 GB for `6 x 10^4` pieces, see Feasibility), the
  label would take about 20 minutes and 25 GB on top of the run; it was not
  run.
- Order. Both Morse graphs give U_Z -> Z, U_Z -> S, S -> C, S -> F among the
  five sets, and every path from U_Z to C or F passes through S. At 1024
  cells U_Z reaches Z through the spurious nodes `M(8)`, `M(5)`, `M(6)`; at
  2048 cells directly.

Spurious nodes. None has a nontrivial label.

- `tau = 1`: four computed trivial nodes, single cells within 0.013 of S:
  `M(4)` on the right unstable branch (toward C), `M(9)` on the left one
  (toward F), `M(11)` on the right stable branch, and `M(12)` next to
  `M(11)`. Six blocked nodes: `M(3)` and `M(7)` (28 and 14 cells in 9
  components each, around F, the combinatorial rotation ring of the focus
  seen at `beta = 0.8`), `M(5)`, `M(6)`, `M(8)` (one or two base cells
  within two cells of the wall at `|v|` from 0.032 to 0.048, with 22 to 41
  handle pieces over `v_G` in `[0.039, 0.077]`, on the rim of Z's node), and
  `M(13)` (41 atoms on the rim of U_Z's node: 32 base cells in 12
  components and 22 handle pieces over `v_G` in `[0.682, 0.703]`). For each of the six, `X` and `A` have the same number
  of components (`M(3)` 9, `M(5)` 4, `M(6)` 4, `M(7)` 9, `M(8)` 4, `M(13)`
  9), all acyclic, and `H_*(X, A; GF(5)) = 0`.
  Their index is therefore trivial although the code reports it blocked: the
  carrier gate fails before the index map is formed, and any map on a zero
  space is trivial.
- `tau = 0.5`: sixteen computed trivial nodes, each of one to three base
  cells within 0.015 of S, on the connections U_Z -> S, S -> C, and S -> F.
  One blocked node, `M(3)`: 38 cells around F, the rotation ring, with `X`
  and `A` of 9 acyclic components each and `H_*(X, A; GF(5)) = 0`, so its
  index is trivial as well.

The relative homology was computed by recomputing the relation of each run
(the Morse sets were reproduced exactly), forming `X` and `A` and the
quotient nerve as `compute_suspension_grid_conley_index` does, and taking
the ranks of the relative boundary matrices over `GF(5)`. The nerve passed
its good-cover audit, so these are the ranks of `H_*(|X|, |A|)`.

Figures. The `nontrivial` figure of the `tau = 1` run shows 11 nodes: the
five sets, the ring nodes `M(3)` and `M(7)` (zoom A), S alone in zoom B,
and Z with `M(5)`, `M(6)`, `M(8)` at the stop in zoom C (window
`[0.782, 0.812] x [-0.067, 0.074]`: `M(6)` and one cell of `M(8)` just
above Z, `M(5)` and the other cell of `M(8)` just below). The 32 base cells
of `M(13)` lie along U_Z, too spread for a zoom that magnifies, and are
outlined in the base chart. In the handle chart, the pieces of `M(5)`,
`M(6)`, `M(8)` lie along the right edge of the band of Z (`v_G` in
`[0.039, 0.077]`) and those of `M(13)` along the right edge of the band of
U_Z (`v_G` in `[0.682, 0.703]`); they are outlined there, and no zoom can
enlarge them, since each spans most of the phase interval. The `nontrivial`
figure of the
`tau = 0.5` run shows 6 nodes: C (blocked, over the limit), F, Z (zoom C),
S (zoom B), U_Z (blocked), and the ring `M(3)` (zoom A); its Morse graph
reads U_Z -> Z, U_Z -> S, S -> C, S -> `M(3)` -> F. In the `all` variants
the labels of the Morse graph overlap.

Which run to use: the `tau = 1` run labels all of C, F, Z, S and leaves
six small blocked nodes whose index is trivial by the computation above;
the `tau = 0.5` run has the smaller `tau` and a cleaner figure but leaves C
unlabeled at this piece limit.

To reproduce, from `code/`:

```bash
.venv/bin/python demo/run_paper_grid_examples.py impact-vdp-duffing-beta076 \
    --level impact-vdp-duffing-beta076=6 --level-offset impact-vdp-duffing-beta076=4 \
    --tau impact-vdp-duffing-beta076=1 --gap-refinement-depth 14 --workers 12 \
    --index-max-pieces 150000
.venv/bin/python demo/run_paper_grid_examples.py impact-vdp-duffing-beta076 \
    --level impact-vdp-duffing-beta076=7 --level-offset impact-vdp-duffing-beta076=4 \
    --tau impact-vdp-duffing-beta076=0.5 --gap-refinement-depth 14 --workers 12 \
    --index-max-pieces 150000
```

### Sweep

Code `8e99989` on a clean tree, corner sampling, `--workers 3`, and
`--index-max-pieces` 100,000 for the ball, wheel, and neuron and 150,000 for
the oscillator, except where a row states another limit. Level `n` has
`2^(n+2)` phase cells, and the base grid is set with `--level-offset`. The
files are in `figures/paper_grid/sweep-<example>/`. Wall time is that of the
runner. Several sweeps ran at once on 14 cores (load average up to about
70), so wall times are inflated and uneven between runs of the same size.

Columns:

- nodes: Morse nodes. nontrivial: index computed, with a nonzero homology
  dimension. blocked: index not computed (a carrier that is not acyclic, or
  `X` over the piece limit). The remaining nodes have trivial index.
- nontrivial labels: each nontrivial node with the invariant set it
  contains. For the ball and neuron the sweep located sampled points of `Z~`
  and of the cycle; for the wheel the label identifies the node, and the
  saddle node contains `(0, 0)`; for the oscillator the JSON record
  `reference_set_identification` names the sets in each node, including the
  blocked ones (`+` joins sets that share a node).
- order: for the ball and neuron, whether every other node reaches the
  labeled node (it is the unique sink in every run where it is labeled). For
  the wheel, how the saddle reaches the gait in the full Morse graph:
  `direct` (an edge), `via trivial` (a path whose intermediate nodes all have
  trivial index), or `via blocked` (every path passes a blocked node). For
  the oscillator, reachability among the nodes that contain named sets,
  blocked ones included, transitively reduced among them.
- spurious nontrivial: nontrivial nodes that contain no invariant set.
- missed probes: probe points (a `5 x 5` lattice in 150 random base cells
  and 7 phases in 20 random guard intervals) whose endpoint lies in `D` but
  outside the computed image, out of those whose endpoint lies in `D`.

#### Ball (40 runs)

| `tau` | level | base cells | gap | nodes | nontrivial | blocked | nontrivial labels | order | spurious nontrivial | missed probes | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 5 | 256 | yes | 4 | 1 | 3 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2795 | 153 |
| 0.25 | 5 | 512 | yes | 11 | 1 | 10 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2840 | 250 |
| 0.25 | 5 | 1024 | yes | 4 | 1 | 3 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2883 | 317 |
| 0.25 | 6 | 256 | yes | 2 | 1 | 1 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2795 | 1,468 |
| 0.25 | 6 | 512 | yes | 5 | 1 | 4 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2840 | 2,225 |
| 0.25 | 6 | 1024 | yes | 4 | 1 | 3 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2883 | 2,276 |
| 0.25 | 7 | 1024 | yes | 5 | 1 | 4 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2883 | 798 |
| 0.375 | 6 | 1024 | yes | 4 | 1 | 3 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2725 | 586 |
| 0.5 | 5 | 256 | yes | 4 | 1 | 3 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2404 | 90 |
| 0.5 | 5 | 512 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/2515 | 208 |
| 0.5 | 5 | 1024 | yes | 2 | 1 | 1 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2515 | 480 |
| 0.5 | 6 | 256 | yes | 2 | 1 | 1 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2404 | 147 |
| 0.5 | 6 | 512 | yes | 2 | 1 | 1 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2515 | 260 |
| 0.5 | 6 | 1024 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/2515 | 528 |
| 0.5 | 6 | 2048 | yes | 2 | 1 | 1 | `Z~` `(x-1, x-1, 0, 0)` | all other nodes reach it | none | 0/2590 | 1,922 |
| 1 | 5 | 256 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/2005 | 50 |
| 1 | 5 | 512 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/1940 | 120 |
| 1 | 5 | 1024 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/1840 | 349 |
| 1 | 6 | 256 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/2005 | 85 |
| 1 | 6 | 512 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/1940 | 130 |
| 1 | 6 | 1024 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | - | none | 0/1840 | 398 |
| 0.25 | 5 | 256 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 0/2795 | 18 |
| 0.25 | 5 | 512 | no | 12 | 0 | 12 | none (`Z~` blocked) | - | none | 4/2840 | 47 |
| 0.25 | 5 | 1024 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 19/2883 | 174 |
| 0.25 | 6 | 256 | no | 2 | 0 | 2 | none (`Z~` blocked) | - | none | 5/2795 | 81 |
| 0.25 | 6 | 512 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 0/2840 | 307 |
| 0.25 | 6 | 1024 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 13/2883 | 1,106 |
| 0.25 | 6 | 2048 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 28/2990 | 649 |
| 0.5 | 5 | 256 | no | 4 | 0 | 4 | none (`Z~` blocked) | - | none | 0/2404 | 36 |
| 0.5 | 5 | 512 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 13/2515 | 95 |
| 0.5 | 5 | 1024 | no | 2 | 0 | 2 | none (`Z~` blocked) | - | none | 35/2515 | 347 |
| 0.5 | 6 | 256 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 5/2404 | 93 |
| 0.5 | 6 | 512 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 0/2515 | 99 |
| 0.5 | 6 | 1024 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 7/2515 | 333 |
| 1 | 5 | 256 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 14/2005 | 162 |
| 1 | 5 | 512 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 14/1940 | 94 |
| 1 | 5 | 1024 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 53/1840 | 389 |
| 1 | 6 | 256 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 17/2005 | 60 |
| 1 | 6 | 512 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 4/1940 | 130 |
| 1 | 6 | 1024 | no | 1 | 0 | 1 | none (`Z~` blocked) | - | none | 20/1840 | 416 |

- Gap refinement is needed for the label: every unrefined run blocks `Z~`
  (a single-vertex carrier), and unrefined runs miss up to 53 probes.
- No node other than the one of `Z~` ever received a computed index. The
  extra nodes lie in a thin shell just outside it, usually the second row of
  base cells above the ground with handle pieces over the matching guard
  intervals, and all reach it. Their number does not decrease monotonically
  with the grid, and at `tau = 0.25` it does not vanish at level 7 or at 2048
  cells.
- A smaller `tau` gives more extra nodes; `tau = 1` gives none. Gap
  refinement can add an extra node (level 6 at `tau = 0.5` with 256 and 512
  cells).

#### Wheel (40 runs)

| `tau` | level | base cells | gap | nodes | nontrivial | blocked | nontrivial labels | order | spurious nontrivial | missed probes | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 6 | 256 | yes | 66 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 367 |
| 0.25 | 6 | 512 | yes | 67 | 2 | 2 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3704 | 605 |
| 0.25 | 6 | 1024 | yes | 67 | 2 | 2 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3640 | 772 |
| 0.25 | 7 | 256 | yes | 66 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 692 |
| 0.25 | 7 | 512 | yes | 72 | 2 | 7 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3704 | 667 |
| 0.25 | 7 | 1024 | yes | 68 | 2 | 3 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` (piece limit 140,000) | saddle -> gait (via blocked) | none | 0/3640 | 945 |
| 0.5 | 6 | 256 | yes | 18 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3436 | 191 |
| 0.5 | 6 | 512 | yes | 27 | 2 | 10 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 267 |
| 0.5 | 6 | 1024 | yes | 17 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3415 | 732 |
| 0.5 | 6 | 2048 | yes | 17 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3565 | 1,438 |
| 0.5 | 7 | 256 | yes | 18 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3436 | 267 |
| 0.5 | 7 | 512 | yes | 34 | 2 | 17 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 377 |
| 0.5 | 7 | 1024 | yes | 17 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3415 | 740 |
| 0.5 | 7 | 2048 | yes | 17 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3565 | 1,797 |
| 1 | 6 | 256 | yes | 5 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 0/3258 | 116 |
| 1 | 6 | 512 | yes | 4 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3291 | 329 |
| 1 | 6 | 1024 | yes | 4 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 0/3233 | 862 |
| 1 | 7 | 256 | yes | 4 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 0/3258 | 175 |
| 1 | 7 | 512 | yes | 4 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3291 | 344 |
| 1 | 7 | 1024 | yes | 4 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 0/3233 | 892 |
| 0.25 | 6 | 256 | no | 66 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3615 | 352 |
| 0.25 | 6 | 512 | no | 68 | 2 | 2 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3704 | 2,363 |
| 0.25 | 6 | 1024 | no | 68 | 2 | 2 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3640 | 926 |
| 0.25 | 7 | 256 | no | 66 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 1/3615 | 624 |
| 0.25 | 7 | 512 | no | 73 | 2 | 7 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3704 | 733 |
| 0.25 | 7 | 1024 | no | 69 | 2 | 3 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` (piece limit 140,000) | saddle -> gait (via blocked) | none | 4/3640 | 1,032 |
| 0.5 | 6 | 256 | no | 19 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 8/3436 | 156 |
| 0.5 | 6 | 512 | no | 29 | 2 | 11 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 1,411 |
| 0.5 | 6 | 1024 | no | 18 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3415 | 539 |
| 0.5 | 6 | 2048 | no | 18 | 1 | 1 | saddle `(0, x-1, 0, 0)`; gait blocked | - | none | 4/3565 | 1,094 |
| 0.5 | 7 | 256 | no | 19 | 2 | 1 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 25/3436 | 276 |
| 0.5 | 7 | 512 | no | 35 | 2 | 17 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 0/3615 | 402 |
| 0.5 | 7 | 1024 | no | 18 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 4/3415 | 727 |
| 0.5 | 7 | 2048 | no | 18 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3565 | 1,691 |
| 1 | 6 | 256 | no | 8 | 2 | 2 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via blocked) | none | 20/3258 | 83 |
| 1 | 6 | 512 | no | 6 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3291 | 185 |
| 1 | 6 | 1024 | no | 6 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 0/3233 | 514 |
| 1 | 7 | 256 | no | 8 | 1 | 3 | saddle `(0, x-1, 0, 0)`; gait blocked | - | none | 124/3258 | 44 |
| 1 | 7 | 512 | no | 6 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (via trivial) | none | 0/3291 | 214 |
| 1 | 7 | 1024 | no | 6 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | saddle -> gait (direct) | none | 4/3233 | 562 |

- The number of trivial nodes depends only on `tau`: 64, 16, and 4 at
  `tau = 0.25`, 0.5, and 1 without gap refinement, and 63, 15, and 2 with it.
  They sit around the saddle and shrink with the cells, since the saddle is
  linear to leading order; a finer grid does not remove them.
- In two unrefined runs (`tau = 0.5`, level 6, 2048 cells; `tau = 1`, level
  7, 256 cells) the gait is blocked by a single-piece carrier; gap refinement
  labels it.
- The other blocked nodes lie in the basin of the gait (`thetadot` from
  0.32 to 0.84), mostly as handle pieces, and are blocked by the carrier of
  a two-piece simplex. Gap refinement does not remove them.
- In 11 runs every path from the saddle to the gait passes a blocked node.

#### Neuron (30 runs)

The row with piece limit 150,000 repeats the run above it
(`sweep-spiking-neuron/index-max-pieces-150000/`).

| `tau` | level | base cells | gap | nodes | nontrivial | blocked | nontrivial labels | order | spurious nontrivial | missed probes | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 256 | yes | 473 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 1,466 |
| 1 | 7 | 512 | yes | 561 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 430 |
| 1 | 7 | 1024 | yes | 332 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 539 |
| 1 | 8 | 256 | yes | 473 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 685 |
| 1 | 8 | 512 | yes | 561 | 0 | 1 | none (cycle blocked, IndexSizeLimitError) | - | none | 0/3890 | 112 |
| 1 | 8 | 1024 | yes | 332 | 0 | 1 | none (cycle blocked, IndexSizeLimitError) | - | none | 0/3890 | 281 |
| 1 | 8 | 1024 | yes | 332 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` (piece limit 150,000) | all other nodes reach it | none | 0/3890 | 740 |
| 2 | 7 | 256 | yes | 143 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 917 |
| 2 | 7 | 512 | yes | 84 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 291 |
| 2 | 7 | 1024 | yes | 15 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 1,671 |
| 2 | 8 | 256 | yes | 143 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 589 |
| 2 | 8 | 512 | yes | 84 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 628 |
| 2 | 8 | 1024 | yes | 15 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 1,089 |
| 5 | 7 | 256 | yes | 10 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 279 |
| 5 | 7 | 512 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 493 |
| 5 | 7 | 1024 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 1,232 |
| 5 | 8 | 256 | yes | 10 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | all other nodes reach it | none | 0/3890 | 693 |
| 5 | 8 | 512 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 943 |
| 5 | 8 | 1024 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 1,611 |
| 10 | 7 | 256 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 507 |
| 10 | 7 | 512 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 1,003 |
| 10 | 7 | 1024 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 2,504 |
| 10 | 8 | 256 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 735 |
| 10 | 8 | 512 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 1,459 |
| 10 | 8 | 1024 | yes | 1 | 1 | 0 | cycle `(x-1, x-1, 0, 0, 0, 0)` | - | none | 0/3890 | 3,377 |
| 1 | 7 | 1024 | no | 332 | 0 | 1 | none (cycle blocked, ValueError) | - | none | 144/3890 | 196 |
| 2 | 7 | 1024 | no | 15 | 0 | 1 | none (cycle blocked, ValueError) | - | none | 242/3890 | 181 |
| 5 | 7 | 1024 | no | 1 | 0 | 1 | none (cycle blocked, ValueError) | - | none | 129/3890 | 449 |
| 10 | 7 | 256 | no | 1 | 0 | 1 | none (cycle blocked, ValueError) | - | none | 97/3890 | 102 |
| 10 | 7 | 1024 | no | 1 | 0 | 1 | none (cycle blocked, ValueError) | - | none | 81/3890 | 811 |

- Trivial nodes per base grid of 256, 512, and 1024 cells, the same at
  levels 7 and 8: 472, 560, 331 at `tau = 1`; 142, 83, 14 at `tau = 2`; 9, 0,
  0 at `tau = 5`; none at `tau = 10`.
- The phase level changes only the number of handle pieces: at level 8 the
  cycle node has about twice as many, and at `tau = 1` with 512 and 1024
  cells it exceeds the 100,000 limit (102,468 and 128,291 pieces). It never
  changes the node count, the base readouts, or the labels.
- Without gap refinement the cycle node is blocked in every run, 81 to 242
  probes are missed, the node splits into 22 to 99 components of `Sigma X`,
  and 730 to 2,101 of the sampled cycle points lie in no Morse set.

#### Impacting oscillator (26 runs)

The rows at `tau = 1` and 2 with 256 cells at level 6 come from an earlier
attempt at the same commit, with piece limit 100,000.

| `tau` | level | base cells | gap | nodes | nontrivial | blocked | nontrivial labels | order | spurious nontrivial | missed probes | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 6 | 256 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 0/3446 | 116 |
| 0.25 | 6 | 512 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 0/3625 | 71 |
| 0.25 | 6 | 1024 | yes | 43 | 1 | 22 | F `(x-1, 0, 0, 0)`; blocked: C, Z+U_Z, S (piece limit 110,000) | Z+U_Z -> S, S -> C, S -> F | none | 0/3515 | 485 |
| 0.25 | 7 | 256 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 1/3446 | 175 |
| 0.25 | 7 | 512 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 0/3625 | 98 |
| 0.5 | 6 | 256 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 0/3444 | 167 |
| 0.5 | 6 | 512 | yes | 23 | 2 | 17 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; blocked: Z+U_Z, S | Z+U_Z -> S, S -> C, S -> F | none | 0/3534 | 806 |
| 0.5 | 6 | 1024 | yes | 24 | 3 | 15 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3540 | 1,419 |
| 0.5 | 7 | 256 | yes | 1 | 0 | 1 | none; blocked: C+F+Z+S+U_Z | - | none | 1/3444 | 237 |
| 0.5 | 7 | 512 | yes | 23 | 2 | 17 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; blocked: Z+U_Z, S | Z+U_Z -> S, S -> C, S -> F | none | 0/3534 | 841 |
| 0.5 | 7 | 1024 | yes | 23 | 3 | 14 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3540 | 1,815 |
| 1 | 6 | 256 | yes | 11 | 2 | 8 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; blocked: Z+U_Z, S (piece limit 100,000) | Z+U_Z -> S, S -> C, S -> F | none | 0/3570 | 1,196 |
| 1 | 6 | 512 | yes | 14 | 3 | 8 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3590 | 675 |
| 1 | 6 | 1024 | yes | 9 | 3 | 3 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3489 | 1,396 |
| 1 | 7 | 256 | yes | 11 | 2 | 8 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; blocked: Z+U_Z, S | Z+U_Z -> S, S -> C, S -> F | none | 1/3570 | 386 |
| 1 | 7 | 512 | yes | 15 | 3 | 9 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3590 | 1,222 |
| 1 | 7 | 1024 | yes | 9 | 3 | 3 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3489 | 1,612 |
| 2 | 6 | 256 | yes | 17 | 3 | 5 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z (piece limit 100,000) | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3640 | 2,321 |
| 2 | 6 | 512 | yes | 7 | 3 | 3 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3640 | 927 |
| 2 | 6 | 1024 | yes | 12 | 3 | 9 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3615 | 2,422 |
| 2 | 7 | 256 | yes | 17 | 3 | 5 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3640 | 2,225 |
| 2 | 7 | 512 | yes | 7 | 3 | 3 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3640 | 1,529 |
| 2 | 7 | 1024 | yes | 10 | 3 | 7 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; Z `(x-1, x-1, 0, 0)`; blocked: S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3615 | 2,822 |
| 1 | 6 | 1024 | no | 8 | 2 | 4 | C `(x-1, x-1, 0, 0)`; F `(x-1, 0, 0, 0)`; blocked: Z, S, U_Z | U_Z -> Z, U_Z -> S, S -> C, S -> F | none | 0/3489 | 890 |
| 2 | 6 | 512 | no | 7 | 1 | 6 | F `(x-1, 0, 0, 0)`; blocked: C, Z, U_Z | U_Z -> Z, U_Z -> C, U_Z -> F | none | 126/3640 | 314 |
| 2 | 6 | 1024 | no | 12 | 1 | 11 | F `(x-1, 0, 0, 0)`; blocked: C, Z, U_Z | U_Z -> Z, U_Z -> C, U_Z -> F | none | 55/3615 | 1,120 |

- At 256 cells for `tau = 0.25` and 0.5, and at 512 cells for `tau = 0.25`,
  all five sets lie in one blocked node. Z and U_Z share a node at 256 cells
  for `tau = 1`, at 512 cells for `tau = 0.5`, and at 1024 cells for
  `tau = 0.25`.
- Refining the base grid mainly removes the blocked rings around F, which
  come from the slow rotation of the focus. At `tau = 2` with 1024 cells the
  Morse set of S shrinks to 38 atoms in 10 components, and small blocked
  nodes appear on the connection U_Z -> S.
- Gap refinement is required: without it, Z is blocked at `tau = 1` and C
  and Z at `tau = 2`, and at `tau = 2` the point S lies in no Morse set.
- At `tau = 0.25` with 1024 cells, the pairs of S (`X` of 113,788 pieces)
  and of Z with U_Z (123,921 pieces) exceeded the lower limit of that run
  and would fit under 150,000; the pair of C (207,714 pieces) would not.

### Runs at the manuscript's `tau` (`figures/paper_grid/large_tau/`)

These corner runs at commit `0706668`, before the seam fix `788063c`, use
the manuscript's `tau` (ball 1.5, wheel 2, neuron 20) and `tau` from 3 to 6.5
for the oscillator, with the base grid of the level (`2^n` cells per axis).
They come from a step that was interrupted. Their summaries (schema
`paper-suspension-grid-run-v2`) store no Morse-set atoms, so they have one
figure each and cannot be redrawn. In the blocked column, `carrier` is a
carrier that is not acyclic and `seam` is the error `AtlasGoodCoverError:
guard seam is not on the boundary of the base Atlas window`. Time is the sum
of the recorded stages.

| example | `tau` | level | base cells | gap | nodes | nontrivial | blocked (cause) | nontrivial labels | missed probes | disconnected padded images / atoms | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ball | 1.5 | 5 | 32 | no | 1 | 0 | 1 (1 carrier) | none | 170/2774 | 1,597/3,043 | 8 |
| ball | 1.5 | 5 | 32 | yes | 1 | 1 | 0 | `Z~` `(x-1, x-1, 0, 0)` | 0/2774 | 0/3,043 | 78 |
| wheel | 2 | 6 | 64 | no | 2 | 0 | 2 (2 carrier) | none | 519/3209 | 4,359/15,095 | 14 |
| wheel | 2 | 6 | 64 | yes | 2 | 2 | 0 | gait `(x-1, x-1, 0, 0)`; saddle `(0, x-1, 0, 0)` | 2/3209 | 0/15,095 | 141 |
| neuron | 20 | 7 | 128 | no | 1 | 0 | 1 (1 seam) | none | 121/3890 | 2,324/26,326 | 99 |
| neuron | 20 | 8 | 256 | no | 1 | 0 | 1 (1 seam) | none | 91/3890 | 8,672/105,396 | 136 |
| impact | 3 | 7 | 128 | no | 25 | 1 | 14 (12 carrier, 2 seam) | F `(x-1, 0, 0, 0)`; blocked: C, Z, S, U_Z | 440/3740 | 17,938/46,622 | 44 |
| impact | 4 | 7 | 128 | no | 8 | 1 | 7 (5 carrier, 2 seam) | F `(x-1, 0, 0, 0)`; blocked: C, Z, S, U_Z | 636/3866 | 17,160/46,622 | 49 |
| impact | 5 | 6 | 64 | no | 19 | 1 | 4 (4 carrier) | F `(x-1, 0, 0, 0)`; blocked: C, Z+U_Z; S in no Morse set | 575/3890 | 5,759/11,780 | 33 |
| impact | 6 | 6 | 64 | no | 7 | 1 | 6 (6 carrier) | F `(x-1, 0, 0, 0)`; blocked: C, Z, U_Z; S in no Morse set | 799/3890 | 7,939/11,780 | 36 |
| impact | 6.5 | 6 | 64 | no | 7 | 1 | 6 (6 carrier) | F `(x-1, 0, 0, 0)`; blocked: C, Z, U_Z; S in no Morse set | 1019/3890 | 7,272/11,780 | 38 |

- Without gap refinement the sampling leaves large gaps: 2% to 26% of the
  probes are missed, and 8% to 67% of the atoms have a disconnected padded
  image. Only F of the oscillator gets a label; every other label is
  blocked.
- With gap refinement, the ball and wheel get their labels on these coarse
  grids: `Z~` `(x-1, x-1, 0, 0)`, and saddle `(0, x-1, 0, 0)` -> gait
  `(x-1, x-1, 0, 0)`.
- For the neuron at `tau = 20` the Morse graph is one node that meets only 4
  of the 512 (1,024) phase intervals, in 4 (7) components of `Sigma X`, and
  its label is blocked by the seam error that `788063c` removed. After the
  fix, runs at `tau = 20`, levels 6 and 7, are blocked by carriers that are
  not acyclic, also with gap refinement at level 6 (those outputs were not
  kept).
- For the oscillator only F is labeled, `(x-1, 0, 0, 0)`. C, Z, and U_Z are
  blocked in every run, and S is blocked at `tau = 3` and 4 and lies in no
  Morse set from `tau = 5` on.

### Small-`tau` runs before the seam fix (`figures/paper_grid/pre_fix/`)

Corner runs at `0706668`, without gap refinement, on the base grid of the
level. Same schema and columns as the previous folder.

| example | `tau` | level | base cells | gap | nodes | nontrivial | blocked (cause) | nontrivial labels | missed probes | disconnected padded images / atoms | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ball | 0.25 | 5 | 32 | no | 1 | 0 | 1 (1 carrier) | none | 5/2755 | 70/3,043 | 7 |
| ball | 0.25 | 6 | 64 | no | 1 | 0 | 1 (1 carrier) | none | 4/2890 | 311/12,230 | 18 |
| ball | 0.5 | 5 | 32 | no | 1 | 0 | 1 (1 carrier) | none | 14/2409 | 270/3,043 | 7 |
| ball | 0.5 | 6 | 64 | no | 1 | 0 | 1 (1 carrier) | none | 14/2631 | 1,114/12,230 | 19 |
| ball | 1 | 5 | 32 | no | 1 | 0 | 1 (1 carrier) | none | 175/1770 | 1,101/3,043 | 8 |
| ball | 1 | 6 | 64 | no | 1 | 0 | 1 (1 carrier) | none | 107/1694 | 4,278/12,230 | 12 |
| wheel | 0.25 | 6 | 64 | no | 33 | 0 | 1 (1 carrier) | none | 69/3689 | 228/15,095 | 22 |
| wheel | 0.5 | 6 | 64 | no | 11 | 0 | 1 (1 carrier) | none | 140/3461 | 454/15,095 | 17 |
| wheel | 1 | 6 | 64 | no | 7 | 1 | 2 (1 carrier, 1 seam) | saddle `(0, x-1, 0, 0)` | 431/3095 | 1,290/15,095 | 13 |
| neuron | 1 | 7 | 128 | no | 235 | 0 | 1 (1 carrier) | none | 425/3890 | 615/26,326 | 23 |
| neuron | 2 | 7 | 128 | no | 115 | 0 | 37 (1 carrier, 36 seam) | none | 244/3890 | 374/26,326 | 23 |
| impact | 0.25 | 7 | 128 | no | 1 | 0 | 1 (1 carrier) | none; blocked: C+F+Z+S+U_Z | 70/3534 | 792/46,622 | 80 |

- No label of a set that meets the guard was computed. The ball's `Z~` and
  the neuron's cycle are blocked at every `tau`, the oscillator's single
  node is blocked, and the wheel's gait is never labeled. The only label is
  the wheel's saddle at `tau = 1`.
- At `tau = 2`, 36 of the neuron's nodes were blocked by the seam error.
  After `788063c` the same configuration gives 114 trivial nodes and one
  node blocked by a carrier that is not acyclic, so those 36 nodes have
  trivial index (from the report of that fix; the outputs were not kept).

## Recorded runs with the tensor rule

All runs below used code commit `350c93e` (clean tree), the `3 x 3` tensor
rule that was then the default (now `--eval-mode tensor --samples-per-axis 3`),
12 worker processes,
and the tolerances of the example classes (`rtol=1e-10`, `atol=1e-12`,
`max_step=0.02`). Level `n` has `2^n` base cells per axis of the ambient
rectangle and phase width `a_n = 2^{-n-2}`. The JSON file next to each figure
in `figures/paper_grid/superseded-tensor3-350c93e/` holds every count quoted
here; these files predate the corner default and carry no sampling suffix in
their names.

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
