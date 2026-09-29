"""Tests for the paper suspension grid ``Xi_n`` and its sampled map.

The checks mirror the statements of the manuscript: the grid axioms and
``prop:suspension-grid`` (``d_n`` injective, ``iota^{-1}(|d_n(mu)|) = |mu|``),
the refinement identities of ``prop:finite-grid-preimage``, and the base
readout as a Boolean homomorphism.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from hybrid_dynamics import (
    DyadicBaseWindow,
    GuardResetSpec,
    HybridSystem,
    SuspensionFlow,
    SuspensionGridProblem,
    UnsupportedSuspensionGridError,
    atom_set_components,
    audit_suspension_grid_endpoints,
    build_suspension_grid,
    check_suspension_grid,
    compute_suspension_grid_conley_index,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
    piece_evaluation_offsets,
    piece_rectangles,
    suspension_grid_gluing,
)
from hybrid_dynamics.examples.paper_examples import (
    PAPER_PROBLEMS,
    bouncing_ball_problem,
    paper_problem_factory,
    rimless_wheel_problem,
    spiking_neuron_problem,
)
from hybrid_dynamics.src import atlas_conley
from hybrid_dynamics.src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
    simulate_suspension_endpoint,
)
from hybrid_dynamics.src.suspension_grid_relation import (
    ENDPOINT_BASE,
    ENDPOINT_HANDLE,
    SuspensionGridRelation,
)


def _grid(problem, level):
    return build_suspension_grid(problem.window, problem.guard, level)


# ---------------------------------------------------------------------------
# Synthetic systems
# ---------------------------------------------------------------------------


def _translation_problem(tau: float = 0.5) -> SuspensionGridProblem:
    """``x' = 1`` on ``[0,1]^2``, guard ``x = 1``, reset ``(1, y) -> (0, y)``.

    The suspension is an annulus on which ``f_tau`` is a rotation.
    """

    def ode(_t, state):
        return np.array([1.0, 0.0])

    def event(_t, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1

    def reset(state):
        return np.array([0.0, float(state[1])])

    system = HybridSystem(ode, event, reset, domain_bounds=None, event_direction=1)
    guard = GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((np.ones_like(u), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.zeros_like(u), u), axis=-1),
        name="translation",
    )
    return SuspensionGridProblem(
        system=system,
        window=DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0))),
        guard=guard,
        tau=tau,
        name="translation",
    )


def _unit_square_problem(
    field, *, tau: float, name: str, level_offset: int = 0
) -> SuspensionGridProblem:
    """``(x, y)' = field(x, y)`` on ``[0,1]^2``, guard ``x = 1``, reset ``(1, y) -> (0, y)``."""

    def ode(_t, state):
        return np.asarray(field(state), dtype=float)

    def event(_t, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1

    def reset(state):
        return np.array([0.0, float(state[1])])

    system = HybridSystem(ode, event, reset, domain_bounds=None, event_direction=1)
    guard = GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((np.ones_like(u), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.zeros_like(u), u), axis=-1),
        name=name,
    )
    return SuspensionGridProblem(
        system=system,
        window=DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0)), level_offset=level_offset),
        guard=guard,
        tau=tau,
        name=name,
    )


def _saddle_problem() -> SuspensionGridProblem:
    """A saddle at ``(0.5, 0.5)`` whose unstable branches end at sinks on ``x = 0`` and ``x = 0.9``.

    The lines ``x = 0`` and ``x = 0.9`` are invariant and ``y`` contracts to
    ``0.5``, so no orbit reaches the guard or leaves the window.
    """

    return _unit_square_problem(
        lambda state: (4.0 * (state[0] - 0.5) * state[0] * (0.9 - state[0]), 0.5 - state[1]),
        tau=1.0,
        name="saddle",
    )


def _repelling_orbit_problem() -> SuspensionGridProblem:
    """``x' = 1`` and ``y' = 12 (y - 1/2) y (1 - y)`` on the annulus of the translation problem.

    The circle ``y = 1/2`` is a repelling periodic orbit between the
    attracting orbits ``y = 0`` and ``y = 1``; no orbit leaves the window.
    """

    return _unit_square_problem(
        lambda state: (1.0, 12.0 * (state[1] - 0.5) * state[1] * (1.0 - state[1])),
        tau=0.5,
        name="repelling-orbit",
    )


def _repelling_cylinder_problem(rate: float = 16.0, tau: float = 2.0) -> SuspensionGridProblem:
    """``x' = 1``, ``y' = rate (y - 1/2) y (1 - y)`` with the reset of the translation cylinder.

    On the annulus the circles ``y = 0`` and ``y = 1`` are attracting
    periodic orbits and ``y = 1/2`` is a repelling one, whose Conley index
    is that of a two-dimensional unstable periodic orbit.  The window is
    forward invariant, so no sample leaves it.
    """

    return _unit_square_problem(
        lambda state: (1.0, rate * (state[1] - 0.5) * state[1] * (1.0 - state[1])),
        tau=tau,
        name="repelling-cylinder",
        level_offset=1,
    )


def _curved_interior_guard() -> tuple[DyadicBaseWindow, GuardResetSpec]:
    """A curved guard through cell interiors and a reset on a vertical line."""

    guard = GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((0.45 + 0.2 * u * u, u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, 0.25), 0.1 + 0.8 * u), axis=-1),
        name="curved",
    )
    return DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0))), guard


# ---------------------------------------------------------------------------
# Grid structure
# ---------------------------------------------------------------------------


def test_level_zero_ball_grid_matches_the_manuscript_figure():
    grid = _grid(bouncing_ball_problem(), 0)
    # One base cell R, one guard interval, four phase intervals.
    assert (grid.n_base, grid.n_guard, grid.n_phase) == (1, 1, 4)
    # Atoms: K_0(R) = base + bottom collar + top collar, Q_0(R,1), Q_0(R,2).
    assert grid.n_atoms == 3
    d_atom = int(grid.d_map[0])
    assert sorted(grid.atom(d_atom).tolist()) == [0, 1, 4]
    assert sorted(len(grid.atom(atom)) for atom in range(3)) == [1, 1, 3]


@pytest.mark.parametrize(
    ("factory", "levels"),
    [
        (bouncing_ball_problem, (0, 1, 2, 3, 4, 5)),
        (rimless_wheel_problem, (0, 2, 4)),
        (spiking_neuron_problem, (6, 7)),
    ],
)
def test_grid_axioms_and_d_n(factory, levels):
    problem = factory()
    for level in levels:
        grid = _grid(problem, level)
        check = check_suspension_grid(grid)
        assert check.passed, (level, check)
        assert grid.n_phase == 2 ** (level + 2)
        # d_n is injective and its atoms have exactly one base piece.
        assert np.unique(grid.d_map).size == grid.n_base
        # Atoms partition the pieces.
        assert np.array_equal(np.sort(grid.atom_pieces), np.arange(grid.n_pieces))


def test_curved_interior_guard_is_supported():
    window, guard = _curved_interior_guard()
    for level in (0, 1, 3):
        grid = build_suspension_grid(window, guard, level)
        assert check_suspension_grid(grid).passed
        assert not grid.guard_profile.straight
        # Every guard interval lies in one column of finest cells.
        for cells in grid.guard_bottom_cells:
            assert 1 <= cells.size <= 2


def test_non_monotone_guard_is_rejected():
    guard = GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((0.5 + 0.2 * np.sin(6.0 * u), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, 0.25), u), axis=-1),
    )
    with pytest.raises(UnsupportedSuspensionGridError):
        build_suspension_grid(DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0))), guard, 2)


def test_reset_leaving_the_window_is_rejected():
    guard = GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((np.ones_like(u), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.zeros_like(u), 2.0 * u), axis=-1),
    )
    with pytest.raises(UnsupportedSuspensionGridError):
        build_suspension_grid(DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0))), guard, 2)


def test_neuron_reset_on_a_cell_face_gives_shared_collar_atoms():
    grid = _grid(spiking_neuron_problem(), 6)
    owner = np.full(grid.n_atoms, -1)
    owner[grid.d_map] = np.arange(grid.n_base)
    for index in range(grid.n_guard):
        # r(gamma(J)) lies on v = -50, a face between two cells of X_6.
        assert grid.guard_top_cells[index].size == 2
        top_atom = grid.atom_of_piece[grid.handle_piece(index, grid.n_phase - 1)]
        assert owner[top_atom] == -1
        # The guard v = 35 is the boundary of X: the bottom collar joins d_n(mu).
        bottom_atom = grid.atom_of_piece[grid.handle_piece(index, 0)]
        assert owner[bottom_atom] == grid.guard_bottom_cells[index][0]


def test_coarse_top_collars_split_fine_middle_cells():
    # r^{-1}(|mu|) for coarse mu is not a union of fine guard cells, so the
    # middle phases below a coarse top collar split the level-n guard cells.
    grid = _grid(bouncing_ball_problem(), 3)
    phase = grid.n_phase - 2  # inside [1 - a_0, 1 - a_3]
    middle = 1  # inside [a_3, a_2]: only Q generators and bottom collars
    atoms_top = {int(grid.atom_of_piece[grid.handle_piece(J, phase)]) for J in range(grid.n_guard)}
    atoms_mid = {int(grid.atom_of_piece[grid.handle_piece(J, middle)]) for J in range(grid.n_guard)}
    assert len(atoms_top) > len(atoms_mid)


# ---------------------------------------------------------------------------
# prop:finite-grid-preimage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "levels"),
    [
        (bouncing_ball_problem, (1, 2, 4)),
        (rimless_wheel_problem, (1, 3)),
        (spiking_neuron_problem, (6, 7)),
    ],
)
def test_refinement_and_d_commute(factory, levels):
    problem = factory()
    grids = {level: _grid(problem, level) for level in levels}
    for n in levels:
        for m in levels:
            if m < n:
                continue
            fine, coarse = grids[m], grids[n]
            q_mn = fine.refinement_map(coarse)  # raises unless Xi_m refines Xi_n
            shift = m - n
            p_mn = coarse.base_cell_index[
                fine.base_addresses[:, 0] >> shift, fine.base_addresses[:, 1] >> shift
            ]
            assert np.array_equal(q_mn[fine.d_map], coarse.d_map[p_mn])


def test_curved_guard_refinement():
    window, guard = _curved_interior_guard()
    coarse = build_suspension_grid(window, guard, 1)
    fine = build_suspension_grid(window, guard, 3)
    q = fine.refinement_map(coarse)
    p = coarse.base_cell_index[fine.base_addresses[:, 0] >> 2, fine.base_addresses[:, 1] >> 2]
    assert np.array_equal(q[fine.d_map], coarse.d_map[p])


def test_base_readout_is_a_boolean_homomorphism():
    grid = _grid(rimless_wheel_problem(), 3)
    rng = np.random.default_rng(0)
    everything = np.arange(grid.n_atoms)
    base_everything = np.arange(grid.n_base)
    assert grid.base_readout([]).size == 0
    assert np.array_equal(grid.base_readout(everything), base_everything)
    for _ in range(20):
        u = rng.choice(grid.n_atoms, size=grid.n_atoms // 3, replace=False)
        v = rng.choice(grid.n_atoms, size=grid.n_atoms // 2, replace=False)
        ru, rv = grid.base_readout(u), grid.base_readout(v)
        assert np.array_equal(grid.base_readout(np.union1d(u, v)), np.union1d(ru, rv))
        assert np.array_equal(grid.base_readout(np.intersect1d(u, v)), np.intersect1d(ru, rv))
        assert np.array_equal(
            grid.base_readout(np.setdiff1d(everything, u)), np.setdiff1d(base_everything, ru)
        )
        # rho_n(|U|): the two-dimensional base pieces of the atoms of U.
        pieces = np.concatenate([grid.atom(atom) for atom in u])
        assert np.array_equal(np.sort(pieces[pieces < grid.n_base]), ru)


# ---------------------------------------------------------------------------
# Incidence and point location
# ---------------------------------------------------------------------------


def test_adjacency_is_symmetric_and_glues_the_zeno_point():
    grid = _grid(bouncing_ball_problem(), 4)
    adjacency = grid.piece_adjacency
    assert (adjacency != adjacency.T).nnz == 0
    zero = int(np.searchsorted(grid.u_edges, 0.0, side="left")) - 1  # J with right end 0
    bottom = grid.handle_piece(zero, 0)
    top = grid.handle_piece(zero, grid.n_phase - 1)
    # pi((0,0), 1) = iota(r(0,0)) = iota(0,0) = pi((0,0), 0).
    assert adjacency[bottom, top] == 1


def test_identifications_contribute_both_sides():
    grid = _grid(rimless_wheel_problem(), 3)
    u = 0.3
    guard_point = grid.guard.gamma([u])
    located = grid.locate_base_points(guard_point)
    kinds = grid.piece_kind(located.piece)
    assert set(kinds.tolist()) == {0, 1}
    handle = located.piece[kinds == 1]
    assert np.all(grid.handle_indices(handle)[1] == 0)
    reset_located = grid.locate_handle_points([u], [1.0])
    reset_kinds = grid.piece_kind(reset_located.piece)
    assert set(reset_kinds.tolist()) == {0, 1}
    base = reset_located.piece[reset_kinds == 0]
    expected = grid.locate_base_cells(grid.guard.reset([u])).piece
    assert set(base.tolist()) == set(expected.tolist())
    # Points outside the window are not located.
    assert grid.locate_base_points([[0.7, 0.0]]).piece.size == 0
    assert grid.locate_handle_points([1.5], [0.5]).piece.size == 0


# ---------------------------------------------------------------------------
# The unit-handle suspension semiflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [bouncing_ball_problem, rimless_wheel_problem])
def test_flow_agrees_with_sampled_suspension(factory):
    problem = factory()
    flow = SuspensionFlow(problem.system, max_step=problem.max_step)
    rng = np.random.default_rng(1)
    lower = np.array([interval[0] for interval in problem.window.ambient_bounds])
    upper = np.array([interval[1] for interval in problem.window.ambient_bounds])
    points = lower + (upper - lower) * rng.uniform(0.05, 0.95, size=(12, 2))
    for point in points:
        reference = simulate_suspension_endpoint(
            problem.system, point, problem.tau, max_step=problem.max_step
        )
        kind, state, phase = flow.path(point, problem.tau).evaluate(problem.tau)
        if isinstance(reference, BaseSuspensionSample):
            assert kind[0] == ENDPOINT_BASE
            assert np.allclose(state[0], reference.state, atol=1e-7)
        else:
            assert isinstance(reference, HandleSuspensionSample)
            assert kind[0] == ENDPOINT_HANDLE
            assert np.allclose(state[0], reference.guard_state, atol=1e-7)
            assert abs(phase[0] - reference.phase) < 1e-7


def test_flow_does_not_reset_off_guard_event_points():
    # theta = alpha + gamma with omega < 0 is not on the guard: it flows back.
    problem = rimless_wheel_problem()
    flow = SuspensionFlow(problem.system, max_step=0.02)
    kind, state, _phase = flow.path(np.array([0.6, -0.3]), 0.1).evaluate(0.1)
    assert kind[0] == ENDPOINT_BASE and state[0, 0] < 0.6


def test_zeno_point_circulates_through_the_handle():
    problem = bouncing_ball_problem()
    flow = SuspensionFlow(problem.system, max_step=0.02)
    path = flow.path(np.zeros(2), 1.5, start_on_handle=True)
    kind, state, phase = path.evaluate(np.array([0.25, 1.5, 2.0]))
    assert kind.tolist() == [ENDPOINT_HANDLE, ENDPOINT_HANDLE, ENDPOINT_BASE]
    assert np.allclose(phase[:2], [0.25, 0.5])
    assert np.allclose(state, 0.0)


def test_example_reset_specifications_match_the_systems():
    for name, factory in PAPER_PROBLEMS.items():
        problem = factory()
        u = np.linspace(*problem.guard.u_bounds, 7)
        for value, guard_point, reset_point in zip(
            u, problem.guard.gamma(u), problem.guard.reset(u)
        ):
            assert np.allclose(problem.system.reset_map(guard_point), reset_point), name
            assert np.isclose(problem.guard.coordinate(guard_point)[0], value), name


# ---------------------------------------------------------------------------
# Relation, Morse graph, probes, and the index
# ---------------------------------------------------------------------------


def test_translation_cylinder_relation_morse_graph_probes_and_index():
    problem = _translation_problem()
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    assert check_suspension_grid(grid).passed
    relation = compute_suspension_grid_relation(grid, problem)
    stats = relation.statistics
    assert stats["failed_endpoints"] == 0
    assert stats["discarded_exit_endpoints"] == 0
    assert (relation.sampled.multiply(relation.matrix) != relation.sampled).nnz == 0
    morse = compute_suspension_morse_graph(relation)
    # f_tau is a rotation of the annulus: one Morse set containing every atom.
    assert len(morse.morse_sets) == 1
    assert morse.morse_sets[0].size == grid.n_atoms
    assert atom_set_components(grid, morse.morse_sets[0]) == 1
    audit = audit_suspension_grid_endpoints(relation, problem, subdivision=4, seed=3)
    assert audit.witnesses and audit.passed
    result = compute_suspension_grid_conley_index(relation, morse.morse_sets[0])
    assert result.computed, result.blocker
    # H_*(annulus) with f_tau homotopic to the identity.
    assert result.shift_class[:2] == ("x-1", "x-1")
    assert set(result.shift_class[2:]) <= {"0"}


def test_ball_relation_is_one_morse_node_at_level_three():
    problem = bouncing_ball_problem()
    grid = _grid(problem, 3)
    relation = compute_suspension_grid_relation(grid, problem)
    morse = compute_suspension_morse_graph(relation)
    assert len(morse.morse_sets) == 1 and morse.edges == ()
    readout = grid.base_readout(morse.morse_sets[0])
    # The base readout contains the cell at the Zeno point.
    origin_cells = grid.locate_base_cells([[0.0, 0.0]]).piece
    assert set(origin_cells.tolist()) <= set(readout.tolist())
    # Forward closure of a Morse set is forward invariant.
    closure = relation.forward_closure(morse.morse_sets[0])
    assert np.all(np.isin(relation.image_of(closure), closure))


def test_gap_refinement_is_opt_in_and_only_adds_edges():
    problem = bouncing_ball_problem()
    grid = _grid(problem, 3)
    for mode in ("corners", "tensor"):
        plain = compute_suspension_grid_relation(grid, problem, eval_mode=mode)
        refined = compute_suspension_grid_relation(
            grid, problem, eval_mode=mode, gap_refinement_depth=6
        )
        assert plain.statistics["default_rule"] == (mode == "corners")
        assert not refined.statistics["default_rule"]
        assert refined.statistics["gap_refinement_samples"] > 0
        assert (plain.matrix.multiply(refined.matrix) != plain.matrix).nnz == 0
        assert refined.statistics["unresolved_gap_segments"] <= plain.n_edges


def test_evaluation_offsets_mirror_cmgdb():
    corners, denominator = piece_evaluation_offsets("corners")
    assert denominator == 1 and corners.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    center, denominator = piece_evaluation_offsets("center")
    assert denominator == 2 and center.tolist() == [[1, 1]]
    first, denominator = piece_evaluation_offsets("random", num_pts=7, sample_depth=3, seed=5)
    again, _ = piece_evaluation_offsets("random", num_pts=7, sample_depth=3, seed=5)
    other, _ = piece_evaluation_offsets("random", num_pts=7, sample_depth=3, seed=6)
    assert denominator == 8 and first.shape == (7, 2)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert first.min() >= 0 and first.max() <= 8
    tensor, denominator = piece_evaluation_offsets("tensor", samples_per_axis=2)
    assert denominator == 1 and np.array_equal(tensor, corners)
    with pytest.raises(ValueError):
        piece_evaluation_offsets("vertices")
    with pytest.raises(ValueError):
        piece_evaluation_offsets("random", num_pts=0)
    cmgdb = pytest.importorskip("CMGDB.PrecomputedBoxMap")
    for mode, options in (
        ("corners", {}),
        ("center", {}),
        ("random", {}),
        ("random", {"num_pts": 7, "sample_depth": 3, "seed": 5}),
    ):
        ours, denominator = piece_evaluation_offsets(mode, **options)
        theirs, depth = cmgdb.evaluation_offsets(mode, 2, **options)
        assert np.array_equal(ours, theirs) and denominator == 2**depth


def test_default_sampling_is_the_corners_of_every_piece():
    problem = bouncing_ball_problem()
    grid = _grid(problem, 3)
    relation = compute_suspension_grid_relation(grid, problem)
    stats = relation.statistics
    assert stats["eval_mode"] == "corners" and stats["default_rule"]
    assert stats["samples_per_piece"] == 4 and stats["padding_forced"] is False
    # Each distinct vertex is evaluated once: the vertices of the base cells,
    # and the (n_guard + 1) x (n_phase + 1) vertices of the handle pieces.
    vertices = np.unique(
        (grid.base_addresses[:, None, :] + np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).reshape(
            -1, 2
        ),
        axis=0,
    )
    assert stats["unique_base_samples"] == vertices.shape[0]
    assert stats["unique_handle_samples"] == (grid.n_guard + 1) * (grid.n_phase + 1)
    assert stats["handle_paths"] == grid.n_guard + 1
    same = compute_suspension_grid_relation(
        grid, problem, eval_mode="tensor", samples_per_axis=2
    )
    assert (relation.matrix != same.matrix).nnz == 0
    with pytest.raises(ValueError, match="samples_per_axis"):
        compute_suspension_grid_relation(grid, problem, samples_per_axis=3)


def test_center_forces_padding_and_random_is_deterministic():
    problem = bouncing_ball_problem()
    grid = _grid(problem, 2)
    center = compute_suspension_grid_relation(grid, problem, eval_mode="center", padding=False)
    padded = compute_suspension_grid_relation(grid, problem, eval_mode="center")
    assert center.statistics["padding_forced"] and not padded.statistics["padding_forced"]
    assert center.statistics["unique_base_samples"] == grid.n_base
    assert (center.matrix != padded.matrix).nnz == 0
    first = compute_suspension_grid_relation(grid, problem, eval_mode="random", num_pts=4, seed=3)
    again = compute_suspension_grid_relation(grid, problem, eval_mode="random", num_pts=4, seed=3)
    assert (first.matrix != again.matrix).nnz == 0
    assert first.statistics["seed"] == 3 and first.statistics["samples_per_piece"] == 4
    for mode in ("center", "random"):
        with pytest.raises(ValueError, match="no lattice edges"):
            compute_suspension_grid_relation(
                grid, problem, eval_mode=mode, gap_refinement_depth=2
            )


# ---------------------------------------------------------------------------
# Conley adapter and the nerve candidate index
# ---------------------------------------------------------------------------


def test_seam_embeddings_of_the_examples():
    for factory in PAPER_PROBLEMS.values():
        problem = factory()
        grid = _grid(problem, 1 if problem.name != "spiking-neuron" else 6)
        gluing = suspension_grid_gluing(grid)
        u = np.linspace(*problem.guard.u_bounds, 5)
        for value, guard_point, reset_point in zip(
            u, problem.guard.gamma(u), problem.guard.reset(u)
        ):
            assert np.allclose(gluing.guard.point(value), guard_point)
            assert np.allclose(gluing.reset.point(value), reset_point)


def test_neuron_index_without_a_base_cell_at_the_guard():
    # The Morse sets of the neuron on Xi_6 and Xi_7 contain handle pieces but
    # no base cell at the guard v = 35 (the guard seam lies outside their
    # base window).  S below is the upper half of the handle strip over one
    # guard interval J and the two base cells whose common face is
    # r(gamma(J)) on v = -50.  With F(S) = S sending every atom onto S, the
    # label is the homology of |S|, a contractible union of pieces.
    grid = _grid(spiking_neuron_problem(), 6)
    index = grid.n_guard // 2
    handle = [grid.handle_piece(index, k) for k in range(grid.n_phase // 2, grid.n_phase)]
    pieces = np.array([*handle, *grid.guard_top_cells[index]], dtype=np.int64)
    atoms = np.unique(grid.atom_of_piece[pieces])
    # Every atom of S is a single piece of the selection above.
    assert np.array_equal(
        np.sort(np.concatenate([grid.atom(atom) for atom in atoms])), np.sort(pieces)
    )
    cells = piece_rectangles(grid, pieces)
    assert max(cell.bounds[2] for cell in cells if cell.chart_id == 0) < 35.0
    matrix = np.zeros((grid.n_atoms, grid.n_atoms), dtype=bool)
    matrix[np.ix_(atoms, atoms)] = True
    relation = SuspensionGridRelation(
        grid=grid,
        tau=1.0,
        matrix=sparse.csr_matrix(matrix),
        sampled=sparse.csr_matrix(matrix),
        statistics={},
    )

    result = compute_suspension_grid_conley_index(relation, atoms)

    assert result.computed, result.blocker
    assert result.pair_atoms == {"S": atoms.size, "X": atoms.size, "A": 0}
    assert result.shift_class[0] == "x-1" and set(result.shift_class[1:]) <= {"0"}
    assert result.homology_dimensions[0] == 1 and not any(result.homology_dimensions[1:])


def test_nerve_candidate_index_matches_the_exhaustive_scan(monkeypatch):
    grid = _grid(bouncing_ball_problem(), 2)
    cells = piece_rectangles(grid, np.arange(grid.n_pieces))
    gluing = suspension_grid_gluing(grid)
    indexed = atlas_conley.AtlasQuotientNerveComplex2D(cells, gluing)

    def exhaustive(self, cells):
        indices = tuple(sorted(cell.index for cell in cells))
        return {index: indices for index in indices}

    monkeypatch.setattr(
        atlas_conley.AtlasQuotientNerveComplex2D, "_candidate_neighbors", exhaustive
    )
    scanned = atlas_conley.AtlasQuotientNerveComplex2D(cells, gluing)
    assert indexed.cell_set == scanned.cell_set


def test_import_cmgdb_does_not_import_hybrid_dynamics():
    pytest.importorskip("CMGDB")
    code = (
        "import sys, CMGDB; "
        "assert not [m for m in sys.modules if m.startswith('hybrid_dynamics')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _relation_on(grid, sources, targets):
    matrix = np.zeros((grid.n_atoms, grid.n_atoms), dtype=bool)
    matrix[np.ix_(sources, targets)] = True
    return SuspensionGridRelation(
        grid=grid,
        tau=1.0,
        matrix=sparse.csr_matrix(matrix),
        sampled=sparse.csr_matrix(matrix),
        statistics={},
    )


def test_zero_relative_homology_gives_the_trivial_label_without_the_index_map(
    monkeypatch,
):
    # S is one base cell and F(S) is S and the cell next to it, so X is two
    # cells sharing an edge and A is one of them: H_*(X, A) = 0.
    from hybrid_dynamics.src import suspension_grid_conley

    grid = _grid(bouncing_ball_problem(), 3)
    pieces = grid.locate_base_cells([[1.1, 0.6], [1.35, 0.6]]).piece
    source, neighbor = (int(grid.atom_of_piece[piece]) for piece in pieces)
    assert source != neighbor
    relation = _relation_on(grid, [source], [source, neighbor])

    def not_formed(*args, **kwargs):
        raise AssertionError("the index map is formed although H_*(X, A) = 0")

    monkeypatch.setattr(suspension_grid_conley, "prepare_atlas_relation_conley_2d", not_formed)
    monkeypatch.setattr(suspension_grid_conley, "native_relation_shift_class", not_formed)
    result = compute_suspension_grid_conley_index(relation, [source])

    assert result.computed and result.homology_computed, result.blocker
    assert result.label_source == "zero relative homology"
    assert not any(result.homology_dimensions)
    assert set(result.shift_class) == {"0"}
    assert len(result.shift_class) == len(result.homology_dimensions)


def test_homology_is_reported_when_the_index_map_fails(monkeypatch):
    from hybrid_dynamics.src import suspension_grid_conley

    problem = _translation_problem()
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    relation = compute_suspension_grid_relation(grid, problem)
    morse = compute_suspension_morse_graph(relation)
    computed = compute_suspension_grid_conley_index(relation, morse.morse_sets[0])
    assert computed.computed and computed.label_source == "index map"

    def fails(*args, **kwargs):
        raise ValueError("carrier image is not acyclic over GF(5)")

    # The failure is put in the Python construction of the chain map.
    monkeypatch.setattr(suspension_grid_conley, "prepare_atlas_relation_conley_2d", fails)
    blocked = compute_suspension_grid_conley_index(
        relation, morse.morse_sets[0], backend="python"
    )

    assert not blocked.computed and blocked.shift_class == ()
    assert blocked.homology_computed
    # The annulus: the same dimensions the index map reports when it succeeds.
    assert blocked.homology_dimensions == computed.homology_dimensions
    assert blocked.homology_dimensions[:2] == (1, 1)
    assert blocked.blocker.startswith("ValueError: carrier image")
    assert blocked.index_map_blocker == blocked.blocker
    record = blocked.to_dict()
    assert record["homology_computed"] and record["label_source"] == ""


def test_parallel_indices_equal_the_serial_ones():
    from hybrid_dynamics.examples.paper_examples import paper_problem_factory
    from hybrid_dynamics.src.suspension_grid_conley import compute_suspension_grid_conley_indices

    # Eight Morse sets: seven single atoms with zero relative homology and the
    # gait, whose index map fails its carrier check on this coarse grid.
    factory = paper_problem_factory("rimless-wheel", tau=0.5, level_offset=2)
    problem = factory()
    grid = _grid(problem, 3)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=2)
    morse = compute_suspension_morse_graph(relation)
    assert len(morse.morse_sets) > 2
    serial = compute_suspension_grid_conley_indices(relation, morse.morse_sets)
    parallel = compute_suspension_grid_conley_indices(
        relation, morse.morse_sets, workers=3, problem_factory=factory
    )

    def record(result):
        entry = result.to_dict()
        entry.pop("seconds")
        return entry

    assert [record(result) for result in parallel] == [record(result) for result in serial]
    assert [result.morse_node for result in parallel] == list(range(len(morse.morse_sets)))
    one_by_one = [
        compute_suspension_grid_conley_index(relation, morse_set, morse_node=node)
        for node, morse_set in enumerate(morse.morse_sets)
    ]
    assert [record(result) for result in one_by_one] == [record(result) for result in serial]
    # The forward-closure pair reads the rows of U, and the excision pair of
    # "auto" the rows of X; the workers receive them.
    for options in (
        {"index_pair": "forward-closure", "excise": True},
        {"index_map": "auto"},
    ):
        other_serial = compute_suspension_grid_conley_indices(
            relation, morse.morse_sets, **options
        )
        other_parallel = compute_suspension_grid_conley_indices(
            relation, morse.morse_sets, workers=3, problem_factory=factory, **options
        )
        assert [record(result) for result in other_parallel] == [
            record(result) for result in other_serial
        ]


# ---------------------------------------------------------------------------
# The excision construction of the index map
# ---------------------------------------------------------------------------


def _both_index_maps(relation, morse_sets):
    """Per Morse set with nonzero relative homology: the default and the excision result."""

    pairs = []
    for node, morse_set in enumerate(morse_sets):
        default = compute_suspension_grid_conley_index(relation, morse_set, morse_node=node)
        excision = compute_suspension_grid_conley_index(
            relation, morse_set, morse_node=node, index_map="excision"
        )
        assert excision.homology_dimensions == default.homology_dimensions
        if any(default.homology_dimensions):
            pairs.append((default, excision))
        else:
            assert excision.shift_class == default.shift_class
            assert excision.label_source == "zero relative homology"
    return pairs


@pytest.mark.parametrize(
    ("case", "level", "tau", "offset", "depth"),
    [
        ("translation", 2, 0.5, 0, 0),
        ("bouncing-ball", 3, 2.0, 1, 2),
        ("saddle", 4, 1.0, 0, 0),
    ],
)
def test_excision_index_map_gives_the_labels_of_the_default_construction(
    case, level, tau, offset, depth
):
    if case == "translation":
        problem = _translation_problem(tau)
    elif case == "saddle":
        problem = _saddle_problem()
    else:
        problem = PAPER_PROBLEMS[case](tau=tau, level_offset=offset)
    grid = _grid(problem, level)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=depth)
    morse = compute_suspension_morse_graph(relation)
    compared = [
        (default, excision)
        for default, excision in _both_index_maps(relation, morse.morse_sets)
        if default.computed
    ]
    assert compared
    for default, excision in compared:
        assert excision.computed, excision.blocker
        assert excision.shift_class == default.shift_class
        assert excision.label_source == "index map" and excision.index_map == "excision"
        record = excision.to_dict()["excision"]
        dimensions = list(default.homology_dimensions)
        assert record["homology_dimensions"][: len(dimensions)] == dimensions
        assert [len(matrix) for matrix in record["index_matrices"]] == dimensions
    if case == "saddle":
        # A saddle with a one-dimensional unstable direction and nonempty exit set.
        assert any(
            default.pair_atoms["A"] and default.shift_class == ("0", "x-1", "0", "0")
            for default, _ in compared
        )


def test_excision_index_map_of_a_repelling_orbit():
    # X minus S is two annuli, so the carrier of an exit atom in the default
    # construction (its whole component of A) is not acyclic.  The true
    # images give the index of a hyperbolic periodic orbit with a
    # two-dimensional unstable manifold and orientation kept.  The two
    # attracting orbits get the labels of the default construction.
    problem = _repelling_orbit_problem()
    grid = _grid(problem, 4)
    relation = compute_suspension_grid_relation(grid, problem)
    morse = compute_suspension_morse_graph(relation)
    assert len(morse.morse_sets) == 3
    results = _both_index_maps(relation, morse.morse_sets)
    attractors = [pair for pair in results if pair[0].homology_dimensions == (1, 1, 0, 0)]
    assert len(attractors) == 2
    for default, excision in attractors:
        assert default.computed and excision.computed, excision.blocker
        assert excision.shift_class == default.shift_class == ("x-1", "x-1", "0", "0")
    repeller = [pair for pair in results if pair[0].homology_dimensions == (0, 1, 1, 0)]
    assert len(repeller) == 1
    default, excision = repeller[0]
    assert default.pair_atoms["A"] > 0
    assert not default.computed and "is not acyclic" in default.index_map_blocker
    assert excision.computed, excision.blocker
    assert excision.shift_class == ("0", "x-1", "x-1", "0")
    record = excision.to_dict()["excision"]
    assert record["homology_dimensions"] == [0, 1, 1, 0]
    assert record["index_matrices"] == [[], [[1]], [[1]], []]
    assert record["pair_atoms"]["Abar"] == record["pair_atoms"]["Xbar"] - default.pair_atoms["S"]


def test_auto_index_map_falls_back_to_the_excision_pair():
    # The exit-components construction is refused on the repelling orbit;
    # "auto" then takes the label of the excision pair and keeps the first
    # blocker.  On the attracting orbits the first construction succeeds, and
    # "auto" gives its label without forming the excision pair.
    problem = _repelling_orbit_problem()
    grid = _grid(problem, 4)
    relation = compute_suspension_grid_relation(grid, problem)
    morse = compute_suspension_morse_graph(relation)
    fallbacks = 0
    for node, morse_set in enumerate(morse.morse_sets):
        default = compute_suspension_grid_conley_index(relation, morse_set, morse_node=node)
        auto = compute_suspension_grid_conley_index(
            relation, morse_set, morse_node=node, index_map="auto"
        )
        record = auto.to_dict()
        assert "index_map" not in default.to_dict()
        assert record["index_map"] == "auto"
        assert auto.homology_dimensions == default.homology_dimensions
        if default.computed:
            assert auto.shift_class == default.shift_class
            assert auto.label_source == "index map (exit components)"
            assert record["exit_components_blocker"] == "" and record["excision"] == {}
            continue
        fallbacks += 1
        assert auto.computed, auto.blocker
        assert auto.shift_class == ("0", "x-1", "x-1", "0")
        assert auto.label_source == "index map (excision pair)"
        assert auto.blocker == auto.index_map_blocker == ""
        assert record["exit_components_blocker"] == default.index_map_blocker
        assert "is not acyclic" in record["exit_components_blocker"]
        assert record["excision"]["index_matrices"] == [[], [[1]], [[1]], []]
        # With both constructions blocked, the blocker is that of the second.
        assert record["excision"]["pair_pieces"]["Xbar"] > auto.pair_pieces["X"]
        limited = compute_suspension_grid_conley_index(
            relation, morse_set, index_map="auto", max_pieces=auto.pair_pieces["X"]
        )
        assert not limited.computed and limited.homology_computed
        assert limited.exit_components_blocker == record["exit_components_blocker"]
        assert limited.blocker.startswith("IndexSizeLimitError: Xbar = X cup F(X)")
        assert limited.index_map_blocker == limited.blocker
    assert fallbacks == 1


def test_excision_reports_an_exit_atom_with_an_empty_image():
    # On this coarse grid an atom of A leaves the window: its image is empty
    # and so is the carrier of its piece.  The homology is still reported.
    problem = PAPER_PROBLEMS["rimless-wheel"](tau=1.0, level_offset=1)
    grid = _grid(problem, 3)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=2)
    morse = compute_suspension_morse_graph(relation)
    result = compute_suspension_grid_conley_index(
        relation, morse.morse_sets[1], morse_node=1, index_map="excision"
    )
    assert not result.computed and result.homology_computed
    assert result.homology_dimensions == (0, 2, 0, 0)
    assert "has an empty image" in result.blocker
    assert result.index_map_blocker == result.blocker
    with pytest.raises(ValueError, match="index_map"):
        compute_suspension_grid_conley_index(relation, morse.morse_sets[1], index_map="other")
    with pytest.raises(ValueError, match="index_map applies to the image pair"):
        compute_suspension_grid_conley_index(
            relation, morse.morse_sets[1], index_pair="forward-closure", index_map="excision"
        )


# ---------------------------------------------------------------------------
# The forward-closure pair of prop:grid-conley-index
# ---------------------------------------------------------------------------


def _labels_of_every_pair(relation, morse_set, node):
    """Records of the image pair, the forward-closure pair, and its excised form."""

    return tuple(
        compute_suspension_grid_conley_index(relation, morse_set, morse_node=node, **options)
        for options in (
            {},
            {"index_pair": "forward-closure"},
            {"index_pair": "forward-closure", "excise": True},
        )
    )


def test_forward_closure_pair_is_forward_invariant():
    from hybrid_dynamics.src.suspension_grid_conley import (
        compute_suspension_grid_conley_indices,
        index_pair_atoms,
    )

    problem = PAPER_PROBLEMS["impact-vdp-duffing"](tau=3.0, level_offset=2)
    grid = _grid(problem, 4)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=2)
    morse = compute_suspension_morse_graph(relation)
    for morse_set in morse.morse_sets:
        s, u, v = index_pair_atoms(relation, morse_set, "forward-closure")
        assert np.array_equal(u, relation.forward_closure(morse_set))
        assert np.array_equal(v, np.setdiff1d(u, s))
        assert np.all(np.isin(relation.image_of(u), u))
        assert np.all(np.isin(relation.image_of(v), v))
        _, x, a = index_pair_atoms(relation, morse_set)
        assert np.all(np.isin(x, u)) and np.all(np.isin(a, v))
    # {1} is not a strongly connected component of 0 <-> 1: V = {0} maps into S.
    cycle = _relation_on(grid, [0, 1], [0, 1])
    with pytest.raises(ValueError, match="not a strongly connected component"):
        index_pair_atoms(cycle, [1], "forward-closure")
    with pytest.raises(ValueError, match="index_pair"):
        index_pair_atoms(relation, morse.morse_sets[0], "closure")
    # The index of such a set is not attempted; the reason is its blocker,
    # in the serial and in the parallel computation.
    for excise in (False, True):
        result = compute_suspension_grid_conley_index(
            cycle, [1], index_pair="forward-closure", excise=excise
        )
        assert not result.computed and not result.homology_computed
        assert result.blocker.startswith("ValueError: F(V) meets S in 1 atoms")
    factory = paper_problem_factory("impact-vdp-duffing", tau=3.0, level_offset=2)
    results = compute_suspension_grid_conley_indices(
        cycle,
        [np.array([1]), np.array([0, 1])],
        workers=2,
        problem_factory=factory,
        index_pair="forward-closure",
    )
    assert "not a strongly connected component" in results[0].blocker
    assert "not a strongly connected component" not in results[1].blocker


def test_excised_neighborhood_is_checked_against_the_rectangles_and_seams():
    from hybrid_dynamics.src.suspension_grid_conley import _pieces_meeting, index_pair_atoms

    # The pieces that meet S by the rectangles and seams of the quotient are
    # those of the grid adjacency, so W contains every piece of U meeting S.
    for problem, level, depth in (
        (PAPER_PROBLEMS["impact-vdp-duffing"](tau=3.0, level_offset=2), 4, 2),
        (_repelling_cylinder_problem(), 2, 0),
    ):
        grid = _grid(problem, level)
        gluing = suspension_grid_gluing(grid)
        relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=depth)
        morse = compute_suspension_morse_graph(relation)
        every_piece = np.arange(grid.n_pieces)
        for morse_set in morse.morse_sets:
            s, u, _ = index_pair_atoms(relation, morse_set, "forward-closure")
            s_pieces = np.concatenate([grid.atom(atom) for atom in s])
            meeting = _pieces_meeting(grid, gluing, every_piece, s_pieces)
            assert np.array_equal(meeting, np.unique(grid.piece_adjacency[s_pieces].indices))
            w = np.union1d(s, np.intersect1d(grid.atom_adjacency[s].indices, u))
            meeting_u = meeting[np.isin(grid.atom_of_piece[meeting], u)]
            assert np.all(np.isin(grid.atom_of_piece[meeting_u], w))

    # With an adjacency that misses the neighbors of S, W = S, and the check
    # reports the pieces of U outside W that meet S instead of a label.
    problem = _repelling_cylinder_problem()
    grid = _grid(problem, 2)
    relation = compute_suspension_grid_relation(grid, problem)
    morse = compute_suspension_morse_graph(relation)
    whole = compute_suspension_grid_conley_index(
        relation, morse.morse_sets[2], index_pair="forward-closure"
    )
    assert whole.shift_class == ("0", "x-1", "x-1", "0")
    grid.atom_adjacency = sparse.identity(grid.n_atoms, format="csr")
    excised = compute_suspension_grid_conley_index(
        relation, morse.morse_sets[2], index_pair="forward-closure", excise=True
    )
    assert excised.pair_atoms["W"] == excised.pair_atoms["S"]
    assert not excised.computed and not excised.homology_computed
    assert excised.blocker.startswith("AssertionError: ")
    assert "pieces of U outside W meet S" in excised.blocker


@pytest.mark.parametrize(
    ("factory", "level", "depth", "labeled_with_larger_u"),
    [
        (_translation_problem, 2, 0, 0),
        (lambda: PAPER_PROBLEMS["bouncing-ball"](tau=1.5, level_offset=2), 4, 0, 1),
        (lambda: PAPER_PROBLEMS["impact-vdp-duffing"](tau=3.0, level_offset=2), 4, 2, 4),
    ],
    ids=["translation-cylinder", "ball-level4-offset2", "oscillator-level4-offset2"],
)
def test_forward_closure_labels_equal_the_image_labels(factory, level, depth, labeled_with_larger_u):
    problem = factory()
    grid = _grid(problem, level)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=depth)
    morse = compute_suspension_morse_graph(relation)
    labeled = larger = 0
    for node, morse_set in enumerate(morse.morse_sets):
        image, closure, excised = _labels_of_every_pair(relation, morse_set, node)
        # The excised computation is that of the whole forward closure.
        assert excised.homology_dimensions == closure.homology_dimensions
        assert excised.computed == closure.computed
        assert excised.shift_class == closure.shift_class
        assert excised.pair_pieces["U"] == closure.pair_pieces["U"]
        assert excised.pair_pieces["W"] <= excised.pair_pieces["Y"] <= closure.pair_pieces["U"]
        if image.computed:
            assert closure.computed, closure.blocker
            assert closure.shift_class == image.shift_class
            labeled += 1
            larger += closure.pair_pieces["U"] > image.pair_pieces["X"]
    assert labeled >= 1
    assert larger == labeled_with_larger_u


def test_forward_closure_labels_the_repelling_periodic_orbit():
    problem = _repelling_cylinder_problem()
    grid = _grid(problem, 2)
    relation = compute_suspension_grid_relation(grid, problem)
    assert relation.statistics["discarded_exit_endpoints"] == 0
    morse = compute_suspension_morse_graph(relation)
    assert [values.size for values in morse.morse_sets] == [44, 44, 44]
    assert set(morse.edges) == {(2, 0), (2, 1)}
    for node in (0, 1):  # the attracting orbits y = 0 and y = 1
        image, closure, excised = _labels_of_every_pair(relation, morse.morse_sets[node], node)
        assert image.pair_pieces["A"] == closure.pair_pieces["V"] == 0
        for result in (image, closure, excised):
            assert result.shift_class == ("x-1", "x-1", "0", "0"), result.blocker
    image, closure, excised = _labels_of_every_pair(relation, morse.morse_sets[2], 2)
    # F(S) minus S is two annuli, so the carrier of an exit piece of the
    # image pair is not acyclic; the forward-closure pair has no exits.
    assert not image.computed
    assert "is not acyclic" in image.index_map_blocker
    assert image.homology_dimensions == (0, 1, 1, 0)
    for result in (closure, excised):
        assert result.computed and result.label_source == "index map", result.blocker
        assert result.homology_dimensions == (0, 1, 1, 0)
        assert result.shift_class == ("0", "x-1", "x-1", "0")
    assert excised.pair_pieces["W"] < closure.pair_pieces["U"]
    assert "index_pair" not in image.to_dict()
    assert closure.to_dict()["index_pair"] == "forward-closure"
    limited = compute_suspension_grid_conley_index(
        relation, morse.morse_sets[2], index_pair="forward-closure", max_pieces=100
    )
    assert limited.blocker.startswith("IndexSizeLimitError: U = the forward closure of S")
    with pytest.raises(ValueError, match="excise"):
        compute_suspension_grid_conley_index(relation, morse.morse_sets[2], excise=True)


def test_forward_closure_reports_atoms_whose_image_left_the_window():
    problem = PAPER_PROBLEMS["rimless-wheel"](tau=1.0, level_offset=1)
    grid = _grid(problem, 3)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=2)
    morse = compute_suspension_morse_graph(relation)
    node = int(np.argmax([values.size for values in morse.morse_sets]))
    for excise, source in ((False, "U"), (True, "W")):
        result = compute_suspension_grid_conley_index(
            relation, morse.morse_sets[node], index_pair="forward-closure", excise=excise
        )
        assert result.homology_computed and any(result.homology_dimensions)
        assert not result.computed
        # The count is of the atoms whose images are read: U, or W when excised.
        assert f"of the {result.pair_atoms[source]} atoms of {source} have an empty image" in (
            result.blocker
        )
