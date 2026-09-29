"""The paper pipeline on fine base grids computes what it computed before.

The atoms of ``Xi_n``, the image components, the endpoint evaluation of the
exit policies, and the relation carriers of the index were rewritten for base
grids with ``2^{n + level_offset}`` cells per axis.  These tests compare the
results with the earlier computations:

* the relations, Morse graphs, and index records of small runs against
  digests recorded with the code of commit ``24ecdda``, which evaluated every
  exit policy anew, assembled every carrier before validating it, counted
  image components atom by atom, and formed atoms from generator signatures;
* each rewritten step against its direct counterpart, which is kept
  (:func:`reference_signatures`, :func:`atom_set_components`) or restated
  here.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from hybrid_dynamics import (
    DyadicBaseWindow,
    GuardResetSpec,
    HybridSystem,
    SuspensionGridProblem,
    atom_set_components,
    build_suspension_grid,
    compute_suspension_grid_conley_index,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
)
from hybrid_dynamics.examples.paper_examples import PAPER_PROBLEMS
from hybrid_dynamics.src import atlas_conley
from hybrid_dynamics.src.carrier_kernel import carrier_kernel_arrays, native_carrier_kernel_available
from hybrid_dynamics.src.suspension_grid import reference_signatures
from hybrid_dynamics.src.suspension_grid_relation import (
    EndpointCache,
    _unique_rows,
    relation_image_connectivity,
    row_set_components,
)


# ---------------------------------------------------------------------------
# Digests of the code of commit 24ecdda
# ---------------------------------------------------------------------------

#: ``(example, level, level_offset, tau, gap_refinement_depth)``.
DIGEST_CASES = {
    "ball-l4-tau0.5": ("bouncing-ball", 4, 0, 0.5, 0),
    "ball-l3-base16-tau0.5-gap2": ("bouncing-ball", 3, 1, 0.5, 2),
    "wheel-l3-base16-tau1-gap2": ("rimless-wheel", 3, 1, 1.0, 2),
    "neuron-l6-tau1": ("spiking-neuron", 6, 0, 1.0, 0),
    "impact-l3-base16-tau0.5-gap2": ("impact-vdp-duffing", 3, 1, 0.5, 2),
}

#: Computed with commit 24ecdda (the window of a positive offset set with
#: ``dataclasses.replace(problem.window, level_offset=...)``), each exit
#: policy in its own call without a cache.  The ``endpoint`` digests were
#: recomputed when the index records gained the relative homology of every
#: pair; the relations, the Morse graphs, and every label computed before are
#: unchanged, and the Morse sets whose index map fails keep no label.
RECORDED_DIGESTS = {
    "ball-l3-base16-tau0.5-gap2/endpoint": "80e84c7d94cd47e6d020b3703149f0bb4168c6896d9b962b0957b5d776d7210e",
    "ball-l3-base16-tau0.5-gap2/path": "abef6506c80a7bf34bfe96887c5c68b0991d656d88fa64c7ec1683ffb8412c09",
    "ball-l4-tau0.5/endpoint": "00020a29a6c20f030999c531e87ab0ed3889d40f0d2257f64c253d4fa4570959",
    "ball-l4-tau0.5/path": "062805eba3b1b48f4211b66cd5541a1bb68bd6604ea76cde210b7b78ea2fff85",
    "impact-l3-base16-tau0.5-gap2/endpoint": "1bae3c7838392bd9a2ee6c02ffdc00b8715b0251598144db83dc1cadeb26043e",
    "impact-l3-base16-tau0.5-gap2/path": "b840c6dfe067dbe7391b8088fa80ebf6c82a472b8cecae2567661b84f11b8253",
    "neuron-l6-tau1/endpoint": "cc56c7d242adcf5aec7423ee9f66d2adc06b5e5c5a1b1fd75b0c4dffd723bc01",
    "neuron-l6-tau1/path": "7cbbb328942863a1d9663c6e589c8cbf95f711efde862375d4d4a22307fbd4a9",
    "wheel-l3-base16-tau1-gap2/endpoint": "5b5618234c110d80c055ce6aa50699392431e48f18d27d92a8a72f5cf65b1bb4",
    "wheel-l3-base16-tau1-gap2/path": "5a007f83217fdf5de7ed91e43d2b4078191936b2c5151efb336b130e65ec9fee",
}


def _run_digest(grid, relation, morse, conley) -> str:
    """SHA-256 of the atoms, both relations, statistics, Morse graph, index records."""

    digest = hashlib.sha256()
    arrays = [grid.atom_of_piece]
    for matrix in (relation.matrix, relation.sampled):
        canonical = sparse.csr_matrix(matrix).sorted_indices()
        arrays += [canonical.indptr, canonical.indices]
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    statistics = {
        key: value for key, value in relation.statistics.items() if not key.startswith("seconds")
    }
    digest.update(json.dumps(statistics, sort_keys=True).encode())
    digest.update(json.dumps([[int(atom) for atom in values] for values in morse.morse_sets]).encode())
    digest.update(json.dumps([list(edge) for edge in morse.edges]).encode())
    digest.update(json.dumps(conley, sort_keys=True).encode())
    return digest.hexdigest()


def _require_backend(backend: str) -> None:
    if backend == "native" and not native_carrier_kernel_available():
        pytest.skip("the installed CMGDB lacks ComputeCarrierChainMap")


@pytest.mark.parametrize("case", sorted(DIGEST_CASES))
@pytest.mark.parametrize("backend", ["python", "native"])
def test_runs_match_the_digests_of_the_earlier_code(backend, case):
    # Both backends of the index map give the recorded index records.
    _require_backend(backend)
    name, level, offset, tau, depth = DIGEST_CASES[case]
    problem = PAPER_PROBLEMS[name](tau=tau, level_offset=offset)
    grid = build_suspension_grid(problem.window, problem.guard, level)
    cache = EndpointCache()
    for policy in ("endpoint", "path"):
        relation = compute_suspension_grid_relation(
            grid, problem, exit_policy=policy, gap_refinement_depth=depth, endpoint_cache=cache
        )
        morse = compute_suspension_morse_graph(relation)
        conley = []
        if policy == "endpoint":
            evaluated = cache.evaluated
            for index, morse_set in enumerate(morse.morse_sets):
                record = compute_suspension_grid_conley_index(
                    relation, morse_set, morse_node=index, backend=backend
                ).to_dict()
                record.pop("seconds")
                conley.append(record)
        else:
            # The path policy evaluated no sample that the endpoint policy had not.
            assert cache.evaluated == evaluated
        assert _run_digest(grid, relation, morse, conley) == RECORDED_DIGESTS[f"{case}/{policy}"]


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def _curved_guard() -> GuardResetSpec:
    return GuardResetSpec(
        u_bounds=(0.0, 1.0),
        guard_point=lambda u: np.stack((0.45 + 0.2 * u * u, u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, 0.25), 0.1 + 0.8 * u), axis=-1),
        name="curved",
    )


@pytest.mark.parametrize(
    ("name", "levels", "offsets"),
    [
        ("bouncing-ball", (0, 1, 3, 4), (0, 1, 2)),
        ("rimless-wheel", (0, 2, 4), (0, 2)),
        ("spiking-neuron", (6,), (0, 1)),
        ("impact-vdp-duffing", (0, 3), (0, 2)),
    ],
)
def test_atoms_are_the_classes_of_generator_signatures(name, levels, offsets):
    for offset in offsets:
        problem = PAPER_PROBLEMS[name](level_offset=offset)
        for level in levels:
            grid = build_suspension_grid(problem.window, problem.guard, level)
            signatures, generators, atoms = reference_signatures(grid)
            assert np.array_equal(grid.atom_of_piece, atoms), (name, level, offset)
            assert grid.n_generators == len(generators)
            assert grid.signatures == signatures
            assert grid.cells_per_axis == 2 ** (level + offset)
            assert grid.n_phase == 2 ** (level + 2)


def test_atoms_of_a_curved_guard_with_a_finer_base():
    for offset in (0, 1, 3):
        window = DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0)), level_offset=offset)
        for level in (0, 1, 3):
            grid = build_suspension_grid(window, _curved_guard(), level)
            _signatures, generators, atoms = reference_signatures(grid)
            assert np.array_equal(grid.atom_of_piece, atoms)
            assert grid.n_generators == len(generators)


# ---------------------------------------------------------------------------
# Image components, lattice keys
# ---------------------------------------------------------------------------


def test_image_components_match_the_count_per_atom():
    problem = PAPER_PROBLEMS["impact-vdp-duffing"](tau=0.5, level_offset=1)
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    relation = compute_suspension_grid_relation(grid, problem)
    for matrix in (relation.matrix, relation.sampled):
        expected = np.array(
            [
                atom_set_components(grid, matrix.indices[matrix.indptr[atom] : matrix.indptr[atom + 1]])
                for atom in range(grid.n_atoms)
            ]
        )
        # Small blocks exercise the split of the rows.
        for chunk in (7, 1000, 1 << 18):
            assert np.array_equal(
                row_set_components(matrix, grid.atom_adjacency, chunk_entries=chunk), expected
            )
    connectivity = relation_image_connectivity(relation)
    assert connectivity["disconnected_padded_images"] == int(
        np.count_nonzero(row_set_components(relation.matrix, grid.atom_adjacency) > 1)
    )
    # Rows given with unsorted and repeated entries.
    rows = sparse.csr_matrix(
        (np.ones(4), np.array([3, 0, 3, 1]), np.array([0, 4, 4])), shape=(2, grid.n_atoms)
    )
    assert row_set_components(rows, grid.atom_adjacency)[1] == 0
    assert row_set_components(rows, grid.atom_adjacency)[0] == atom_set_components(grid, [0, 1, 3])


def test_unique_rows_is_numpy_unique_along_rows():
    rng = np.random.default_rng(3)
    for values in (
        rng.integers(0, 50, size=(500, 2)),
        rng.integers(0, 3, size=(40, 2)),
        np.array([[5, 0]]),
        np.array([[-1, 2], [0, 1], [-1, 2]]),
    ):
        unique, inverse = _unique_rows(values)
        expected, expected_inverse = np.unique(values, axis=0, return_inverse=True)
        assert np.array_equal(unique, expected)
        assert np.array_equal(inverse, expected_inverse.reshape(-1))


# ---------------------------------------------------------------------------
# Endpoint cache
# ---------------------------------------------------------------------------


def _translation_with_failures(tau: float = 0.5) -> SuspensionGridProblem:
    """``x' = 1`` on the square with the flow undefined above ``y = 0.95``.

    Base samples and handle paths above ``y = 0.95`` fail, so the relation
    carries failure messages from both kinds of samples.
    """

    def ode(_t, state):
        if state[1] > 0.95:
            raise ValueError("vector field undefined above y = 0.95")
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
        window=DyadicBaseWindow(((0.0, 1.0), (0.0, 1.0)), level_offset=1),
        guard=guard,
        tau=tau,
        name="translation-with-failures",
    )


def _statistics(relation):
    return {key: value for key, value in relation.statistics.items() if not key.startswith("seconds")}


@pytest.mark.parametrize(("eval_mode", "depth"), [("corners", 2), ("tensor", 1), ("random", 0)])
def test_endpoint_cache_reproduces_relations_and_failure_messages(eval_mode, depth):
    problem = _translation_with_failures()
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    cache = EndpointCache()
    for policy in ("endpoint", "path", "endpoint"):
        direct = compute_suspension_grid_relation(
            grid, problem, eval_mode=eval_mode, exit_policy=policy, gap_refinement_depth=depth
        )
        before = cache.evaluated
        cached = compute_suspension_grid_relation(
            grid,
            problem,
            eval_mode=eval_mode,
            exit_policy=policy,
            gap_refinement_depth=depth,
            endpoint_cache=cache,
        )
        assert (cached.matrix != direct.matrix).nnz == 0
        assert (cached.sampled != direct.sampled).nnz == 0
        assert _statistics(cached) == _statistics(direct)
        if policy != "endpoint" or before:
            assert cache.evaluated == before
    statistics = _statistics(direct)
    assert statistics["failed_endpoints"] > 0
    messages = statistics["failure_messages"]
    assert any(message.startswith("base sample") for message in messages)
    if eval_mode != "random":
        assert any(message.startswith("handle sample") for message in messages)


def test_endpoint_cache_serves_one_grid_and_problem():
    problem = _translation_with_failures()
    grid = build_suspension_grid(problem.window, problem.guard, 1)
    cache = EndpointCache()
    compute_suspension_grid_relation(grid, problem, endpoint_cache=cache)
    other = build_suspension_grid(problem.window, problem.guard, 1)
    with pytest.raises(ValueError, match="one grid and one problem"):
        compute_suspension_grid_relation(other, problem, endpoint_cache=cache)


# ---------------------------------------------------------------------------
# Relation carriers of the index
# ---------------------------------------------------------------------------


def _assembled_carrier_generators(complex_, vertex_images, **_options):
    """Every carrier assembled before validation (the earlier construction)."""

    by_first_vertex = {}
    for cell in complex_.cells:
        by_first_vertex.setdefault(cell.vertices[0], []).append(cell)
    induced_cache = {}
    generators = {}
    for source in complex_.cells:
        key = frozenset(target for vertex in source.vertices for target in vertex_images[vertex])
        value = induced_cache.get(key)
        if value is None:
            value = frozenset(
                cell
                for vertex in key
                for cell in by_first_vertex.get(vertex, ())
                if key.issuperset(cell.vertices)
            )
            induced_cache[key] = value
        if not value:
            raise ValueError(f"relation carrier of {source!r} is empty")
        generators[source] = value
    return generators


@pytest.mark.parametrize(
    ("name", "level", "offset", "tau", "depth"),
    [
        ("bouncing-ball", 3, 1, 0.5, 2),
        ("rimless-wheel", 3, 1, 1.0, 2),
        ("impact-vdp-duffing", 3, 1, 0.5, 0),
    ],
)
def test_carriers_on_demand_match_the_assembled_carriers(monkeypatch, name, level, offset, tau, depth):
    problem = PAPER_PROBLEMS[name](tau=tau, level_offset=offset)
    grid = build_suspension_grid(problem.window, problem.guard, level)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=depth)
    morse = compute_suspension_morse_graph(relation)

    on_demand_class = atlas_conley._InducedCarrierGenerators
    recorded = []

    def recording(complex_, vertex_images, **options):
        recorded.append((complex_, vertex_images))
        return on_demand_class(complex_, vertex_images, **options)

    def records(generators_class):
        # The carrier generators are those of the Python construction.
        monkeypatch.setattr(atlas_conley, "_InducedCarrierGenerators", generators_class)
        result = []
        for index, morse_set in enumerate(morse.morse_sets):
            record = compute_suspension_grid_conley_index(
                relation, morse_set, morse_node=index, backend="python"
            ).to_dict()
            record.pop("seconds")
            result.append(record)
        return result

    on_demand = records(recording)
    assembled = records(_assembled_carrier_generators)
    assert on_demand == assembled
    assert recorded
    for complex_, vertex_images in recorded:
        generators = on_demand_class(complex_, vertex_images, cache_size=2)
        expected = _assembled_carrier_generators(complex_, vertex_images)
        assert list(generators) == list(complex_.cells) == list(expected)
        assert all(generators[cell] == expected[cell] for cell in complex_.cells)


def test_index_size_limit_reports_a_blocker():
    problem = PAPER_PROBLEMS["bouncing-ball"](tau=0.5, level_offset=1)
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    relation = compute_suspension_grid_relation(grid, problem)
    morse_set = compute_suspension_morse_graph(relation).morse_sets[0]
    unlimited = compute_suspension_grid_conley_index(relation, morse_set)
    pieces = unlimited.pair_pieces["X"]
    limited = compute_suspension_grid_conley_index(relation, morse_set, max_pieces=pieces - 1)
    assert not limited.computed
    assert limited.blocker.startswith("IndexSizeLimitError: X = S cup F(S) has")
    assert limited.pair_pieces == unlimited.pair_pieces
    at_limit = compute_suspension_grid_conley_index(relation, morse_set, max_pieces=pieces)
    assert at_limit.blocker == unlimited.blocker
    assert at_limit.shift_class == unlimited.shift_class


def _scanned_chain_map_images(carrier):
    """The chain selector with carrier cells found by scanning the complex."""

    from hybrid_dynamics.src.suspension_complex import _solve_linear_system_mod_prime

    complex_ = carrier.complex
    images = {}
    for dimension in range(complex_.max_dimension + 1):
        for source in complex_.cells_of_dimension(dimension):
            value = carrier.image(source)
            if dimension == 0:
                vertices = [cell for cell in complex_.cells_of_dimension(0) if cell in value]
                images[source] = {vertices[0]: 1}
                continue
            right_hand_side = {}
            for face, incidence in complex_.boundary(source).items():
                for target, coefficient in images[face].items():
                    entry = (right_hand_side.get(target, 0) + incidence * coefficient) % carrier.modulus
                    if entry:
                        right_hand_side[target] = entry
                    else:
                        right_hand_side.pop(target, None)
            rows = [cell for cell in complex_.cells_of_dimension(dimension - 1) if cell in value]
            columns = [cell for cell in complex_.cells_of_dimension(dimension) if cell in value]
            images[source] = _solve_linear_system_mod_prime(
                rows,
                columns,
                {cell: complex_.boundary(cell) for cell in columns},
                right_hand_side,
                carrier.modulus,
            )
    return images


@pytest.mark.parametrize("backend", ["python", "native"])
def test_chain_selector_matches_the_scan_of_the_complex(monkeypatch, backend):
    _require_backend(backend)
    problem = PAPER_PROBLEMS["rimless-wheel"](tau=1.0, level_offset=1)
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=2)
    morse = compute_suspension_morse_graph(relation)

    from hybrid_dynamics.src import suspension_grid_conley

    if backend == "native":
        _check_native_chain_selector(monkeypatch, relation, morse, suspension_grid_conley)
        return
    prepare = atlas_conley.prepare_atlas_relation_conley_2d
    preparations = []

    def recording(*arguments, **options):
        preparations.append(prepare(*arguments, **options))
        return preparations[-1]

    monkeypatch.setattr(suspension_grid_conley, "prepare_atlas_relation_conley_2d", recording)
    for index, morse_set in enumerate(morse.morse_sets):
        assert compute_suspension_grid_conley_index(
            relation, morse_set, morse_node=index, backend="python"
        ).computed
    assert preparations
    for preparation in preparations:
        expected = _scanned_chain_map_images(preparation.carrier)
        for cell in preparation.carrier.complex.cells:
            assert dict(preparation.chain_map.image(cell)) == expected[cell]


def _check_native_chain_selector(monkeypatch, relation, morse, suspension_grid_conley):
    """The chain map of the native kernel against the scan of the complex.

    The carrier of the scan is assembled here from the vertex images (the
    induced subcomplexes), not taken from the Python construction.
    """

    native = suspension_grid_conley.native_relation_shift_class
    calls = []

    def recording(pair, **options):
        calls.append((pair, options))
        return native(pair, **options)

    monkeypatch.setattr(suspension_grid_conley, "native_relation_shift_class", recording)
    for index, morse_set in enumerate(morse.morse_sets):
        assert compute_suspension_grid_conley_index(
            relation, morse_set, morse_node=index, backend="native"
        ).computed
    assert calls
    for pair, options in calls:
        complex_ = pair.complex
        vertex_images = atlas_conley._relation_vertex_images(
            pair,
            options["top_relation"],
            use_exit_component_carrier=options["use_exit_component_carrier"],
        )
        values = _assembled_carrier_generators(complex_, vertex_images)
        carrier = SimpleNamespace(complex=complex_, modulus=5, image=values.__getitem__)
        expected = _scanned_chain_map_images(carrier)
        result = carrier_kernel_arrays(complex_, vertex_images, pair.p0_atlas_cells).compute()
        assert result["status"] == "ok"
        for degree in range(complex_.max_dimension + 1):
            cells = complex_.cells_of_dimension(degree)
            found = {cell: {} for cell in cells}
            entries = np.asarray(result["chain_map"][degree], dtype=np.int64).reshape(-1, 3)
            for source, target, value in entries.tolist():
                found[cells[source]][cells[target]] = value
            for cell in cells:
                assert found[cell] == expected[cell]
