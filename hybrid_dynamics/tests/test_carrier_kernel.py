"""The native carrier kernel of CMGDB against the Python construction.

``CMGDB.ComputeCarrierChainMap`` forms the carriers and the chain map of
:func:`atlas_conley.prepare_atlas_relation_conley_2d` natively, and those of
the excision construction and of the excised forward-closure pair, whose
carriers lie in another complex.  These tests check the arrays handed to it
(:mod:`carrier_kernel`), the exceptions rebuilt from its failures, and the
choice of backend, and they run the Conley problems of
``test_suspension_grid.py`` with the Python and the native backend in every
construction: the records and every payload passed to the shift-class
function of CMGDB must be equal.  The runs with the native kernel are
skipped when the installed CMGDB lacks it.  The same
comparisons also run with :func:`reference_carrier_chain_map`, a Python
statement of the kernel, in its place; they check the arrays, the
dispatch, and the rebuilt exceptions, but not the kernel itself.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from test_suspension_grid import (
    _repelling_cylinder_problem,
    _repelling_orbit_problem,
    _saddle_problem,
    _translation_problem,
)

from hybrid_dynamics import (
    AffineBoundaryEmbedding2D,
    AtlasNerveSimplex,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    FiniteCellComplex,
    build_suspension_grid,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
    prepare_atlas_relation_conley_2d,
)
from hybrid_dynamics.examples.paper_examples import (
    PAPER_PROBLEMS,
    bouncing_ball_problem,
    paper_problem_factory,
    spiking_neuron_problem,
)
from hybrid_dynamics.src import carrier_kernel, suspension_grid_conley
from hybrid_dynamics.src.carrier_kernel import (
    KERNEL_FUNCTION,
    _image_csr,
    carrier_kernel_arrays,
    cmgdb_provenance,
    cross_complex_carrier_arrays,
    kernel_failure,
    native_carrier_kernel_available,
    native_cross_complex_chain_map,
    native_relation_payload,
    native_subcomplex_chain_entries,
    relation_carrier_arrays,
    resolve_conley_backend,
    simplex_arrays,
)
from hybrid_dynamics.src.suspension_complex import (
    SHIFT_CLASS_FUNCTION,
    _eliminate_columns_mod_prime,
    _rank_mod_prime,
    _solve_with_pivots,
)
from hybrid_dynamics.src.suspension_grid_conley import (
    compute_suspension_grid_conley_index,
    compute_suspension_grid_conley_indices,
)
from hybrid_dynamics.src.suspension_grid_relation import SuspensionGridRelation

CMGDB = pytest.importorskip("CMGDB")

CODE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# A Python statement of CMGDB.ComputeCarrierChainMap
# ---------------------------------------------------------------------------


def _betti_numbers(simplices, row_of, cells):
    """Betti numbers over GF(5) of the subcomplex with the given rows per degree."""

    ranks = [0] * (len(cells) + 1)
    for degree in range(1, len(cells)):
        position = {row: index for index, row in enumerate(cells[degree - 1])}
        columns = []
        for row in cells[degree]:
            simplex = simplices[degree][row]
            columns.append(
                [
                    (
                        position[row_of[degree - 1][simplex[:index] + simplex[index + 1 :]]],
                        1 if index % 2 == 0 else -1,
                    )
                    for index in range(degree + 1)
                ]
            )
        ranks[degree] = _rank_mod_prime(columns, 5)
    return [len(cells[d]) - ranks[d] - ranks[d + 1] for d in range(len(cells))]


def reference_carrier_chain_map(
    source_simplices,
    vertex_image_indptr,
    vertex_image_indices,
    source_exit,
    *,
    target_simplices=None,
    target_exit=None,
    modulus=5,
    return_carriers=False,
):
    """``CMGDB.ComputeCarrierChainMap`` in Python.

    The carriers are the subcomplexes of the target (the source when
    ``target_simplices`` is ``None``) induced on the unions of the vertex
    images; the checks run in the order of the kernel (every carrier
    nonempty, every carrier acyclic, the carriers of ``P0`` in ``P0``, the
    chain map, then its validation), each reporting the first failing cell
    in the order of the complex.  The chain map is solved with the
    elimination of :meth:`suspension_complex.FixedTimeCarrier.construct_chain_map`
    on the rows of the cells, so its entries come in the order of the Python
    construction.  The relative payload is returned when the target is the
    source.
    """

    if modulus != 5:
        raise ValueError("modulus must be 5")
    if (target_simplices is None) != (target_exit is None):
        raise ValueError("target_simplices and target_exit go together")

    def read(arrays):
        simplices = [
            [tuple(int(v) for v in row) for row in np.asarray(array).reshape(-1, degree + 1)]
            for degree, array in enumerate(arrays)
        ]
        return simplices, [{simplex: row for row, simplex in enumerate(group)} for group in simplices]

    simplices, row_of = read(source_simplices)
    same = target_simplices is None
    targets, target_row_of = (simplices, row_of) if same else read(target_simplices)
    labels = [simplex[0] for simplex in simplices[0]]
    target_labels = [simplex[0] for simplex in targets[0]]
    vertex_row = {label: row for row, label in enumerate(labels)}
    indptr = np.asarray(vertex_image_indptr, dtype=np.int64)
    indices = np.asarray(vertex_image_indices, dtype=np.int64)
    if indptr.shape != (len(labels) + 1,) or indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("vertex_image_indptr does not match the vertices")
    images = [
        frozenset(int(v) for v in indices[indptr[row] : indptr[row + 1]])
        for row in range(len(labels))
    ]
    unknown = set().union(*images).difference(target_labels)
    if unknown:
        raise IndexError(f"vertex images outside the target: {sorted(unknown)!r}")
    exit_labels = frozenset(label for label, flag in zip(labels, np.asarray(source_exit)) if flag)
    target_exit_labels = (
        exit_labels
        if same
        else frozenset(
            label for label, flag in zip(target_labels, np.asarray(target_exit)) if flag
        )
    )

    def failure(status, degree, row):
        return {"status": status, "failure_degree": degree, "failure_row": row}

    keys = [
        [frozenset().union(*(images[vertex_row[v]] for v in simplex)) for simplex in group]
        for group in simplices
    ]
    for degree, group in enumerate(keys):
        for row, key in enumerate(group):
            if not key:
                return failure("empty_carrier", degree, row)
    by_first: dict[int, list[tuple[int, int]]] = {}
    for degree, group in enumerate(targets):
        for row, simplex in enumerate(group):
            by_first.setdefault(simplex[0], []).append((degree, row))
    carrier_of: dict[frozenset, int] = {}
    carrier_cells: list[list[list[int]]] = []
    carrier_ids = [[0] * len(group) for group in simplices]
    for degree, group in enumerate(keys):
        for row, key in enumerate(group):
            identifier = carrier_of.get(key)
            if identifier is None:
                cells = [[] for _ in targets]
                for vertex in key:
                    for cell_degree, cell_row in by_first.get(vertex, ()):
                        if key.issuperset(targets[cell_degree][cell_row]):
                            cells[cell_degree].append(cell_row)
                cells = [sorted(rows) for rows in cells]
                betti = _betti_numbers(targets, target_row_of, cells)
                if betti[0] != 1 or any(betti[1:]):
                    return failure("not_acyclic", degree, row)
                identifier = carrier_of[key] = len(carrier_cells)
                carrier_cells.append(cells)
            carrier_ids[degree][row] = identifier
    in_p0 = [[exit_labels.issuperset(simplex) for simplex in group] for group in simplices]
    for degree, group in enumerate(keys):
        for row, key in enumerate(group):
            if in_p0[degree][row] and not target_exit_labels.issuperset(key):
                return failure("pair_violation", degree, row)

    def faces(simplices_, row_of_, degree, simplex):
        return [
            (
                row_of_[degree - 1][simplex[:index] + simplex[index + 1 :]],
                1 if index % 2 == 0 else -1,
            )
            for index in range(degree + 1)
        ]

    def add(accumulator, chain, scale):
        for target, coefficient in chain.items():
            value = (accumulator.get(target, 0) + scale * coefficient) % 5
            if value:
                accumulator[target] = value
            else:
                accumulator.pop(target, None)

    images_of = [[None] * len(group) for group in simplices]
    systems = {}
    for degree, group in enumerate(simplices):
        for row, simplex in enumerate(group):
            identifier = carrier_ids[degree][row]
            cells = carrier_cells[identifier]
            if degree == 0:
                images_of[0][row] = {cells[0][0]: 1}
                continue
            right_hand_side = {}
            for face, incidence in faces(simplices, row_of, degree, simplex):
                add(right_hand_side, images_of[degree - 1][face], incidence)
            system = systems.get((identifier, degree))
            if system is None:
                rows = cells[degree - 1] if degree - 1 < len(cells) else []
                row_index = {cell: index for index, cell in enumerate(rows)}
                columns = cells[degree] if degree < len(cells) else []
                entries = {
                    column: dict(faces(targets, target_row_of, degree, targets[degree][column]))
                    for column in columns
                }
                pivots = _eliminate_columns_mod_prime(row_index, columns, entries, 5)
                system = (row_index, pivots, columns)
                systems[(identifier, degree)] = system
            row_index, pivots, columns = system
            if not set(right_hand_side) <= set(row_index):
                return failure("chain_map_invalid", degree, row)
            try:
                images_of[degree][row] = _solve_with_pivots(
                    row_index, pivots, columns, right_hand_side, 5
                )
            except ValueError:
                return failure("no_solution", degree, row)

    for degree in range(1, len(simplices)):
        for row, simplex in enumerate(simplices[degree]):
            boundary_after_map = {}
            for target, coefficient in images_of[degree][row].items():
                add(
                    boundary_after_map,
                    dict(faces(targets, target_row_of, degree, targets[degree][target])),
                    coefficient,
                )
            map_after_boundary = {}
            for face, incidence in faces(simplices, row_of, degree, simplex):
                add(map_after_boundary, images_of[degree - 1][face], incidence)
            if boundary_after_map != map_after_boundary:
                return failure("chain_map_invalid", degree, row)
    for degree, group in enumerate(simplices):
        for row in range(len(group)):
            carrier = set(carrier_cells[carrier_ids[degree][row]][degree])
            if not set(images_of[degree][row]) <= carrier:
                return failure("chain_map_invalid", degree, row)
    target_in_p0 = [
        [target_exit_labels.issuperset(simplex) for simplex in group] for group in targets
    ]
    for degree, group in enumerate(simplices):
        for row in range(len(group)):
            if in_p0[degree][row] and not all(
                target_in_p0[degree][t] for t in images_of[degree][row]
            ):
                return failure("chain_map_invalid", degree, row)

    result = {
        "status": "ok",
        "failure_degree": -1,
        "failure_row": -1,
        "chain_map": [
            np.array(
                [
                    (row, target, coefficient)
                    for row in range(len(group))
                    for target, coefficient in images_of[degree][row].items()
                ],
                dtype=np.int64,
            ).reshape(-1, 3)
            for degree, group in enumerate(simplices)
        ],
        "carrier_count": len(carrier_cells),
    }
    if same:
        basis = [
            [row for row in range(len(group)) if not in_p0[degree][row]]
            for degree, group in enumerate(simplices)
        ]
        position = [{row: index for index, row in enumerate(rows)} for rows in basis]
        boundary_entries = [[]]
        for degree in range(1, len(simplices)):
            entries = []
            for column, row in enumerate(basis[degree]):
                for face, incidence in faces(simplices, row_of, degree, simplices[degree][row]):
                    if face in position[degree - 1]:
                        entries.append((position[degree - 1][face], column, incidence % 5))
            boundary_entries.append(entries)
        chain_map_entries = []
        for degree in range(len(simplices)):
            entries = []
            for column, row in enumerate(basis[degree]):
                for target, coefficient in images_of[degree][row].items():
                    if target in position[degree]:
                        entries.append((position[degree][target], column, coefficient % 5))
            chain_map_entries.append(entries)
        result["payload"] = {
            "cell_counts": [len(rows) for rows in basis],
            "boundary_entries": boundary_entries,
            "chain_map_entries": chain_map_entries,
        }
    if return_carriers:
        result["carrier_ids"] = np.array(
            [identifier for group in carrier_ids for identifier in group], dtype=np.int64
        )
    return result


def _real_native_kernel() -> bool:
    return (
        native_carrier_kernel_available()
        and getattr(CMGDB, KERNEL_FUNCTION) is not reference_carrier_chain_map
    )


@pytest.fixture(params=["native", "reference"])
def kernel(request, monkeypatch):
    """``CMGDB.ComputeCarrierChainMap``: the installed native kernel or the Python statement."""

    if request.param == "native":
        if not _real_native_kernel():
            pytest.skip("the installed CMGDB lacks ComputeCarrierChainMap")
    else:
        monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, reference_carrier_chain_map, raising=False)
    return request.param


# ---------------------------------------------------------------------------
# Problems of test_suspension_grid.py
# ---------------------------------------------------------------------------


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


def _sampled(factory, level, depth=0):
    def build():
        problem = factory()
        grid = build_suspension_grid(problem.window, problem.guard, level)
        relation = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=depth)
        return relation, compute_suspension_morse_graph(relation).morse_sets

    return build


def _neuron_without_a_base_cell_at_the_guard():
    # The relation of test_neuron_index_without_a_base_cell_at_the_guard.
    grid = build_suspension_grid(spiking_neuron_problem().window, spiking_neuron_problem().guard, 6)
    index = grid.n_guard // 2
    handle = [grid.handle_piece(index, k) for k in range(grid.n_phase // 2, grid.n_phase)]
    pieces = np.array([*handle, *grid.guard_top_cells[index]], dtype=np.int64)
    atoms = np.unique(grid.atom_of_piece[pieces])
    return _relation_on(grid, atoms, atoms), [atoms]


def _zero_relative_homology():
    # The relation of test_zero_relative_homology_gives_the_trivial_label_without_the_index_map.
    problem = bouncing_ball_problem()
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    pieces = grid.locate_base_cells([[1.1, 0.6], [1.35, 0.6]]).piece
    source, neighbor = (int(grid.atom_of_piece[piece]) for piece in pieces)
    return _relation_on(grid, [source], [source, neighbor]), [np.array([source])]


#: The Conley problems of test_suspension_grid.py: ``name -> build``.
PROBLEMS = {
    "translation-l2": _sampled(_translation_problem, 2),
    "saddle-l4": _sampled(_saddle_problem, 4),
    "repelling-orbit-l4": _sampled(_repelling_orbit_problem, 4),
    "repelling-cylinder-l2": _sampled(_repelling_cylinder_problem, 2),
    "neuron-l6-synthetic": _neuron_without_a_base_cell_at_the_guard,
    "ball-l3-zero-homology": _zero_relative_homology,
    "wheel-l3-offset2-tau0.5-gap2": _sampled(
        paper_problem_factory("rimless-wheel", tau=0.5, level_offset=2), 3, 2
    ),
    "wheel-l3-offset1-tau1-gap2": _sampled(
        lambda: PAPER_PROBLEMS["rimless-wheel"](tau=1.0, level_offset=1), 3, 2
    ),
    "ball-l3-offset1-tau2-gap2": _sampled(
        lambda: PAPER_PROBLEMS["bouncing-ball"](tau=2.0, level_offset=1), 3, 2
    ),
    "ball-l4-offset2-tau1.5": _sampled(
        lambda: PAPER_PROBLEMS["bouncing-ball"](tau=1.5, level_offset=2), 4
    ),
    "oscillator-l4-offset2-tau3-gap2": _sampled(
        lambda: PAPER_PROBLEMS["impact-vdp-duffing"](tau=3.0, level_offset=2), 4, 2
    ),
}

#: The runs with the Python statement of the kernel, which is slow: every
#: construction on the small problems, and a carrier that is not acyclic.
REFERENCE_RUNS = {
    *(
        (name, option)
        for name in (
            "translation-l2",
            "saddle-l4",
            "repelling-cylinder-l2",
            "neuron-l6-synthetic",
            "ball-l3-zero-homology",
        )
        for option in ("exit-components", "auto", "forward-closure", "excision", "excised")
    ),
    ("repelling-orbit-l4", "exit-components"),
}

#: The constructions whose carrier and chain map the backend forms.
OPTIONS = {
    "exit-components": {"index_map": "exit-components"},
    "auto": {"index_map": "auto"},
    "forward-closure": {"index_pair": "forward-closure"},
    "excision": {"index_map": "excision"},
    "excised": {"index_pair": "forward-closure", "excise": True},
}

#: The Python and the native forms of each construction in suspension_grid_conley.
CONSTRUCTIONS = {
    "python": (
        "prepare_atlas_relation_conley_2d",
        "_excision_chain_map",
        "_excised_chain_map_entries",
    ),
    "native": (
        "native_relation_shift_class",
        "native_cross_complex_chain_map",
        "native_subcomplex_chain_entries",
    ),
}

_BUILT: dict[str, tuple] = {}
_PYTHON_RUNS: dict[tuple[str, str], tuple] = {}


def _problem(name):
    if name not in _BUILT:
        _BUILT[name] = PROBLEMS[name]()
    return _BUILT[name]


def _python_runs(name, option):
    """:func:`_runs` of the Python backend, computed once per problem and construction."""

    if (name, option) not in _PYTHON_RUNS:
        _PYTHON_RUNS[name, option] = _runs(*_problem(name), "python", **OPTIONS[option])
    return _PYTHON_RUNS[name, option]


def _payload_arguments(arguments):
    counts, boundary, chain = arguments
    return (
        [int(value) for value in counts],
        [[tuple(int(v) for v in entry) for entry in entries] for entries in boundary],
        [[tuple(int(v) for v in entry) for entry in entries] for entries in chain],
    )


def _runs(relation, morse_sets, backend, **options):
    """Records (without ``seconds``) and CMGDB payloads of every Morse set, and the calls made.

    ``calls["python"]`` and ``calls["native"]`` count the calls of the Python
    and of the native forms of the constructions (:data:`CONSTRUCTIONS`).
    """

    payloads: list = []
    calls = {"python": 0, "native": 0}

    def recording(shift_class):
        def recorded(*arguments):
            payloads.append(_payload_arguments(arguments))
            return shift_class(*arguments)

        return recorded

    def counted(kind, function):
        def construction(*arguments, **keywords):
            calls[kind] += 1
            return function(*arguments, **keywords)

        return construction

    runs = []
    with pytest.MonkeyPatch.context() as patch:
        # The payload goes to one of the two shift-class functions.
        for name in ("ComputeRelativeHomologyShiftClass", SHIFT_CLASS_FUNCTION):
            if hasattr(CMGDB, name):
                patch.setattr(CMGDB, name, recording(getattr(CMGDB, name)))
        for kind, names in CONSTRUCTIONS.items():
            for name in names:
                function = getattr(suspension_grid_conley, name)
                patch.setattr(suspension_grid_conley, name, counted(kind, function))
        for node, morse_set in enumerate(morse_sets):
            payloads.clear()
            record = compute_suspension_grid_conley_index(
                relation, morse_set, morse_node=node, backend=backend, **options
            ).to_dict()
            record.pop("seconds")
            runs.append((record, list(payloads)))
    return runs, calls


# ---------------------------------------------------------------------------
# Native and Python backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("options", sorted(OPTIONS))
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_native_and_python_backends_give_the_same_records_and_payloads(kernel, name, options):
    if kernel == "reference" and (name, options) not in REFERENCE_RUNS:
        pytest.skip("run with the native kernel only")
    python, python_calls = _python_runs(name, options)
    native, native_calls = _runs(*_problem(name), "native", **OPTIONS[options])
    assert python_calls["native"] == native_calls["python"] == 0
    assert native_calls["native"] == python_calls["python"]
    assert len(native) == len(python) == len(_problem(name)[1])
    for node, ((python_record, python_payloads), (native_record, native_payloads)) in enumerate(
        zip(python, native)
    ):
        assert native_record == python_record, node
        assert len(native_payloads) == len(python_payloads), node
        for python_payload, native_payload in zip(python_payloads, native_payloads):
            assert native_payload[:2] == python_payload[:2], node
            # Equal chain-map entries in another order are reported as such.
            for degree, (ours, theirs) in enumerate(zip(native_payload[2], python_payload[2])):
                assert sorted(ours) == sorted(theirs), (node, degree)
            assert native_payload == python_payload, (node, "same entries in another order")


def test_problems_reach_every_construction():
    # The problems above form chain maps with and without exit components,
    # and a carrier of some exit component is not acyclic.
    image, calls = _python_runs("repelling-orbit-l4", "exit-components")
    assert calls["python"] == 3
    assert [record["label_source"] for record, _ in image].count("index map") == 2
    assert sum(len(payloads) for _, payloads in image) == 2
    assert any("is not acyclic" in record["index_map_blocker"] for record, _ in image)
    closure, calls = _python_runs("repelling-cylinder-l2", "forward-closure")
    assert calls["python"] == 3
    assert all(record["label_source"] == "index map" for record, _ in closure)
    # The excision construction (also the fallback of "auto") and the excised
    # forward-closure pair: labels, a carrier that is not acyclic, and an
    # atom with an empty image.
    for name, option, labels in (
        ("repelling-orbit-l4", "excision", 3),
        ("repelling-orbit-l4", "auto", 3),
        ("repelling-orbit-l4", "excised", 3),
        ("saddle-l4", "excision", 2),
        ("ball-l3-offset1-tau2-gap2", "excised", 1),
    ):
        runs, calls = _python_runs(name, option)
        assert calls["python"] >= labels, (name, option)
        sources = [record["label_source"] for record, _ in runs]
        assert sum(source.startswith("index map") for source in sources) == labels, (name, option)
    auto, calls = _python_runs("repelling-orbit-l4", "auto")
    assert calls["python"] == 4  # three exit-components maps and one excision map
    assert [record["label_source"] for record, _ in auto].count("index map (excision pair)") == 1
    for option in ("excision", "excised"):
        runs, _ = _python_runs("oscillator-l4-offset2-tau3-gap2", option)
        assert any("is not acyclic" in record["blocker"] for record, _ in runs), option
    runs, _ = _python_runs("wheel-l3-offset1-tau1-gap2", "excision")
    assert any("has an empty image" in record["blocker"] for record, _ in runs)


def _preparations(relation, morse_sets, **options):
    """The Python preparations of the index maps, with their arguments."""

    prepare = suspension_grid_conley.prepare_atlas_relation_conley_2d
    recorded = []

    def recording(*arguments, **keywords):
        preparation = prepare(*arguments, **keywords)
        recorded.append((preparation, keywords))
        return preparation

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(suspension_grid_conley, "prepare_atlas_relation_conley_2d", recording)
        for node, morse_set in enumerate(morse_sets):
            compute_suspension_grid_conley_index(
                relation, morse_set, morse_node=node, backend="python", **options
            )
    return recorded


def test_native_chain_map_and_carriers_are_those_of_the_python_construction(kernel):
    # The problem of test_chain_selector_matches_the_scan_of_the_complex, and
    # one with exit components.
    recorded = _preparations(*_problem("wheel-l3-offset1-tau1-gap2"))
    recorded += _preparations(*_problem("saddle-l4"))
    assert any(preparation.pair.p0_atlas_cells for preparation, _ in recorded)
    for preparation, keywords in recorded:
        pair = preparation.pair
        complex_ = pair.complex
        arrays = relation_carrier_arrays(
            pair,
            top_relation=keywords["top_relation"],
            use_exit_component_carrier=keywords["use_exit_component_carrier"],
        )
        result = arrays.compute(return_carriers=True)
        assert result["status"] == "ok"
        assert int(result["failure_degree"]) == int(result["failure_row"]) == -1
        for degree in range(complex_.max_dimension + 1):
            cells = complex_.cells_of_dimension(degree)
            row_of = {cell: row for row, cell in enumerate(cells)}
            expected = {
                row: {
                    row_of[target]: value
                    for target, value in preparation.chain_map.image(cell).items()
                }
                for row, cell in enumerate(cells)
            }
            found: dict[int, dict[int, int]] = {row: {} for row in range(len(cells))}
            entries = np.asarray(result["chain_map"][degree], dtype=np.int64).reshape(-1, 3)
            for source, target, value in entries.tolist():
                assert target not in found[source]
                found[source][target] = value
            assert found == expected, degree
        # Equal carrier ids exactly for equal carriers.
        values = [preparation.carrier.image(cell) for cell in complex_.cells]
        identifiers = [int(value) for value in np.asarray(result["carrier_ids"])]
        assert len(identifiers) == len(values)
        assert len(set(zip(identifiers, values))) == len(set(values)) == len(set(identifiers))
        assert int(result["carrier_count"]) == len(set(values))
        # The relative payload is the Python one, entry for entry.
        payload = native_relation_payload(
            pair,
            top_relation=keywords["top_relation"],
            use_exit_component_carrier=keywords["use_exit_component_carrier"],
        )
        assert payload.as_compute_args() == preparation.payload.as_compute_args()
        assert payload.basis_by_dimension == preparation.payload.basis_by_dimension


# ---------------------------------------------------------------------------
# The arrays
# ---------------------------------------------------------------------------


def test_arrays_are_the_cells_in_complex_order_and_the_carrier_vertex_images():
    recorded = _preparations(*_problem("saddle-l4"))
    recorded += _preparations(*_problem("repelling-cylinder-l2"), index_pair="forward-closure")
    assert any(preparation.pair.p0_atlas_cells for preparation, _ in recorded)
    assert any(not keywords["use_exit_component_carrier"] for _, keywords in recorded)
    for preparation, keywords in recorded:
        pair = preparation.pair
        complex_ = pair.complex
        arrays = relation_carrier_arrays(
            pair,
            top_relation=keywords["top_relation"],
            use_exit_component_carrier=keywords["use_exit_component_carrier"],
        )
        assert len(arrays.source_simplices) == complex_.max_dimension + 1
        flattened = []
        for degree, array in enumerate(arrays.source_simplices):
            assert array.dtype == np.int32 and array.flags.c_contiguous
            assert array.shape == (len(complex_.cells_of_dimension(degree)), degree + 1)
            flattened += [AtlasNerveSimplex(tuple(row)) for row in array.tolist()]
        assert tuple(flattened) == complex_.cells
        labels = arrays.source_simplices[0][:, 0].tolist()
        indptr, indices = arrays.vertex_image_indptr, arrays.vertex_image_indices
        assert indptr.dtype == np.int64 and indices.dtype == np.int32
        assert indptr.shape == (len(labels) + 1,) and indptr[-1] == indices.size
        images = {
            label: frozenset(indices[indptr[row] : indptr[row + 1]].tolist())
            for row, label in enumerate(labels)
        }
        for row, label in enumerate(labels):
            values = indices[indptr[row] : indptr[row + 1]].tolist()
            assert values == sorted(preparation.relation_vertex_images[label])
        assert arrays.source_exit.dtype == np.uint8
        exits = [int(label in pair.p0_atlas_cells) for label in labels]
        assert arrays.source_exit.tolist() == exits
        # The induced carriers of the arrays are those of the Python construction.
        by_first: dict[int, list] = {}
        for cell in complex_.cells:
            by_first.setdefault(cell.vertices[0], []).append(cell)
        for cell in complex_.cells:
            vertices = frozenset().union(*(images[vertex] for vertex in cell.vertices))
            induced = frozenset(
                other
                for vertex in vertices
                for other in by_first.get(vertex, ())
                if vertices.issuperset(other.vertices)
            )
            assert induced == preparation.carrier.image(cell)


def _four_cell_quotient() -> AtlasQuotientNerveComplex2D:
    # The quotient of test_atlas_conley.py: two base rectangles and two
    # phase slabs glued into an annulus.
    cells = (
        AtlasRectangleCell2D(0, 0, (0.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(1, 0, (0.0, 1.0, 1.0, 2.0)),
        AtlasRectangleCell2D(2, 1, (0.0, 0.0, 1.0, 0.5)),
        AtlasRectangleCell2D(3, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    return AtlasQuotientNerveComplex2D(
        cells,
        AtlasResetGluing2D(
            0,
            1,
            guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0, 0.0),
            reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0, 1.0),
        ),
    )


def test_arrays_of_the_exit_components_of_a_small_quotient():
    nerve = _four_cell_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0, 1})
    relation = {0: {2}, 1: {3}, 2: {0}, 3: {0}}
    arrays = relation_carrier_arrays(pair, top_relation=relation)
    assert [array.tolist() for array in arrays.source_simplices] == [
        [list(cell.vertices) for cell in pair.complex.cells_of_dimension(degree)]
        for degree in range(pair.complex.max_dimension + 1)
    ]
    # The exit vertices 0 and 1 are carried by their component {0, 1} of P0.
    assert arrays.vertex_image_indptr.tolist() == [0, 2, 4, 5, 6]
    assert arrays.vertex_image_indices.tolist() == [0, 1, 0, 1, 0, 0]
    assert arrays.source_exit.tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError, match="does not preserve P0"):
        relation_carrier_arrays(pair, top_relation=relation, use_exit_component_carrier=False)
    with pytest.raises(ValueError, match="missing P1 sources"):
        relation_carrier_arrays(pair, top_relation={0: {0}})


def test_simplex_arrays_refuse_other_complexes():
    def simplex(*vertices):
        return AtlasNerveSimplex(tuple(vertices))

    # d[0, 1] = [1] - [0]: the faces by removed vertex are [1], then [0].
    edge = simplex(0, 1)
    good = FiniteCellComplex(
        {simplex(0): 0, simplex(1): 0, edge: 1},
        {edge: {simplex(1): 1, simplex(0): -1}},
    )
    assert [array.tolist() for array in simplex_arrays(good)] == [[[0], [1]], [[0, 1]]]
    unordered = FiniteCellComplex({simplex(1): 0, simplex(0): 0}, {})
    with pytest.raises(ValueError, match="lexicographic order"):
        simplex_arrays(unordered)
    interleaved = FiniteCellComplex(
        {simplex(0): 0, edge: 1, simplex(1): 0},
        {edge: {simplex(1): 1, simplex(0): -1}},
    )
    with pytest.raises(ValueError, match="ordered by dimension"):
        simplex_arrays(interleaved)
    # The other orientation, and the faces listed in another order.
    for boundary in ({simplex(1): -1, simplex(0): 1}, {simplex(0): -1, simplex(1): 1}):
        other = FiniteCellComplex({simplex(0): 0, simplex(1): 0, edge: 1}, {edge: boundary})
        with pytest.raises(ValueError, match="alternating sum"):
            simplex_arrays(other)
    large = FiniteCellComplex({simplex(2**31): 0}, {})
    with pytest.raises(ValueError, match="int32"):
        simplex_arrays(large)
    with pytest.raises(ValueError, match="'a' of dimension 0 is not a simplex with 1 vertices"):
        simplex_arrays(FiniteCellComplex({"a": 0}, {}))
    long_edge = simplex(0, 1, 2)
    with pytest.raises(ValueError, match="is not a simplex with 2 vertices"):
        simplex_arrays(
            FiniteCellComplex(
                {simplex(0): 0, simplex(1): 0, long_edge: 1},
                {long_edge: {simplex(1): 1, simplex(0): -1}},
            )
        )
    # A triangle, and the same triangle with its faces listed in another order.
    vertices = {simplex(v): 0 for v in range(3)}
    edges = {simplex(a, b): {simplex(b): 1, simplex(a): -1} for a, b in ((0, 1), (0, 2), (1, 2))}
    triangle = simplex(0, 1, 2)
    faces = {simplex(1, 2): 1, simplex(0, 2): -1, simplex(0, 1): 1}
    dimensions = {**vertices, **dict.fromkeys(edges, 1), triangle: 2}
    filled = FiniteCellComplex(dimensions, {**edges, triangle: faces})
    assert [array.tolist() for array in simplex_arrays(filled)] == [
        [[0], [1], [2]],
        [[0, 1], [0, 2], [1, 2]],
        [[0, 1, 2]],
    ]
    reordered = dict(reversed(list(faces.items())))
    with pytest.raises(ValueError, match="alternating sum"):
        simplex_arrays(FiniteCellComplex(dimensions, {**edges, triangle: reordered}))
    with pytest.raises(ValueError, match="has no image"):
        carrier_kernel_arrays(good, {0: {1}})
    with pytest.raises(ValueError, match="outside the complex"):
        carrier_kernel_arrays(good, {0: {1}, 1: {0}}, exit_vertices={5})


def test_vertex_images_shared_by_vertices_are_repeated():
    shared = frozenset({5, 3})
    images = {0: shared, 1: [4, 3, 4], 2: shared, 3: {7}}
    indptr, indices = _image_csr([0, 1, 2, 3], images)
    assert indptr.tolist() == [0, 2, 4, 6, 7]
    assert indices.tolist() == [3, 5, 3, 4, 3, 5, 7]
    assert (indptr.dtype, indices.dtype) == (np.int64, np.int32)
    indptr, indices = _image_csr([0, 1, 2, 3], images, targets={3, 7})
    assert indptr.tolist() == [0, 1, 2, 3, 4]
    assert indices.tolist() == [3, 3, 3, 7]
    indptr, indices = _image_csr([3, 1], images)
    assert (indptr.tolist(), indices.tolist()) == ([0, 1, 3], [7, 3, 4])
    indptr, indices = _image_csr([], images)
    assert (indptr.tolist(), indices.tolist(), indices.dtype) == ([0], [], np.int32)
    with pytest.raises(ValueError, match="the vertex 9 of the complex has no image"):
        _image_csr([0, 9], images)
    with pytest.raises(ValueError, match="int32"):
        _image_csr([0], {0: {2**31}})


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def _raised(function):
    try:
        function()
    except Exception as error:  # the exception is the result
        return error
    raise AssertionError("no exception was raised")


def test_a_carrier_that_is_not_acyclic_gives_the_python_exception(kernel):
    nerve = _four_cell_quotient()
    assert nerve.betti_numbers() == (1, 1, 0)  # the annulus
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices)
    # The carrier of the edges at vertex 2 is the whole annulus.
    relation = {0: {0}, 1: {1}, 2: {0, 1, 2, 3}, 3: {3}}
    python = _raised(lambda: prepare_atlas_relation_conley_2d(pair, top_relation=relation))
    native = _raised(lambda: native_relation_payload(pair, top_relation=relation))
    assert type(native) is type(python) is ValueError
    assert str(native) == str(python)
    assert str(native) == (
        "carrier image of AtlasNerveSimplex(vertices=(2,)) is not acyclic over GF(5)"
    )


def test_kernel_failures_give_the_exceptions_of_the_python_construction():
    nerve = _four_cell_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0, 1})
    images = {0: frozenset({0, 1}), 1: frozenset({0, 1}), 2: frozenset({0}), 3: frozenset({0})}
    edge = pair.complex.cells_of_dimension(1)[2]

    def failure(status, degree=1, row=2, vertex_images=images, reference=None):
        result = {"status": status, "failure_degree": degree, "failure_row": row}
        return kernel_failure(result, pair, vertex_images, reference or (lambda: None))

    error = failure("empty_carrier")
    assert type(error) is ValueError and str(error) == f"relation carrier of {edge!r} is empty"
    error = failure("not_acyclic", 0, 1)
    assert str(error) == (
        "carrier image of AtlasNerveSimplex(vertices=(1,)) is not acyclic over GF(5)"
    )
    error = failure("no_solution")
    assert type(error) is ValueError and str(error) == (
        f"acyclic-carrier chain selection failed on {edge!r}: the boundary equation has no "
        "solution in the carrier"
    )
    # The Python construction checks the pair before it forms the chain map.
    leaving = {**images, 0: frozenset({0, 2})}
    for status in ("no_solution", "pair_violation", "chain_map_invalid"):
        error = failure(status, vertex_images=leaving)
        assert type(error) is ValueError
        assert str(error) == "the acyclic carrier does not preserve the relative pair"
    # Other failures of the chain map give the exception of the Python construction.
    expected = ValueError("dF != Fd on the cell")

    def raising():
        raise expected

    assert failure("chain_map_invalid", reference=raising) is expected
    assert failure("pair_violation", reference=raising) is expected
    assert type(failure("chain_map_invalid")) is AssertionError
    assert type(failure("unknown")) is AssertionError
    assert type(failure("not_acyclic", 1, 99)) is AssertionError
    assert type(failure("not_acyclic", -1, -1)) is AssertionError


# ---------------------------------------------------------------------------
# Carriers into another complex: the excision construction and the excised pair
# ---------------------------------------------------------------------------


def _checked_native_constructions(relation, morse_sets, **options):
    """Run the native backend, comparing each native chain map with the Python one.

    The chain maps of the excision construction are compared cell by cell,
    with the entries of every image in order; the entries of the excised
    pair are compared as they go to CMGDB.  Returns the number of chain maps
    compared, and of those whose target is the source (``Xbar = X``).
    """

    compared = {"excision": 0, "excision into X": 0, "excised": 0}
    cross = suspension_grid_conley.native_cross_complex_chain_map
    subcomplex = suspension_grid_conley.native_subcomplex_chain_entries

    def checked_cross(source, target, vertex_images, reference):
        native = cross(source, target, vertex_images, reference)
        python = reference()
        for cell in source.complex.cells:
            assert list(native.image(cell).items()) == list(python.image(cell).items()), cell
        compared["excision"] += 1
        compared["excision into X"] += target is source
        return native

    def checked_subcomplex(pair, sources, vertex_images, reference):
        native = subcomplex(pair, sources, vertex_images, reference)
        assert native == reference()
        compared["excised"] += 1
        return native

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(suspension_grid_conley, "native_cross_complex_chain_map", checked_cross)
        patch.setattr(
            suspension_grid_conley, "native_subcomplex_chain_entries", checked_subcomplex
        )
        for node, morse_set in enumerate(morse_sets):
            compute_suspension_grid_conley_index(
                relation, morse_set, morse_node=node, backend="native", **options
            )
    return compared


def test_native_excision_and_excised_chain_maps_are_those_of_the_python_construction(kernel):
    names = ["saddle-l4", "repelling-cylinder-l2"]
    if kernel == "native":
        names += ["repelling-orbit-l4", "ball-l3-offset1-tau2-gap2"]
    totals = {"excision": 0, "excision into X": 0, "excised": 0}
    for name in names:
        for option in ("excision", "excised"):
            compared = _checked_native_constructions(*_problem(name), **OPTIONS[option])
            for key, value in compared.items():
                totals[key] += value
    assert totals["excision"] >= 4 and totals["excised"] >= 4
    assert 0 < totals["excision into X"] < totals["excision"]


def _annulus_pair(vertices=(0, 1, 2, 3), exits=()):
    return AtlasRelativeIndexPair2D(_four_cell_quotient(), vertices, exits)


def _excised(pair, images, backend):
    return suspension_grid_conley._excised_shift_class(
        pair, sorted(pair.p1_atlas_cells), images, backend=backend
    )


_NOT_PRESERVED = "the carrier of AtlasNerveSimplex(vertices=({},)) does not preserve P0"
_NOT_ACYCLIC = "carrier image of AtlasNerveSimplex(vertices={}) is not acyclic over GF(5)"


@pytest.mark.parametrize(
    ("exits", "images", "message"),
    [
        # A vertex of P0 whose carrier leaves P0, before a carrier that is not acyclic.
        ({0}, {0: {1}, 1: {1}, 2: {0, 1, 2, 3}, 3: {3}}, _NOT_PRESERVED.format(0)),
        # A carrier that is not acyclic before such a vertex, and at one.
        ({2}, {0: {0, 1, 2, 3}, 1: {1}, 2: {1}, 3: {3}}, _NOT_ACYCLIC.format("(0,)")),
        ({0}, {0: {0, 1, 2, 3}, 1: {1}, 2: {2}, 3: {3}}, _NOT_ACYCLIC.format("(0,)")),
        # Such a vertex after the others, before an edge carried by the annulus.
        ({3}, {0: {0, 1}, 1: {2, 3}, 2: {2}, 3: {0}}, _NOT_PRESERVED.format(3)),
        ({3}, {0: {0}, 1: {1}, 2: {2}, 3: {0}}, _NOT_PRESERVED.format(3)),
        ((), {0: {0, 1}, 1: {2, 3}, 2: {2}, 3: {3}}, _NOT_ACYCLIC.format("(0, 1)")),
    ],
)
def test_excised_failures_are_those_of_the_python_construction(kernel, exits, images, message):
    pair = _annulus_pair(exits=exits)
    python = _raised(lambda: _excised(pair, images, "python"))
    native = _raised(lambda: _excised(pair, images, "native"))
    assert type(native) is type(python) is ValueError
    assert str(native) == str(python) == message


def test_excised_shift_class_is_that_of_the_python_construction(kernel):
    pair = _annulus_pair(exits={3})
    for images in (
        {0: {0}, 1: {1}, 2: {2}, 3: {3}},
        {0: {0}, 1: {0, 1}, 2: {2}, 3: {3}},
    ):
        python = _excised(pair, images, "python")
        assert _excised(pair, images, "native") == python
        assert python["homology_dimensions"] == [0, 1, 0]


def _excision_failure(source, target, images):
    """The exceptions of the Python and the native excision chain maps."""

    def reference():
        return suspension_grid_conley._excision_chain_map(source, target, images)

    python = _raised(reference)
    native = _raised(lambda: native_cross_complex_chain_map(source, target, images, reference))
    return python, native


@pytest.mark.parametrize(
    ("source", "target", "images", "message"),
    [
        (
            ((0, 1, 2), ()),
            ((0, 1, 2, 3), ()),
            {0: {0}, 1: {1, 2, 3}, 2: {2}},
            _NOT_ACYCLIC.format("(1,)"),
        ),
        # Every carrier is checked before the pair.
        (
            ((0, 1, 2), (0,)),
            ((0, 1, 2, 3), (2, 3)),
            {0: {1}, 1: {1}, 2: {1, 2, 3}},
            _NOT_ACYCLIC.format("(2,)"),
        ),
        (
            ((0, 1, 2), (2,)),
            ((0, 1, 2, 3), (2, 3)),
            {0: {0}, 1: {1}, 2: {0}},
            "the carrier of AtlasNerveSimplex(vertices=(2,)) in A does not lie in Abar",
        ),
        # The image vertex 3 is not a vertex of the target: an empty carrier.
        (
            ((0, 1), ()),
            ((0, 1, 2), ()),
            {0: {3}, 1: {1}},
            "carrier image of AtlasNerveSimplex(vertices=(0,)) must be nonempty",
        ),
    ],
)
def test_excision_failures_are_those_of_the_python_construction(
    kernel, source, target, images, message
):
    python, native = _excision_failure(
        _annulus_pair(*source), _annulus_pair(*target), images
    )
    assert type(native) is type(python) is ValueError
    assert str(native) == str(python) == message


def test_excision_chain_map_is_that_of_the_python_construction(kernel):
    source = _annulus_pair((0, 1, 2), (2,))
    target = _annulus_pair(exits=(2, 3))
    cases = [
        (source, target, {0: {0, 1}, 1: {1}, 2: {2}}),
        # Image vertices outside the target span nothing.
        (_annulus_pair((0, 1), ()), _annulus_pair((0, 1, 2)), {0: {0, 3}, 1: {1, 2}}),
    ]
    into_itself = _annulus_pair(exits=(3,))
    cases.append((into_itself, into_itself, {0: {0}, 1: {0, 1}, 2: {2}, 3: {3}}))
    for source, target, images in cases:

        def reference(source=source, target=target, images=images):
            return suspension_grid_conley._excision_chain_map(source, target, images)

        python = reference()
        native = native_cross_complex_chain_map(source, target, images, reference)
        for cell in source.complex.cells:
            assert list(native.image(cell).items()) == list(python.image(cell).items()), cell
        with pytest.raises(KeyError, match="not a source cell"):
            native.image(AtlasNerveSimplex((0, 1, 2, 3)))
    arrays = cross_complex_carrier_arrays(
        into_itself.complex, into_itself.complex, {0: {0}, 1: {1}, 2: {2}, 3: {3}}, {3}, {3}
    )
    assert arrays.target_simplices is None and arrays.target_exit is None
    with pytest.raises(ValueError, match="one set of exit vertices"):
        cross_complex_carrier_arrays(
            into_itself.complex, into_itself.complex, {0: {0}, 1: {1}, 2: {2}, 3: {3}}, {3}, {2}
        )


def test_other_kernel_failures_run_the_python_construction(monkeypatch):
    source = _annulus_pair((0, 1, 2))
    target = _annulus_pair()
    images = {0: {0}, 1: {1}, 2: {2}}
    pair = _annulus_pair(exits={3})
    pair_images = {0: {0}, 1: {1}, 2: {2}, 3: {3}}
    expected = ValueError("the exception of the Python construction")

    def raising():
        raise expected

    def forming():
        return None

    def returning(result):
        def kernel(*arguments, **keywords):
            return dict(result)

        return kernel

    def cross(reference):
        return _raised(lambda: native_cross_complex_chain_map(source, target, images, reference))

    def excised(reference):
        return _raised(
            lambda: native_subcomplex_chain_entries(
                pair, pair.complex.cell_set, pair_images, reference
            )
        )

    for status, degree, row in (
        ("empty_carrier", 0, 1),
        ("no_solution", 1, 0),
        ("chain_map_invalid", 1, 0),
        ("pair_violation", 1, 0),
    ):
        result = {"status": status, "failure_degree": degree, "failure_row": row}
        monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, returning(result), raising=False)
        assert cross(raising) is expected and excised(raising) is expected
        assert type(cross(forming)) is AssertionError
        assert type(excised(forming)) is AssertionError
    # The failures rebuilt without the Python construction.
    result = {"status": "not_acyclic", "failure_degree": 0, "failure_row": 2}
    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, returning(result), raising=False)
    assert str(cross(raising)) == str(excised(raising)) == _NOT_ACYCLIC.format("(2,)")
    result = {"status": "pair_violation", "failure_degree": 0, "failure_row": 2}
    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, returning(result), raising=False)
    assert cross(raising) is expected
    assert str(excised(raising)) == _NOT_PRESERVED.format(2)
    for result in (
        {"status": "unknown", "failure_degree": 0, "failure_row": 0},
        {"status": "not_acyclic", "failure_degree": 1, "failure_row": 99},
        {"status": "not_acyclic", "failure_degree": 7, "failure_row": 0},
    ):
        monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, returning(result), raising=False)
        assert type(cross(raising)) is AssertionError
        assert type(excised(raising)) is AssertionError


# ---------------------------------------------------------------------------
# The choice of backend
# ---------------------------------------------------------------------------


def test_auto_uses_python_when_cmgdb_lacks_the_native_kernel(monkeypatch):
    monkeypatch.delattr(CMGDB, KERNEL_FUNCTION, raising=False)
    assert not native_carrier_kernel_available()
    assert resolve_conley_backend("auto") == resolve_conley_backend("python") == "python"
    with pytest.raises(RuntimeError, match="ComputeCarrierChainMap"):
        resolve_conley_backend("native")
    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_conley_backend("fast")

    def not_used(*arguments, **keywords):
        raise AssertionError("the native kernel is used although CMGDB lacks it")

    monkeypatch.setattr(suspension_grid_conley, "native_relation_shift_class", not_used)
    relation, morse_sets = _problem("translation-l2")
    records, calls = _runs(relation, morse_sets, "auto")
    assert calls == {"python": 1, "native": 0}
    assert records[0][0]["computed"] and records[0][0]["label_source"] == "index map"
    with pytest.raises(RuntimeError, match="ComputeCarrierChainMap"):
        compute_suspension_grid_conley_index(relation, morse_sets[0], backend="native")
    with pytest.raises(RuntimeError, match="ComputeCarrierChainMap"):
        compute_suspension_grid_conley_indices(relation, morse_sets, backend="native")
    with pytest.raises(ValueError, match="backend must be one of"):
        compute_suspension_grid_conley_index(relation, morse_sets[0], backend="fast")


def test_auto_uses_the_native_kernel_when_cmgdb_has_it(monkeypatch):
    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, reference_carrier_chain_map, raising=False)
    assert resolve_conley_backend("auto") == resolve_conley_backend("native") == "native"
    relation, morse_sets = _problem("translation-l2")
    records, calls = _runs(relation, morse_sets, "auto")
    assert calls == {"python": 0, "native": 1}
    assert records[0][0]["computed"] and records[0][0]["shift_class"][:2] == ["x-1", "x-1"]


def test_auto_is_resolved_only_to_form_an_index_map(monkeypatch):
    # Resolving "auto" imports CMGDB; a label of zero homology needs neither.
    calls = []
    available = carrier_kernel.native_carrier_kernel_available

    def counted():
        calls.append(None)
        return available()

    monkeypatch.setattr(carrier_kernel, "native_carrier_kernel_available", counted)
    relation, morse_sets = _problem("ball-l3-zero-homology")
    result = compute_suspension_grid_conley_index(relation, morse_sets[0], backend="auto")
    assert result.label_source == "zero relative homology"
    results = compute_suspension_grid_conley_indices(relation, morse_sets, backend="auto")
    assert results[0].to_dict() | {"seconds": 0} == result.to_dict() | {"seconds": 0}
    assert not calls
    relation, morse_sets = _problem("translation-l2")
    result = compute_suspension_grid_conley_index(relation, morse_sets[0], backend="auto")
    assert result.label_source == "index map"
    assert len(calls) == 1


def test_cmgdb_provenance_names_the_installed_cmgdb_and_functions(monkeypatch):
    try:
        version = importlib.metadata.version("cmgdb")
    except importlib.metadata.PackageNotFoundError:
        version = None
    old_function = "ComputeRelativeHomologyShiftClass"
    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, reference_carrier_chain_map, raising=False)
    monkeypatch.setattr(
        CMGDB, SHIFT_CLASS_FUNCTION, getattr(CMGDB, old_function), raising=False
    )
    assert cmgdb_provenance("auto") == {
        "version": version,
        "shift_class_function": SHIFT_CLASS_FUNCTION,
        "conley_backend": "native",
    }
    assert cmgdb_provenance("python")["conley_backend"] == "python"
    monkeypatch.delattr(CMGDB, KERNEL_FUNCTION)
    monkeypatch.delattr(CMGDB, SHIFT_CLASS_FUNCTION)
    assert cmgdb_provenance("auto") == {
        "version": version,
        "shift_class_function": old_function,
        "conley_backend": "python",
    }
    assert cmgdb_provenance(None) == {
        "version": version,
        "shift_class_function": None,
        "conley_backend": None,
    }


def _other_cell_counts(result):
    payload = dict(result["payload"])
    payload["cell_counts"] = [count + 1 for count in payload["cell_counts"]]
    return {**result, "payload": payload}


def _other_boundary(result):
    payload = dict(result["payload"])
    boundary = [list(entries) for entries in payload["boundary_entries"]]
    degree = next(degree for degree, entries in enumerate(boundary) if entries)
    boundary[degree] = boundary[degree][1:]
    payload["boundary_entries"] = boundary
    return {**result, "payload": payload}


def _failure_the_python_construction_does_not_have(result):
    return {"status": "chain_map_invalid", "failure_degree": 1, "failure_row": 0}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_other_cell_counts, "gives the relative cell counts"),
        (_other_boundary, "gives another relative boundary"),
        (
            _failure_the_python_construction_does_not_have,
            "but the Python construction forms the chain map",
        ),
    ],
)
def test_auto_index_map_does_not_replace_a_kernel_disagreement(monkeypatch, change, message):
    # The kernel and the Python construction disagree.  With index_map="auto"
    # the AssertionError is raised and recorded, and the excision
    # construction is not tried in its place.
    def disagreeing(*arguments, **keywords):
        return change(dict(reference_carrier_chain_map(*arguments, **keywords)))

    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, disagreeing, raising=False)
    native = suspension_grid_conley.native_relation_shift_class
    raised = []

    def recording(*arguments, **keywords):
        try:
            return native(*arguments, **keywords)
        except Exception as error:
            raised.append(error)
            raise

    excision_calls = []

    def excision(*arguments, **keywords):
        excision_calls.append(arguments)
        return ()

    monkeypatch.setattr(suspension_grid_conley, "native_relation_shift_class", recording)
    monkeypatch.setattr(suspension_grid_conley, "_excision_index_map", excision)
    relation, morse_sets = _problem("translation-l2")
    record = compute_suspension_grid_conley_index(
        relation, morse_sets[0], backend="native", index_map="auto"
    ).to_dict()
    assert len(raised) == 1 and type(raised[0]) is AssertionError
    assert message in str(raised[0])
    assert excision_calls == []
    assert record["homology_computed"] and any(record["homology_dimensions"])
    assert not record["computed"] and not record["label_source"]
    assert record["index_map_blocker"] == record["blocker"] == f"AssertionError: {raised[0]}"
    assert record["exit_components_blocker"] == ""


def _load_runner_script():
    path = CODE_ROOT / "demo" / "run_paper_examples.py"
    spec = importlib.util.spec_from_file_location("run_paper_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_conley_backend_option(monkeypatch):
    runner = _load_runner_script()

    def parse(*argv):
        monkeypatch.setattr("sys.argv", ["run_paper_examples.py", *argv])
        return runner._arguments()

    assert parse().conley_backend == "auto"
    arguments = parse("--conley-backend", "python")
    assert arguments.conley_backend == "python"
    # The backend is not an option of the pair, so the output names keep theirs.
    assert runner._index_options(arguments) == {"index_pair": "image", "index_map": "auto"}
    for argv in (("--conley-backend", "fast"), ("--no-conley", "--conley-backend", "python")):
        with pytest.raises(SystemExit):
            parse(*argv)
    monkeypatch.delattr(CMGDB, KERNEL_FUNCTION, raising=False)
    with pytest.raises(SystemExit):
        parse("--conley-backend", "native")
    monkeypatch.setattr(CMGDB, KERNEL_FUNCTION, reference_carrier_chain_map, raising=False)
    assert parse("--conley-backend", "native").conley_backend == "native"
