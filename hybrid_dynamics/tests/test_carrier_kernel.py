"""The native carrier kernel of CMGDB against the Python construction.

``CMGDB.ComputeCarrierChainMap`` forms the carriers and the chain map of
:func:`atlas_conley.prepare_atlas_relation_conley_2d` natively.  These tests
check the arrays handed to it (:mod:`carrier_kernel`), the exceptions
rebuilt from its failures, and the choice of backend, and they run the
Conley problems of ``test_suspension_grid.py`` with the Python and the
native backend: the records and every payload passed to
``CMGDB.ComputeRelativeHomologyShiftClass`` must be equal.  The runs with
the native kernel are skipped when the installed CMGDB lacks it.  The same
comparisons also run with :func:`reference_carrier_chain_map`, a Python
statement of the kernel, in its place; they check the arrays, the
dispatch, and the rebuilt exceptions, but not the kernel itself.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

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
from hybrid_dynamics.src import suspension_grid_conley
from hybrid_dynamics.src.carrier_kernel import (
    KERNEL_FUNCTION,
    carrier_kernel_arrays,
    kernel_failure,
    native_carrier_kernel_available,
    native_relation_payload,
    relation_carrier_arrays,
    resolve_conley_backend,
    simplex_arrays,
)
from hybrid_dynamics.src.suspension_complex import (
    _eliminate_columns_mod_prime,
    _rank_mod_prime,
    _solve_with_pivots,
)
from hybrid_dynamics.src.suspension_grid_conley import (
    compute_suspension_grid_conley_index,
    compute_suspension_grid_conley_indices,
)
from hybrid_dynamics.src.suspension_grid_relation import SuspensionGridRelation
from test_suspension_grid import (
    _repelling_cylinder_problem,
    _repelling_orbit_problem,
    _saddle_problem,
    _translation_problem,
)

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
    """``CMGDB.ComputeCarrierChainMap`` for a complex that is its own target, in Python.

    The carriers are the subcomplexes induced on the unions of the vertex
    images; the checks run in the order of the specification (every carrier
    nonempty, every carrier acyclic, the chain map, then its validation),
    each reporting the first failing cell in the order of the complex.  The
    chain map is solved with the elimination of
    :meth:`suspension_complex.FixedTimeCarrier.construct_chain_map` on the
    rows of the cells, so its entries come in the order of the Python
    construction.
    """

    if modulus != 5:
        raise ValueError("modulus must be 5")
    if target_simplices is not None or target_exit is not None:
        raise NotImplementedError("the reference covers a complex that is its own target")
    simplices = [
        [tuple(int(v) for v in row) for row in np.asarray(array).reshape(-1, degree + 1)]
        for degree, array in enumerate(source_simplices)
    ]
    row_of = [{simplex: row for row, simplex in enumerate(group)} for group in simplices]
    labels = [simplex[0] for simplex in simplices[0]]
    vertex_row = {label: row for row, label in enumerate(labels)}
    indptr = np.asarray(vertex_image_indptr, dtype=np.int64)
    indices = np.asarray(vertex_image_indices, dtype=np.int64)
    if indptr.shape != (len(labels) + 1,) or indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("vertex_image_indptr does not match the vertices")
    images = [
        frozenset(int(v) for v in indices[indptr[row] : indptr[row + 1]])
        for row in range(len(labels))
    ]
    unknown = set().union(*images).difference(labels)
    if unknown:
        raise IndexError(f"vertex images outside the target: {sorted(unknown)!r}")
    exit_labels = frozenset(label for label, flag in zip(labels, np.asarray(source_exit)) if flag)

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
    for degree, group in enumerate(simplices):
        for row, simplex in enumerate(group):
            by_first.setdefault(simplex[0], []).append((degree, row))
    carrier_of: dict[frozenset, int] = {}
    carrier_cells: list[list[list[int]]] = []
    carrier_ids = [[0] * len(group) for group in simplices]
    for degree, group in enumerate(keys):
        for row, key in enumerate(group):
            identifier = carrier_of.get(key)
            if identifier is None:
                cells = [[] for _ in simplices]
                for vertex in key:
                    for cell_degree, cell_row in by_first.get(vertex, ()):
                        if key.issuperset(simplices[cell_degree][cell_row]):
                            cells[cell_degree].append(cell_row)
                cells = [sorted(rows) for rows in cells]
                betti = _betti_numbers(simplices, row_of, cells)
                if betti[0] != 1 or any(betti[1:]):
                    return failure("not_acyclic", degree, row)
                identifier = carrier_of[key] = len(carrier_cells)
                carrier_cells.append(cells)
            carrier_ids[degree][row] = identifier

    def faces(degree, simplex):
        return [
            (
                row_of[degree - 1][simplex[:index] + simplex[index + 1 :]],
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
            for face, incidence in faces(degree, simplex):
                add(right_hand_side, images_of[degree - 1][face], incidence)
            system = systems.get((identifier, degree))
            if system is None:
                row_index = {cell: index for index, cell in enumerate(cells[degree - 1])}
                columns = cells[degree]
                entries = {
                    column: dict(faces(degree, simplices[degree][column])) for column in columns
                }
                pivots = _eliminate_columns_mod_prime(row_index, columns, entries, 5)
                system = (row_index, pivots, columns)
                systems[(identifier, degree)] = system
            row_index, pivots, columns = system
            if not set(right_hand_side) <= set(row_index):
                return failure("no_solution", degree, row)
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
                add(boundary_after_map, dict(faces(degree, simplices[degree][target])), coefficient)
            map_after_boundary = {}
            for face, incidence in faces(degree, simplex):
                add(map_after_boundary, images_of[degree - 1][face], incidence)
            if boundary_after_map != map_after_boundary:
                return failure("chain_map_invalid", degree, row)
    for degree, group in enumerate(simplices):
        for row in range(len(group)):
            carrier = set(carrier_cells[carrier_ids[degree][row]][degree])
            if not set(images_of[degree][row]) <= carrier:
                return failure("chain_map_invalid", degree, row)
    in_p0 = [[exit_labels.issuperset(simplex) for simplex in group] for group in simplices]
    for degree, group in enumerate(simplices):
        for row in range(len(group)):
            if in_p0[degree][row] and not all(in_p0[degree][t] for t in images_of[degree][row]):
                return failure("pair_violation", degree, row)

    basis = [
        [row for row in range(len(group)) if not in_p0[degree][row]]
        for degree, group in enumerate(simplices)
    ]
    position = [{row: index for index, row in enumerate(rows)} for rows in basis]
    boundary_entries = [[]]
    for degree in range(1, len(simplices)):
        entries = []
        for column, row in enumerate(basis[degree]):
            for face, incidence in faces(degree, simplices[degree][row]):
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
        "payload": {
            "cell_counts": [len(rows) for rows in basis],
            "boundary_entries": boundary_entries,
            "chain_map_entries": chain_map_entries,
        },
        "carrier_count": len(carrier_cells),
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
        for option in ("exit-components", "auto", "forward-closure")
    ),
    ("repelling-orbit-l4", "exit-components"),
}

#: The constructions whose carrier and chain map the backend forms.
OPTIONS = {
    "exit-components": {"index_map": "exit-components"},
    "auto": {"index_map": "auto"},
    "forward-closure": {"index_pair": "forward-closure"},
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
    """Records (without ``seconds``) and CMGDB payloads of every Morse set, and the calls made."""

    payloads: list = []
    calls = {"python": 0, "native": 0}
    shift_class = CMGDB.ComputeRelativeHomologyShiftClass
    prepare = suspension_grid_conley.prepare_atlas_relation_conley_2d
    native = suspension_grid_conley.native_relation_shift_class

    def recording(*arguments):
        payloads.append(_payload_arguments(arguments))
        return shift_class(*arguments)

    def python_construction(*arguments, **keywords):
        calls["python"] += 1
        return prepare(*arguments, **keywords)

    def native_construction(*arguments, **keywords):
        calls["native"] += 1
        return native(*arguments, **keywords)

    runs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CMGDB, "ComputeRelativeHomologyShiftClass", recording)
        patch.setattr(
            suspension_grid_conley, "prepare_atlas_relation_conley_2d", python_construction
        )
        patch.setattr(suspension_grid_conley, "native_relation_shift_class", native_construction)
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
    assert any("is not acyclic" in record["index_map_blocker"] for record, _ in image)
    closure, calls = _python_runs("repelling-cylinder-l2", "forward-closure")
    assert calls["python"] == 3
    assert all(record["label_source"] == "index map" for record, _ in closure)


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
    with pytest.raises(ValueError, match="has no image"):
        carrier_kernel_arrays(good, {0: {1}})
    with pytest.raises(ValueError, match="outside the complex"):
        carrier_kernel_arrays(good, {0: {1}, 1: {0}}, exit_vertices={5})


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
