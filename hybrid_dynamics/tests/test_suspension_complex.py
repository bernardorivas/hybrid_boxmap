from __future__ import annotations

import json
from itertools import product

import pytest

from hybrid_dynamics.src.suspension_complex import (
    _solve_linear_system_mod_prime,
)

from hybrid_dynamics import (
    BaseCell,
    CMGDBRelativeHomologyPayload,
    CellularAttachmentMap,
    CellularChainMap,
    CrossComplexAcyclicCarrier,
    CellularResetMap,
    CubicalCell,
    CubicalGridComplex,
    DoubleMappingCylinderComplex,
    DoubleMappingCylinderHandle,
    FiniteCellComplex,
    FixedTimeCarrier,
    FixedTimeCellRelation,
    GuardPrismCell,
    PhaseSliceCell,
    PhaseCell,
    RelativeCellPair,
    ResetHandle,
    SampledSuspensionCellAdapter,
    SparseCubicalGridComplex,
    SuspensionBaseCell,
    SuspensionCellComplex,
    audit_cubical_hyperplane_attachment,
)


def _cubical_face_embedding(
    guard_cell: CubicalCell, side: int
) -> CubicalCell:
    return CubicalCell(
        (side, *guard_cell.anchor),
        (False, *guard_cell.spanning),
    )


def four_dimensional_double_mapping_cylinder():
    base = CubicalGridComplex((1, 1, 1, 1))
    guard = CubicalGridComplex((1, 1, 1))
    bottom_carrier = CrossComplexAcyclicCarrier(
        guard,
        base,
        {
            cell: (_cubical_face_embedding(cell, 0),)
            for cell in guard.cells
        },
    )
    top_carrier = CrossComplexAcyclicCarrier(
        guard,
        base,
        {
            cell: (_cubical_face_embedding(cell, 1),)
            for cell in guard.cells
        },
    )
    bottom = bottom_carrier.construct_integral_attachment_map()
    top = top_carrier.construct_integral_attachment_map()
    cylinder = DoubleMappingCylinderComplex(
        base,
        [
            DoubleMappingCylinderHandle(
                "nonlinear-reset-placeholder",
                guard,
                bottom,
                top,
                slabs=2,
            )
        ],
    )
    return base, guard, bottom_carrier, top_carrier, cylinder


def interval_with_reset(*, slabs: int = 2, fixed_reset: bool = False):
    base = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    reset_target = right if fixed_reset else left
    reset = CellularResetMap.from_cell_map(base, {right}, {right: reset_target})
    suspension = SuspensionCellComplex(
        base, [ResetHandle("impact", reset, slabs)]
    )
    return base, suspension, left, right


def test_cubical_grid_has_oriented_boundary_and_expected_homology():
    square = CubicalGridComplex((1, 1))
    top = square.top_cell(0)

    assert top.dimension == 2
    assert len(square.boundary(top)) == 4
    assert set(square.boundary(top).values()) == {-1, 1}
    assert square.betti_numbers(modulus=5) == (1, 0, 0)


def test_sparse_cubical_grid_materializes_only_selected_cube_closures():
    sparse = SparseCubicalGridComplex((4, 4, 4, 4), ((1, 1, 1, 1), (2, 1, 1, 1)))

    assert len(sparse.cells_of_dimension(4)) == 2
    assert sparse.top_cell_at((1, 1, 1, 1)) in sparse.cell_set
    assert sparse.top_cell_at((2, 1, 1, 1)) in sparse.cell_set
    assert sparse.betti_numbers(modulus=5) == (1, 0, 0, 0, 0)
    with pytest.raises(KeyError, match="not a selected top cube"):
        sparse.top_cell_at((0, 0, 0, 0))


def test_finite_complex_rejects_nonzero_boundary_squared():
    with pytest.raises(ValueError, match="boundary squared"):
        FiniteCellComplex(
            {"v": 0, "e": 1, "f": 2},
            {"e": {"v": 1}, "f": {"e": 1}},
        )


def test_sparse_modular_solver_sets_free_variables_to_zero_deterministically():
    result = _solve_linear_system_mod_prime(
        ("row",),
        ("pivot", "free"),
        {
            "pivot": {"row": 1},
            "free": {"row": 1},
        },
        {"row": 1},
        5,
    )

    assert result == {"pivot": 1}
    assert _solve_linear_system_mod_prime(
        (), ("unconstrained",), {"unconstrained": {}}, {}, 5
    ) == {}


def test_sparse_modular_solver_rejects_an_inconsistent_system():
    with pytest.raises(ValueError, match="no solution"):
        _solve_linear_system_mod_prime(
            ("first", "second"),
            ("x",),
            {"x": {"first": 1, "second": 1}},
            {"second": 1},
            5,
        )


def test_sparse_modular_solver_matches_all_two_by_two_gf5_systems():
    rows = ("r0", "r1")
    columns = ("x0", "x1")
    modulus = 5

    for a00, a01, a10, a11 in product(range(modulus), repeat=4):
        column_entries = {
            "x0": {
                row: coefficient
                for row, coefficient in zip(rows, (a00, a10), strict=True)
                if coefficient
            },
            "x1": {
                row: coefficient
                for row, coefficient in zip(rows, (a01, a11), strict=True)
                if coefficient
            },
        }
        for b0, b1 in product(range(modulus), repeat=2):
            right_hand_side = {
                row: coefficient
                for row, coefficient in zip(rows, (b0, b1), strict=True)
                if coefficient
            }
            brute_force_solution = next(
                (
                    candidate
                    for candidate in product(range(modulus), repeat=2)
                    if (a00 * candidate[0] + a01 * candidate[1] - b0)
                    % modulus
                    == 0
                    and (a10 * candidate[0] + a11 * candidate[1] - b1)
                    % modulus
                    == 0
                ),
                None,
            )

            try:
                result = _solve_linear_system_mod_prime(
                    rows,
                    columns,
                    column_entries,
                    right_hand_side,
                    modulus,
                )
            except ValueError:
                assert brute_force_solution is None
                continue

            assert brute_force_solution is not None
            x0 = result.get("x0", 0)
            x1 = result.get("x1", 0)
            assert (a00 * x0 + a01 * x1 - b0) % modulus == 0
            assert (a10 * x0 + a11 * x1 - b1) % modulus == 0


def test_reset_gluing_turns_interval_plus_handle_into_circle():
    _, suspension, left, right = interval_with_reset(slabs=2)
    left_base = suspension.base_cell(left)
    right_base = suspension.base_cell(right)
    middle = suspension.slice_cell("impact", right, 1)
    first = suspension.prism_cell("impact", right, 0)
    last = suspension.prism_cell("impact", right, 1)

    assert suspension.betti_numbers(modulus=5) == (1, 1)
    assert suspension.boundary(first) == {right_base: -1, middle: 1}
    assert suspension.boundary(last) == {middle: -1, left_base: 1}


def test_independent_guard_double_mapping_cylinder_is_four_dimensional():
    base, guard, bottom_carrier, top_carrier, cylinder = (
        four_dimensional_double_mapping_cylinder()
    )
    guard_top = guard.top_cell(0)
    first = cylinder.prism_cell("nonlinear-reset-placeholder", guard_top, 0)
    middle = cylinder.slice_cell("nonlinear-reset-placeholder", guard_top, 1)
    last = cylinder.prism_cell("nonlinear-reset-placeholder", guard_top, 1)
    bottom = cylinder.base_cell(_cubical_face_embedding(guard_top, 0))
    top = cylinder.base_cell(_cubical_face_embedding(guard_top, 1))

    assert base.max_dimension == 4
    assert guard.max_dimension == 3
    assert cylinder.max_dimension == 4
    assert cylinder.metadata["integral_attachment_maps_verified"] is True
    assert bottom_carrier.acyclicity_validated
    assert top_carrier.acyclicity_validated
    assert bottom_carrier.carries(bottom_carrier.construct_chain_map())
    assert top_carrier.carries(top_carrier.construct_chain_map())

    # For dim(g)=3, d(g x I)=d(g)xI - top(g) + bottom(g).
    assert cylinder.boundary(first)[bottom] == 1
    assert cylinder.boundary(first)[middle] == -1
    assert cylinder.boundary(last)[middle] == 1
    assert cylinder.boundary(last)[top] == -1

    # FiniteCellComplex already rejected d^2 != 0 at construction.  Check it
    # directly on every cell so the 4D path cannot bypass that gate.
    for cell in cylinder.cells:
        second_boundary: dict[object, int] = {}
        for face, coefficient in cylinder.boundary(cell).items():
            for subface, incidence in cylinder.boundary(face).items():
                second_boundary[subface] = (
                    second_boundary.get(subface, 0) + coefficient * incidence
                )
                if second_boundary[subface] == 0:
                    second_boundary.pop(subface)
        assert second_boundary == {}


def test_interior_hyperplane_audit_rejects_missing_opposite_coface():
    # The selected sparse base contains only the q<0 cube.  Its q=0 face is a
    # valid cellular attachment target, but an interior suspension chart needs
    # the incident q>0 coface too; the audit must not add it implicitly.
    base = SparseCubicalGridComplex(
        (1, 1, 2, 1),
        ((0, 0, 0, 0),),
    )
    guard = CubicalGridComplex((1, 1, 1))

    def q_zero(cell: CubicalCell) -> CubicalCell:
        return CubicalCell(
            (cell.anchor[0], cell.anchor[1], 1, cell.anchor[2]),
            (cell.spanning[0], cell.spanning[1], False, cell.spanning[2]),
        )

    carrier = CrossComplexAcyclicCarrier(
        guard,
        base,
        {cell: (q_zero(cell),) for cell in guard.cells},
    )
    attachment = carrier.construct_integral_attachment_map()

    with pytest.raises(ValueError, match="missing incident base top cofaces"):
        audit_cubical_hyperplane_attachment(
            carrier,
            attachment,
            axis=2,
            coordinate=1,
        )


def test_four_dimensional_cylinder_relative_carrier_selects_nonidentity_chain_map():
    base, _guard, _bottom_carrier, _top_carrier, cylinder = (
        four_dimensional_double_mapping_cylinder()
    )
    target = cylinder.base_cell(CubicalCell((0, 0, 0, 0), (False,) * 4))
    pair = RelativeCellPair(cylinder, cylinder.cells, {target})
    fixed_time_carrier = FixedTimeCarrier(
        cylinder,
        {cell: (target,) for cell in cylinder.cells},
        modulus=5,
        validate_acyclic=True,
    )
    chain_map = fixed_time_carrier.construct_chain_map(pair=pair)

    assert fixed_time_carrier.preserves_pair(pair)
    assert fixed_time_carrier.carries(chain_map)
    assert chain_map.preserves(pair.p1_cells)
    assert chain_map.preserves(pair.p0_cells)
    assert any(
        chain_map.image(vertex) != {vertex: 1}
        for vertex in cylinder.cells_of_dimension(0)
        if vertex != target
    )
    assert all(
        not chain_map.image(cell)
        for dimension in range(1, cylinder.max_dimension + 1)
        for cell in cylinder.cells_of_dimension(dimension)
    )
    payload = chain_map.to_cmgdb_payload(pair)
    assert payload.cell_counts == pair.cell_counts


def test_double_mapping_cylinder_rejects_nonchain_attachment():
    base = CubicalGridComplex((1, 1))
    guard = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    edge = CubicalCell((0,), (True,))
    bottom_left = CubicalCell((0, 0), (False, False))
    bottom_right = CubicalCell((0, 1), (False, False))
    bottom_edge = CubicalCell((0, 0), (False, True))

    with pytest.raises(ValueError, match="integral chain map"):
        CellularAttachmentMap(
            guard,
            base,
            {
                left: {bottom_left: 1},
                right: {bottom_right: 1},
                edge: {bottom_edge: -1},
            },
        )


def test_cross_carrier_never_silently_uses_only_a_modular_attachment():
    source = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    edge = CubicalCell((0,), (True,))
    target = FiniteCellComplex(
        {"a": 0, "b": 0, "double-edge": 1},
        {"double-edge": {"a": -2, "b": 2}},
    )
    carrier = CrossComplexAcyclicCarrier(
        source,
        target,
        {
            left: ("a",),
            right: ("b",),
            edge: ("double-edge",),
        },
        modulus=5,
    )

    modular = carrier.construct_chain_map()
    assert carrier.carries(modular)
    with pytest.raises(ValueError, match="no verified integral lift"):
        carrier.construct_integral_attachment_map()


def test_phase_registry_returns_the_same_global_cell_object():
    _, suspension, _, right = interval_with_reset(slabs=2)

    registered = suspension.phase_registry.prism("impact", right, 0)
    looked_up = suspension.prism_cell("impact", right, 0)

    assert registered is looked_up
    assert suspension.phase_cells.count(looked_up) == 1


def test_phase_cells_are_namespaced_by_handle():
    base = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    reset = CellularResetMap.from_cell_map(base, {right}, {right: left})
    suspension = SuspensionCellComplex(
        base,
        [ResetHandle("a", reset, 2), ResetHandle("b", reset, 2)],
    )

    assert suspension.prism_cell("a", right, 0) != suspension.prism_cell(
        "b", right, 0
    )


def test_reset_chain_map_is_checked_before_building_the_quotient():
    base = CubicalGridComplex((1, 1))
    lower_right = CubicalCell((1, 0), (False, False))
    upper_right = CubicalCell((1, 1), (False, False))
    right_edge = CubicalCell((1, 0), (False, True))
    lower_left = CubicalCell((0, 0), (False, False))
    upper_left = CubicalCell((0, 1), (False, False))
    left_edge = CubicalCell((0, 0), (False, True))
    guard = {lower_right, upper_right, right_edge}

    good = CellularResetMap.from_cell_map(
        base,
        guard,
        {
            lower_right: lower_left,
            upper_right: upper_left,
            right_edge: left_edge,
        },
    )
    suspension = SuspensionCellComplex(base, [ResetHandle("edge", good, 2)])
    assert suspension.max_dimension == 2

    with pytest.raises(ValueError, match="do not define a chain map"):
        CellularResetMap.from_cell_map(
            base,
            guard,
            {
                lower_right: upper_left,
                upper_right: lower_left,
                right_edge: left_edge,
            },
        )


def test_relative_pair_requires_subcomplexes_and_builds_quotient_basis():
    _, suspension, left, _ = interval_with_reset(slabs=2)
    left_base = suspension.base_cell(left)

    pair = RelativeCellPair(suspension, suspension.cells, {left_base})

    assert pair.cell_counts == (2, 3)
    assert len(pair.boundary_entries(modulus=5)) == 2
    assert pair.boundary_entries(modulus=5)[0] == ()

    edge = next(
        cell
        for cell in suspension.cells
        if isinstance(cell, SuspensionBaseCell)
        and suspension.dimension(cell) == 1
    )
    with pytest.raises(ValueError, match="not a subcomplex"):
        RelativeCellPair(suspension, {edge})


def test_identity_carrier_and_chain_map_are_certified_separately():
    _, suspension, _, _ = interval_with_reset(slabs=2)
    carrier = FixedTimeCarrier.identity(suspension, modulus=5)
    chain_map = CellularChainMap.identity(suspension, modulus=5)

    assert carrier.acyclicity_validated
    assert carrier.carries(chain_map)
    assert not carrier.carries(CellularChainMap.identity(suspension, modulus=7))


def test_nonregular_one_cell_loop_prevents_identity_acyclic_carrier():
    # With one slab and a fixed reset, the handle edge has one endpoint twice.
    # Its closed cell is a loop, not an acyclic carrier value.  Subdivision or a
    # larger separately certified carrier is required.
    _, suspension, _, _ = interval_with_reset(slabs=1, fixed_reset=True)

    assert suspension.betti_numbers(modulus=5) == (1, 1)
    with pytest.raises(ValueError, match="not acyclic"):
        FixedTimeCarrier.identity(suspension, modulus=5)


def test_carrier_rejects_failure_of_face_nesting():
    base = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    edge = CubicalCell((0,), (True,))
    images = {left: (left,), right: (right,), edge: (left,)}

    with pytest.raises(ValueError, match="carrier nesting"):
        FixedTimeCarrier(base, images)


def test_chain_map_rejects_d_f_not_equal_f_d():
    base = CubicalGridComplex((1,))
    left = CubicalCell((0,), (False,))
    right = CubicalCell((1,), (False,))
    edge = CubicalCell((0,), (True,))

    with pytest.raises(ValueError, match="dF != Fd"):
        CellularChainMap(
            base,
            {left: {left: 1}, right: {right: 1}, edge: {}},
            modulus=5,
        )


def test_cmgdb_payload_has_exact_degreewise_sparse_shape():
    _, suspension, _, _ = interval_with_reset(slabs=2)
    pair = RelativeCellPair(suspension, suspension.cells)
    payload = CellularChainMap.identity(suspension).to_cmgdb_payload(pair)
    counts, boundaries, chain_map = payload.as_compute_args()

    assert counts == [3, 3]
    assert boundaries[0] == []
    assert len(boundaries[1]) == 6
    assert chain_map == [
        [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
        [(0, 0, 1), (1, 1, 1), (2, 2, 1)],
    ]
    serialized = json.loads(payload.to_json())
    assert serialized["coefficient_field"] == 5
    assert serialized["cell_counts"] == counts

    with pytest.raises(ValueError, match="coefficient_field=5"):
        CMGDBRelativeHomologyPayload(
            cell_counts=(1,),
            boundary_entries=((),),
            chain_map_entries=(((0, 0, 1),),),
            basis_by_dimension=(("v",),),
            coefficient_field=7,
        )


def test_payload_runs_through_cmgdb_relative_homology_bridge_if_installed():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "ComputeRelativeHomologyShiftClass"):
        pytest.skip("installed CMGDB predates the generalized relative bridge")

    _, suspension, _, _ = interval_with_reset(slabs=2)
    pair = RelativeCellPair(suspension, suspension.cells)
    payload = CellularChainMap.identity(suspension).to_cmgdb_payload(pair)

    result = cmgdb.ComputeRelativeHomologyShiftClass(*payload.as_compute_args())

    assert result["coefficient_field"] == 5
    assert result["homology_dimensions"] == [1, 1]
    assert result["shift_class"] == ["x-1", "x-1"]
    assert all(result["validation"].values())


def test_fixed_time_relation_materializes_and_serializes_without_losing_cells():
    _, suspension, _, right = interval_with_reset(slabs=2)
    base = suspension.base_cell(right)
    phase = suspension.prism_cell("impact", right, 0)
    relation = FixedTimeCellRelation(
        suspension,
        (base, phase),
        {base: {phase}, phase: {base}},
    )

    graph = relation.to_networkx()
    payload = relation.to_integer_relation()

    assert set(graph.edges) == {(base, phase), (phase, base)}
    assert payload.as_adjacency_dict() == {0: [1], 1: [0]}
    assert json.loads(payload.to_json())["adjacency"] == [[1], [0]]


def test_expected_public_phase_cell_types_are_used():
    _, suspension, _, right = interval_with_reset(slabs=2)

    assert isinstance(suspension.prism_cell("impact", right, 0), GuardPrismCell)
    assert isinstance(suspension.slice_cell("impact", right, 1), PhaseSliceCell)


def test_sampled_token_adapter_uses_registered_quotient_cells():
    base, suspension, _, right = interval_with_reset(slabs=2)
    adapter = SampledSuspensionCellAdapter(
        suspension,
        base,
        handle_id="impact",
        guard_cell_by_box={0: right},
    )
    base_token = BaseCell(0)
    phase_token = PhaseCell(0, 0)

    relation = adapter.relation_from_tokens(
        (base_token, phase_token),
        {base_token: {phase_token}, phase_token: {base_token}},
    )

    phase_cell = suspension.prism_cell("impact", right, 0)
    assert adapter.cell_for_token(phase_token) is phase_cell
    assert set(relation.to_networkx().edges) == {
        (suspension.base_cell(base.top_cell(0)), phase_cell),
        (phase_cell, suspension.base_cell(base.top_cell(0))),
    }


def test_sampled_token_adapter_unions_sources_collapsed_to_one_guard_prism():
    base = CubicalGridComplex((2,))
    left = CubicalCell((0,), (False,))
    middle = CubicalCell((1,), (False,))
    reset = CellularResetMap.from_cell_map(base, {middle}, {middle: left})
    suspension = SuspensionCellComplex(
        base, [ResetHandle("impact", reset, slabs=2)]
    )
    adapter = SampledSuspensionCellAdapter(
        suspension,
        base,
        handle_id="impact",
        # The middle guard vertex belongs to both adjacent top boxes.
        guard_cell_by_box={0: middle, 1: middle},
    )
    base_0 = BaseCell(0)
    base_1 = BaseCell(1)
    phase_from_0 = PhaseCell(0, 0)
    phase_from_1 = PhaseCell(1, 0)

    relation = adapter.relation_from_tokens(
        (base_0, base_1, phase_from_0, phase_from_1),
        {
            base_0: {phase_from_0},
            base_1: {phase_from_1},
            phase_from_0: {base_0},
            phase_from_1: {base_1},
        },
    )

    prism = suspension.prism_cell("impact", middle, 0)
    assert relation.image(prism) == frozenset(
        {
            suspension.base_cell(base.top_cell(0)),
            suspension.base_cell(base.top_cell(1)),
        }
    )

    with pytest.raises(ValueError, match="must be supplied exactly"):
        adapter.relation_from_tokens(
            (phase_from_0, phase_from_1),
            {phase_from_0: {phase_from_1}},
        )
