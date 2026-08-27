"""Algebraic and physical-geometry tests for the honest Atlas Conley front end."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from hybrid_dynamics import (
    AffineBoundaryEmbedding2D,
    AtlasGoodCoverError,
    AtlasMappingCylinderRegistry,
    AtlasMappingCylinderRelativePair,
    AtlasMappingCylinderTopCell,
    AtlasNerveSimplex,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    CrossComplexAcyclicCarrier,
    CubicalCell,
    CubicalGridComplex,
    DoubleMappingCylinderComplex,
    DoubleMappingCylinderHandle,
    PhysicalConleyCertificationError,
    audit_cubical_hyperplane_attachment,
    atlas_cells_from_phase_space,
    prepare_atlas_mapping_cylinder_conley,
    prepare_atlas_physical_conley_2d,
    prepare_atlas_relation_conley_2d,
)
from hybrid_dynamics.examples.bouncing_ball_atlas import (
    bouncing_ball_atlas_reset_gluing,
    build_bouncing_ball_atlas_model,
)
from hybrid_dynamics.examples.rimless_wheel_atlas import (
    build_rimless_wheel_atlas_model,
    rimless_wheel_atlas_reset_gluing,
)


def _four_cell_quotient() -> AtlasQuotientNerveComplex2D:
    # Two actual base rectangles and two actual phase-slab rectangles.  The
    # handle bottom attaches to x=1, y in [0,1], while its top attaches to
    # x=0, y in [1,2].
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


def _interior_reset_quotient() -> AtlasQuotientNerveComplex2D:
    # The guard is the outer face x=1.  The reset x=0 is an interior grid
    # hyperplane shared by base cells on both sides.  Two base faces cover each
    # of the two selected handle top faces exactly.
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 0.0, 0.5)),
        AtlasRectangleCell2D(1, 0, (-1.0, 0.5, 0.0, 1.0)),
        AtlasRectangleCell2D(2, 0, (0.0, 0.0, 1.0, 0.5)),
        AtlasRectangleCell2D(3, 0, (0.0, 0.5, 1.0, 1.0)),
        AtlasRectangleCell2D(4, 1, (0.0, 0.0, 0.5, 0.5)),
        AtlasRectangleCell2D(5, 1, (0.5, 0.0, 1.0, 0.5)),
        AtlasRectangleCell2D(6, 1, (0.0, 0.5, 0.5, 1.0)),
        AtlasRectangleCell2D(7, 1, (0.5, 0.5, 1.0, 1.0)),
    )
    return AtlasQuotientNerveComplex2D(
        cells,
        AtlasResetGluing2D(
            0,
            1,
            guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
            reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
        ),
        interior_seam_subcomplexes=("reset",),
    )


def _four_dimensional_atlas_mapping_cylinder():
    base = CubicalGridComplex((1, 1, 1, 1))
    guard = CubicalGridComplex((1, 1, 1))

    def face(cell: CubicalCell, side: int) -> CubicalCell:
        return CubicalCell(
            (side, *cell.anchor),
            (False, *cell.spanning),
        )

    bottom = CrossComplexAcyclicCarrier(
        guard,
        base,
        {cell: (face(cell, 0),) for cell in guard.cells},
    ).construct_integral_attachment_map()
    reset = CrossComplexAcyclicCarrier(
        guard,
        base,
        {cell: (face(cell, 1),) for cell in guard.cells},
    ).construct_integral_attachment_map()
    cylinder = DoubleMappingCylinderComplex(
        base,
        [DoubleMappingCylinderHandle("impact", guard, bottom, reset, slabs=2)],
    )
    guard_top = guard.top_cell(0)
    registry = AtlasMappingCylinderRegistry(
        cylinder,
        (
            AtlasMappingCylinderTopCell(10, 0, cylinder.base_cell(base.top_cell(0))),
            AtlasMappingCylinderTopCell(
                20, 1, cylinder.prism_cell("impact", guard_top, 0)
            ),
            AtlasMappingCylinderTopCell(
                21, 1, cylinder.prism_cell("impact", guard_top, 1)
            ),
        ),
        base_chart_id=0,
        handle_chart_id=1,
    )
    return base, cylinder, registry


def _guard_aligned_four_dimensional_mapping_cylinder():
    # Coordinates are (theta, omega, q, nu).  q=1 is the interior grid
    # hyperplane representing physical q=0.  Guard and reset attach to
    # disjoint theta patches [0,1] and [2,3] of that same hyperplane.
    base = CubicalGridComplex((3, 1, 2, 1))
    guard = CubicalGridComplex((1, 1, 1))

    def patch(cell: CubicalCell, theta_offset: int) -> CubicalCell:
        return CubicalCell(
            (
                theta_offset + cell.anchor[0],
                cell.anchor[1],
                1,
                cell.anchor[2],
            ),
            (
                cell.spanning[0],
                cell.spanning[1],
                False,
                cell.spanning[2],
            ),
        )

    guard_carrier = CrossComplexAcyclicCarrier(
        guard,
        base,
        {cell: (patch(cell, 0),) for cell in guard.cells},
    )
    reset_carrier = CrossComplexAcyclicCarrier(
        guard,
        base,
        {cell: (patch(cell, 2),) for cell in guard.cells},
    )
    guard_attachment = guard_carrier.construct_integral_attachment_map()
    reset_attachment = reset_carrier.construct_integral_attachment_map()
    guard_audit = audit_cubical_hyperplane_attachment(
        guard_carrier,
        guard_attachment,
        axis=2,
        coordinate=1,
    )
    reset_audit = audit_cubical_hyperplane_attachment(
        reset_carrier,
        reset_attachment,
        axis=2,
        coordinate=1,
    )
    cylinder = DoubleMappingCylinderComplex(
        base,
        (
            DoubleMappingCylinderHandle(
                "guard-aligned-impact",
                guard,
                guard_attachment,
                reset_attachment,
                slabs=2,
            ),
        ),
    )

    records = []
    atlas_index_by_anchor = {}
    for atlas_index, top_cell in enumerate(base.cells_of_dimension(4), start=10):
        atlas_index_by_anchor[top_cell.anchor] = atlas_index
        records.append(
            AtlasMappingCylinderTopCell(
                atlas_index,
                0,
                cylinder.base_cell(top_cell),
            )
        )
    guard_top = guard.top_cell(0)
    records.extend(
        (
            AtlasMappingCylinderTopCell(
                100,
                1,
                cylinder.prism_cell("guard-aligned-impact", guard_top, 0),
            ),
            AtlasMappingCylinderTopCell(
                101,
                1,
                cylinder.prism_cell("guard-aligned-impact", guard_top, 1),
            ),
        )
    )
    registry = AtlasMappingCylinderRegistry(
        cylinder,
        records,
        base_chart_id=0,
        handle_chart_id=1,
    )
    return (
        base,
        cylinder,
        registry,
        atlas_index_by_anchor,
        guard_audit,
        reset_audit,
    )


class _ClosedRectangleAtlas:
    def __init__(self, cells: Sequence[AtlasRectangleCell2D]) -> None:
        self.cells = tuple(cells)

    def cover(self, chart_id: int, bounds: Sequence[float]) -> list[int]:
        x0, y0, x1, y1 = (float(value) for value in bounds)
        return [
            cell.index
            for cell in self.cells
            if cell.chart_id == int(chart_id)
            and cell.bounds[0] <= x1
            and x0 <= cell.bounds[2]
            and cell.bounds[1] <= y1
            and y0 <= cell.bounds[3]
        ]


def test_actual_atlas_nerve_uses_reset_quotient_and_has_d_squared_zero():
    nerve = _four_cell_quotient()

    assert len(nerve.cells_of_dimension(0)) == 4
    assert nerve.simplex({0, 2}) in nerve.cell_set  # bottom/guard gluing
    assert nerve.simplex({1, 3}) in nerve.cell_set  # top/reset gluing
    assert nerve.metadata["finite_intersections_verified_contractible"] is True
    assert nerve.metadata["kind"] == "actual-atlas-reset-quotient-nerve"

    # FiniteCellComplex already rejects d^2 != 0.  Check it explicitly too so
    # this physical adapter cannot silently bypass the algebraic gate.
    for cell in nerve.cells:
        second_boundary: dict[AtlasNerveSimplex, int] = {}
        for face, first_coefficient in nerve.boundary(cell).items():
            for subface, second_coefficient in nerve.boundary(face).items():
                second_boundary[subface] = (
                    second_boundary.get(subface, 0)
                    + first_coefficient * second_coefficient
                )
        assert all(value == 0 for value in second_boundary.values())


def test_interior_reset_requires_opt_in_verified_subcomplex_certificate():
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 0.0, 1.0)),
        AtlasRectangleCell2D(1, 0, (0.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(2, 1, (0.0, 0.0, 1.0, 0.5)),
        AtlasRectangleCell2D(3, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(AtlasGoodCoverError, match="reset seam is not on the boundary"):
        AtlasQuotientNerveComplex2D(cells, gluing)

    nerve = _interior_reset_quotient()
    audit = nerve.seam_subcomplex_audits["reset"]
    assert audit.no_base_top_cell_straddles
    assert audit.complete_handle_face_attachment_cover
    assert audit.all_incident_top_cofaces_present
    assert audit.base_face_cell_indices == (0, 1, 2, 3)
    assert audit.negative_side_base_cell_indices == (0, 1)
    assert audit.positive_side_base_cell_indices == (2, 3)
    assert audit.handle_face_cell_indices == (6, 7)
    assert nerve.metadata["interior_seam_subcomplexes"] == ("reset",)
    assert nerve.simplex({0, 2, 6}) in nerve.cell_set


@pytest.mark.parametrize(
    ("requested", "error_type", "message"),
    (
        ("reset", TypeError, "collection of seam names"),
        (("reset", "reset"), ValueError, "duplicate-free"),
        (("other",), ValueError, "unknown interior seam names"),
    ),
)
def test_interior_seam_opt_in_has_strict_canonical_names(
    requested, error_type, message
):
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 0.0, 1.0)),
        AtlasRectangleCell2D(1, 0, (0.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(2, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(error_type, match=message):
        AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            interior_seam_subcomplexes=requested,
        )


def test_boundary_seam_cannot_be_declared_interior():
    cells = (
        AtlasRectangleCell2D(0, 0, (0.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(1, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="declared interior but lies on the boundary"):
        AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            interior_seam_subcomplexes=("reset",),
        )


def test_interior_reset_rejects_straddling_base_top_cell():
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(1, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(AtlasGoodCoverError, match="crossed by base top cells"):
        AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            interior_seam_subcomplexes=("reset",),
        )


def test_interior_reset_rejects_incomplete_handle_face_attachment_cover():
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 0.0, 0.4)),
        AtlasRectangleCell2D(1, 0, (0.0, 0.0, 1.0, 0.4)),
        AtlasRectangleCell2D(2, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(AtlasGoodCoverError, match="not completely covered"):
        AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            interior_seam_subcomplexes=("reset",),
        )


def test_interior_reset_rejects_one_sided_local_seam_support():
    # The union of seam faces covers the handle endpoint, but the lower half
    # has only a negative-side coface and the upper half only a positive-side
    # coface.  Treating that union as an interior subcomplex would silently
    # turn a local open boundary into a quotient attachment.
    cells = (
        AtlasRectangleCell2D(0, 0, (-1.0, 0.0, 0.0, 0.5)),
        AtlasRectangleCell2D(1, 0, (0.0, 0.5, 1.0, 1.0)),
        AtlasRectangleCell2D(2, 1, (0.0, 0.5, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(AtlasGoodCoverError, match="missing incident base top cofaces"):
        AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            interior_seam_subcomplexes=("reset",),
        )


def test_interior_reset_relation_pipeline_checks_every_algebraic_gate():
    nerve = _interior_reset_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0})
    relation = {source: {0} for source in nerve.atlas_indices}
    preparation = prepare_atlas_relation_conley_2d(
        pair,
        top_relation=relation,
        use_exit_component_carrier=False,
    )

    assert preparation.carrier.acyclicity_validated
    assert preparation.carrier.preserves_pair(pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    assert preparation.chain_map.preserves(pair.relative_pair.p0_cells)
    assert preparation.finite_relation_algebra_validated
    for cell in pair.complex.cells:
        second_boundary: dict[object, int] = {}
        for face, coefficient in pair.complex.boundary(cell).items():
            for subface, incidence in pair.complex.boundary(face).items():
                second_boundary[subface] = (
                    second_boundary.get(subface, 0) + coefficient * incidence
                )
                if second_boundary[subface] == 0:
                    second_boundary.pop(subface)
        assert second_boundary == {}


def test_base_only_induced_family_needs_no_dummy_handle_cell():
    nerve = AtlasQuotientNerveComplex2D(
        (
            AtlasRectangleCell2D(4, 0, (0.0, 0.0, 1.0, 1.0)),
            AtlasRectangleCell2D(7, 0, (1.0, 0.0, 2.0, 1.0)),
        ),
        AtlasResetGluing2D(
            0,
            1,
            guard=AffineBoundaryEmbedding2D(0, 2.0, 1.0),
            reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
        ),
    )

    assert nerve.simplex({4, 7}) in nerve.cell_set
    assert nerve.metadata["chart_cell_counts"] == {0: 2}


def test_disconnected_guard_and_reset_intersection_is_a_hard_good_cover_failure():
    cells = (
        AtlasRectangleCell2D(0, 0, (0.0, 0.0, 1.0, 1.0)),
        AtlasRectangleCell2D(1, 1, (0.0, 0.0, 1.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 1.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
    )

    with pytest.raises(AtlasGoodCoverError, match="disconnected seam components"):
        AtlasQuotientNerveComplex2D(cells, gluing)


def test_ball_handle_cells_meet_through_bottom_top_quotient_at_rest():
    cells = (
        AtlasRectangleCell2D(0, 0, (0.0, -2.0, 1.0, 2.0)),
        AtlasRectangleCell2D(1, 1, (-1.0, 0.0, 0.0, 0.4)),
        AtlasRectangleCell2D(2, 1, (-1.0, 0.6, 0.0, 1.0)),
    )
    nerve = AtlasQuotientNerveComplex2D(
        cells,
        AtlasResetGluing2D(
            0,
            1,
            guard=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
            reset=AffineBoundaryEmbedding2D(0, 0.0, -0.8),
        ),
    )

    assert nerve.simplex({1, 2}) in nerve.cell_set
    assert nerve.simplex({0, 1, 2}) in nerve.cell_set
    audit = nerve.intersection_audit(nerve.simplex({0, 1, 2}))
    assert audit.contractible
    assert any(representative.bounds == (0.0, 0.0, 0.0, 0.0) for representative in audit.representatives)


def test_unsplit_ball_handle_is_rejected_when_rest_corners_are_identified():
    cells = (
        AtlasRectangleCell2D(0, 0, (0.0, -1.0, 1.0, 1.0)),
        AtlasRectangleCell2D(1, 1, (-1.0, 0.0, 0.0, 1.0)),
    )
    gluing = AtlasResetGluing2D(
        0,
        1,
        guard=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, -0.8),
    )

    with pytest.raises(AtlasGoodCoverError, match="spans both seams"):
        AtlasQuotientNerveComplex2D(cells, gluing)


def test_face_raw_covers_construct_a_nonidentity_subordinate_chain_map():
    nerve = _four_cell_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0})
    atlas = _ClosedRectangleAtlas(nerve.atlas_cells)
    target_point = (0.25, 0.25, 0.25, 0.25)

    def constant_physical_map(_chart_id: int, _bounds: Sequence[float]):
        return [(0, target_point)]

    expected_relation = {index: {0} for index in nerve.atlas_indices}
    preparation = prepare_atlas_physical_conley_2d(
        pair,
        box_map=constant_physical_map,
        atlas=atlas,
        expected_top_relation=expected_relation,
    )

    assert len(preparation.raw_covers) == len(pair.complex.cells)
    assert any(source.dimension > 0 for source in preparation.raw_covers)
    assert preparation.carrier.acyclicity_validated
    assert preparation.carrier.preserves_pair(pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    assert preparation.chain_map.preserves(pair.relative_pair.p1_cells)
    assert preparation.chain_map.preserves(pair.relative_pair.p0_cells)
    assert any(
        not preparation.chain_map.image(cell)
        for cell in pair.complex.cells
        if pair.complex.dimension(cell) > 0
    )
    assert preparation.payload.cell_counts == pair.relative_pair.cell_counts
    assert preparation.finite_relation_algebra_validated
    assert not preparation.continuous_system_conley_index_certified

    cmgdb = pytest.importorskip("CMGDB")
    if hasattr(cmgdb, "ComputeRelativeHomologyShiftClass"):
        finite_result = preparation.compute_finite_relation_shift_class()
        assert finite_result["result_scope"] == "finite_reset_quotient_relation"
        assert finite_result["finite_relation_algebra_validated"] is True
        assert finite_result["continuous_system_conley_index_certified"] is False
        assert all(finite_result["validation"].values())

    with pytest.raises(PhysicalConleyCertificationError, match="outer enclosure"):
        preparation.compute_cmgdb_shift_class()


def test_mapgraph_provenance_and_pair_preservation_are_checked_before_selection():
    nerve = _four_cell_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0})
    atlas = _ClosedRectangleAtlas(nerve.atlas_cells)

    def to_zero(_chart_id: int, _bounds: Sequence[float]):
        return [(0, (0.25, 0.25, 0.25, 0.25))]

    wrong_relation = {index: {0} for index in nerve.atlas_indices}
    wrong_relation[1] = {1}
    with pytest.raises(ValueError, match="does not reproduce MapGraph"):
        prepare_atlas_physical_conley_2d(
            pair,
            box_map=to_zero,
            atlas=atlas,
            expected_top_relation=wrong_relation,
        )

    leaves_p0 = {index: {0} for index in nerve.atlas_indices}
    leaves_p0[0] = {1}
    with pytest.raises(ValueError, match="does not preserve P0"):
        prepare_atlas_physical_conley_2d(
            pair,
            box_map=to_zero,
            atlas=atlas,
            expected_top_relation=leaves_p0,
        )


def test_relation_carrier_uses_connected_exit_components_without_identity_map():
    nerve = _four_cell_quotient()
    pair = AtlasRelativeIndexPair2D(nerve, nerve.atlas_indices, {0, 1})
    relation = {
        0: {2},  # The raw relation leaves P0; the relative extension replaces it.
        1: {3},
        2: {0},
        3: {0},
    }

    preparation = prepare_atlas_relation_conley_2d(
        pair,
        top_relation=relation,
        use_exit_component_carrier=True,
    )

    assert preparation.relation_vertex_images[0] == frozenset({0, 1})
    assert preparation.relation_vertex_images[1] == frozenset({0, 1})
    assert preparation.relation_vertex_images[2] == frozenset({0})
    assert "P0-component" in preparation.carrier_construction
    assert preparation.carrier.preserves_pair(pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    assert preparation.chain_map.preserves(pair.relative_pair.p0_cells)

    with pytest.raises(ValueError, match="does not preserve P0"):
        prepare_atlas_relation_conley_2d(
            pair,
            top_relation=relation,
            use_exit_component_carrier=False,
        )


def test_four_dimensional_atlas_mapping_cylinder_relative_chain_pipeline():
    _base, cylinder, registry = _four_dimensional_atlas_mapping_cylinder()
    pair = AtlasMappingCylinderRelativePair(
        registry,
        registry.atlas_indices,
        {10},
    )
    target = registry.complex_cell(10)
    relation = {10: {10}, 20: {10}, 21: {10}}
    preparation = prepare_atlas_mapping_cylinder_conley(
        pair,
        top_relation=relation,
        cell_carrier_generators={cell: (target,) for cell in pair.complex.cells},
    )

    assert cylinder.metadata["kind"] == "independent-guard-double-mapping-cylinder"
    assert preparation.carrier.acyclicity_validated
    assert preparation.carrier.preserves_pair(pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    assert preparation.chain_map.preserves(pair.relative_pair.p1_cells)
    assert preparation.chain_map.preserves(pair.relative_pair.p0_cells)
    assert preparation.finite_relation_algebra_validated
    assert not preparation.continuous_system_conley_index_certified
    assert preparation.payload.cell_counts == pair.relative_pair.cell_counts
    assert any(
        preparation.chain_map.image(cell) != {cell: 1}
        for cell in pair.complex.cells
        if cell not in pair.relative_pair.p0_cells
    )

    cmgdb = pytest.importorskip("CMGDB")
    if hasattr(cmgdb, "ComputeRelativeHomologyShiftClass"):
        result = preparation.compute_finite_relation_shift_class()
        assert result["result_scope"] == "finite_reset_quotient_relation"
        assert result["finite_relation_algebra_validated"] is True
        assert all(result["validation"].values())


def test_guard_aligned_four_dimensional_disjoint_interior_attachments():
    (
        base,
        cylinder,
        registry,
        atlas_index_by_anchor,
        guard_audit,
        reset_audit,
    ) = _guard_aligned_four_dimensional_mapping_cylinder()

    assert guard_audit.interior and reset_audit.interior
    assert guard_audit.no_base_top_cell_straddles
    assert reset_audit.no_base_top_cell_straddles
    assert guard_audit.complete_source_cell_images
    assert reset_audit.complete_source_cell_images
    assert guard_audit.all_incident_top_cofaces_present
    assert reset_audit.all_incident_top_cofaces_present
    assert guard_audit.carrier_face_nesting_and_acyclicity_verified
    assert reset_audit.carrier_face_nesting_and_acyclicity_verified
    assert guard_audit.integral_chain_map_verified
    assert reset_audit.integral_chain_map_verified
    assert guard_audit.carrier_subordination_verified
    assert reset_audit.carrier_subordination_verified
    assert set(guard_audit.target_support_cells).isdisjoint(
        reset_audit.target_support_cells
    )
    assert len(guard_audit.incident_base_top_cells) == 2
    assert len(reset_audit.incident_base_top_cells) == 2

    selected_base_anchors = {
        (theta, 0, q_side, 0)
        for theta in (0, 2)
        for q_side in (0, 1)
    }
    selected = {
        atlas_index_by_anchor[anchor] for anchor in selected_base_anchors
    } | {100, 101}
    gap = {
        atlas_index_by_anchor[(1, 0, q_side, 0)] for q_side in (0, 1)
    }
    p0 = {atlas_index_by_anchor[(0, 0, 0, 0)]}
    pair = AtlasMappingCylinderRelativePair(registry, selected, p0)

    # Cellular closure adds faces, never remote top-dimensional cofaces.  In
    # particular, sharing q=0 does not percolate the local attachments through
    # the unselected theta-gap boxes.
    assert pair.p1_atlas_cells == frozenset(selected)
    assert pair.p1_atlas_cells.isdisjoint(gap)
    assert all(
        registry.complex_cell(index) not in pair.complex.cell_set for index in gap
    )

    target_index = next(iter(p0))
    target = registry.complex_cell(target_index)
    preparation = prepare_atlas_mapping_cylinder_conley(
        pair,
        top_relation={source: {target_index} for source in selected},
        cell_carrier_generators={cell: (target,) for cell in pair.complex.cells},
    )
    assert preparation.carrier.acyclicity_validated
    assert preparation.carrier.preserves_pair(pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    assert preparation.chain_map.preserves(pair.relative_pair.p0_cells)
    assert preparation.finite_relation_algebra_validated
    assert cylinder.metadata["integral_attachment_maps_verified"] is True

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


def test_mapping_cylinder_preserves_raw_p0_exits_without_fake_component_edges():
    _base, cylinder, registry = _four_dimensional_atlas_mapping_cylinder()
    pair = AtlasMappingCylinderRelativePair(
        registry,
        registry.atlas_indices,
        {10},
    )
    target = registry.complex_cell(10)
    preparation = prepare_atlas_mapping_cylinder_conley(
        pair,
        top_relation={10: {999}, 20: {10}, 21: {10}},
        cell_carrier_generators={cell: (target,) for cell in pair.complex.cells},
    )

    assert preparation.relation_vertex_images[10] == frozenset({999})
    assert "algebraic extension on P0" in preparation.carrier_construction

    with pytest.raises(ValueError, match=r"enters P1\\P0"):
        prepare_atlas_mapping_cylinder_conley(
            pair,
            top_relation={10: {20}, 20: {10}, 21: {10}},
            cell_carrier_generators={cell: (target,) for cell in pair.complex.cells},
        )


def test_mapping_cylinder_carrier_must_contain_actual_mapgraph_targets():
    _base, cylinder, registry = _four_dimensional_atlas_mapping_cylinder()
    pair = AtlasMappingCylinderRelativePair(registry, registry.atlas_indices)
    target_vertex = cylinder.base_cell(
        CubicalCell((0, 0, 0, 0), (False, False, False, False))
    )

    with pytest.raises(ValueError, match="does not contain the actual MapGraph"):
        prepare_atlas_mapping_cylinder_conley(
            pair,
            top_relation={10: {10}, 20: {10}, 21: {10}},
            cell_carrier_generators={
                cell: (target_vertex,) for cell in pair.complex.cells
            },
        )

@pytest.mark.parametrize("model_name", ["ball", "wheel"])
def test_physical_two_dimensional_atlas_has_a_verified_actual_cell_nerve(model_name):
    pytest.importorskip("CMGDB")
    if model_name == "ball":
        setup = build_bouncing_ball_atlas_model(depth=4)
        gluing = bouncing_ball_atlas_reset_gluing(
            restitution=setup.ball.c,
        )
    else:
        setup = build_rimless_wheel_atlas_model(depth=4)
        gluing = rimless_wheel_atlas_reset_gluing(
            alpha=setup.wheel.alpha,
            gamma=setup.wheel.gamma,
        )

    phase_space = setup.model.phaseSpace()
    # AtlasModel stores chart roots and applies its declared depth inside
    # ComputeMorseGraph.  Reproduce that fixed-depth geometry here without
    # evaluating the physical box map.
    for _ in range(4):
        phase_space.subdivide()
    cells = atlas_cells_from_phase_space(phase_space)
    nerve = AtlasQuotientNerveComplex2D(cells, gluing)

    assert len(cells) == 32
    assert len(nerve.cells_of_dimension(0)) == 32
    assert nerve.metadata["finite_intersections_verified_contractible"] is True
    assert {cell.chart_id for cell in nerve.atlas_cells} == {0, 1}
