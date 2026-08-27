from __future__ import annotations

import gzip
import hashlib
import json
import numpy as np
import pytest

from hybrid_dynamics.examples.spiking_neuron import (
    U_MAX,
    U_MIN,
    U_NOTCH,
    V_NOTCH,
    V_PEAK,
    V_RESET,
    SpikingNeuron,
    in_spiking_neuron_domain,
    spiking_neuron_analytic_audit,
)
from hybrid_dynamics.examples.spiking_neuron_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    AtlasNeuronCell,
    NeuronDyadicCell,
    NeuronSuspensionBoxMap,
    SpikingNeuronQuotientIncidence,
    _reference_endpoint_from_base,
    _reference_endpoint_from_handle,
    audit_spiking_neuron_mixed_incidence_parity,
    audit_spiking_neuron_sparse_incidence_parity,
    build_spiking_neuron_adaptive_family,
    encoded_line_for_validation,
    build_spiking_neuron_active_family,
    build_spiking_neuron_atlas_model,
    build_spiking_neuron_quotient_nerve,
    compute_neuron_reference_cycle,
    spiking_neuron_atlas_charts,
    spiking_neuron_atlas_reset_gluing,
    spiking_neuron_dyadic_bounds,
    validate_spiking_neuron_provenance,
)
from hybrid_dynamics.src.sampled_suspension import BaseSuspensionSample


def test_compact_domain_exact_analytic_audit() -> None:
    audit = spiking_neuron_analytic_audit()
    assert audit.passed
    assert audit.guard_v_dot_min == pytest.approx(48.975)
    assert audit.guard_v_dot_max == pytest.approx(53.575)
    assert audit.equilibrium_discriminant == pytest.approx(-52.0)
    assert audit.dwell_time_lower_bound > 1.58


def test_l_shaped_domain_excludes_only_upper_right_notch() -> None:
    assert in_spiking_neuron_domain((-50.0, 500.0))
    assert in_spiking_neuron_domain((0.0, 100.0))
    assert in_spiking_neuron_domain((V_NOTCH, U_MAX))
    assert not in_spiking_neuron_domain((0.0, 200.0))
    assert not in_spiking_neuron_domain((-81.0, 0.0))


def test_neuron_event_direction_is_increasing_and_reset_is_interior() -> None:
    neuron = SpikingNeuron()
    assert neuron.system.event_function.direction == 1
    reset = neuron.system.apply_reset_map(np.asarray((V_PEAK, U_NOTCH)))
    assert tuple(reset) == pytest.approx((V_RESET, U_NOTCH + 100.0))
    assert in_spiking_neuron_domain(reset)
    assert V_RESET < V_NOTCH


@pytest.mark.parametrize(
    ("axis_depth", "base_count", "handle_count"),
    ((6, 705, 1472), (7, 2820, 5888), (8, 11280, 23552)),
)
def test_frozen_active_families_are_complete_cubical_subcomplexes(
    axis_depth: int,
    base_count: int,
    handle_count: int,
) -> None:
    family = build_spiking_neuron_active_family(axis_depth)
    assert family.chart_counts() == {
        BASE_CHART_ID: base_count,
        HANDLE_CHART_ID: handle_count,
    }
    assert len(family.fingerprint) == 64


def test_adaptive_family_replaces_each_parent_by_exactly_four_children() -> None:
    coarse = build_spiking_neuron_active_family(7)
    parent = coarse.cells[300]
    adaptive = build_spiking_neuron_adaptive_family(
        coarse_axis_depth=7,
        refined_parents=(parent,),
    )
    children = tuple(cell for cell in adaptive.cells if cell.axis_depth == 8)
    assert len(adaptive.cells) == len(coarse.cells) + 3
    assert len(children) == 4
    assert adaptive.depth_counts() == {7: len(coarse.cells) - 1, 8: 4}
    parent_bounds = spiking_neuron_dyadic_bounds(parent)
    child_bounds = tuple(spiking_neuron_dyadic_bounds(cell) for cell in children)
    assert min(bounds[0] for bounds in child_bounds) == parent_bounds[0]
    assert min(bounds[1] for bounds in child_bounds) == parent_bounds[1]
    assert max(bounds[2] for bounds in child_bounds) == parent_bounds[2]
    assert max(bounds[3] for bounds in child_bounds) == parent_bounds[3]


def test_boxmap_splits_against_notch_and_emits_both_guard_representatives() -> None:
    setup = build_spiking_neuron_atlas_model(total_depth=12)
    split = setup.box_map._clip_piece(  # noqa: SLF001 - geometric regression
        BASE_CHART_ID,
        (-45.0, 140.0, -35.0, 180.0),
    )
    assert len(split) == 2
    assert (BASE_CHART_ID, [-45.0, 140.0, -40.0, 180.0]) in split
    assert (BASE_CHART_ID, [-40.0, 140.0, -35.0, 160.0]) in split

    pieces = setup.box_map(BASE_CHART_ID, (15.0, -60.0, 20.0, -40.0))
    assert {chart_id for chart_id, _bounds in pieces} == {
        BASE_CHART_ID,
        HANDLE_CHART_ID,
    }
    assert any(
        chart_id == BASE_CHART_ID and bounds[2] == pytest.approx(V_PEAK)
        for chart_id, bounds in pieces
    )
    assert any(
        chart_id == HANDLE_CHART_ID and bounds[1] == pytest.approx(0.0)
        for chart_id, bounds in pieces
    )


def test_neuron_single_source_terminal_carrier_bridge_is_provenance_bearing() -> None:
    box_map = NeuronSuspensionBoxMap(
        SpikingNeuron(max_jumps=10).system,
        spiking_neuron_atlas_charts(),
        20.0,
        samples_per_axis=3,
        padding_cells=1.0,
        max_jumps=10,
        max_step=0.02,
        require_domain_path=True,
        single_handle_bridge=True,
        single_handle_bridge_max_bisections=12,
    )
    pieces = box_map(
        BASE_CHART_ID,
        (-47.5, -55.0, -46.25, -50.0),
    )
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.unresolved_stage_edges == ()
    assert len(record.single_handle_bridge_attempts) == 1
    bridge = record.single_handle_bridge_attempts[0]
    assert bridge.synthesized
    assert bridge.incident_grid_indices == ((0, 1), (1, 1))
    assert bridge.witness_source_point == pytest.approx((-47.1875, -52.5))
    assert bridge.raw_handle_carrier_bounds[1::2] == pytest.approx((0.0, 1.0))
    encoded = bridge.to_dict()
    assert encoded["terminal_image_carrier_only"] is True
    assert encoded["intermediate_time_graph_edges_added"] is False
    json.dumps(encoded, allow_nan=False)

    rectangles = tuple(
        AtlasNeuronCell(index, chart_id, tuple(bounds))
        for index, (chart_id, bounds) in enumerate(pieces)
    )
    components = SpikingNeuronQuotientIncidence(
        spiking_neuron_atlas_charts()
    ).connected_components(rectangles)
    assert len(components) == 1


@pytest.mark.parametrize("total_depth", (12, 14))
def test_sparse_incidence_has_exact_full_family_parity(total_depth: int) -> None:
    setup = build_spiking_neuron_atlas_model(total_depth=total_depth)
    atlas = setup.model.phaseSpace()
    cells = tuple(
        AtlasNeuronCell(
            index,
            int(atlas.cell(index).chart_id),
            tuple(float(value) for value in atlas.cell(index).bounds),
        )
        for index in range(int(atlas.size()))
    )
    audit = audit_spiking_neuron_sparse_incidence_parity(
        cells, setup.charts, axis_depth=total_depth // 2
    )
    assert audit.passed
    assert audit.compared_cell_rows == len(cells)


def test_interior_reset_quotient_incidence_is_exact() -> None:
    incidence = SpikingNeuronQuotientIncidence(spiking_neuron_atlas_charts())
    handle_top = AtlasNeuronCell(0, HANDLE_CHART_ID, (-40.0, 0.98, -20.0, 1.0))
    reset_base = AtlasNeuronCell(1, BASE_CHART_ID, (-50.0, 60.0, -45.0, 80.0))
    far_base = AtlasNeuronCell(2, BASE_CHART_ID, (-50.0, 300.0, -45.0, 320.0))
    assert incidence.intersects(handle_top, reset_base)
    assert not incidence.intersects(handle_top, far_base)

    gluing = spiking_neuron_atlas_reset_gluing()
    assert gluing.guard.point(-40.0) == pytest.approx((35.0, -40.0))
    assert gluing.reset.point(-40.0) == pytest.approx((-50.0, 60.0))


def test_mixed_sparse_incidence_has_exact_reset_seam_parity() -> None:
    addresses = (
        NeuronDyadicCell(HANDLE_CHART_ID, 8, (80, 255)),
        NeuronDyadicCell(BASE_CHART_ID, 8, (55, 100)),
        NeuronDyadicCell(BASE_CHART_ID, 8, (56, 100)),
        # This unused guard-face cell makes v=35 the selected base boundary;
        # only reset is intentionally treated as an interior seam.
        NeuronDyadicCell(BASE_CHART_ID, 8, (123, 80)),
    )
    cells = tuple(
        AtlasNeuronCell(index, address.chart_id, spiking_neuron_dyadic_bounds(address))
        for index, address in enumerate(addresses)
    )
    audit = audit_spiking_neuron_mixed_incidence_parity(
        cells, spiking_neuron_atlas_charts(), fine_axis_depth=8
    )
    assert audit.passed
    assert audit.leaf_depth_counts == ((8, 4),)
    assert audit.sparse_undirected_edges == 3
    nerve = build_spiking_neuron_quotient_nerve(cells)
    seam = nerve.seam_subcomplex_audits["reset"]
    assert seam.no_base_top_cell_straddles
    assert seam.complete_handle_face_attachment_cover
    assert seam.all_incident_top_cofaces_present


def test_neuron_nerve_constructor_certifies_two_sided_interior_reset() -> None:
    cells = (
        AtlasNeuronCell(0, BASE_CHART_ID, (30.0, -40.0, 35.0, -20.0)),
        AtlasNeuronCell(1, BASE_CHART_ID, (-55.0, 60.0, -50.0, 80.0)),
        AtlasNeuronCell(2, BASE_CHART_ID, (-50.0, 60.0, -45.0, 80.0)),
        AtlasNeuronCell(3, HANDLE_CHART_ID, (-40.0, 0.0, -20.0, 1.0)),
    )
    nerve = build_spiking_neuron_quotient_nerve(cells)
    audit = nerve.seam_subcomplex_audits["reset"]
    assert audit.no_base_top_cell_straddles
    assert audit.complete_handle_face_attachment_cover
    assert audit.all_incident_top_cofaces_present

    with pytest.raises(ValueError, match="missing incident base top cofaces"):
        build_spiking_neuron_quotient_nerve(tuple(cell for cell in cells if cell.index != 2))


def test_reference_cycle_closes_under_reset_and_obeys_dwell_bound() -> None:
    cycle = compute_neuron_reference_cycle()
    assert cycle.pre_reset_u + 100.0 == pytest.approx(
        cycle.post_reset_u, abs=1.0e-8
    )
    assert cycle.flight_time > spiking_neuron_analytic_audit().dwell_time_lower_bound
    assert U_MIN < cycle.u_range[0] <= cycle.u_range[1] < U_MAX
    assert cycle.v_range[1] == pytest.approx(V_PEAK, abs=1.0e-8)


@pytest.mark.parametrize("t_star", (20.0, 40.0))
def test_scientific_clock_reference_endpoints_wrap_through_handle(
    t_star: float,
) -> None:
    cycle = compute_neuron_reference_cycle()
    endpoint = _reference_endpoint_from_base(
        cycle, cycle.flight_time - 5.0, t_star
    )
    assert isinstance(endpoint, BaseSuspensionSample)
    expected_clock = (cycle.flight_time - 5.0 + t_star) % cycle.suspension_period
    assert endpoint.state == pytest.approx(cycle.solution(expected_clock), abs=1.0e-9)

    handle_endpoint = _reference_endpoint_from_handle(cycle, 0.8, t_star)
    assert isinstance(handle_endpoint, BaseSuspensionSample)
    handle_clock = (
        cycle.flight_time + 0.8 + t_star
    ) % cycle.suspension_period
    assert handle_endpoint.state == pytest.approx(
        cycle.solution(handle_clock), abs=1.0e-9
    )


def test_diagnostic_provenance_trailer_is_strict_and_csr_bound(tmp_path) -> None:
    records = (
        {"record": "header", "schema": "test"},
        {"record": "source", "index": 0},
    )
    content = b"".join(encoded_line_for_validation(record) for record in records)
    trailer = {
        "record": "trailer",
        "schema": "spiking-neuron-source-provenance-trailer-v1",
        "checkpoint_complete": True,
        "diagnostic_only": True,
        "source_records": 1,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "relation_csr_fingerprint": "csr-test",
    }
    trailer["fingerprint"] = hashlib.sha256(
        json.dumps(
            trailer,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "provenance.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(content)
        stream.write(encoded_line_for_validation(trailer))
    reference = validate_spiking_neuron_provenance(
        path, expected_relation_csr_fingerprint="csr-test"
    )
    assert reference["source_records"] == 1
    assert reference["diagnostic_only"] is True
    with pytest.raises(ValueError, match="different CSR"):
        validate_spiking_neuron_provenance(
            path, expected_relation_csr_fingerprint="other"
        )
