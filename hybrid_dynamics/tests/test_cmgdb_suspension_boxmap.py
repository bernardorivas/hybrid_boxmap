"""Tests for the tagged fixed-time suspension callback used by AtlasModel."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    SuspensionAtlasCharts,
    build_cmgdb_atlas_model,
)
from hybrid_dynamics.src.hybrid_system import HybridSystem
from hybrid_dynamics.src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
)


BASE = 7
HANDLE = 11


def _interior_impact_system() -> HybridSystem:
    """A system whose impact branch is invisible at every box corner.

    The flow is ``x' = 4 y (1-y), y' = 0`` with guard ``x=1``.  On the source
    rectangle ``[0,.1] x [0,1]``, every corner has ``y in {0,1}`` and is fixed.
    The interior line ``y=1/2`` instead has unit x-speed, reaches the guard at
    time ``1-x``, and lies inside the reset handle at suspension time 1.2.
    """

    def ode(_time, state):
        _x, y = state
        return np.asarray((4.0 * y * (1.0 - y), 0.0))

    def event(_time, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1
    return HybridSystem(
        ode=ode,
        event_function=event,
        reset_map=lambda state: np.asarray((0.0, state[1])),
        domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
        event_direction=1,
        max_jumps=4,
    )


def _charts() -> SuspensionAtlasCharts:
    return SuspensionAtlasCharts(
        base_bounds=((0.0, 1.0), (0.0, 1.0)),
        guard_bounds=((0.0, 1.0),),
        guard_coordinates=lambda guard: np.asarray((guard[1],)),
        guard_embedding=lambda intrinsic: np.asarray((1.0, intrinsic[0])),
        base_chart_id=BASE,
        handle_chart_id=HANDLE,
    )


def _adapter(
    t_star: float,
    *,
    samples_per_axis: int = 3,
    diagnostics_limit: int = 2048,
    single_handle_bridge: bool = False,
    single_handle_bridge_max_bisections: int = 12,
) -> CMGDBSuspensionBoxMap:
    return CMGDBSuspensionBoxMap(
        _interior_impact_system(),
        _charts(),
        t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=0.0,
        max_step=0.01,
        diagnostics_limit=diagnostics_limit,
        single_handle_bridge=single_handle_bridge,
        single_handle_bridge_max_bisections=(
            single_handle_bridge_max_bisections
        ),
    )


def _piece_is_point_at(piece, point, *, atol=1e-10):
    _chart, bounds = piece
    dimension = len(bounds) // 2
    return np.allclose(bounds[:dimension], point, atol=atol) and np.allclose(
        bounds[dimension:], point, atol=atol
    )


def _unit_reset_quotient_value_is_connected(pieces, *, atol=1e-9):
    """Connectivity in the base/handle cover with x=1~s=0, x=0~s=1."""

    def intervals(piece):
        chart_id, bounds = piece
        dimension = len(bounds) // 2
        return chart_id, np.asarray(bounds[:dimension]), np.asarray(bounds[dimension:])

    rectangles = [intervals(piece) for piece in pieces]
    adjacency = [set() for _ in rectangles]
    for first_index, (first_chart, first_lower, first_upper) in enumerate(rectangles):
        for second_index in range(first_index + 1, len(rectangles)):
            second_chart, second_lower, second_upper = rectangles[second_index]
            meets = False
            if first_chart == second_chart:
                meets = bool(
                    np.all(first_lower <= second_upper + atol)
                    and np.all(second_lower <= first_upper + atol)
                )
            else:
                base = (
                    (first_lower, first_upper)
                    if first_chart == BASE
                    else (second_lower, second_upper)
                )
                handle = (
                    (first_lower, first_upper)
                    if first_chart == HANDLE
                    else (second_lower, second_upper)
                )
                base_lower, base_upper = base
                handle_lower, handle_upper = handle
                intrinsic_overlaps = bool(
                    base_lower[1] <= handle_upper[0] + atol
                    and handle_lower[0] <= base_upper[1] + atol
                )
                at_guard = (
                    base_lower[0] - atol <= 1.0 <= base_upper[0] + atol
                    and handle_lower[1] - atol <= 0.0 <= handle_upper[1] + atol
                )
                at_reset = (
                    base_lower[0] - atol <= 0.0 <= base_upper[0] + atol
                    and handle_lower[1] - atol <= 1.0 <= handle_upper[1] + atol
                )
                meets = intrinsic_overlaps and (at_guard or at_reset)
            if meets:
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)

    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == len(rectangles)


def test_interior_tensor_sample_recovers_branch_missed_by_every_corner():
    box_map = _adapter(1.2)
    source_bounds = [0.0, 0.0, 0.1, 1.0]

    # This is an analytic counterexample to corner-only sampling: all four
    # corners have zero velocity and remain in the base chart.
    corner_samples = [
        box_map.evaluate_point(BASE, (x, y))
        for x in (0.0, 0.1)
        for y in (0.0, 1.0)
    ]
    assert all(isinstance(sample, BaseSuspensionSample) for sample in corner_samples)

    center = box_map.evaluate_point(BASE, (0.05, 0.5))
    assert isinstance(center, HandleSuspensionSample)
    assert center.phase == pytest.approx(0.25, abs=2e-8)

    pieces = box_map(BASE, source_bounds)
    record = box_map.diagnostics().retained_source_records[-1]

    assert {chart_id for chart_id, _bounds in pieces} == {BASE, HANDLE}
    assert any(
        chart_id == HANDLE
        and bounds[0] <= 0.5 <= bounds[2]
        and bounds[1] <= 0.25 <= bounds[3]
        for chart_id, bounds in pieces
    )
    assert [stratum.kind for stratum in record.interior_only_strata] == ["handle"]
    assert record.guard_face_witnesses == 1
    assert not record.unresolved_stage_edges

    # The observed base(0) <-> handle(0) transition contributes both closed
    # representatives of the quotient guard/phase-zero face.
    assert any(
        chart_id == BASE and _piece_is_point_at(piece, (1.0, 0.5))
        for piece in pieces
        for chart_id in (piece[0],)
    )
    assert any(
        chart_id == HANDLE and _piece_is_point_at(piece, (0.5, 0.0))
        for piece in pieces
        for chart_id in (piece[0],)
    )
    # Face coordinates are incorporated into their incident branch hulls, not
    # merely appended as isolated point pieces.  Consequently the complete
    # tagged value is connected in the reset-glued quotient incidence graph.
    assert _unit_reset_quotient_value_is_connected(pieces)
    assert record.whole_cell_outer_enclosure_certified is False


def test_terminal_handle_transition_inserts_handle_one_and_reset_faces():
    box_map = _adapter(0.4)

    # At source phases .4, .6, .8 the final stages are respectively handle,
    # exactly the quotient terminal face, and post-reset base.  The target
    # union must contain both (guard,1) and reset(guard), without hulling the
    # two coordinate charts together.
    pieces = box_map(HANDLE, [0.5, 0.4, 0.5, 0.8])
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.reset_face_witnesses == 1
    assert any(
        chart_id == HANDLE and _piece_is_point_at(piece, (0.5, 1.0))
        for piece in pieces
        for chart_id in (piece[0],)
    )
    assert any(
        chart_id == BASE and _piece_is_point_at(piece, (0.0, 0.5))
        for piece in pieces
        for chart_id in (piece[0],)
    )


def test_a_past_handle_crossing_does_not_add_an_intermediate_seam():
    box_map = _adapter(0.4)

    # Every endpoint is strictly after the current handle's terminal face.
    # Intermediate orbit seams are not part of a fixed-time image and must not
    # be inserted merely because the trajectory crossed them.
    pieces = box_map(HANDLE, [0.5, 0.8, 0.5, 0.9])
    record = box_map.diagnostics().retained_source_records[-1]

    assert {chart_id for chart_id, _bounds in pieces} == {BASE}
    assert record.reset_face_witnesses == 0


def test_adjacent_samples_that_skip_a_stage_are_flagged_not_hulled():
    box_map = _adapter(2.2)
    pieces = box_map(BASE, [0.0, 0.0, 0.1, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    # Corner bands remain in base stage 0 while the y=.5 interior has passed
    # one handle and lies in base stage 2.  No observed handle endpoint exists
    # between them at this sampling depth, so the event gate must fail loudly.
    assert record.unresolved_stage_edges == ((0, 2),)
    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.single_handle_bridge_attempts == ()
    assert box_map.diagnostics().single_handle_bridge_enabled is False
    assert not record.passed_sampled_event_gate
    assert record.guard_face_witnesses == 0
    assert record.reset_face_witnesses == 0
    assert {chart_id for chart_id, _bounds in pieces} == {BASE}
    assert len(pieces) >= 3  # two stage-0 bands plus the stage-2 component


def test_opt_in_single_handle_bridge_completes_only_the_terminal_value():
    box_map = _adapter(2.2, single_handle_bridge=True)
    pieces = box_map(BASE, [0.0, 0.0, 0.1, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.unresolved_stage_edges == ()
    assert record.passed_sampled_event_gate
    assert record.single_handle_bridge_attempts
    assert all(
        bridge.algorithm_revision == SINGLE_HANDLE_BRIDGE_ALGORITHM
        and bridge.synthesized
        and bridge.lower_stage == 0
        and bridge.skipped_handle_stage == 1
        and bridge.upper_stage == 2
        and bridge.raw_handle_carrier_bounds is not None
        and bridge.emitted_handle_carrier_bounds is not None
        and bridge.to_dict()["terminal_image_carrier_only"] is True
        and bridge.to_dict()["intermediate_time_graph_edges_added"] is False
        for bridge in record.single_handle_bridge_attempts
    )
    assert all(
        json.loads(json.dumps(bridge.to_dict()))["status"] == "synthesized"
        for bridge in record.single_handle_bridge_attempts
    )
    assert {chart_id for chart_id, _bounds in pieces} == {BASE, HANDLE}
    assert any(
        chart_id == HANDLE
        and bounds[1] == pytest.approx(0.0)
        and bounds[3] == pytest.approx(1.0)
        for chart_id, bounds in pieces
    )
    assert _unit_reset_quotient_value_is_connected(pieces)

    diagnostics = box_map.diagnostics()
    assert diagnostics.single_handle_bridge_enabled is True
    assert (
        diagnostics.single_handle_bridge_algorithm_revision
        == SINGLE_HANDLE_BRIDGE_ALGORITHM
    )
    assert diagnostics.single_handle_bridge_max_bisections == 12
    assert diagnostics.single_handle_bridge_assumptions
    assert diagnostics.raw_unresolved_stage_edges == 1
    assert diagnostics.unresolved_stage_edges == 0
    assert diagnostics.single_handle_bridge_attempts == len(
        record.single_handle_bridge_attempts
    )
    assert diagnostics.synthesized_single_handle_bridges == len(
        record.single_handle_bridge_attempts
    )
    assert diagnostics.single_handle_bridge_probe_points >= len(
        record.single_handle_bridge_attempts
    )
    assert diagnostics.single_handle_bridge_failed_probes == 0
    assert diagnostics.single_handle_bridge_records == (record,)


def test_bridge_provenance_is_not_truncated_by_the_ordinary_record_limit():
    box_map = _adapter(
        2.2,
        single_handle_bridge=True,
        diagnostics_limit=0,
    )
    box_map(BASE, [0.0, 0.0, 0.1, 1.0])

    diagnostics = box_map.diagnostics()
    assert diagnostics.retained_source_records == ()
    assert len(diagnostics.single_handle_bridge_records) == 1
    assert diagnostics.single_handle_bridge_records[0].single_handle_bridge_attempts


def test_partial_bridge_success_does_not_clear_a_shared_raw_stage_gap():
    box_map = _adapter(
        2.2,
        single_handle_bridge=True,
        single_handle_bridge_max_bisections=1,
    )
    box_map(BASE, [0.0, 0.0, 0.1, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.unresolved_stage_edges == ((0, 2),)
    assert any(
        bridge.synthesized for bridge in record.single_handle_bridge_attempts
    )
    assert any(
        bridge.status == "intermediate_handle_not_found"
        for bridge in record.single_handle_bridge_attempts
    )
    assert not record.passed_sampled_event_gate


def _one_dimensional_synthetic_skip_adapter(
    *,
    upper_jumps: int = 1,
    emit_intermediate_handle: bool = True,
) -> CMGDBSuspensionBoxMap:
    system = HybridSystem(
        ode=lambda _time, _state: np.asarray((0.0,)),
        event_function=lambda _time, state: float(state[0] - 1.0),
        reset_map=lambda _state: np.asarray((0.0,)),
        domain_bounds=[(0.0, 1.0)],
        max_jumps=4,
    )
    charts = SuspensionAtlasCharts(
        base_bounds=((0.0, 1.0),),
        guard_bounds=(),
        guard_coordinates=lambda _guard: np.asarray((), dtype=float),
        guard_embedding=lambda _intrinsic: np.asarray((1.0,)),
        base_chart_id=BASE,
        handle_chart_id=HANDLE,
    )

    class SyntheticContinuousQuotientMap(CMGDBSuspensionBoxMap):
        def _evaluate_point(self, source_chart_id, point):
            assert source_chart_id == BASE
            coordinate = float(point[0])
            lower_cutoff = 0.74 if emit_intermediate_handle else 0.75
            upper_cutoff = 0.76 if emit_intermediate_handle else 0.75
            lower_state = (
                0.26 + coordinate
                if emit_intermediate_handle
                else 0.2 + coordinate
            )
            if coordinate < lower_cutoff or (
                emit_intermediate_handle and coordinate == lower_cutoff
            ):
                return (
                    BaseSuspensionSample(
                        state=np.asarray((lower_state,)),
                        total_time=2.0,
                        continuous_time=2.0,
                        jumps_completed=0,
                    ),
                    [],
                )
            if coordinate >= upper_cutoff:
                return (
                    BaseSuspensionSample(
                        state=np.asarray((coordinate - upper_cutoff,)),
                        total_time=2.0,
                        continuous_time=2.0 - upper_jumps,
                        jumps_completed=upper_jumps,
                    ),
                    [],
                )
            return (
                HandleSuspensionSample(
                    guard_state=np.asarray((1.0,)),
                    reset_state=np.asarray((0.0,)),
                    phase=(coordinate - 0.74) / 0.02,
                    total_time=2.0,
                    continuous_time=1.5,
                    jump_index=0,
                ),
                [],
            )

    return SyntheticContinuousQuotientMap(
        system,
        charts,
        2.0,
        samples_per_axis=3,
        padding_cells=0.0,
        single_handle_bridge=True,
        single_handle_bridge_max_bisections=8,
    )


def test_one_dimensional_bridge_is_quotient_connected_without_base_convexification():
    box_map = _one_dimensional_synthetic_skip_adapter()
    pieces = box_map(BASE, [0.0, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.unresolved_stage_edges == ()
    assert len(record.single_handle_bridge_attempts) == 1
    bridge = record.single_handle_bridge_attempts[0]
    assert bridge.synthesized
    assert bridge.witness_source_point == pytest.approx((0.75,))
    assert bridge.witness_phase == pytest.approx(0.5)

    base_pieces = [bounds for chart, bounds in pieces if chart == BASE]
    handle_pieces = [bounds for chart, bounds in pieces if chart == HANDLE]
    assert any(bounds[0] <= 1.0 <= bounds[1] for bounds in base_pieces)
    assert any(bounds[0] <= 0.0 <= bounds[1] for bounds in base_pieces)
    assert any(bounds[0] <= 0.0 and bounds[1] >= 1.0 for bounds in handle_pieces)
    # The reset-separated base branches remain distinct; no Euclidean hull
    # from reset x=0 to guard x=1 is manufactured in the base chart.
    assert not any(bounds[0] <= 0.0 and bounds[1] >= 1.0 for bounds in base_pieces)


def test_single_handle_bridge_rejects_a_gap_larger_than_two():
    box_map = _one_dimensional_synthetic_skip_adapter(upper_jumps=2)
    pieces = box_map(BASE, [0.0, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    assert pieces
    assert record.raw_unresolved_stage_edges == ((0, 4),)
    assert record.unresolved_stage_edges == ((0, 4),)
    assert record.single_handle_bridge_attempts == ()
    assert {chart_id for chart_id, _bounds in pieces} == {BASE}


def test_bridge_stays_unresolved_when_bisection_never_observes_a_handle():
    box_map = _one_dimensional_synthetic_skip_adapter(
        emit_intermediate_handle=False
    )
    pieces = box_map(BASE, [0.0, 1.0])
    record = box_map.diagnostics().retained_source_records[-1]

    assert record.raw_unresolved_stage_edges == ((0, 2),)
    assert record.unresolved_stage_edges == ((0, 2),)
    assert len(record.single_handle_bridge_attempts) == 1
    bridge = record.single_handle_bridge_attempts[0]
    assert bridge.status == "intermediate_handle_not_found"
    assert not bridge.synthesized
    assert len(bridge.probes) == 8
    assert all(probe.failure is None for probe in bridge.probes)
    assert {chart_id for chart_id, _bounds in pieces} == {BASE}


def test_all_failed_samples_produce_the_valid_empty_atlas_union():
    box_map = _adapter(1.2)
    pieces = box_map(BASE, [0.0, 0.0, 0.1, 0.0])

    # y=0 is stationary and valid, so first demonstrate that a degenerate
    # source is accepted by the Python callback itself.
    assert pieces

    def exiting_ode(_time, _state):
        return np.asarray((2.0, 0.0))

    system = _interior_impact_system()
    system.ode = exiting_ode
    # Disable the guard so every final state exits x in [0,1].
    system.event_function = lambda _time, _state: 1.0
    exiting = CMGDBSuspensionBoxMap(
        system,
        _charts(),
        1.0,
        samples_per_axis=3,
        padding_cells=0.0,
    )
    assert exiting(BASE, [0.0, 0.0, 0.1, 0.1]) == []
    diagnostics = exiting.diagnostics()
    assert diagnostics.empty_images == 1
    assert diagnostics.failed_samples == 9


def test_handle_endpoint_outside_declared_guard_chart_is_a_diagnostic_failure():
    narrow_handle = SuspensionAtlasCharts(
        base_bounds=((0.0, 1.0), (0.0, 1.0)),
        guard_bounds=((0.0, 0.4),),
        guard_coordinates=lambda guard: np.asarray((guard[1],)),
        guard_embedding=lambda intrinsic: np.asarray((1.0, intrinsic[0])),
        base_chart_id=BASE,
        handle_chart_id=HANDLE,
    )
    box_map = CMGDBSuspensionBoxMap(
        _interior_impact_system(),
        narrow_handle,
        1.2,
        samples_per_axis=3,
        padding_cells=0.0,
        max_step=0.01,
    )

    # Every source sample has y=.5 and reaches the handle, but y=.5 lies
    # outside the declared intrinsic handle patch [0,.4].  Returning a tagged
    # rectangle outside Atlas would silently cover no cell, so this is instead
    # an explicit failed source image and an empty union.
    assert box_map(BASE, [0.0, 0.5, 0.1, 0.5]) == []
    record = box_map.diagnostics().retained_source_records[-1]
    assert len(record.failures) == 9
    assert all("outside target Atlas chart" in failure.reason for failure in record.failures)
    assert not record.passed_sampled_event_gate


def test_public_chart_round_trip_and_endpoint_encoding():
    box_map = _adapter(0.2)
    coordinates = box_map.charts.encode_handle((1.0, 0.35), 0.4)
    guard, reset, phase = box_map.decode_handle(coordinates)

    assert np.allclose(guard, (1.0, 0.35))
    assert np.allclose(reset, (0.0, 0.35))
    assert phase == pytest.approx(0.4)
    sample = box_map.evaluate_point(HANDLE, coordinates)
    chart_id, encoded, stratum = box_map.encode_endpoint(sample)
    assert chart_id == HANDLE
    assert encoded == pytest.approx((0.35, 0.6))
    assert stratum.kind == "handle"


def test_point_map_respects_both_source_quotient_identifications():
    box_map = _adapter(0.4)

    from_base_guard = box_map.evaluate_point(BASE, (1.0, 0.5))
    from_handle_zero = box_map.evaluate_point(HANDLE, (0.5, 0.0))
    from_base_reset = box_map.evaluate_point(BASE, (0.0, 0.5))
    from_handle_one = box_map.evaluate_point(HANDLE, (0.5, 1.0))

    assert isinstance(from_base_guard, HandleSuspensionSample)
    assert isinstance(from_handle_zero, HandleSuspensionSample)
    base_guard_chart, base_guard_coordinates, _ = box_map.encode_endpoint(
        from_base_guard
    )
    handle_zero_chart, handle_zero_coordinates, _ = box_map.encode_endpoint(
        from_handle_zero
    )
    assert base_guard_chart == handle_zero_chart
    assert base_guard_coordinates == pytest.approx(handle_zero_coordinates)
    assert isinstance(from_base_reset, BaseSuspensionSample)
    assert isinstance(from_handle_one, BaseSuspensionSample)
    assert np.allclose(from_base_reset.state, from_handle_one.state, atol=1e-9)


def test_requires_interior_tensor_nodes():
    with pytest.raises(ValueError, match="at least three"):
        _adapter(0.5, samples_per_axis=2)


def test_single_handle_bridge_options_are_strict():
    with pytest.raises(TypeError, match="must be a boolean"):
        CMGDBSuspensionBoxMap(
            _interior_impact_system(),
            _charts(),
            0.5,
            single_handle_bridge=1,
        )
    with pytest.raises(ValueError, match="must be a positive integer"):
        CMGDBSuspensionBoxMap(
            _interior_impact_system(),
            _charts(),
            0.5,
            single_handle_bridge=True,
            single_handle_bridge_max_bisections=0,
        )


def test_callback_runs_through_native_cmgdb_atlas_model():
    cmgdb = pytest.importorskip("CMGDB")
    box_map = _adapter(0.4)
    model = build_cmgdb_atlas_model(box_map, depth=1)

    morse_graph, map_graph = cmgdb.ComputeMorseGraph(model)

    assert model.chart_ids() == [BASE, HANDLE]
    assert map_graph.num_vertices() == 4
    assert morse_graph.num_vertices() >= 1
    assert {
        chart_id
        for node in morse_graph.vertices()
        for chart_id, _bounds in morse_graph.morse_set_chart_boxes(node)
    } == {BASE, HANDLE}
    diagnostics = box_map.diagnostics()
    assert diagnostics.source_boxes > 0
    assert diagnostics.failed_samples == 0
    assert diagnostics.whole_cell_outer_enclosure_certified is False


def test_native_builder_accepts_a_selected_tagged_dyadic_family():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb.AtlasModel, "set_active_subgrid"):
        pytest.skip("installed CMGDB predates the active-subgrid API")
    box_map = _adapter(0.1)
    model = build_cmgdb_atlas_model(
        box_map,
        depth=0,
        active_dyadic_cells=[
            (BASE, 2, (0, 0)),
            (HANDLE, 2, (1, 1)),
        ],
    )

    atlas = model.phaseSpace()
    assert model.active_subgrid_configured()
    assert model.initial_cell_count() == 2
    assert atlas.num_charts() == 2
    assert [atlas.cell(index).chart_id for index in range(2)] == [BASE, HANDLE]
