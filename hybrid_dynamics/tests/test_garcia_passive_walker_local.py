"""Tests for the exit-aware local Garcia walker relation workflow."""

from __future__ import annotations

import gzip
import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_dynamics.examples.garcia_passive_walker_local import (
    _quotient_cross_chart_pairs,
    _relation_strongly_connected_components,
    _validate_raw_morse_snapshot,
    GarciaDyadicCell,
    GarciaLocalCandidateAudit,
    GarciaMixedDyadicFamily,
    GarciaRefinementSizeLimitExceeded,
    GarciaRelationSizeLimitExceeded,
    audit_garcia_sparse_relation_connectivity,
    _dyadic_rectangle_cover,
    _garcia_resume_fingerprint,
    _piece_has_unmatched_ambient_exit,
    _raw_checkpoint_scope_claims,
    _raw_fingerprint,
    _safe_refinement_locator,
    _update_raw_content_hash,
    GarciaSourceExitProvenance,
    attach_garcia_resume_fingerprint,
    compute_garcia_local_relation,
    estimated_callback_point_evaluations,
    exit_aware_index_pair_refinement,
    full_garcia_dyadic_family,
    migrate_garcia_relation_to_raw_checkpoint,
    persisted_refinement_size_bounds,
    read_garcia_raw_relation_checkpoint,
    relation_saturated_refinement_from_payload,
    resume_garcia_local_relation_from_raw_checkpoint,
)
from hybrid_dynamics.examples.garcia_endpoint_precompute import PARITY_PROVENANCE
from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    AtlasWalkerCell,
    GarciaWalkerQuotientIncidence,
    garcia_passive_walker_atlas_charts,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    _GarciaHybridSystem,
    InadmissibleHeelstrikeError,
    GarciaPassiveWalker,
    PERIOD_TWO_POINT_A,
    post_impact_state,
)
from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    build_cmgdb_atlas_model,
)


def _rehash_raw_checkpoint_records(records: list[dict[str, object]]) -> None:
    header = records[0]
    trailer = records[-1]
    digest = hashlib.sha256()
    _update_raw_content_hash(digest, "scope_claims", _raw_checkpoint_scope_claims())
    _update_raw_content_hash(digest, "configuration", header["configuration"])
    _update_raw_content_hash(digest, "compute_metadata", header["compute_metadata"])
    _update_raw_content_hash(
        digest, "raw_elapsed_seconds", header["raw_elapsed_seconds"]
    )
    for record in records[1:-1]:
        _update_raw_content_hash(digest, "cell", record)
    _update_raw_content_hash(
        digest, "quotient_neighbor_pairs", trailer["quotient_neighbor_pairs"]
    )
    trailer["raw_relation_fingerprint"] = _raw_fingerprint(
        configuration=header["configuration"],
        content_sha256=digest.hexdigest(),
        cell_count=trailer["cell_count"],
        relation_edges=trailer["relation_edges"],
    )


def test_guard_validation_rejects_a_branch_switch_that_is_not_contact():
    walker = GarciaPassiveWalker()
    false_event = np.asarray(
        (
            -walker.guard_delta,
            -0.2,
            0.25,  # phi - 2 theta is positive, not zero
            -0.3,
        ),
        dtype=float,
    )
    assert false_event[3] - 2.0 * false_event[1] >= walker.transversality_eta
    with pytest.raises(InadmissibleHeelstrikeError):
        walker.system.apply_reset_map(false_event)


def test_reference_gait_still_uses_only_admissible_heel_strikes():
    walker = GarciaPassiveWalker(max_jumps=2)
    trajectory = walker.system.simulate(
        post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=0.01,
    )
    assert len(trajectory.jump_states) == 2
    assert all(walker.is_admissible_heelstrike(guard) for guard, _ in trajectory.jump_states)


def test_full_family_and_cost_are_four_dimensional():
    family = full_garcia_dyadic_family(2)
    assert len(family.cells) == 2 * 4**4
    assert family.chart_counts() == {0: 4**4, 1: 4**4}
    assert estimated_callback_point_evaluations(family, 3) == len(family.cells) * 3**4


def test_safe_locator_never_promotes_itself_to_scientific_acceptance():
    clean = GarciaSourceExitProvenance(
        source_index=0,
        callback_invocations=2,
        successful_samples=81,
        failed_samples=0,
        failure_reasons=(),
        unresolved_stage_edges=(),
        returned_pieces=1,
        explicit_empty_callback=False,
        mapgraph_empty_image=False,
        active_target_cells=1,
        missing_in_domain_target_cells=0,
        missing_in_domain_witnesses=(),
        target_pieces=(),
        ambient_boundary_pieces=0,
        wholly_outside_active_family_pieces=0,
    )
    locator = _safe_refinement_locator(
        {0: frozenset((0,))},
        {0: clean},
        (),
        (0,),
    )
    assert locator.selected_component == frozenset((0,))
    assert locator.selected_reference_hits == frozenset((0,))
    assert locator.to_dict()["scientific_result_accepted"] is False
    assert locator.to_dict()["acceptance_relation"] == (
        "complete_unpruned_open_exit_relation"
    )


def test_exit_aware_support_expands_only_N_minus_L_and_preserves_L_exit():
    source = GarciaDyadicCell(0, 1, (0, 0, 0, 0))
    exit_cell = GarciaDyadicCell(0, 1, (1, 0, 0, 0))
    source_target = GarciaDyadicCell(0, 1, (0, 1, 0, 0))
    ignored_exit_target = GarciaDyadicCell(0, 1, (1, 1, 0, 0))
    family = GarciaMixedDyadicFamily(
        tuple(sorted((source, exit_cell))),
        role="synthetic local open pair",
    )

    def provenance(index, witness):
        return GarciaSourceExitProvenance(
            source_index=index,
            callback_invocations=2,
            successful_samples=81,
            failed_samples=0,
            failure_reasons=(),
            unresolved_stage_edges=(),
            returned_pieces=1,
            explicit_empty_callback=False,
            mapgraph_empty_image=False,
            active_target_cells=1,
            missing_in_domain_target_cells=1,
            missing_in_domain_witnesses=(witness,),
            target_pieces=(),
            ambient_boundary_pieces=0,
            wholly_outside_active_family_pieces=0,
        )


    run = SimpleNamespace(
        family=family,
        candidate=SimpleNamespace(
            morse_node=0,
            s_cells=frozenset((0,)),
            x_cells=frozenset((0, 1)),
            a_cells=frozenset((1,)),
            reference_recovered=True,
            pair_second_condition_violations=(),
            touches_nonglued_ambient_boundary=(),
            disconnected_sources_in_s=(),
        ),
        source_provenance={
            0: provenance(0, source_target),
            1: provenance(1, ignored_exit_target),
        },
        relation={0: frozenset((1,)), 1: frozenset()},
    )
    expanded, audit = exit_aware_index_pair_refinement(
        run,
        target_axis_depth=1,
    )
    assert source_target in expanded.cells
    assert ignored_exit_target not in expanded.cells
    assert audit.expandable_sources == frozenset((0,))
    assert audit.preserved_l_exit_sources == frozenset((1,))
    assert audit.to_dict()["scientific_result_accepted"] is False
    with pytest.raises(GarciaRefinementSizeLimitExceeded):
        exit_aware_index_pair_refinement(
            run,
            target_axis_depth=1,
            max_output_cells=2,
        )


def test_candidate_gate_allows_missing_support_only_on_exit_set_L():
    audit = GarciaLocalCandidateAudit(
        morse_node=0,
        s_cells=frozenset((0,)),
        x_cells=frozenset((0, 1)),
        a_cells=frozenset((1,)),
        reference_source_cells=frozenset((0,)),
        reference_source_cells_in_candidate=frozenset((0,)),
        reference_labels_total=1,
        reference_labels_recovered=1,
        reference_missing_labels=(),
        reference_endpoint_misses=0,
        reference_evaluation_failures=0,
        failed_sources_in_s=(),
        unresolved_stage_sources_in_s=(),
        open_exit_sources_in_s=(),
        failed_sources_in_a=(),
        empty_sources_in_s=(),
        disconnected_sources_in_s=(),
        disconnected_sources_in_a=(),
        disconnected_image_components=(),
        missing_in_domain_sources_in_s=(),
        missing_in_domain_sources_in_x=(1,),
        a_exit_sources=(1,),
        pair_second_condition_violations=(),
        recurrent_components_in_x=((0,),),
        touches_nonglued_ambient_boundary=(),
        minimum_boundary_margin_by_chart_axis={},
    )
    assert audit.discovery_gates_passed


def test_handle_phase_boundary_is_exit_only_without_matching_base_seam():
    walker = GarciaPassiveWalker()
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )
    atlas = build_cmgdb_atlas_model(
        CMGDBSuspensionBoxMap(walker.system, charts, 0.5),
        depth=0,
        active_dyadic_cells=full_garcia_dyadic_family(2).tagged_cells(),
    ).phaseSpace()
    guard = np.asarray((-0.2, -0.2, -0.4, 0.0))
    handle_point = charts.encode_handle(guard, 0.0)
    handle_bounds = (*handle_point[:3], -0.01, *handle_point[:3], 0.01)
    _cover, crossings = _dyadic_rectangle_cover(
        charts,
        charts.handle_chart_id,
        handle_bounds,
        2,
    )
    base_lower = tuple(interval[0] for interval in charts.base_bounds)
    base_upper = tuple(interval[1] for interval in charts.base_bounds)
    guard_bounds = tuple(float(value) for value in (*base_lower, *base_upper))
    assert (3, -1) in crossings
    assert _piece_has_unmatched_ambient_exit(
        charts,
        incidence,
        atlas,
        charts.handle_chart_id,
        handle_bounds,
        crossings,
        ((charts.handle_chart_id, handle_bounds),),
    )


    assert not _piece_has_unmatched_ambient_exit(
        charts,
        incidence,
        atlas,
        charts.handle_chart_id,
        handle_bounds,
        crossings,
        (
            (charts.handle_chart_id, handle_bounds),
            (charts.base_chart_id, guard_bounds),
        ),
    )

    reset_bounds = guard_bounds
    upper_handle = (*handle_point[:3], 0.99, *handle_point[:3], 1.01)
    _cover, upper_crossings = _dyadic_rectangle_cover(
        charts,
        charts.handle_chart_id,
        upper_handle,
        2,
    )
    assert (3, 1) in upper_crossings
    assert not _piece_has_unmatched_ambient_exit(
        charts,
        incidence,
        atlas,
        charts.handle_chart_id,
        upper_handle,
        upper_crossings,
        (
            (charts.handle_chart_id, upper_handle),
            (charts.base_chart_id, reset_bounds),
        ),
    )

    rho_exit = list(handle_bounds)
    rho_exit[2] = -0.01
    _cover, rho_crossings = _dyadic_rectangle_cover(
        charts,
        charts.handle_chart_id,
        rho_exit,
        2,
    )
    assert (2, -1) in rho_crossings
    assert _piece_has_unmatched_ambient_exit(
        charts,
        incidence,
        atlas,
        charts.handle_chart_id,
        rho_exit,
        rho_crossings,
        (
            (charts.handle_chart_id, tuple(rho_exit)),
            (charts.base_chart_id, guard_bounds),
        ),
    )


def test_fast_reset_quotient_incidence_matches_scalar_reference():
    walker = GarciaPassiveWalker()
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )
    atlas = build_cmgdb_atlas_model(
        CMGDBSuspensionBoxMap(walker.system, charts, 0.5),
        depth=0,
        active_dyadic_cells=full_garcia_dyadic_family(1).tagged_cells(),
    ).phaseSpace()
    cells = tuple(
        AtlasWalkerCell(
            index,
            int(atlas.cell(index).chart_id),
            tuple(float(value) for value in atlas.cell(index).bounds),
        )
        for index in range(int(atlas.size()))
    )

    def scalar_reset_reference(base, handle):
        base_lower, base_upper = base.lower, base.upper
        handle_lower, handle_upper = handle.lower, handle.upper
        theta = incidence._closed_interval(
            (float(handle_lower[0]), float(handle_upper[0])),
            (-float(base_upper[0]), -float(base_lower[0])),
            (-0.5 * float(base_upper[2]), -0.5 * float(base_lower[2])),
            atol=incidence.atol,
        )
        if theta is None:
            return False

        def gap(value):
            collision = float(np.cos(2.0 * value))
            transverse = collision * (1.0 - collision)
            intervals = (
                (float(handle_lower[1]), float(handle_upper[1])),
                (
                    float(base_lower[1]) / collision,
                    float(base_upper[1]) / collision,
                ),
                (
                    float(base_lower[3]) / transverse,
                    float(base_upper[3]) / transverse,
                ),
            )
            return max(interval[0] for interval in intervals) - min(
                interval[1] for interval in intervals
            )

        return incidence._scalar_feasible(theta, gap)

    def scalar_guard_reference(base, handle):
        base_lower, base_upper = base.lower, base.upper
        handle_lower, handle_upper = handle.lower, handle.upper
        theta = incidence._closed_interval(
            (float(handle_lower[0]), float(handle_upper[0])),
            (float(base_lower[0]), float(base_upper[0])),
            (0.5 * float(base_lower[2]), 0.5 * float(base_upper[2])),
            atol=incidence.atol,
        )
        if theta is None:
            return False
        velocity = incidence._closed_interval(
            (float(handle_lower[1]), float(handle_upper[1])),
            (float(base_lower[1]), float(base_upper[1])),
            atol=incidence.atol,
        )
        if velocity is None:
            return False

        def gap(value):
            lower_phi = incidence._lower_guard_phi_dot(value)
            width = incidence.phi_dot_max - lower_phi
            image_lower = lower_phi + float(handle_lower[2]) * width
            image_upper = lower_phi + float(handle_upper[2]) * width
            return max(
                float(base_lower[3]) - image_upper,
                image_lower - float(base_upper[3]),
            )

        return incidence._scalar_feasible(velocity, gap)

    bases = [cell for cell in cells if cell.chart_id == charts.base_chart_id]
    reset_handles = [
        cell
        for cell in cells
        if cell.chart_id == charts.handle_chart_id
        and cell.upper[-1] >= 1.0 - incidence.atol
    ]
    for handle in reset_handles:
        for base in bases:
            assert incidence._reset_face_intersects(
                base, handle
            ) == scalar_reset_reference(base, handle)
    guard_handles = [
        cell
        for cell in cells
        if cell.chart_id == charts.handle_chart_id
        and cell.lower[-1] <= incidence.atol
    ]
    for handle in guard_handles:
        for base in bases:
            assert incidence._guard_face_intersects(
                base, handle
            ) == scalar_guard_reference(base, handle)

    pairs = _quotient_cross_chart_pairs(cells, incidence, charts, atlas)
    assert pairs


def test_small_local_run_persists_complete_relation_and_exit_provenance(tmp_path):
    pytest.importorskip("CMGDB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run = compute_garcia_local_relation(
            full_garcia_dyadic_family(1),
            reference_base_samples_per_stride=7,
            reference_handle_samples=5,
        )

    assert len(run.cells) == 32
    assert set(run.relation) == set(range(32))
    assert set(run.source_provenance) == set(range(32))
    assert all(record.callback_invocations == 2 for record in run.source_provenance.values())
    assert run.box_map.diagnostics().source_boxes == 32
    assert run.reference_miss_count == 0
    # The deliberately coarse relation fills its ambient family and is not an
    # isolated-gait result; the new workflow must state that rather than label it.
    assert not run.candidate.discovery_gates_passed

    target = run.write(tmp_path / "walker.json.gz")
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["schema"] == "garcia-walker-open-exit-atlas-relation-v1"
    assert payload["metadata"]["legacy_cemetery_used"] is False
    assert payload["metadata"]["base_chart_id"] == 0
    assert payload["metadata"]["handle_chart_id"] == 1
    assert payload["metadata"]["cmgdb_callback_invocations"] == 64
    assert payload["metadata"]["box_map_unique_source_boxes_evaluated"] == 32
    assert len(payload["cells"]) == 32
    assert all("open_exit" in cell for cell in payload["cells"])
    assert all("target_pieces" in cell["open_exit"] for cell in payload["cells"])
    assert all(
        "unresolved_stage_edges" in cell["open_exit"] for cell in payload["cells"]
    )
    assert "disconnected_image_components" in payload["candidate"]
    assert "open_exit_sources_in_S" in payload["candidate"]
    assert "reference_endpoint_misses" in payload["candidate"]
    assert isinstance(payload["quotient_neighbor_pairs"], list)
    assert payload["candidate"]["discovery_gates_passed"] is False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replay = compute_garcia_local_relation(
            run.family,
            reference_base_samples_per_stride=7,
            reference_handle_samples=5,
            reuse_evaluations_from=run,
        )
    assert replay.relation == run.relation
    assert replay.box_map.source_cache_counts() == (0, len(run.family.cells))
    with pytest.raises(ValueError, match="identical active family"):
        compute_garcia_local_relation(
            full_garcia_dyadic_family(2),
            reuse_evaluations_from=run,
        )

    resumed = relation_saturated_refinement_from_payload(
        payload,
        target_axis_depth=2,
    )
    assert resumed.max_axis_depth == 2
    assert resumed.min_axis_depth == 2
    assert len(resumed.cells) == 2 * 4**4
    preflight = persisted_refinement_size_bounds(payload, target_axis_depth=2)
    assert preflight["complete_ambient_cover"] is True
    assert preflight["candidate_only_cell_lower_bound"] == len(resumed.cells)
    assert preflight["complete_ambient_cell_upper_bound"] == len(resumed.cells)
    with pytest.raises(GarciaRefinementSizeLimitExceeded):
        relation_saturated_refinement_from_payload(
            payload,
            target_axis_depth=2,
            max_output_cells=10,
        )

    tampered = deepcopy(payload)
    tampered["cells"][0]["image"] = []
    with pytest.raises(ValueError, match="fingerprint"):
        relation_saturated_refinement_from_payload(
            tampered,
            target_axis_depth=2,
        )

    inconsistent = deepcopy(payload)
    source = next(cell for cell in inconsistent["cells"] if cell["image"])
    source["image"] = []
    source["open_exit"]["mapgraph_empty_image"] = True
    attach_garcia_resume_fingerprint(inconsistent)
    with pytest.raises(ValueError, match="raw target cover"):
        relation_saturated_refinement_from_payload(
            inconsistent,
            target_axis_depth=2,
        )

    false_candidate = deepcopy(payload)
    false_candidate["candidate"]["X"] = []
    attach_garcia_resume_fingerprint(false_candidate)
    with pytest.raises(ValueError, match="candidate S/X/A"):
        relation_saturated_refinement_from_payload(
            false_candidate,
            target_axis_depth=2,
        )

    false_exit = deepcopy(payload)
    false_exit["cells"][0]["open_exit"]["has_open_exit"] = not bool(
        false_exit["cells"][0]["open_exit"]["has_open_exit"]
    )
    attach_garcia_resume_fingerprint(false_exit)
    with pytest.raises(ValueError, match="open-exit flag"):
        relation_saturated_refinement_from_payload(
            false_exit,
            target_axis_depth=2,
        )


def test_sparse_mixed_dyadic_adjacency_matches_exhaustive_closed_boxes():
    coarse = list(full_garcia_dyadic_family(1).cells)
    refined_parent = next(cell for cell in coarse if cell.chart_id == 0)
    coarse.remove(refined_parent)
    coarse.extend(refined_parent.descendants(2))
    family = GarciaMixedDyadicFamily(tuple(sorted(coarse)), role="mixed parity test")
    chart_targets = {
        chart_id: frozenset(
            index for index, cell in enumerate(family.cells) if cell.chart_id == chart_id
        )
        for chart_id in (0, 1)
    }
    relation = {
        index: chart_targets[cell.chart_id]
        for index, cell in enumerate(family.cells)
    }
    audit = audit_garcia_sparse_relation_connectivity(
        relation,
        family.cells,
        (),
    )
    expected_edges = 0
    for first_index, first in enumerate(family.cells):
        for second in family.cells[first_index + 1 :]:
            if first.chart_id != second.chart_id:
                continue
            common_depth = max(first.axis_depth, second.axis_depth)
            first_scale = 2 ** (common_depth - first.axis_depth)
            second_scale = 2 ** (common_depth - second.axis_depth)
            first_lower = tuple(value * first_scale for value in first.coordinates)
            first_upper = tuple(value + first_scale for value in first_lower)
            second_lower = tuple(value * second_scale for value in second.coordinates)
            second_upper = tuple(value + second_scale for value in second_lower)
            if all(
                first_lower[axis] <= second_upper[axis]
                and second_lower[axis] <= first_upper[axis]
                for axis in range(4)
            ):
                expected_edges += 1
    assert audit.algorithm == "mixed_antichain_dyadic_trie_v1"
    assert audit.same_chart_undirected_adjacencies == expected_edges
    assert not audit.disconnected_image_components


def test_iterative_relation_scc_handles_back_cross_and_induced_edges():
    relation = {
        0: frozenset({1}),
        1: frozenset({2, 4}),
        2: frozenset({0, 3}),
        3: frozenset({3}),
        4: frozenset({5}),
        5: frozenset({4, 6}),
        6: frozenset(),
    }
    assert _relation_strongly_connected_components(relation) == (
        (0, 1, 2),
        (3,),
        (4, 5),
        (6,),
    )
    assert _relation_strongly_connected_components(
        relation, frozenset({0, 1, 2, 4, 5, 6})
    ) == ((0, 1, 2), (4, 5), (6,))

    # The implementation is intentionally iterative; this is well beyond
    # Python's recursion limit and guards against a recursive regression.
    chain_length = 20_000
    chain = {
        index: frozenset({index + 1}) if index + 1 < chain_length else frozenset()
        for index in range(chain_length)
    }
    chain_components = _relation_strongly_connected_components(chain)
    assert len(chain_components) == chain_length
    assert chain_components[0] == (0,)
    assert chain_components[-1] == (chain_length - 1,)


def test_raw_morse_validator_accepts_hasse_edges_and_checks_full_reachability():
    relation = {
        0: frozenset({0, 1}),
        1: frozenset({3}),
        2: frozenset(),
        3: frozenset({3, 4}),
        4: frozenset({5}),
        5: frozenset({5}),
    }
    snapshot = {
        "nodes": [0, 1, 2],
        "morse_sets": {"0": [0], "1": [3], "2": [5]},
        "edges": [[0, 1], [1, 2]],
    }
    _validate_raw_morse_snapshot(snapshot, relation)
    corrupted = deepcopy(snapshot)
    corrupted["edges"] = [[0, 1]]
    with pytest.raises(ValueError, match="Morse order disagrees"):
        _validate_raw_morse_snapshot(corrupted, relation)
    duplicate = deepcopy(snapshot)
    duplicate["edges"] = [[0, 1], [0, 1], [1, 2]]
    with pytest.raises(ValueError, match="Morse edges are not canonical"):
        _validate_raw_morse_snapshot(duplicate, relation)


def test_sparse_depth12_connectivity_matches_persisted_exhaustive_audit():
    source = (
        Path(__file__).resolve().parents[2]
        / "data/garcia_passive_walker_atlas/local_relation_tau050_depth12.json.gz"
    )
    if not source.exists():
        pytest.skip("persisted depth-12 Garcia discovery relation is unavailable")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    dyadic_cells = tuple(
        GarciaDyadicCell(
            int(raw["chart_id"]),
            int(raw["axis_depth"]),
            tuple(int(value) for value in raw["coordinates"]),
        )
        for raw in payload["cells"]
    )
    relation = {
        int(raw["index"]): frozenset(int(value) for value in raw["image"])
        for raw in payload["cells"]
    }
    audit = audit_garcia_sparse_relation_connectivity(
        relation,
        dyadic_cells,
        tuple(tuple(int(value) for value in pair) for pair in payload["quotient_neighbor_pairs"]),
    )
    expected = tuple(
        (
            int(item["source"]),
            tuple(tuple(int(value) for value in component) for component in item["components"]),
        )
        for item in payload["candidate"]["disconnected_image_components"]
    )
    assert audit.relation_edges == 1_992_616
    assert audit.total_undirected_adjacencies == 234_946
    assert audit.estimated_peak_adjacency_storage_bytes == 12_850_272
    # The earlier 128V+32E model undercounted an isolated measured d12 audit
    # peak of 10.81 MB; this explicit margin protects the hard memory cap.
    assert audit.estimated_peak_adjacency_storage_bytes > 10_813_000
    assert audit.disconnected_image_components == expected


def test_atomic_raw_checkpoint_roundtrip_is_derived_audit_exact(
    tmp_path, monkeypatch
):
    pytest.importorskip("CMGDB")
    checkpoint = tmp_path / "raw-depth4.jsonl.gz"
    family = full_garcia_dyadic_family(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fresh = compute_garcia_local_relation(
            family,
            reference_base_samples_per_stride=7,
            reference_handle_samples=5,
            precompute_workers=1,
            raw_checkpoint_path=checkpoint,
        )
        resumed = resume_garcia_local_relation_from_raw_checkpoint(
            checkpoint,
            expected_family=family,
            t_star=0.5,
            samples_per_axis=3,
            padding_cells=1.0,
            gamma=0.0172,
            guard_delta=0.05,
            transversality_eta=0.1,
            max_step=0.02,
            max_jumps=20,
            reference_base_samples_per_stride=7,
            reference_handle_samples=5,
        )
    assert fresh.relation == resumed.relation
    assert fresh.reference_node_hits == resumed.reference_node_hits
    assert fresh.candidate.to_dict() == resumed.candidate.to_dict()
    assert fresh.refinement_locator.to_dict() == resumed.refinement_locator.to_dict()
    assert fresh.connectivity_audit == resumed.connectivity_audit
    assert fresh.box_map.point_cache_counts() == (0, 0)
    assert fresh.box_map.source_cache_counts() == (0, 0)
    assert resumed.to_dict()["metadata"]["raw_checkpoint_resumed"] is True
    with gzip.open(checkpoint, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert records[0]["checkpoint_complete"] is False
    assert records[0]["audit_complete"] is False
    assert records[-1]["checkpoint_complete"] is True
    assert records[-1]["scientific_result_accepted"] is False

    # A separately hash-consistent but noncanonical Morse relabel must not
    # alter the resumed candidate identifier.
    corrupted = deepcopy(records)
    morse = corrupted[-1]["morse_graph"]
    old_node = str(morse["nodes"][0])
    morse["nodes"] = [99]
    morse["morse_sets"] = {"99": morse["morse_sets"][old_node]}
    morse["edges"] = [
        [99 if source == int(old_node) else source, 99 if target == int(old_node) else target]
        for source, target in morse["edges"]
    ]
    corrupted[-1]["morse_snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            morse,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    corrupted_path = tmp_path / "raw-corrupted.jsonl.gz"
    with gzip.open(corrupted_path, "wt", encoding="utf-8") as stream:
        for record in corrupted:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="Morse sets disagree"):
        read_garcia_raw_relation_checkpoint(corrupted_path)

    with pytest.raises(GarciaRelationSizeLimitExceeded, match="raw_relation_edge"):
        resume_garcia_local_relation_from_raw_checkpoint(
            checkpoint,
            expected_family=family,
            max_relation_edges=0,
        )
    with pytest.raises(
        GarciaRelationSizeLimitExceeded,
        match="raw_relation_estimated_storage_byte",
    ):
        read_garcia_raw_relation_checkpoint(
            checkpoint,
            max_relation_storage_bytes=0,
        )
    for invalid_cap in (True, 1.0, float("nan")):
        with pytest.raises(ValueError, match="nonnegative exact integer"):
            read_garcia_raw_relation_checkpoint(
                checkpoint,
                max_relation_edges=invalid_cap,  # type: ignore[arg-type]
            )

    mis_scoped = deepcopy(records)
    mis_scoped[0]["whole_cell_outer_enclosure_certified"] = True
    mis_scoped_path = tmp_path / "raw-mis-scoped.jsonl.gz"
    with gzip.open(mis_scoped_path, "wt", encoding="utf-8") as stream:
        for record in mis_scoped:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="malformed or mis-scoped"):
        read_garcia_raw_relation_checkpoint(mis_scoped_path)

    bogus_precompute = deepcopy(records)
    logical_samples = bogus_precompute[0]["compute_metadata"][
        "box_map_sampled_points_evaluated"
    ]
    unique_points = bogus_precompute[0]["compute_metadata"][
        "unique_physical_endpoint_evaluations"
    ]
    bogus_precompute[0]["compute_metadata"]["endpoint_precompute"] = {
        "logical_tensor_points": logical_samples,
        "unique_points": unique_points,
        "duplicate_tensor_points": logical_samples - unique_points,
        "requested_workers": 8,
        "used_workers": 8,
        "process_count": 8,
        "mode": "rigorous_certified",
        "elapsed_seconds": 1.0,
        "fallback_reason": None,
        "parity_provenance": dict(PARITY_PROVENANCE),
    }
    _rehash_raw_checkpoint_records(bogus_precompute)
    bogus_precompute_path = tmp_path / "raw-bogus-precompute.jsonl.gz"
    with gzip.open(bogus_precompute_path, "wt", encoding="utf-8") as stream:
        for record in bogus_precompute:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="precompute mode is unsupported"):
        read_garcia_raw_relation_checkpoint(bogus_precompute_path)

    duplicate_morse_edge = deepcopy(records)
    if duplicate_morse_edge[-1]["morse_graph"]["edges"]:
        duplicate_morse_edge[-1]["morse_graph"]["edges"].append(
            duplicate_morse_edge[-1]["morse_graph"]["edges"][0]
        )
        duplicate_morse_edge[-1]["morse_snapshot_sha256"] = hashlib.sha256(
            json.dumps(
                duplicate_morse_edge[-1]["morse_graph"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        duplicate_edge_path = tmp_path / "raw-duplicate-morse-edge.jsonl.gz"
        with gzip.open(duplicate_edge_path, "wt", encoding="utf-8") as stream:
            for record in duplicate_morse_edge:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="Morse edges are not canonical"):
            read_garcia_raw_relation_checkpoint(duplicate_edge_path)

    # Derived-audit revision changes do not invalidate the immutable raw data:
    # a fully provenanced v3 relation migrates without another ODE call.
    legacy_payload = fresh.to_dict()
    legacy_payload["metadata"]["resume_fingerprint"] = _garcia_resume_fingerprint(
        legacy_payload,
        algorithm_revision="garcia-local-open-safe-locator-v3",
    )
    legacy_path = tmp_path / "legacy-v3.json.gz"
    with gzip.open(legacy_path, "wt", encoding="utf-8") as stream:
        json.dump(legacy_payload, stream, sort_keys=True)
    migrated_path = tmp_path / "migrated-v3.jsonl.gz"
    with monkeypatch.context() as no_ode:
        def forbidden_ode(*_args, **_kwargs):
            raise AssertionError("migration/readback must not evaluate an ODE")

        no_ode.setattr(_GarciaHybridSystem, "simulate", forbidden_ode)
        no_ode.setattr(CMGDBSuspensionBoxMap, "_evaluate_point", forbidden_ode)
        migrate_garcia_relation_to_raw_checkpoint(legacy_path, migrated_path)
        migrated = read_garcia_raw_relation_checkpoint(
            migrated_path,
            expected_family=family,
        )
    assert migrated.relation == fresh.relation
