"""Tests for the conditional Garcia mapping-cylinder Conley adapter."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    garcia_passive_walker_atlas_charts,
)
from hybrid_dynamics.examples.garcia_passive_walker_conley import (
    GARCIA_RELATION_SCHEMA,
    build_garcia_mapping_cylinder_topology,
    load_persisted_garcia_relation,
    prepare_persisted_garcia_finite_conley,
)


def _uniform_depth_two_payload() -> dict[str, object]:
    charts = garcia_passive_walker_atlas_charts()
    records: list[dict[str, object]] = []
    for chart_id in (BASE_CHART_ID, HANDLE_CHART_ID):
        bounds = charts.bounds_for(chart_id)
        for coordinates in itertools.product(range(4), repeat=4):
            lower = [
                interval[0] + (interval[1] - interval[0]) * value / 4
                for interval, value in zip(bounds, coordinates)
            ]
            upper = [
                interval[0] + (interval[1] - interval[0]) * (value + 1) / 4
                for interval, value in zip(bounds, coordinates)
            ]
            records.append(
                {
                    "index": len(records),
                    "chart_id": chart_id,
                    "axis_depth": 2,
                    "coordinates": list(coordinates),
                    "bounds": [*lower, *upper],
                    "image": [],
                    "morse_node": 0,
                    "relation_evaluated": True,
                    "open_exit": {
                        "has_open_exit": False,
                        "unresolved_stage_edges": [],
                        "failed_samples_per_invocation": 0,
                        "missing_in_domain_target_cells": 0,
                        "ambient_boundary_pieces": 0,
                        "wholly_outside_active_family_pieces": 0,
                        "explicit_empty_callback": False,
                        "mapgraph_empty_image": False,
                    },
                }
            )
    # The candidate contains only the two small base blocks needed by the
    # sampled guard/reset attachment covers at padding 0.01.  Keeping the
    # remaining Atlas records ambient (but outside X) exercises sparse-family
    # loading without making the chain-selector regression unnecessarily huge.
    attachment_target_coordinates = {
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 1, 0, 1),
        (0, 1, 0, 2),
        (3, 1, 3, 2),
        (3, 1, 3, 3),
        (3, 2, 3, 2),
        (3, 2, 3, 3),
    }
    base_indices = [
        int(record["index"])
        for record in records
        if record["chart_id"] == BASE_CHART_ID
        and tuple(record["coordinates"]) in attachment_target_coordinates
    ]
    recurrent_indices = [
        int(record["index"])
        for record in records
        if record["chart_id"] == HANDLE_CHART_ID
        and record["coordinates"][:3] == [1, 1, 1]
    ]
    x_indices = base_indices + recurrent_indices
    for record in records:
        if int(record["index"]) in recurrent_indices:
            record["image"] = x_indices
        else:
            record["image"] = []
        if int(record["index"]) not in recurrent_indices:
            record["open_exit"] = {
                "has_open_exit": True,
                "unresolved_stage_edges": [],
                "failed_samples_per_invocation": 1,
                "missing_in_domain_target_cells": 0,
                "ambient_boundary_pieces": 0,
                "wholly_outside_active_family_pieces": 0,
                "explicit_empty_callback": False,
                "mapgraph_empty_image": True,
            }
    return {
        "schema": GARCIA_RELATION_SCHEMA,
        "metadata": {
            "t_star": 0.5,
            "base_chart_id": BASE_CHART_ID,
            "handle_chart_id": HANDLE_CHART_ID,
            "min_axis_depth": 2,
            "max_axis_depth": 2,
            "mixed_depth": False,
            "whole_cell_outer_enclosure_certified": False,
        },
        "charts": {
            "base": {
                "chart_id": BASE_CHART_ID,
                "bounds": [list(interval) for interval in charts.base_bounds],
            },
            "handle": {
                "chart_id": HANDLE_CHART_ID,
                "bounds": [list(interval) for interval in charts.handle_bounds],
            },
        },
        "candidate": {
            "morse_node": 0,
            "S": list(recurrent_indices),
            "X": list(x_indices),
            "A": list(base_indices),
            "pair_second_condition_violations": [],
            "discovery_gates_passed": True,
            "reference_labels_total": 1,
            "reference_labels_recovered": 1,
            "reference_missing_labels": [],
        },
        "morse_graph": {
            "nodes": [0],
            "edges": [],
            "morse_sets": {"0": list(recurrent_indices)},
        },
        "reference_audit": {
            "probes": 1,
            "misses": 0,
            "evaluation_failures": 0,
            "morse_node_hits": {"0": 1},
        },
        "cells": records,
    }


def test_garcia_attachment_and_fixed_time_carriers_pass_synthetic_4d_pipeline():
    topology = build_garcia_mapping_cylinder_topology(
        _uniform_depth_two_payload(),
        attachment_samples_per_axis=3,
        attachment_padding_cells=0.01,
    )
    preparation = prepare_persisted_garcia_finite_conley(topology)

    assert topology.axis_depth == 2
    assert topology.base_complex.max_dimension == 4
    assert topology.guard_complex.max_dimension == 3
    assert topology.cylinder.max_dimension == 4
    assert topology.guard_carrier.acyclicity_validated
    assert topology.reset_carrier.acyclicity_validated
    assert topology.cylinder.metadata["integral_attachment_maps_verified"] is True
    assert len(topology.attachment_records) == 2 * len(topology.guard_complex.cells)
    assert all(record.target_top_coordinates for record in topology.attachment_records)
    assert preparation.finite_relation_algebra_validated
    assert preparation.carrier.preserves_pair(topology.pair.relative_pair)
    assert preparation.carrier.carries(preparation.chain_map)
    exit_source = next(iter(topology.relation.a_cells))
    assert preparation.relation_vertex_images[exit_source] == frozenset()
    assert not preparation.continuous_system_conley_index_certified

    pytest.importorskip("CMGDB")
    result = preparation.compute_finite_relation_shift_class()
    assert result["homology_dimensions"] == [0, 1, 0, 0, 0]
    assert result["shift_class"] == ["0", "x-1", "0", "0", "0"]
    assert result["continuous_system_conley_index_certified"] is False


def test_mixed_depth_candidate_stops_before_nonconforming_topology_is_claimed():
    payload = _uniform_depth_two_payload()
    cells = payload["cells"]
    assert isinstance(cells, list)
    source = payload["candidate"]["A"][0]
    coordinates = tuple(int(value) for value in cells[source]["coordinates"])
    cells[source]["axis_depth"] = 3
    cells[source]["coordinates"] = [2 * value for value in coordinates]

    with pytest.raises(ValueError, match="conforming adaptive"):
        build_garcia_mapping_cylinder_topology(payload)


def test_interior_handle_family_uses_auxiliary_seam_support_outside_x():
    payload = _uniform_depth_two_payload()
    cells = payload["cells"]
    interior_s = [
        int(record["index"])
        for record in cells
        if record["chart_id"] == HANDLE_CHART_ID
        and record["coordinates"][:3] == [1, 1, 1]
        and record["coordinates"][3] in (1, 2)
    ]
    guard_block = [
        int(record["index"])
        for record in cells
        if record["chart_id"] == BASE_CHART_ID
        and tuple(record["coordinates"])
        in {
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 1, 0, 1),
            (0, 1, 0, 2),
        }
    ]
    x_indices = guard_block + interior_s
    for record in cells:
        if int(record["index"]) in interior_s:
            record["image"] = list(x_indices)
            record["open_exit"] = {
                "has_open_exit": False,
                "unresolved_stage_edges": [],
                "failed_samples_per_invocation": 0,
                "missing_in_domain_target_cells": 0,
                "ambient_boundary_pieces": 0,
                "wholly_outside_active_family_pieces": 0,
                "explicit_empty_callback": False,
                "mapgraph_empty_image": False,
            }
        else:
            record["image"] = []
            record["open_exit"] = {
                "has_open_exit": True,
                "unresolved_stage_edges": [],
                "failed_samples_per_invocation": 1,
                "missing_in_domain_target_cells": 0,
                "ambient_boundary_pieces": 0,
                "wholly_outside_active_family_pieces": 0,
                "explicit_empty_callback": False,
                "mapgraph_empty_image": True,
            }
    payload["candidate"]["S"] = list(interior_s)
    payload["candidate"]["X"] = list(x_indices)
    payload["candidate"]["A"] = list(guard_block)
    payload["morse_graph"]["morse_sets"] = {"0": list(interior_s)}

    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )

    assert topology.auxiliary_base_coordinates
    assert all(
        coordinates[0] == 3
        for coordinates in topology.auxiliary_base_coordinates
    )
    for coordinates in topology.auxiliary_base_coordinates:
        auxiliary_top = topology.cylinder.base_cell(
            topology.base_complex.top_cell_at(coordinates)
        )
        assert auxiliary_top not in topology.pair.complex.cell_set


def test_mixed_depth_ambient_with_uniform_candidate_uses_sparse_candidate_only():
    payload = _uniform_depth_two_payload()
    cells = payload["cells"]
    candidate = payload["candidate"]
    charts_payload = payload["charts"]
    assert isinstance(cells, list)
    assert isinstance(candidate, dict)
    assert isinstance(charts_payload, dict)

    parent = cells.pop()
    parent_index = int(parent["index"])
    parent_coordinates = tuple(int(value) for value in parent["coordinates"])
    handle_bounds = charts_payload["handle"]["bounds"]
    for offsets in itertools.product((0, 1), repeat=4):
        coordinates = tuple(
            2 * value + offset
            for value, offset in zip(parent_coordinates, offsets)
        )
        lower = [
            interval[0] + (interval[1] - interval[0]) * value / 8
            for interval, value in zip(handle_bounds, coordinates)
        ]
        upper = [
            interval[0] + (interval[1] - interval[0]) * (value + 1) / 8
            for interval, value in zip(handle_bounds, coordinates)
        ]
        cells.append(
            {
                **parent,
                "index": len(cells),
                "axis_depth": 3,
                "coordinates": list(coordinates),
                "bounds": [*lower, *upper],
            }
        )
    payload["metadata"]["mixed_depth"] = True
    payload["metadata"]["max_axis_depth"] = 3

    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )

    assert topology.relation.metadata["mixed_depth"] is True
    assert topology.axis_depth == 2
    assert parent_index not in topology.registry.atlas_indices


def test_current_persisted_depth_eight_artifact_is_consumed_but_not_promoted():
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "garcia_passive_walker_atlas"
        / "local_relation_tau050_depth8.json.gz"
    )
    relation = load_persisted_garcia_relation(path)

    assert len(relation.cells) == 512
    assert relation.x_cells
    assert not relation.discovery_gates_passed


def test_loader_rejects_coerced_booleans_and_stale_candidate_equations():
    payload = _uniform_depth_two_payload()
    payload["cells"][0]["relation_evaluated"] = "true"
    with pytest.raises(ValueError, match="JSON boolean"):
        load_persisted_garcia_relation(payload)

    payload = _uniform_depth_two_payload()
    payload["candidate"]["X"] = payload["candidate"]["S"]
    payload["candidate"]["A"] = []
    with pytest.raises(ValueError, match="candidate equations"):
        load_persisted_garcia_relation(payload)


def test_loader_recomputes_exit_provenance_pair_audit_and_scc():
    payload = _uniform_depth_two_payload()
    source = payload["candidate"]["S"][0]
    payload["cells"][source]["open_exit"]["has_open_exit"] = True
    with pytest.raises(ValueError, match="raw exit provenance"):
        load_persisted_garcia_relation(payload)

    payload = _uniform_depth_two_payload()
    exit_source = payload["candidate"]["A"][0]
    payload["candidate"]["pair_second_condition_violations"] = [exit_source]
    with pytest.raises(ValueError, match="pair-second-condition audit"):
        load_persisted_garcia_relation(payload)

    payload = _uniform_depth_two_payload()
    source = payload["candidate"]["S"][0]
    payload["cells"][source]["image"] = payload["candidate"]["A"]
    with pytest.raises(ValueError, match="recurrent SCC"):
        load_persisted_garcia_relation(payload)


def test_public_preparation_recomputes_s_open_exits_instead_of_trusting_gate():
    payload = _uniform_depth_two_payload()
    source = payload["candidate"]["S"][0]
    payload["cells"][source]["open_exit"]["has_open_exit"] = True
    payload["cells"][source]["open_exit"]["failed_samples_per_invocation"] = 1
    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )

    with pytest.raises(ValueError, match="explicit open-exit sources"):
        prepare_persisted_garcia_finite_conley(topology)


def test_open_pair_allows_missing_support_on_a_but_rejects_it_on_s():
    payload = _uniform_depth_two_payload()
    exit_source = payload["candidate"]["A"][0]
    payload["cells"][exit_source]["open_exit"][
        "missing_in_domain_target_cells"
    ] = 1
    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )
    preparation = prepare_persisted_garcia_finite_conley(topology)
    assert preparation.relation_vertex_images[exit_source] == frozenset()

    payload = _uniform_depth_two_payload()
    recurrent_source = payload["candidate"]["S"][0]
    payload["cells"][recurrent_source]["open_exit"][
        "missing_in_domain_target_cells"
    ] = 1
    payload["cells"][recurrent_source]["open_exit"]["has_open_exit"] = True
    payload["candidate"]["missing_in_domain_sources_in_S"] = [recurrent_source]
    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )
    with pytest.raises(ValueError, match="candidate S has in-domain targets omitted"):
        prepare_persisted_garcia_finite_conley(topology)


def test_public_preparation_rejects_missing_reference_labels():
    payload = _uniform_depth_two_payload()

    payload["candidate"]["reference_labels_recovered"] = 0
    topology = build_garcia_mapping_cylinder_topology(
        payload, attachment_padding_cells=0.01
    )
    with pytest.raises(ValueError, match="reference labels"):
        prepare_persisted_garcia_finite_conley(topology)
