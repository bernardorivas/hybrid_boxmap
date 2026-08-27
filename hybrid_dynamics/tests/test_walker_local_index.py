"""Tests for exit-aware local pairs on persisted Atlas relations."""

from __future__ import annotations

import gzip
import json

import pytest

from demo.run_walker_local_index_audit import main as run_local_index_audit
from hybrid_dynamics.examples.physical_conley import (
    AtlasRelationSnapshot,
    AtlasTopCellRecord,
)
from hybrid_dynamics.examples.walker_local_index import (
    EXIT_AWARE_LOCAL_INDEX_CERTIFICATE_SCHEMA,
    GARCIA_WALKER_OPEN_EXIT_RELATION_SCHEMA,
    ExitAwareLocalIndexInput,
    build_geometric_collar,
    construct_exit_aware_local_index_pair,
    load_exit_aware_relation,
    merge_relation_inputs,
)


def _cell(
    index: int,
    interval: tuple[float, float],
    image: tuple[int, ...],
    *,
    chart_id: int = 0,
    morse_node: int | None = None,
    dimension: int = 4,
    relation_evaluated: bool = True,
) -> AtlasTopCellRecord:
    lower, upper = interval
    bounds = (lower, *(0.0 for _ in range(dimension - 1))) + (
        upper,
        *(1.0 for _ in range(dimension - 1)),
    )
    return AtlasTopCellRecord(
        index=index,
        chart_id=chart_id,
        bounds=bounds,
        image=image,
        morse_node=morse_node,
        relation_evaluated=relation_evaluated,
    )


def _snapshot(
    cells: tuple[AtlasTopCellRecord, ...],
    *,
    depth: int = 16,
) -> AtlasRelationSnapshot:
    return AtlasRelationSnapshot(
        model="adaptive-walker-test",
        depth=depth,
        t_star=0.5,
        cells=cells,
        morse_nodes=(0,),
        morse_edges=(),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=0,
        box_map_empty_images=0,
        box_map_unresolved_stage_edges=0,
    )


def _mixed_depth_snapshot(*, depth: int = 16) -> AtlasRelationSnapshot:
    return _snapshot(
        (
            _cell(0, (0.0, 0.5), (0,)),
            _cell(1, (0.5, 1.5), (1,), morse_node=0),
            _cell(2, (1.5, 2.0), (3,)),
            _cell(3, (2.0, 2.25), (3,)),
        ),
        depth=depth,
    )


@pytest.mark.parametrize("depth", (16, 20))
def test_mixed_depth_adaptive_collar_yields_exit_aware_pair(depth):
    snapshot = _mixed_depth_snapshot(depth=depth)
    specification = ExitAwareLocalIndexInput(
        morse_node=0,
        collar_layers=1,
        allowed_cells=frozenset(range(4)),
        open_exit_flags={0: True, 1: False, 2: False},
    )

    certificate = construct_exit_aware_local_index_pair(snapshot, specification)

    assert certificate.n_cells == frozenset({0, 1, 2})
    assert certificate.l_cells == frozenset({0, 2})
    assert certificate.relation_exit_edges == ((2, (3,)),)
    assert certificate.first_condition_violations == ()
    assert certificate.second_condition_violations == ()
    assert certificate.s_intersection_l == ()
    assert certificate.s_is_recurrent_component
    assert certificate.unexpected_recurrent_components == ()
    assert certificate.collar.mixed_cell_sizes
    assert certificate.finite_relation_pair_certified
    assert not certificate.continuous_system_index_pair_certified


def test_exit_set_is_closed_forward_until_stable():
    snapshot = _snapshot(
        (
            _cell(0, (0.0, 1.0), (0,), morse_node=0, dimension=1),
            _cell(1, (1.0, 2.0), (1,), dimension=1),
            _cell(2, (2.0, 3.0), (1,), dimension=1),
            _cell(3, (3.0, 4.0), (2, 4), dimension=1),
            _cell(4, (4.0, 5.0), (4,), dimension=1),
        )
    )
    specification = ExitAwareLocalIndexInput(
        morse_node=0,
        collar_layers=3,
        open_exit_flags={0: False, 1: False, 2: False, 3: False},
    )

    certificate = construct_exit_aware_local_index_pair(snapshot, specification)

    assert certificate.n_cells == frozenset({0, 1, 2, 3})
    assert certificate.initial_l_cells == frozenset({3})
    assert certificate.closure_rounds == ((2,), (1,))
    assert certificate.l_cells == frozenset({1, 2, 3})
    assert certificate.finite_relation_pair_certified


def test_closure_reaching_gait_is_a_persisted_failure_witness():
    snapshot = _snapshot(
        (
            _cell(0, (0.0, 1.0), (0,), morse_node=0, dimension=1),
            _cell(1, (1.0, 2.0), (0,), dimension=1),
            _cell(2, (2.0, 3.0), (1, 3), dimension=1),
            _cell(3, (3.0, 4.0), (3,), dimension=1),
        )
    )
    specification = ExitAwareLocalIndexInput(
        morse_node=0,
        collar_layers=2,
        open_exit_flags={0: False, 1: False, 2: False},
    )

    certificate = construct_exit_aware_local_index_pair(snapshot, specification)
    payload = certificate.to_json_dict()

    assert certificate.closure_rounds == ((1,), (0,))
    assert certificate.s_intersection_l == (0,)
    assert not certificate.finite_relation_pair_certified
    assert "S_intersects_L" in payload["finite_relation_pair_blockers"]


def test_explicit_quotient_neighbor_adds_cross_chart_collar_cell():
    snapshot = _snapshot(
        (
            _cell(0, (0.0, 1.0), (0,), chart_id=0, morse_node=0, dimension=1),
            _cell(1, (0.0, 1.0), (1,), chart_id=1, dimension=1),
        )
    )

    without_seam = build_geometric_collar(snapshot, {0}, collar_layers=1)
    with_seam = build_geometric_collar(
        snapshot,
        {0},
        collar_layers=1,
        quotient_neighbor_pairs=((0, 1),),
    )

    assert without_seam.n_cells == frozenset({0})
    assert with_seam.n_cells == frozenset({0, 1})
    assert with_seam.quotient_neighbor_edges == ((0, 1),)


def test_missing_flags_and_unevaluated_sources_prevent_certificate():
    snapshot = _snapshot(
        (
            _cell(0, (0.0, 1.0), (0,), morse_node=0, dimension=1),
            _cell(
                1,
                (1.0, 2.0),
                (1,),
                dimension=1,
                relation_evaluated=False,
            ),
        )
    )
    specification = ExitAwareLocalIndexInput(
        morse_node=0,
        collar_layers=1,
        open_exit_flags={0: False},
    )

    certificate = construct_exit_aware_local_index_pair(snapshot, specification)

    assert certificate.missing_open_exit_flags == (1,)
    assert certificate.unevaluated_n_sources == (1,)
    assert not certificate.finite_relation_pair_certified


def test_runner_persists_pass_certificate_and_relation_digest(tmp_path):
    snapshot = _mixed_depth_snapshot(depth=20)
    relation_path = tmp_path / "walker-depth20-relation.json.gz"
    snapshot.write_gzip_json(relation_path)
    specification = ExitAwareLocalIndexInput(
        morse_node=0,
        collar_layers=1,
        allowed_cells=frozenset(range(4)),
        open_exit_flags={0: True, 1: False, 2: False},
        metadata={
            "expected_relation": {
                "model": "adaptive-walker-test",
                "depth": 20,
                "t_star": 0.5,
            }
        },
    )
    specification_path = specification.write_json(tmp_path / "input.json")
    output = tmp_path / "certificate.json"

    exit_code = run_local_index_audit(
        (str(relation_path), str(specification_path), "--output", str(output))
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["schema"] == EXIT_AWARE_LOCAL_INDEX_CERTIFICATE_SCHEMA
    assert payload["finite_relation_pair_certified"] is True
    assert payload["continuous_system_index_pair_certified"] is False
    assert len(payload["relation_provenance"]["sha256"]) == 64
    assert payload["geometric_collar"]["uniform_grid_assumed"] is False


def test_planned_walker_schema_supplies_embedded_open_exit_flags(tmp_path):
    relation_path = tmp_path / "walker-open-exit.json.gz"
    chart_domain = ((0.0, 4.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    descriptors = (
        (2, (0, 0, 0, 0), (0,), None),
        (2, (1, 0, 0, 0), (1,), 0),
        (3, (4, 0, 0, 0), (3,), None),
        (3, (5, 0, 0, 0), (3,), None),
    )
    raw_cells = []
    for index, (axis_depth, coordinates, image, morse_node) in enumerate(
        descriptors
    ):
        subdivisions = 2**axis_depth
        lower = tuple(
            left + (right - left) * coordinate / subdivisions
            for (left, right), coordinate in zip(chart_domain, coordinates)
        )
        upper = tuple(
            left + (right - left) * (coordinate + 1) / subdivisions
            for (left, right), coordinate in zip(chart_domain, coordinates)
        )
        raw_cells.append(
            {
                "index": index,
                "chart_id": 0,
                "axis_depth": axis_depth,
                "coordinates": list(coordinates),
                "bounds": [*lower, *upper],
                "image": list(image),
                "morse_node": morse_node,
                "relation_evaluated": True,
                "open_exit": {
                    "callback_invocations": 1,
                    "successful_samples_per_invocation": [81],
                    "failed_samples_per_invocation": [0],
                    "failure_reasons": {},
                    "returned_pieces": 1,
                    "explicit_empty_callback": False,
                    "mapgraph_empty_image": False,
                    "active_target_cells": len(image),
                    "missing_in_domain_target_cells": 0,
                    "missing_in_domain_witnesses": [],
                    "ambient_boundary_pieces": 1 if index == 0 else 0,
                    "wholly_outside_active_family_pieces": 0,
                    "has_open_exit": index == 0,
                },
            }
        )
    payload = {
        "schema": GARCIA_WALKER_OPEN_EXIT_RELATION_SCHEMA,
        "model": "adaptive-walker-test",
        "depth": 20,
        "tau": 0.5,
        "relation_scope": "adaptive_local_atlas_relation",
        "whole_cell_outer_enclosure_certified": False,
        "morse_graph": {"morse_sets": {"0": [1]}},
        "candidate": {"morse_node": 0, "S": [1]},
        "charts": {
            "base": {
                "chart_id": 0,
                "bounds": [list(interval) for interval in chart_domain],
            }
        },
        "cells": raw_cells,
    }
    with gzip.open(relation_path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)

    loaded = load_exit_aware_relation(relation_path)
    specification = merge_relation_inputs(
        ExitAwareLocalIndexInput(
            morse_node=0,
            collar_layers=1,
            open_exit_flags={},
        ),
        loaded,
    )
    certificate = construct_exit_aware_local_index_pair(
        loaded.snapshot,
        specification,
    )

    assert loaded.persistence_schema == GARCIA_WALKER_OPEN_EXIT_RELATION_SCHEMA
    assert specification.open_exit_flags[0] is True
    assert specification.open_exit_flags[1] is False
    assert certificate.collar.contact_algorithm == "dyadic-coordinate lookup"
    assert certificate.finite_relation_pair_certified

    output = tmp_path / "embedded-certificate.json"
    assert run_local_index_audit((str(relation_path), "--output", str(output))) == 0
    assert json.loads(output.read_text())["finite_relation_pair_certified"] is True


def test_external_flags_cannot_contradict_persisted_walker_flags(tmp_path):
    relation_path = tmp_path / "walker-open-exit.json.gz"
    cell = _cell(0, (0.0, 1.0), (0,), morse_node=0, dimension=1)
    payload = {
        "schema": GARCIA_WALKER_OPEN_EXIT_RELATION_SCHEMA,
        "model": "adaptive-walker-test",
        "depth": 20,
        "tau": 0.5,
        "morse_graph": {"morse_sets": {"0": [0]}},
        "cells": [
            {
                **cell.to_json_dict(),
                "axis_depth": 0,
                "coordinates": [0],
                "open_exit": {"has_open_exit": False},
            }
        ],
        "charts": {"base": {"chart_id": 0, "bounds": [[0.0, 1.0]]}},
    }
    with gzip.open(relation_path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    loaded = load_exit_aware_relation(relation_path)

    with pytest.raises(ValueError, match="contradict persisted relation flags"):
        merge_relation_inputs(
            ExitAwareLocalIndexInput(
                morse_node=0,
                collar_layers=0,
                open_exit_flags={0: True},
            ),
            loaded,
        )


def test_programmatic_open_exit_flags_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        ExitAwareLocalIndexInput(
            morse_node=0,
            collar_layers=1,
            open_exit_flags={0: 1},  # type: ignore[dict-item]
        )


def test_input_round_trip_and_duplicate_flags_are_checked(tmp_path):
    specification = ExitAwareLocalIndexInput(
        morse_node=3,
        collar_layers=2,
        open_exit_flags={4: False, 8: True},
        allowed_cells=frozenset({4, 8}),
        quotient_neighbor_pairs=((8, 4),),
    )
    path = specification.write_json(tmp_path / "input.json")
    loaded = ExitAwareLocalIndexInput.read_json(path)
    assert loaded == specification

    payload = specification.to_json_dict()
    payload["source_exit_flags"].append({"source": 4, "open_exit": True})
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source-exit flag"):
        ExitAwareLocalIndexInput.read_json(path)
