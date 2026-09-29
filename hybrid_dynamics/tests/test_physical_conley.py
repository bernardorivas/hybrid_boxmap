"""Tests for physical Atlas index-pair and carrier gates."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hybrid_dynamics.examples.physical_conley import (
    AtlasRelationSnapshot,
    AtlasTopCellRecord,
    audit_atlas_nerve_finite_relation,
    audit_cellular_carrier,
    audit_physical_conley_candidate,
    build_ball_mapping_cylinder,
    extract_top_cell_index_pair,
)
from hybrid_dynamics.examples.bouncing_ball_atlas import (
    bouncing_ball_atlas_reset_gluing,
)
from hybrid_dynamics.examples.rimless_wheel_atlas import (
    rimless_wheel_atlas_reset_gluing,
)

#: Keys of a stored candidate audit that record the run, not the audit.
_RUN_KEYS = ("elapsed_seconds", "relation_cache", "shared_model_elapsed_seconds")


def _tiny_ball_snapshot(
    images,
    nodes,
    *,
    outer_certified=False,
):
    cells = (
        AtlasTopCellRecord(0, 0, (0.0, -1.0, 1.0, 0.0), tuple(images[0]), nodes.get(0)),
        AtlasTopCellRecord(1, 0, (0.0, 0.0, 1.0, 1.0), tuple(images[1]), nodes.get(1)),
        AtlasTopCellRecord(2, 1, (-1.0, 0.0, 0.0, 1.0), tuple(images[2]), nodes.get(2)),
    )
    return AtlasRelationSnapshot(
        model="tiny-ball",
        depth=2,
        t_star=0.5,
        cells=cells,
        morse_nodes=(0,),
        morse_edges=(),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=outer_certified,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=0,
        box_map_empty_images=0,
        box_map_unresolved_stage_edges=0,
    )


def test_standard_top_cell_pair_is_extracted_before_cellular_closure():
    snapshot = AtlasRelationSnapshot(
        model="pair-test",
        depth=2,
        t_star=1.0,
        cells=(
            AtlasTopCellRecord(0, 0, (0, 0, 1, 1), (0, 1), 0),
            AtlasTopCellRecord(1, 0, (1, 0, 2, 1), (2,), None),
            AtlasTopCellRecord(2, 0, (2, 0, 3, 1), (2,), 1),
        ),
        morse_nodes=(0, 1),
        morse_edges=((0, 1),),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=0,
        box_map_empty_images=0,
        box_map_unresolved_stage_edges=0,
    )

    pair = extract_top_cell_index_pair(snapshot, 0)

    assert pair.s_cells == {0}
    assert pair.f_s_cells == {0, 1}
    assert pair.x_cells == {0, 1}
    assert pair.a_cells == {1}
    assert pair.a_targets_outside_x == ((1, (2,)),)
    assert pair.pair_invariance_passed
    assert pair.combinatorial_isolation_passed
    assert pair.summary()["closures_taken_at_this_stage"] is False


def test_mapping_cylinder_validates_both_reset_attachments_without_new_edges():
    snapshot = _tiny_ball_snapshot(
        {0: (2,), 1: (0,), 2: (1,)},
        {0: 0, 1: 0, 2: 0},
    )

    physical = build_ball_mapping_cylinder(snapshot, restitution=1.0)

    assert physical.attachment_chain_maps_validated
    assert physical.complex.max_dimension == 2
    assert len(physical.top_cells) == len(snapshot.cells)
    assert set(physical.top_to_atlas.values()) == {0, 1, 2}
    assert physical.complex.betti_numbers(modulus=5) == (1, 1, 0)
    assert physical.summary()["atlas_edges_inferred"] == 0


def test_relative_fiber_reports_the_nonacyclic_rest_handle_exactly():
    snapshot = _tiny_ball_snapshot(
        {0: (2,), 1: (0,), 2: (1,)},
        {0: 0, 1: 0, 2: 0},
    )
    pair = extract_top_cell_index_pair(snapshot, 0)
    physical = build_ball_mapping_cylinder(snapshot, restitution=1.0)

    audit = audit_cellular_carrier(snapshot, pair, physical, witness_limit=20)

    assert audit.fiber_classification_counts["nonacyclic"] > 0
    assert not audit.all_relative_fibers_acyclic
    assert any(
        witness.relative_betti_numbers == (1, 1, 0)
        for witness in audit.nonacyclic_witnesses
    )
    # A quotient chain selector can still satisfy dF=Fd; that does not repair
    # the failed acyclic-carrier hypothesis.
    assert audit.chain_map_constructed
    assert audit.carried_chain_map.chain_map_equation_validated


def test_shift_class_is_withheld_when_any_physical_or_carrier_gate_fails():
    snapshot = _tiny_ball_snapshot(
        {0: (2,), 1: (0,), 2: (1,)},
        {0: 0, 1: 0, 2: 0},
    )
    physical = build_ball_mapping_cylinder(snapshot, restitution=1.0)

    result = audit_physical_conley_candidate(
        snapshot,
        physical,
        morse_node=0,
        candidate_name="tiny-M0",
    )

    assert result.cmgdb_shift_result is None
    assert "whole_cell_outer_enclosure_not_certified" in result.withheld_reasons
    assert "relative_carrier_fibers_not_all_acyclic" in result.withheld_reasons
    assert result.summary()["analytic_conley_label_attached"] is False


def test_relation_snapshot_cache_retains_full_adjacency_and_failure_flags(tmp_path):
    snapshot = _tiny_ball_snapshot(
        {0: (2,), 1: (0,), 2: (1,)},
        {0: 0, 1: 0, 2: 0},
    )
    target = tmp_path / "relation.json.gz"

    snapshot.write_gzip_json(target)

    with gzip.open(target, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["schema"] == "physical-conley-atlas-relation-v1"
    assert [cell["image"] for cell in payload["cells"]] == [[2], [0], [1]]
    assert all("callback_failed_samples" in cell for cell in payload["cells"])

    loaded = AtlasRelationSnapshot.read_gzip_json(target)
    assert loaded == snapshot


def test_unevaluated_morse_source_is_not_treated_as_an_empty_map_value():
    snapshot = _tiny_ball_snapshot(
        {0: (2,), 1: (0,), 2: (1,)},
        {0: 0, 1: 0, 2: 0},
    )
    cells = list(snapshot.cells)
    cells[1] = AtlasTopCellRecord(
        index=1,
        chart_id=cells[1].chart_id,
        bounds=cells[1].bounds,
        image=(),
        morse_node=0,
        relation_evaluated=False,
    )
    targeted = AtlasRelationSnapshot(
        model=snapshot.model,
        depth=snapshot.depth,
        t_star=snapshot.t_star,
        cells=tuple(cells),
        morse_nodes=snapshot.morse_nodes,
        morse_edges=snapshot.morse_edges,
        base_chart_id=snapshot.base_chart_id,
        handle_chart_id=snapshot.handle_chart_id,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=0,
        box_map_empty_images=0,
        box_map_unresolved_stage_edges=0,
        relation_scope="targeted_index_candidates",
    )

    with pytest.raises(ValueError, match="relation was not evaluated"):
        extract_top_cell_index_pair(targeted, 0)


def test_actual_nerve_finite_relation_result_keeps_continuous_gate_separate():
    snapshot = AtlasRelationSnapshot(
        model="one-box-finite-relation",
        depth=0,
        t_star=1.0,
        cells=(AtlasTopCellRecord(0, 0, (0.0, 0.0, 1.0, 1.0), (0,), 0),),
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

    audit = audit_atlas_nerve_finite_relation(
        snapshot,
        morse_node=0,
        candidate_name="one-box-M0",
        gluing=bouncing_ball_atlas_reset_gluing(restitution=0.8),
    )
    summary = audit.summary()

    assert audit.finite_relation_conley_index_computed
    assert summary["finite_relation_shift_class"] == ["x-1"]
    assert summary["finite_relation_blockers"] == []
    assert summary["continuous_system_conley_index_certified"] is False
    assert summary["carrier_certificate"]["chain_equation_dF_equals_Fd"] is True


def test_unevaluated_exit_source_blocks_finite_relation_provenance():
    snapshot = AtlasRelationSnapshot(
        model="unevaluated-exit",
        depth=0,
        t_star=1.0,
        cells=(
            AtlasTopCellRecord(0, 0, (0.0, 0.0, 1.0, 1.0), (0, 1), 0),
            AtlasTopCellRecord(
                1,
                0,
                (1.0, 0.0, 2.0, 1.0),
                (),
                None,
                relation_evaluated=False,
            ),
        ),
        morse_nodes=(0,),
        morse_edges=(),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=0,
        box_map_empty_images=0,
        box_map_unresolved_stage_edges=0,
        relation_scope="targeted_index_candidates",
    )

    audit = audit_atlas_nerve_finite_relation(
        snapshot,
        morse_node=0,
        candidate_name="unevaluated-exit-M0",
        gluing=bouncing_ball_atlas_reset_gluing(restitution=0.8),
    )

    assert not audit.finite_relation_conley_index_computed
    assert "mapgraph_relation_not_evaluated_on_all_X_sources" in (
        audit.finite_relation_blockers
    )


@pytest.mark.parametrize(("model", "nodes"), [("ball", (0,)), ("wheel", (0, 1))])
def test_stored_physical_audits_are_the_audits_of_the_stored_relations(model, nodes):
    root = Path(__file__).resolve().parents[2]
    path = root / f"data/physical_conley/physical_conley_audit_{model}.json"
    if not path.exists():
        pytest.skip("stored physical Conley audit is not installed")
    gluing = (
        bouncing_ball_atlas_reset_gluing(restitution=0.8)
        if model == "ball"
        else rimless_wheel_atlas_reset_gluing(alpha=0.4, gamma=0.2)
    )
    candidates = json.loads(path.read_text(encoding="utf-8"))["candidates"]
    assert len(candidates) == len(nodes)
    for node, stored in zip(nodes, candidates):
        snapshot = AtlasRelationSnapshot.read_gzip_json(root / stored["relation_cache"])
        summary = audit_atlas_nerve_finite_relation(
            snapshot,
            morse_node=node,
            candidate_name=stored["candidate"],
            gluing=gluing,
        ).summary()
        expected = {key: value for key, value in stored.items() if key not in _RUN_KEYS}
        assert json.loads(json.dumps(summary)) == expected
