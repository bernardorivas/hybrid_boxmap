"""Tests for the pinned Garcia orbit tube and compact relation bundle."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_dynamics.examples import garcia_passive_walker_csr as csr_module
from hybrid_dynamics.examples import garcia_passive_walker_local as local_module
from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    read_garcia_csr_provenance_checkpoint,
    resume_garcia_local_relation_from_csr_checkpoint,
    validate_garcia_csr_bundle,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaSourceExitProvenance,
    compute_garcia_local_relation,
    full_garcia_dyadic_family,
    garcia_failure_reason_is_explicit_domain_exit,
    garcia_non_domain_failure_sources,
)
from hybrid_dynamics.examples.garcia_passive_walker_tube import (
    DEFAULT_TUBE_SHA256,
    audit_garcia_orbit_tube_candidate_boundary,
    garcia_orbit_tube_acceptance_blockers,
    read_garcia_orbit_tube,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUBE_PATH = (
    PROJECT_ROOT
    / "data/garcia_passive_walker_atlas/"
    "tube_geometry_tau050_depth20_r00625.json.gz"
)


def _caps() -> dict[str, int]:
    return {
        "max_vertices": 100,
        "max_relation_edges": 100_000,
        "max_csr_payload_bytes": 10_000_000,
        "max_relation_storage_bytes": 100_000_000,
        "max_undirected_adjacencies": 100_000,
        "max_adjacency_storage_bytes": 100_000_000,
    }


def _run_configuration() -> dict[str, object]:
    return {
        "t_star": 0.5,
        "gamma": 0.0172,
        "guard_delta": 0.05,
        "transversality_eta": 0.1,
        "samples_per_axis": 3,
        "padding_cells": 1.0,
        "max_step": 0.02,
        "max_jumps": 20,
        "require_domain_path": True,
    }


def _compute_small_bundle(path: Path):
    family = full_garcia_dyadic_family(1)
    family_provenance = {
        "schema": "garcia-test-family-v1",
        "fingerprint": "0" * 64,
    }
    run = compute_garcia_local_relation(
        family,
        precompute_workers=0,
        csr_bundle_path=path,
        family_provenance=family_provenance,
        max_relation_edges=100_000,
        max_csr_payload_bytes=10_000_000,
        max_native_cache_bytes=10_000_000,
        max_relation_storage_bytes=100_000_000,
        max_undirected_adjacencies=100_000,
        max_adjacency_storage_bytes=100_000_000,
    )
    return family, family_provenance, run


def test_pinned_depth20_tube_has_complete_nonpercolating_attachment():
    tube = read_garcia_orbit_tube(
        TUBE_PATH,
        expected_fingerprint=DEFAULT_TUBE_SHA256,
    )
    assert len(tube.family.cells) == 29_906
    assert tube.family.chart_counts() == {0: 21_906, 1: 8_000}
    assert tube.added_base_attachment_cells == 60
    assert tube.active_quotient_incidences == 2_921
    assert tube.reverse_open_quotient_incidences == 28_938
    assert len(tube.same_chart_open_boundary_cells) == 22_924
    assert len(tube.reverse_open_base_boundary_cells) == 1_237
    assert len(tube.attachment_carriers) == 500
    assert all(bases for _handle, bases in tube.attachment_carriers)
    boundary_index = tube.family.cells.index(tube.same_chart_open_boundary_cells[0])
    audit = audit_garcia_orbit_tube_candidate_boundary(
        tube,
        tube.family.cells,
        frozenset((boundary_index,)),
    )
    assert not audit.passed
    assert audit.same_chart_witnesses_in_s == frozenset((boundary_index,))


def test_exit_set_failure_policy_allows_only_explicit_domain_exit_reasons():
    def source(index: int, reasons: tuple[tuple[str, int], ...]):
        failures = sum(count for _reason, count in reasons)
        return GarciaSourceExitProvenance(
            source_index=index,
            callback_invocations=2,
            successful_samples=81 - failures,
            failed_samples=failures,
            failure_reasons=reasons,
            unresolved_stage_edges=(),
            returned_pieces=1,
            explicit_empty_callback=False,
            mapgraph_empty_image=False,
            active_target_cells=1,
            missing_in_domain_target_cells=0,
            missing_in_domain_witnesses=(),
            target_pieces=(),
            ambient_boundary_pieces=int(bool(failures)),
            wholly_outside_active_family_pieces=0,
        )

    explicit = "ValueError: trajectory leaves the declared state-space bounds"
    incomplete = "ValueError: recorded trajectory does not cover suspension time T=0.5"
    inadmissible = "InadmissibleHeelstrikeError: initial state lies beyond contact"
    assert garcia_failure_reason_is_explicit_domain_exit(explicit)
    assert not garcia_failure_reason_is_explicit_domain_exit(incomplete)
    assert not garcia_failure_reason_is_explicit_domain_exit(inadmissible)
    provenance = {
        0: source(0, ((explicit, 2),)),
        1: source(1, ((explicit, 1), (incomplete, 1))),
        2: source(2, ()),
    }
    assert garcia_non_domain_failure_sources(provenance, (0, 1, 2)) == frozenset((1,))


def test_tube_blockers_allow_only_predeclared_A_exit_failures():
    def provenance(reason: str) -> GarciaSourceExitProvenance:
        return GarciaSourceExitProvenance(
            source_index=0,
            callback_invocations=1,
            successful_samples=80,
            failed_samples=1,
            failure_reasons=((reason, 1),),
            unresolved_stage_edges=(),
            returned_pieces=1,
            explicit_empty_callback=False,
            mapgraph_empty_image=False,
            active_target_cells=1,
            missing_in_domain_target_cells=0,
            missing_in_domain_witnesses=(),
            target_pieces=(),
            ambient_boundary_pieces=1,
            wholly_outside_active_family_pieces=0,
        )

    candidate = SimpleNamespace(
        reference_endpoint_misses=0,
        reference_evaluation_failures=0,
        failed_sources_in_s=(),
        unresolved_stage_sources_in_s=(),
        open_exit_sources_in_s=(),
        empty_sources_in_s=(),
        disconnected_sources_in_s=(),
        missing_in_domain_sources_in_s=(),
        touches_nonglued_ambient_boundary=(),
        pair_second_condition_violations=(),
        reference_recovered=True,
        a_cells=frozenset((0,)),
        disconnected_sources_in_a=(0,),
    )
    support = SimpleNamespace(terminal_blockers=(), added_cells=frozenset())
    explicit = "ValueError: trajectory leaves the declared state-space bounds"
    non_domain = "ValueError: recorded trajectory does not cover suspension time T=0.5"
    allowed = SimpleNamespace(candidate=candidate, source_provenance={0: provenance(explicit)})
    blocked = SimpleNamespace(candidate=candidate, source_provenance={0: provenance(non_domain)})
    assert garcia_orbit_tube_acceptance_blockers(
        allowed,
        support,
        same_chart_boundary_in_s=set(),
        reverse_seam_boundary_in_s=set(),
    ) == []
    assert garcia_orbit_tube_acceptance_blockers(
        blocked,
        support,
        same_chart_boundary_in_s=set(),
        reverse_seam_boundary_in_s=set(),
    ) == ["non_domain_sample_failure_in_A"]


def test_csr_bundle_fresh_resume_parity_and_no_json_edges(tmp_path: Path):
    bundle = tmp_path / "bundle"
    family, provenance, fresh = _compute_small_bundle(bundle)
    assert {item.name for item in bundle.iterdir()} == {
        "manifest.json",
        "provenance.jsonl.gz",
        "relation.csr",
    }
    validate_garcia_csr_bundle(bundle)
    with gzip.open(bundle / "provenance.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    cell_records = [record for record in records if record["record"] == "cell"]
    assert cell_records
    assert all("image" not in record for record in cell_records)

    resumed_stage = read_garcia_csr_provenance_checkpoint(
        bundle,
        expected_family=family,
        expected_family_provenance=provenance,
        expected_run_configuration=_run_configuration(),
        **_caps(),
    )
    resumed = resume_garcia_local_relation_from_csr_checkpoint(
        bundle,
        expected_family=family,
        expected_family_provenance=provenance,
        expected_run_configuration=_run_configuration(),
        **_caps(),
    )
    wrong_configuration = {**_run_configuration(), "t_star": 0.75}
    with pytest.raises(ValueError, match="physical/run configuration differs"):
        read_garcia_csr_provenance_checkpoint(
            bundle,
            expected_family=family,
            expected_family_provenance=provenance,
            expected_run_configuration=wrong_configuration,
            **_caps(),
        )
    assert resumed_stage.map_graph is resumed_stage.relation
    assert resumed.connectivity_audit == fresh.connectivity_audit
    assert resumed.candidate.to_dict() == fresh.candidate.to_dict()
    assert resumed.refinement_locator.to_dict() == fresh.refinement_locator.to_dict()
    assert resumed.morse_graph.edges() == fresh.morse_graph.edges()
    assert resumed.morse_graph.vertices() == fresh.morse_graph.vertices()
    assert all(
        tuple(resumed.relation[source]) == tuple(fresh.relation[source])
        for source in range(len(family.cells))
    )
    # Fresh completion itself exercises the enforced native-MapGraph weakref
    # lifetime gate before any global CSR postprocessing begins.


def test_csr_bundle_rejects_partial_corrupt_and_mismatched_inputs(tmp_path: Path):
    bundle = tmp_path / "bundle"
    family, provenance, _run = _compute_small_bundle(bundle)

    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copytree(bundle / "relation.csr", partial / "relation.csr")
    with pytest.raises(ValueError, match="partial"):
        validate_garcia_csr_bundle(partial)

    corrupt = tmp_path / "corrupt"
    shutil.copytree(bundle, corrupt)
    provenance_path = corrupt / "provenance.jsonl.gz"
    encoded = bytearray(provenance_path.read_bytes())
    encoded[len(encoded) // 2] ^= 1
    provenance_path.write_bytes(encoded)
    with pytest.raises(ValueError, match="checksum"):
        read_garcia_csr_provenance_checkpoint(
            corrupt,
            expected_family=family,
            expected_family_provenance=provenance,
            expected_run_configuration=_run_configuration(),
            **_caps(),
        )

    with pytest.raises(ValueError, match="family manifest"):
        read_garcia_csr_provenance_checkpoint(
            bundle,
            expected_family=family,
            expected_family_provenance={**provenance, "fingerprint": "1" * 64},
            expected_run_configuration=_run_configuration(),
            **_caps(),
        )


@pytest.mark.parametrize("failure_stage", ["after_csr", "provenance", "before_rename"])
def test_bundle_transaction_cleans_ordinary_failure_and_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
):
    target = tmp_path / "bundle"
    if failure_stage == "after_csr":
        original = csr_module.write_and_load_garcia_map_graph_csr

        def fail_after_csr(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected after-CSR failure")

        monkeypatch.setattr(
            csr_module, "write_and_load_garcia_map_graph_csr", fail_after_csr
        )
    elif failure_stage == "provenance":
        def fail_provenance(*_args, **_kwargs):
            raise RuntimeError("injected provenance failure")

        monkeypatch.setattr(
            local_module, "_collect_garcia_source_provenance", fail_provenance
        )
    else:
        def fail_completion(*_args, **_kwargs):
            raise RuntimeError("injected before-rename failure")

        monkeypatch.setattr(csr_module, "complete_garcia_csr_bundle", fail_completion)
    with pytest.raises(RuntimeError, match="injected"):
        _compute_small_bundle(target)
    assert not target.exists()
    assert not tuple(tmp_path.glob(".bundle.*.tmp"))
    monkeypatch.undo()
    _compute_small_bundle(target)
    validate_garcia_csr_bundle(target)
