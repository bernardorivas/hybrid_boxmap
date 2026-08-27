from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import CMGDB
import numpy as np
import pytest

import hybrid_dynamics.examples.garcia_passive_walker_csr_derived as derived_module
from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    CSR_BUNDLE_SCHEMA,
    CSR_CONFIGURATION_SCHEMA,
    CSR_PROVENANCE_SCHEMA,
    CSR_REFERENCE_SCHEMA,
    CSR_RELATION_REVISION,
)
from hybrid_dynamics.examples.garcia_passive_walker_csr_derived import (
    DERIVED_BUNDLE_REFERENCE_SCHEMA,
    attach_garcia_csr_derived_artifact_fingerprint,
    garcia_csr_bundle_reference,
    load_garcia_csr_derived_artifact,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    _GARCIA_BOXMAP_REVISION,
    _GARCIA_GUARD_ALIGNED_MODEL_REVISION,
    _GARCIA_PHYSICAL_MODEL_REVISION,
    _GARCIA_SAMPLE_PROVENANCE_POLICY,
    _relation_recurrent_components,
)


ARTIFACT_SCHEMA = "garcia-derived-csr-unit-test-v1"
ARTIFACT_CAP = 1 << 20
HEADER_CAP = 1 << 20


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _map_graph():
    model = CMGDB.AtlasModel(0)
    model.add_chart(10, [0.0], [1.0])
    model.add_chart(20, [-1.0], [1.0])
    model.set_active_subgrid(
        [
            (10, 2, [0]),
            (10, 2, [3]),
            (20, 2, [1]),
        ]
    )

    def box_map(chart_id, bounds):
        if chart_id == 20:
            return [(10, [0.9, 0.9])]
        if bounds[0] < 0.5:
            return [(10, [0.1, 0.1]), (10, [0.9, 0.9])]
        return []

    model.set_map(box_map)
    _morse_graph, graph = CMGDB.ComputeMorseGraph(model)
    return graph


def _configuration() -> dict[str, object]:
    base_bounds = [
        [-0.35, 0.35],
        [-2.0, 2.0],
        [-0.7, 0.7],
        [-3.0, 3.0],
    ]
    handle_bounds = [
        [-0.35, 0.35],
        [-2.0, 2.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    return {
        "schema": CSR_CONFIGURATION_SCHEMA,
        "relation_revision": CSR_RELATION_REVISION,
        "physical_model_revision": _GARCIA_PHYSICAL_MODEL_REVISION,
        "boxmap_revision": _GARCIA_BOXMAP_REVISION,
        "model": "garcia_passive_walker_fixed_time_suspension_atlas",
        "relation_scope": "local_open_exit_fixed_time_suspension",
        "t_star": 0.5,
        "gamma": 0.011,
        "guard_delta": 1.0e-6,
        "transversality_eta": 1.0e-8,
        "samples_per_axis": 3,
        "padding_cells": 1.0,
        "max_step": 0.02,
        "max_jumps": 20,
        "require_domain_path": True,
        "base_bounds": base_bounds,
        "handle_bounds": handle_bounds,
        "family_role": "unit-test active family",
        "min_axis_depth": 0,
        "max_axis_depth": 2,
        "active_cells": 3,
        "family_cells_sha256": "a" * 64,
        "family_provenance": {
            "schema": "garcia-unit-test-family-v1",
            "sha256": "b" * 64,
        },
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
    }


def _provenance_configuration(configuration: dict[str, object]) -> dict[str, object]:
    excluded = {
        "schema",
        "relation_scope",
        "whole_cell_outer_enclosure_certified",
        "continuous_system_conley_index_certified",
    }
    return {
        "schema": CSR_PROVENANCE_SCHEMA,
        "relation_scope": "post_mapgraph_pre_audit_checkpoint",
        **{key: value for key, value in configuration.items() if key not in excluded},
    }


def _write_bundle(
    tmp_path: Path,
    configuration: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "relation-bundle"
    bundle.mkdir()
    relation_path = bundle / "relation.csr"
    configuration = _configuration() if configuration is None else configuration
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=10,
        max_edges=10,
        max_payload_bytes=10_000,
    )
    CMGDB.write_map_graph_csr_checkpoint(
        _map_graph(),
        relation_path,
        configuration=configuration,
        caps=caps,
    )
    metadata = CMGDB.read_map_graph_csr_metadata(relation_path)
    relation_reference = {
        "schema": CSR_REFERENCE_SCHEMA,
        "path": "relation.csr",
        "fingerprint": metadata["fingerprint"]["sha256"],
        "configuration": configuration,
        "configuration_sha256": metadata["configuration_sha256"],
        "vertices": metadata["vertices"],
        "edges": metadata["edges"],
        "payload_bytes": metadata["payload_bytes"],
        "offsets_dtype": metadata["files"]["offsets"]["dtype"],
        "targets_dtype": metadata["files"]["targets"]["dtype"],
    }
    header = {
        "record": "header",
        "schema": CSR_PROVENANCE_SCHEMA,
        "checkpoint_scope": "post_mapgraph_pre_audit",
        "checkpoint_complete": False,
        "audit_complete": False,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "sample_failure_provenance_validation": _GARCIA_SAMPLE_PROVENANCE_POLICY,
        "configuration": _provenance_configuration(configuration),
        "relation_csr": relation_reference,
        "compute_metadata": {},
        "raw_elapsed_seconds": 0.0,
    }
    provenance = bundle / "provenance.jsonl.gz"
    with gzip.open(provenance, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
    provenance_hash = hashlib.sha256(provenance.read_bytes()).hexdigest()
    manifest = {
        "schema": CSR_BUNDLE_SCHEMA,
        "bundle_complete": True,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
        "relation_directory": "relation.csr",
        "relation_fingerprint": relation_reference["fingerprint"],
        "relation_configuration_sha256": relation_reference[
            "configuration_sha256"
        ],
        "provenance_file": "provenance.jsonl.gz",
        "provenance_sha256": provenance_hash,
    }
    manifest["fingerprint"] = _sha256_json(manifest)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle, configuration


def _write_artifact(path: Path, bundle: Path) -> dict[str, object]:
    reference = garcia_csr_bundle_reference(
        path,
        bundle,
        max_provenance_header_bytes=HEADER_CAP,
    )
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "relation_bundle": reference,
        "candidate": {"S": [0], "X": [0, 1], "A": [1]},
        "note": "there are deliberately no cell.image edge lists here",
    }
    attach_garcia_csr_derived_artifact_fingerprint(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load(path: Path, configuration: dict[str, object], **overrides):
    arguments = {
        "expected_artifact_schema": ARTIFACT_SCHEMA,
        "expected_configuration": configuration,
        "max_vertices": 10,
        "max_relation_edges": 10,
        "max_csr_payload_bytes": 10_000,
        "max_derived_artifact_bytes": ARTIFACT_CAP,
        "max_provenance_header_bytes": HEADER_CAP,
    }
    arguments.update(overrides)
    return load_garcia_csr_derived_artifact(path, **arguments)


def test_derived_artifact_resolves_lazy_csr_without_edge_lists(tmp_path):
    bundle, configuration = _write_bundle(tmp_path)
    artifact_path = tmp_path / "derived-audit.json"
    payload = _write_artifact(artifact_path, bundle)

    loaded = _load(artifact_path, configuration)
    assert loaded.payload == payload
    assert loaded.bundle_reference["schema"] == DERIVED_BUNDLE_REFERENCE_SCHEMA
    assert loaded.bundle_path == bundle
    assert isinstance(loaded.relation.offsets, np.memmap)
    assert isinstance(loaded.relation.targets, np.memmap)
    assert [list(loaded.relation[index]) for index in loaded.relation] == [
        [0, 1],
        [],
        [1],
    ]
    assert "cells" not in loaded.payload

    legacy = {0: frozenset((0, 1)), 1: frozenset(), 2: frozenset((1,))}
    assert _relation_recurrent_components(loaded.relation) == (
        _relation_recurrent_components(legacy)
    )
    s_cells = frozenset(payload["candidate"]["S"])
    mmap_x = s_cells | frozenset(
        target for source in s_cells for target in loaded.relation[source]
    )
    legacy_x = s_cells | frozenset(
        target for source in s_cells for target in legacy[source]
    )
    assert mmap_x == legacy_x == frozenset(payload["candidate"]["X"])


def test_expected_configuration_is_external_and_checked_before_csr_load(
    tmp_path,
    monkeypatch,
):
    bundle, configuration = _write_bundle(tmp_path)
    artifact_path = tmp_path / "derived-audit.json"
    _write_artifact(artifact_path, bundle)
    wrong = {**configuration, "t_star": 0.75}

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("configuration mismatch was checked too late")

    monkeypatch.setattr(derived_module, "load_csr_reference", must_not_load)
    with pytest.raises(ValueError, match="configuration differs"):
        _load(artifact_path, wrong)


def test_guard_aligned_coordinate_system_is_canonical_and_strictly_checked(
    tmp_path,
):
    configuration = {
        **_configuration(),
        "coordinate_system": "guard_aligned",
        "physical_model_revision": _GARCIA_GUARD_ALIGNED_MODEL_REVISION,
    }
    bundle, configuration = _write_bundle(tmp_path, configuration)
    artifact_path = tmp_path / "derived-audit.json"
    _write_artifact(artifact_path, bundle)

    loaded = _load(artifact_path, configuration)
    assert loaded.relation_reference["configuration"]["coordinate_system"] == (
        "guard_aligned"
    )
    with pytest.raises(ValueError, match="configuration differs"):
        _load(artifact_path, {**configuration, "coordinate_system": "physical"})


def test_derived_reference_rejects_wrong_schema_path_digest_and_caps(tmp_path):
    bundle, configuration = _write_bundle(tmp_path)
    artifact_path = tmp_path / "derived-audit.json"
    payload = _write_artifact(artifact_path, bundle)

    with pytest.raises(ValueError, match="artifact schema"):
        _load(
            artifact_path,
            configuration,
            expected_artifact_schema="another-derived-schema-v1",
        )

    escaped = json.loads(json.dumps(payload))
    escaped["relation_bundle"]["path"] = "../relation-bundle"
    escaped.pop("fingerprint")
    attach_garcia_csr_derived_artifact_fingerprint(escaped)
    artifact_path.write_text(json.dumps(escaped), encoding="utf-8")
    with pytest.raises(ValueError, match="sibling directory"):
        _load(artifact_path, configuration)

    altered = json.loads(json.dumps(payload))
    altered["relation_bundle"]["bundle_fingerprint"] = "0" * 64
    altered.pop("fingerprint")
    attach_garcia_csr_derived_artifact_fingerprint(altered)
    artifact_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with the bundle"):
        _load(artifact_path, configuration)

    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MemoryError, match="edges"):
        _load(artifact_path, configuration, max_relation_edges=2)


def test_derived_loader_rejects_corrupted_mmap_and_bounded_artifacts(tmp_path):
    bundle, configuration = _write_bundle(tmp_path)
    artifact_path = tmp_path / "derived-audit.json"
    payload = _write_artifact(artifact_path, bundle)

    tampered = json.loads(json.dumps(payload))
    tampered["note"] = "changed without refreshing the fingerprint"
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint does not match"):
        _load(artifact_path, configuration)
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    targets = bundle / "relation.csr" / "targets.npy"
    with targets.open("r+b") as stream:
        stream.truncate(targets.stat().st_size - 1)
    with pytest.raises(ValueError, match="file size"):
        _load(artifact_path, configuration)

    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit byte cap"):
        _load(artifact_path, configuration, max_derived_artifact_bytes=8)
