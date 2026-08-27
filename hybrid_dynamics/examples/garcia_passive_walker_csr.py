"""Fingerprint-bound compact relation checkpoints for the Garcia walker.

The CMGDB MapGraph adjacency is authoritative and remains in memory-mapped
CSR form.  This module stores cell geometry and open-exit provenance in a
separate atomic JSON-lines file; no edge row is copied into JSON.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .garcia_passive_walker_local import (
    AtlasWalkerCell,
    GarciaDyadicCell,
    GarciaMixedDyadicFamily,
    GarciaRelationSizeLimitExceeded,
    _AuditedGarciaSuspensionBoxMap,
    _GARCIA_BOXMAP_REVISION,
    _GARCIA_GUARD_ALIGNED_MODEL_REVISION,
    _GARCIA_PHYSICAL_MODEL_REVISION,
    _GARCIA_SAMPLE_PROVENANCE_POLICY,
    _GarciaRawRelationStage,
    _StoredMorseGraph,
    _canonical_json_bytes,
    _canonicalize_morse_snapshot,
    _morse_snapshot,
    _piece_has_unmatched_ambient_exit,
    _quotient_cross_chart_pairs,
    _sha256_json,
    _source_provenance_from_dict,
    _validate_endpoint_precompute_metadata,
    _validate_raw_morse_snapshot,
    _dyadic_rectangle_cover,
    _finalize_garcia_raw_relation,
    _garcia_coordinate_system,
    _garcia_model_revision,
    audit_garcia_sparse_relation_connectivity,
)
from ..src.cmgdb_suspension_boxmap import build_cmgdb_atlas_model
from .garcia_passive_walker_atlas import (
    GarciaWalkerQuotientIncidence,
    GuardAlignedGarciaWalkerQuotientIncidence,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from .garcia_passive_walker_suspension import (
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
)


CSR_CONFIGURATION_SCHEMA = "garcia-walker-mapgraph-csr-configuration-v1"
CSR_REFERENCE_SCHEMA = "garcia-walker-mapgraph-csr-reference-v1"
CSR_PROVENANCE_SCHEMA = "garcia-walker-open-exit-csr-provenance-v1"
CSR_RELATION_REVISION = "garcia-raw-mapgraph-csr-relation-v1"
CSR_BUNDLE_SCHEMA = "garcia-walker-csr-relation-bundle-v1"

_OPEN_EXIT_KEYS = {
    "callback_invocations",
    "successful_samples_per_invocation",
    "failed_samples_per_invocation",
    "failure_reasons",
    "unresolved_stage_edges",
    "returned_pieces",
    "explicit_empty_callback",
    "mapgraph_empty_image",
    "active_target_cells",
    "missing_in_domain_target_cells",
    "missing_in_domain_witnesses",
    "target_pieces",
    "ambient_boundary_pieces",
    "wholly_outside_active_family_pieces",
    "has_open_exit",
}


def _family_cells_payload(stage: _GarciaRawRelationStage) -> list[dict[str, object]]:
    return [
        {
            "chart_id": int(cell.chart_id),
            "axis_depth": int(cell.axis_depth),
            "coordinates": [int(value) for value in cell.coordinates],
        }
        for cell in stage.family.cells
    ]


def garcia_csr_configuration(
    stage: _GarciaRawRelationStage,
    *,
    family_provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the exact physical/family configuration bound into CMGDB CSR."""

    family_cells = _family_cells_payload(stage)
    return {
        "schema": CSR_CONFIGURATION_SCHEMA,
        "relation_revision": CSR_RELATION_REVISION,
        "physical_model_revision": _garcia_model_revision(stage.walker),
        "coordinate_system": _garcia_coordinate_system(stage.walker),
        "boxmap_revision": _GARCIA_BOXMAP_REVISION,
        "model": "garcia_passive_walker_fixed_time_suspension_atlas",
        "relation_scope": "local_open_exit_fixed_time_suspension",
        "t_star": float(stage.box_map.t_star),
        "gamma": float(stage.walker.gamma),
        "guard_delta": float(stage.walker.guard_delta),
        "transversality_eta": float(stage.walker.transversality_eta),
        "samples_per_axis": int(stage.box_map.samples_per_axis),
        "padding_cells": float(stage.box_map.padding_cells),
        "max_step": float(stage.box_map.max_step),
        "max_jumps": int(stage.box_map.max_jumps),
        "require_domain_path": bool(stage.box_map.require_domain_path),
        "base_bounds": [list(value) for value in stage.charts.base_bounds],
        "handle_bounds": [list(value) for value in stage.charts.handle_bounds],
        "family_role": stage.family.role,
        "min_axis_depth": int(stage.family.min_axis_depth),
        "max_axis_depth": int(stage.family.max_axis_depth),
        "active_cells": len(stage.family.cells),
        "family_cells_sha256": _sha256_json(family_cells),
        "family_provenance": (
            None if family_provenance is None else dict(family_provenance)
        ),
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
    }


def write_and_load_garcia_map_graph_csr(
    stage: _GarciaRawRelationStage,
    path: str | Path,
    *,
    family_provenance: Mapping[str, object] | None,
    max_vertices: int,
    max_edges: int,
    max_payload_bytes: int,
) -> tuple[Any, dict[str, object]]:
    """Persist native CSR immediately and return its strict mmap relation."""

    import CMGDB

    configuration = garcia_csr_configuration(
        stage,
        family_provenance=family_provenance,
    )
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=max_vertices,
        max_edges=max_edges,
        max_payload_bytes=max_payload_bytes,
    )
    target = CMGDB.write_map_graph_csr_checkpoint(
        stage.map_graph,
        path,
        configuration=configuration,
        caps=caps,
        target_dtype="auto",
    )
    metadata = CMGDB.read_map_graph_csr_metadata(target)
    fingerprint = metadata["fingerprint"]["sha256"]
    relation = CMGDB.load_map_graph_csr_checkpoint(
        target,
        expected_configuration=configuration,
        caps=caps,
        expected_fingerprint=fingerprint,
    )
    reference = {
        "schema": CSR_REFERENCE_SCHEMA,
        "path": Path(target).name,
        "fingerprint": fingerprint,
        "configuration": configuration,
        "configuration_sha256": metadata["configuration_sha256"],
        "vertices": metadata["vertices"],
        "edges": metadata["edges"],
        "payload_bytes": metadata["payload_bytes"],
        "offsets_dtype": metadata["files"]["offsets"]["dtype"],
        "targets_dtype": metadata["files"]["targets"]["dtype"],
    }
    return relation, reference


def _provenance_configuration(stage: _GarciaRawRelationStage) -> dict[str, object]:
    reference = stage.relation_reference
    if not isinstance(reference, Mapping):
        raise ValueError("CSR provenance requires an authoritative relation reference")
    return {
        "schema": CSR_PROVENANCE_SCHEMA,
        "relation_revision": CSR_RELATION_REVISION,
        "physical_model_revision": _garcia_model_revision(stage.walker),
        "coordinate_system": _garcia_coordinate_system(stage.walker),
        "boxmap_revision": _GARCIA_BOXMAP_REVISION,
        "model": "garcia_passive_walker_fixed_time_suspension_atlas",
        "relation_scope": "post_mapgraph_pre_audit_checkpoint",
        "t_star": float(stage.box_map.t_star),
        "gamma": float(stage.walker.gamma),
        "guard_delta": float(stage.walker.guard_delta),
        "transversality_eta": float(stage.walker.transversality_eta),
        "samples_per_axis": int(stage.box_map.samples_per_axis),
        "padding_cells": float(stage.box_map.padding_cells),
        "max_step": float(stage.box_map.max_step),
        "max_jumps": int(stage.box_map.max_jumps),
        "require_domain_path": bool(stage.box_map.require_domain_path),
        "base_bounds": [list(value) for value in stage.charts.base_bounds],
        "handle_bounds": [list(value) for value in stage.charts.handle_bounds],
        "family_role": stage.family.role,
        "min_axis_depth": int(stage.family.min_axis_depth),
        "max_axis_depth": int(stage.family.max_axis_depth),
        "active_cells": len(stage.family.cells),
        "family_cells_sha256": reference["configuration"]["family_cells_sha256"],
        "family_provenance": (
            None if stage.family_provenance is None else dict(stage.family_provenance)
        ),
    }


def _cell_record(
    stage: _GarciaRawRelationStage,
    index: int,
) -> dict[str, object]:
    cell = stage.cells[index]
    dyadic = stage.dyadic_cells[index]
    return {
        "record": "cell",
        "index": index,
        "chart_id": int(cell.chart_id),
        "axis_depth": int(dyadic.axis_depth),
        "coordinates": [int(value) for value in dyadic.coordinates],
        "bounds": [float(value) for value in cell.bounds],
        "relation_evaluated": True,
        "open_exit": stage.source_provenance[index].to_dict(),
    }


def _update_digest(digest: Any, kind: str, value: object) -> None:
    kind_bytes = kind.encode("ascii")
    value_bytes = _canonical_json_bytes(value)
    digest.update(len(kind_bytes).to_bytes(4, "big"))
    digest.update(kind_bytes)
    digest.update(len(value_bytes).to_bytes(8, "big"))
    digest.update(value_bytes)


def write_garcia_csr_provenance_checkpoint(
    stage: _GarciaRawRelationStage,
    path: str | Path,
) -> Path:
    """Atomically store geometry/provenance bound to the separate CSR graph."""

    reference = stage.relation_reference
    if not isinstance(reference, Mapping):
        raise ValueError("stage has no CSR relation reference")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csr_path = target.parent / str(reference["path"])
    if csr_path.parent != target.parent or not csr_path.is_dir():
        raise ValueError("CSR relation must be a sibling checkpoint directory")
    configuration = _provenance_configuration(stage)
    scope = {
        "checkpoint_scope": "post_mapgraph_pre_audit",
        "checkpoint_complete": False,
        "audit_complete": False,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "sample_failure_provenance_validation": _GARCIA_SAMPLE_PROVENANCE_POLICY,
    }
    header = {
        "record": "header",
        "schema": CSR_PROVENANCE_SCHEMA,
        **scope,
        "configuration": configuration,
        "relation_csr": dict(reference),
        "compute_metadata": dict(stage.compute_metadata),
        "raw_elapsed_seconds": float(stage.raw_elapsed_seconds),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        opener = gzip.open if target.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
            for kind, value in (
                ("scope", scope),
                ("configuration", configuration),
                ("relation_csr", dict(reference)),
                ("compute_metadata", dict(stage.compute_metadata)),
                ("raw_elapsed_seconds", float(stage.raw_elapsed_seconds)),
            ):
                _update_digest(digest, kind, value)
            for index in range(len(stage.cells)):
                record = _cell_record(stage, index)
                _update_digest(digest, "cell", record)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            pairs = [list(pair) for pair in stage.quotient_neighbor_pairs]
            _update_digest(digest, "quotient_neighbor_pairs", pairs)
            morse = _canonicalize_morse_snapshot(_morse_snapshot(stage.morse_graph))
            trailer_fields = {
                "record": "trailer",
                "schema": CSR_PROVENANCE_SCHEMA,
                "checkpoint_complete": True,
                "audit_complete": False,
                "scientific_result_accepted": False,
                "cell_count": len(stage.cells),
                "relation_edges": int(reference["edges"]),
                "quotient_neighbor_pairs": pairs,
                "morse_graph": morse,
                "morse_snapshot_sha256": _sha256_json(morse),
                "content_sha256": digest.hexdigest(),
                "relation_csr_fingerprint": reference["fingerprint"],
            }
            trailer_fields["fingerprint"] = _sha256_json(trailer_fields)
            stream.write(
                json.dumps(trailer_fields, sort_keys=True, allow_nan=False) + "\n"
            )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o644)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def complete_garcia_csr_bundle(
    stage: _GarciaRawRelationStage,
    temporary_bundle: str | Path,
    target_bundle: str | Path,
) -> Path:
    """Complete one all-or-nothing CSR + provenance bundle directory."""

    temporary = Path(temporary_bundle)
    target = Path(target_bundle)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite CSR bundle {target}")
    if not temporary.is_dir() or set(item.name for item in temporary.iterdir()) != {
        "relation.csr"
    }:
        raise ValueError("temporary CSR bundle is not at its pre-provenance stage")
    provenance = temporary / "provenance.jsonl.gz"
    write_garcia_csr_provenance_checkpoint(stage, provenance)
    reference = stage.relation_reference
    if not isinstance(reference, Mapping) or reference["path"] != "relation.csr":
        raise ValueError("CSR bundle relation reference is not internal and canonical")
    manifest_fields: dict[str, object] = {
        "schema": CSR_BUNDLE_SCHEMA,
        "bundle_complete": True,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
        "relation_directory": "relation.csr",
        "relation_fingerprint": reference["fingerprint"],
        "relation_configuration_sha256": reference["configuration_sha256"],
        "provenance_file": "provenance.jsonl.gz",
        "provenance_sha256": _sha256_file(provenance),
    }
    manifest_fields["fingerprint"] = _sha256_json(manifest_fields)
    manifest = temporary / "manifest.json"
    manifest.write_bytes(
        json.dumps(
            manifest_fields,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with manifest.open("rb") as stream:
        os.fsync(stream.fileno())
    directory = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.replace(temporary, target)
    parent = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return target


def validate_garcia_csr_bundle(path: str | Path) -> tuple[Path, dict[str, object]]:
    """Reject partial, extra-file, rehashed-claim, or corrupted bundles."""

    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("CSR bundle must be a real directory")
    if set(item.name for item in source.iterdir()) != {
        "relation.csr",
        "provenance.jsonl.gz",
        "manifest.json",
    }:
        raise ValueError("CSR bundle is partial or contains unknown entries")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("CSR bundle manifest must be a regular file")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "bundle_complete",
        "scientific_result_accepted",
        "whole_cell_outer_enclosure_certified",
        "continuous_system_conley_index_certified",
        "relation_directory",
        "relation_fingerprint",
        "relation_configuration_sha256",
        "provenance_file",
        "provenance_sha256",
        "fingerprint",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw["schema"] != CSR_BUNDLE_SCHEMA
        or raw["bundle_complete"] is not True
        or raw["scientific_result_accepted"] is not False
        or raw["whole_cell_outer_enclosure_certified"] is not False
        or raw["continuous_system_conley_index_certified"] is not False
        or raw["relation_directory"] != "relation.csr"
        or raw["provenance_file"] != "provenance.jsonl.gz"
    ):
        raise ValueError("CSR bundle manifest is malformed or mis-scoped")
    unsigned = {key: value for key, value in raw.items() if key != "fingerprint"}
    if raw["fingerprint"] != _sha256_json(unsigned):
        raise ValueError("CSR bundle manifest fingerprint is invalid")
    provenance = source / "provenance.jsonl.gz"
    if (
        not provenance.is_file()
        or provenance.is_symlink()
        or _sha256_file(provenance) != raw["provenance_sha256"]
    ):
        raise ValueError("CSR bundle provenance checksum is invalid")
    return provenance, raw


def load_csr_reference(
    provenance_path: str | Path,
    reference: Mapping[str, object],
    *,
    max_vertices: int,
    max_edges: int,
    max_payload_bytes: int,
) -> Any:
    """Strictly resolve and load a sibling CSR reference."""

    import CMGDB

    required = {
        "schema",
        "path",
        "fingerprint",
        "configuration",
        "configuration_sha256",
        "vertices",
        "edges",
        "payload_bytes",
        "offsets_dtype",
        "targets_dtype",
    }
    if not isinstance(reference, Mapping) or set(reference) != required:
        raise ValueError("CSR relation reference is not canonical")
    if reference["schema"] != CSR_REFERENCE_SCHEMA:
        raise ValueError("CSR relation reference has the wrong schema")
    csr_configuration_keys = {
        "schema",
        "relation_revision",
        "physical_model_revision",
        "boxmap_revision",
        "model",
        "relation_scope",
        "t_star",
        "gamma",
        "guard_delta",
        "transversality_eta",
        "samples_per_axis",
        "padding_cells",
        "max_step",
        "max_jumps",
        "require_domain_path",
        "base_bounds",
        "handle_bounds",
        "family_role",
        "min_axis_depth",
        "max_axis_depth",
        "active_cells",
        "family_cells_sha256",
        "family_provenance",
        "whole_cell_outer_enclosure_certified",
        "continuous_system_conley_index_certified",
    }
    configuration = reference["configuration"]
    configuration_keys = set(configuration) if isinstance(configuration, Mapping) else set()
    coordinate_system = (
        configuration.get("coordinate_system", "physical")
        if isinstance(configuration, Mapping)
        else None
    )
    expected_model_revision = (
        _GARCIA_GUARD_ALIGNED_MODEL_REVISION
        if coordinate_system == "guard_aligned"
        else _GARCIA_PHYSICAL_MODEL_REVISION
    )
    if (
        not isinstance(configuration, Mapping)
        or configuration_keys
        not in (csr_configuration_keys, csr_configuration_keys | {"coordinate_system"})
        or coordinate_system not in {"physical", "guard_aligned"}
        or configuration["schema"] != CSR_CONFIGURATION_SCHEMA
        or configuration["relation_revision"] != CSR_RELATION_REVISION
        or configuration["physical_model_revision"]
        != expected_model_revision
        or configuration["boxmap_revision"] != _GARCIA_BOXMAP_REVISION
        or configuration["relation_scope"]
        != "local_open_exit_fixed_time_suspension"
        or configuration["require_domain_path"] is not True
        or configuration["whole_cell_outer_enclosure_certified"] is not False
        or configuration["continuous_system_conley_index_certified"] is not False
    ):
        raise ValueError("CSR relation configuration is not canonical or is mis-scoped")
    name = reference["path"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("CSR relation reference must name one sibling directory")
    source = Path(provenance_path)
    csr_path = source.parent / name
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=max_vertices,
        max_edges=max_edges,
        max_payload_bytes=max_payload_bytes,
    )
    relation = CMGDB.load_map_graph_csr_checkpoint(
        csr_path,
        expected_configuration=reference["configuration"],
        caps=caps,
        expected_fingerprint=reference["fingerprint"],
    )
    metadata = relation.metadata
    expected_summary = {
        "configuration_sha256": metadata["configuration_sha256"],
        "vertices": metadata["vertices"],
        "edges": metadata["edges"],
        "payload_bytes": metadata["payload_bytes"],
        "offsets_dtype": metadata["files"]["offsets"]["dtype"],
        "targets_dtype": metadata["files"]["targets"]["dtype"],
    }
    if any(reference[key] != value for key, value in expected_summary.items()):
        raise ValueError("CSR relation reference summary disagrees with checkpoint")
    return relation


def _strict_configuration(raw: object) -> dict[str, object]:
    required = {
        "schema",
        "relation_revision",
        "physical_model_revision",
        "boxmap_revision",
        "model",
        "relation_scope",
        "t_star",
        "gamma",
        "guard_delta",
        "transversality_eta",
        "samples_per_axis",
        "padding_cells",
        "max_step",
        "max_jumps",
        "require_domain_path",
        "base_bounds",
        "handle_bounds",
        "family_role",
        "min_axis_depth",
        "max_axis_depth",
        "active_cells",
        "family_cells_sha256",
        "family_provenance",
    }
    if not isinstance(raw, Mapping) or set(raw) not in (
        required,
        required | {"coordinate_system"},
    ):
        raise ValueError("CSR provenance configuration is not canonical")
    result = dict(raw)
    coordinate_system = result.get("coordinate_system", "physical")
    expected_model_revision = (
        _GARCIA_GUARD_ALIGNED_MODEL_REVISION
        if coordinate_system == "guard_aligned"
        else _GARCIA_PHYSICAL_MODEL_REVISION
    )
    if (
        coordinate_system not in {"physical", "guard_aligned"}
        or result["schema"] != CSR_PROVENANCE_SCHEMA
        or result["relation_revision"] != CSR_RELATION_REVISION
        or result["physical_model_revision"] != expected_model_revision
        or result["boxmap_revision"] != _GARCIA_BOXMAP_REVISION
        or result["model"]
        != "garcia_passive_walker_fixed_time_suspension_atlas"
        or result["relation_scope"] != "post_mapgraph_pre_audit_checkpoint"
        or result["require_domain_path"] is not True
    ):
        raise ValueError("CSR provenance configuration has incompatible semantics")
    for key in (
        "samples_per_axis",
        "max_jumps",
        "min_axis_depth",
        "max_axis_depth",
        "active_cells",
    ):
        if type(result[key]) is not int or int(result[key]) < 0:
            raise ValueError(f"CSR provenance {key} must be a nonnegative integer")
    for key in (
        "t_star",
        "gamma",
        "guard_delta",
        "transversality_eta",
        "padding_cells",
        "max_step",
    ):
        if type(result[key]) not in (int, float) or not np.isfinite(float(result[key])):
            raise ValueError(f"CSR provenance {key} must be finite")
    if float(result["t_star"]) <= 0 or float(result["max_step"]) <= 0:
        raise ValueError("CSR provenance has a nonpositive time parameter")
    if int(result["samples_per_axis"]) < 3 or int(result["max_jumps"]) < 1:
        raise ValueError("CSR provenance sampling/jump configuration is invalid")
    if float(result["padding_cells"]) < 0:
        raise ValueError("CSR provenance padding must be nonnegative")
    if not isinstance(result["family_role"], str):
        raise ValueError("CSR provenance family role must be a string")
    for label in ("base_bounds", "handle_bounds"):
        bounds = result[label]
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError(f"CSR provenance {label} is malformed")
        for interval in bounds:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(type(value) not in (int, float) for value in interval)
                or not all(np.isfinite(float(value)) for value in interval)
                or float(interval[0]) >= float(interval[1])
            ):
                raise ValueError(f"CSR provenance {label} is malformed")
    digest = result["family_cells_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("CSR provenance family hash is malformed")
    return result


def _validate_compute_metadata(
    raw: object,
    provenance: Mapping[int, object],
) -> dict[str, object]:
    required = {
        "box_map_unique_source_boxes_evaluated",
        "new_whole_source_values_evaluated",
        "reused_whole_source_values",
        "box_map_sampled_points_evaluated",
        "unique_physical_endpoint_evaluations",
        "reused_lattice_endpoint_evaluations",
        "box_map_failed_samples",
        "box_map_empty_images",
        "box_map_unresolved_stage_edges",
        "endpoint_precompute",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("CSR provenance compute metadata is not canonical")
    metadata = dict(raw)
    for key in required - {"endpoint_precompute"}:
        if type(metadata[key]) is not int or int(metadata[key]) < 0:
            raise ValueError(f"CSR provenance compute metric {key} is invalid")
    values = tuple(provenance.values())
    logical = sum(
        int(item.successful_samples) + int(item.failed_samples) for item in values
    )
    if metadata["box_map_unique_source_boxes_evaluated"] != len(values):
        raise ValueError("CSR provenance source count is inconsistent")
    if (
        metadata["new_whole_source_values_evaluated"]
        + metadata["reused_whole_source_values"]
        != len(values)
    ):
        raise ValueError("CSR provenance whole-source totals are inconsistent")
    if metadata["box_map_sampled_points_evaluated"] != logical:
        raise ValueError("CSR provenance sampled-point total is inconsistent")
    if (
        metadata["unique_physical_endpoint_evaluations"]
        + metadata["reused_lattice_endpoint_evaluations"]
        != logical
    ):
        raise ValueError("CSR provenance endpoint-cache totals are inconsistent")
    if metadata["box_map_failed_samples"] != sum(
        int(item.failed_samples) for item in values
    ):
        raise ValueError("CSR provenance failure total is inconsistent")
    if metadata["box_map_empty_images"] != sum(
        bool(item.explicit_empty_callback) for item in values
    ):
        raise ValueError("CSR provenance empty-image total is inconsistent")
    if metadata["box_map_unresolved_stage_edges"] != sum(
        len(item.unresolved_stage_edges) for item in values
    ):
        raise ValueError("CSR provenance unresolved-stage total is inconsistent")
    _validate_endpoint_precompute_metadata(
        metadata["endpoint_precompute"],
        logical_samples=logical,
        unique_endpoint_evaluations=int(
            metadata["unique_physical_endpoint_evaluations"]
        ),
        reused_endpoint_evaluations=int(
            metadata["reused_lattice_endpoint_evaluations"]
        ),
        new_source_values=int(metadata["new_whole_source_values_evaluated"]),
        reused_source_values=int(metadata["reused_whole_source_values"]),
    )
    return metadata


def read_garcia_csr_provenance_checkpoint(
    path: str | Path,
    *,
    expected_family: GarciaMixedDyadicFamily | None = None,
    expected_family_provenance: Mapping[str, object] | None = None,
    expected_run_configuration: Mapping[str, object] | None = None,
    max_vertices: int,
    max_relation_edges: int,
    max_csr_payload_bytes: int,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
) -> _GarciaRawRelationStage:
    """Strictly resume a complete CSR plus geometry/provenance bundle."""

    for name, value in (
        ("max_vertices", max_vertices),
        ("max_relation_edges", max_relation_edges),
        ("max_csr_payload_bytes", max_csr_payload_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative exact integer")
    requested_source = Path(path)
    bundle_manifest: dict[str, object] | None = None
    if requested_source.is_dir():
        source, bundle_manifest = validate_garcia_csr_bundle(requested_source)
    else:
        source = requested_source
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        line = stream.readline()
        if not line:
            raise ValueError("CSR provenance checkpoint is empty")
        header = json.loads(line)
        scope = {
            "checkpoint_scope": "post_mapgraph_pre_audit",
            "checkpoint_complete": False,
            "audit_complete": False,
            "scientific_result_accepted": False,
            "whole_cell_outer_enclosure_certified": False,
            "sample_failure_provenance_validation": _GARCIA_SAMPLE_PROVENANCE_POLICY,
        }
        expected_header = {
            "record",
            "schema",
            *scope,
            "configuration",
            "relation_csr",
            "compute_metadata",
            "raw_elapsed_seconds",
        }
        if (
            not isinstance(header, Mapping)
            or set(header) != expected_header
            or header.get("record") != "header"
            or header.get("schema") != CSR_PROVENANCE_SCHEMA
            or any(header.get(key) != value for key, value in scope.items())
        ):
            raise ValueError("CSR provenance header is malformed or mis-scoped")
        configuration = _strict_configuration(header["configuration"])
        if expected_run_configuration is not None:
            if not isinstance(expected_run_configuration, Mapping):
                raise ValueError(
                    "CSR provenance physical/run configuration differs from request"
                )
            expected_run_keys = {
                "t_star",
                "gamma",
                "guard_delta",
                "transversality_eta",
                "samples_per_axis",
                "padding_cells",
                "max_step",
                "max_jumps",
                "require_domain_path",
            }
            expected_keys = set(expected_run_configuration)
            expected_coordinate_system = expected_run_configuration.get(
                "coordinate_system",
                "physical",
            )
            if (
                expected_keys
                not in (expected_run_keys, expected_run_keys | {"coordinate_system"})
                or expected_coordinate_system not in {"physical", "guard_aligned"}
                or configuration.get("coordinate_system", "physical")
                != expected_coordinate_system
                or any(
                    configuration[key] != expected_run_configuration[key]
                    for key in expected_run_keys
                )
            ):
                raise ValueError(
                    "CSR provenance physical/run configuration differs from request"
                )
        if expected_family_provenance is not None and configuration[
            "family_provenance"
        ] != dict(expected_family_provenance):
            raise ValueError("CSR provenance family manifest differs from request")
        relation = load_csr_reference(
            source,
            header["relation_csr"],
            max_vertices=max_vertices,
            max_edges=max_relation_edges,
            max_payload_bytes=max_csr_payload_bytes,
        )
        reference = dict(header["relation_csr"])
        if bundle_manifest is not None and (
            bundle_manifest["relation_fingerprint"] != reference["fingerprint"]
            or bundle_manifest["relation_configuration_sha256"]
            != reference["configuration_sha256"]
        ):
            raise ValueError("CSR bundle manifest disagrees with provenance reference")
        csr_configuration = reference["configuration"]
        for key in (
            "relation_revision",
            "physical_model_revision",
            "boxmap_revision",
            "model",
            "t_star",
            "gamma",
            "guard_delta",
            "transversality_eta",
            "samples_per_axis",
            "padding_cells",
            "max_step",
            "max_jumps",
            "require_domain_path",
            "base_bounds",
            "handle_bounds",
            "family_role",
            "min_axis_depth",
            "max_axis_depth",
            "active_cells",
            "family_cells_sha256",
            "family_provenance",
        ):
            if csr_configuration[key] != configuration[key]:
                raise ValueError(f"CSR and provenance configurations differ at {key}")
        if csr_configuration.get("coordinate_system", "physical") != configuration.get(
            "coordinate_system",
            "physical",
        ):
            raise ValueError("CSR and provenance coordinate systems differ")
        elapsed = header["raw_elapsed_seconds"]
        if type(elapsed) not in (int, float) or not np.isfinite(float(elapsed)) or elapsed < 0:
            raise ValueError("CSR provenance elapsed time is invalid")

        digest = hashlib.sha256()
        for kind, value in (
            ("scope", scope),
            ("configuration", configuration),
            ("relation_csr", reference),
            ("compute_metadata", dict(header["compute_metadata"])),
            ("raw_elapsed_seconds", float(elapsed)),
        ):
            _update_digest(digest, kind, value)
        cells: list[AtlasWalkerCell] = []
        dyadic_cells: list[GarciaDyadicCell] = []
        provenance: dict[int, object] = {}
        raw_open_exit: dict[int, bool] = {}
        trailer: Mapping[str, object] | None = None
        cell_keys = {
            "record",
            "index",
            "chart_id",
            "axis_depth",
            "coordinates",
            "bounds",
            "relation_evaluated",
            "open_exit",
        }
        for line in stream:
            if not line.strip():
                raise ValueError("CSR provenance contains a blank record")
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("CSR provenance record is not an object")
            if record.get("record") == "trailer":
                if trailer is not None:
                    raise ValueError("CSR provenance contains two trailers")
                trailer = record
                continue
            if trailer is not None:
                raise ValueError("CSR provenance has data after its trailer")
            index = len(cells)
            if index >= max_vertices or index >= relation.num_vertices():
                raise ValueError("CSR provenance contains more cells than its graph/cap")
            if set(record) != cell_keys or record.get("record") != "cell":
                raise ValueError("CSR provenance cell record is not canonical")
            if type(record["index"]) is not int or record["index"] != index:
                raise ValueError("CSR provenance cells are not complete and ordered")
            if type(record["chart_id"]) is not int or type(record["axis_depth"]) is not int:
                raise ValueError("CSR provenance dyadic fields must be integers")
            if record["relation_evaluated"] is not True:
                raise ValueError(f"CSR relation was not evaluated at source {index}")
            coordinates = record["coordinates"]
            bounds = record["bounds"]
            if (
                not isinstance(coordinates, list)
                or len(coordinates) != 4
                or any(type(value) is not int for value in coordinates)
                or not isinstance(bounds, list)
                or len(bounds) != 8
                or any(type(value) not in (int, float) for value in bounds)
                or not all(np.isfinite(float(value)) for value in bounds)
                or any(float(a) >= float(b) for a, b in zip(bounds[:4], bounds[4:], strict=True))
            ):
                raise ValueError(f"CSR provenance cell {index} has invalid geometry")
            open_exit = record["open_exit"]
            if not isinstance(open_exit, Mapping) or set(open_exit) != _OPEN_EXIT_KEYS:
                raise ValueError(f"CSR source {index} exit provenance is not canonical")
            dyadic = GarciaDyadicCell(
                int(record["chart_id"]),
                int(record["axis_depth"]),
                tuple(coordinates),
            )
            cells.append(
                AtlasWalkerCell(index, dyadic.chart_id, tuple(float(v) for v in bounds))
            )
            dyadic_cells.append(dyadic)
            provenance[index] = _source_provenance_from_dict(index, open_exit)
            raw_open_exit[index] = bool(open_exit["has_open_exit"])
            _update_digest(digest, "cell", record)
    if trailer is None:
        raise ValueError("CSR provenance lacks a complete trailer")
    trailer_keys = {
        "record",
        "schema",
        "checkpoint_complete",
        "audit_complete",
        "scientific_result_accepted",
        "cell_count",
        "relation_edges",
        "quotient_neighbor_pairs",
        "morse_graph",
        "morse_snapshot_sha256",
        "content_sha256",
        "relation_csr_fingerprint",
        "fingerprint",
    }
    if (
        set(trailer) != trailer_keys
        or trailer.get("record") != "trailer"
        or trailer.get("schema") != CSR_PROVENANCE_SCHEMA
        or trailer.get("checkpoint_complete") is not True
        or trailer.get("audit_complete") is not False
        or trailer.get("scientific_result_accepted") is not False
    ):
        raise ValueError("CSR provenance trailer is malformed or claims acceptance")
    unsigned = {key: value for key, value in trailer.items() if key != "fingerprint"}
    if trailer["fingerprint"] != _sha256_json(unsigned):
        raise ValueError("CSR provenance trailer fingerprint is invalid")
    if (
        trailer["relation_csr_fingerprint"] != reference["fingerprint"]
        or type(trailer["cell_count"]) is not int
        or trailer["cell_count"] != len(cells)
        or type(trailer["relation_edges"]) is not int
        or trailer["relation_edges"] != relation.num_cached_edges()
        or len(cells) != relation.num_vertices()
    ):
        raise ValueError("CSR provenance graph summary is inconsistent")
    raw_pairs = trailer["quotient_neighbor_pairs"]
    if not isinstance(raw_pairs, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(value) is not int for value in pair)
        for pair in raw_pairs
    ):
        raise ValueError("CSR provenance quotient pairs are malformed")
    pairs = tuple((pair[0], pair[1]) for pair in raw_pairs)
    if pairs != tuple(sorted(set(pairs))) or any(
        first < 0 or first >= second or second >= len(cells) for first, second in pairs
    ):
        raise ValueError("CSR provenance quotient pairs are not canonical")
    _update_digest(digest, "quotient_neighbor_pairs", raw_pairs)
    if trailer["content_sha256"] != digest.hexdigest():
        raise ValueError("CSR provenance content hash is invalid")
    morse = trailer["morse_graph"]
    if not isinstance(morse, Mapping) or trailer["morse_snapshot_sha256"] != _sha256_json(dict(morse)):
        raise ValueError("CSR provenance Morse snapshot hash is invalid")

    family = GarciaMixedDyadicFamily(
        tuple(sorted(dyadic_cells)), role=str(configuration["family_role"])
    )
    family_payload = [
        {
            "chart_id": int(cell.chart_id),
            "axis_depth": int(cell.axis_depth),
            "coordinates": [int(value) for value in cell.coordinates],
        }
        for cell in family.cells
    ]
    if (
        len(family.cells) != configuration["active_cells"]
        or family.min_axis_depth != configuration["min_axis_depth"]
        or family.max_axis_depth != configuration["max_axis_depth"]
        or _sha256_json(family_payload) != configuration["family_cells_sha256"]
    ):
        raise ValueError("CSR provenance family summary is inconsistent")
    if expected_family is not None and family != expected_family:
        raise ValueError("CSR provenance family differs from the requested family")

    coordinate_system = configuration.get("coordinate_system", "physical")
    if coordinate_system == "guard_aligned":
        walker = GuardAlignedGarciaPassiveWalker(
            gamma=float(configuration["gamma"]),
            guard_delta=float(configuration["guard_delta"]),
            transversality_eta=float(configuration["transversality_eta"]),
            domain_bounds=[
                tuple(map(float, interval)) for interval in configuration["base_bounds"]
            ],
            max_jumps=int(configuration["max_jumps"]),
        )
        charts = garcia_guard_aligned_atlas_charts(
            base_bounds=walker.domain_bounds,
            guard_delta=walker.guard_delta,
            transversality_eta=walker.transversality_eta,
        )
        incidence = GuardAlignedGarciaWalkerQuotientIncidence(
            charts,
            transversality_eta=walker.transversality_eta,
        )
    else:
        walker = GarciaPassiveWalker(
            gamma=float(configuration["gamma"]),
            guard_delta=float(configuration["guard_delta"]),
            transversality_eta=float(configuration["transversality_eta"]),
            domain_bounds=[
                tuple(map(float, interval)) for interval in configuration["base_bounds"]
            ],
            max_jumps=int(configuration["max_jumps"]),
        )
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
    if not np.allclose(charts.handle_bounds, configuration["handle_bounds"], atol=1e-13, rtol=0):
        raise ValueError("CSR provenance handle chart disagrees with physical parameters")
    box_map = _AuditedGarciaSuspensionBoxMap(
        walker.system,
        charts,
        float(configuration["t_star"]),
        samples_per_axis=int(configuration["samples_per_axis"]),
        padding_cells=float(configuration["padding_cells"]),
        max_step=float(configuration["max_step"]),
        max_jumps=int(configuration["max_jumps"]),
        require_domain_path=True,
        diagnostics_limit=0,
    )

    class _NoDynamics:
        def __init__(self) -> None:
            self.charts = charts

        def __call__(self, *_args: object) -> list[object]:
            raise AssertionError("CSR resume must not evaluate the source box map")

    model = build_cmgdb_atlas_model(
        _NoDynamics(),  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    )
    atlas = model.phaseSpace()
    if int(atlas.size()) != len(cells):
        raise ValueError("CSR provenance family disagrees with native Atlas size")
    for index, (cell, dyadic) in enumerate(zip(cells, dyadic_cells, strict=True)):
        atlas_cell = atlas.cell(index)
        if (
            int(atlas_cell.chart_id) != cell.chart_id
            or not np.allclose(atlas_cell.bounds, cell.bounds, atol=1e-13, rtol=0)
            or not np.allclose(dyadic.bounds(charts), cell.bounds, atol=1e-13, rtol=0)
        ):
            raise ValueError(f"CSR provenance cell {index} disagrees with native Atlas order")
    if _quotient_cross_chart_pairs(cells, incidence, charts, atlas) != pairs:
        raise ValueError("CSR provenance quotient pairs disagree with geometry")
    neighbors_by_handle: dict[int, set[int]] = {}
    for first, second in pairs:
        handle = first if cells[first].chart_id == charts.handle_chart_id else second
        base = second if handle == first else first
        neighbors_by_handle.setdefault(handle, set()).add(base)
    frozen_neighbors = {
        handle: frozenset(values) for handle, values in neighbors_by_handle.items()
    }
    expected_samples = int(configuration["samples_per_axis"]) ** 4
    for index, item in provenance.items():
        if item.callback_invocations < 1 or item.successful_samples + item.failed_samples != expected_samples:
            raise ValueError(f"CSR source {index} has inconsistent sample counts")
        if sum(count for _reason, count in item.failure_reasons) != item.failed_samples:
            raise ValueError(f"CSR source {index} has inconsistent failure counts")
        recovered = frozenset(
            int(target)
            for chart_id, bounds in item.target_pieces
            for target in atlas.cover(chart_id, bounds)
        )
        if recovered != frozenset(relation[index]):
            raise ValueError(f"CSR source {index} pieces disagree with adjacency row")
        missing: set[GarciaDyadicCell] = set()
        ambient = outside = 0
        for chart_id, bounds in item.target_pieces:
            covered = tuple(int(value) for value in atlas.cover(chart_id, bounds))
            outside += int(not covered)
            fine_cover, crossings = _dyadic_rectangle_cover(
                charts, chart_id, bounds, family.max_axis_depth
            )
            missing.update(cell for cell in fine_cover if family.covering_cell(cell) is None)
            ambient += int(
                _piece_has_unmatched_ambient_exit(
                    charts,
                    incidence,
                    atlas,
                    chart_id,
                    bounds,
                    crossings,
                    item.target_pieces,
                    quotient_neighbors_by_handle=frozen_neighbors,
                )
            )
        if (
            item.mapgraph_empty_image != (not recovered)
            or item.returned_pieces != len(item.target_pieces)
            or item.explicit_empty_callback != (not item.target_pieces)
            or item.active_target_cells != len(recovered)
            or item.missing_in_domain_target_cells != len(missing)
            or frozenset(item.missing_in_domain_witnesses) != frozenset(missing)
            or item.ambient_boundary_pieces != ambient
            or item.wholly_outside_active_family_pieces != outside
            or raw_open_exit[index] != item.has_open_exit
        ):
            raise ValueError(f"CSR source {index} has inconsistent derived exit provenance")
    compute_metadata = _validate_compute_metadata(header["compute_metadata"], provenance)
    _validate_raw_morse_snapshot(morse, relation)
    connectivity = audit_garcia_sparse_relation_connectivity(
        relation,
        tuple(dyadic_cells),
        pairs,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )
    resumed_metadata = dict(compute_metadata)
    resumed_metadata["raw_checkpoint_resumed"] = True
    return _GarciaRawRelationStage(
        family=family,
        walker=walker,
        charts=charts,
        box_map=box_map,
        model=model,
        morse_graph=_StoredMorseGraph(morse),
        map_graph=relation,
        cells=tuple(cells),
        dyadic_cells=tuple(dyadic_cells),
        relation=relation,
        quotient_neighbor_pairs=pairs,
        source_provenance=provenance,  # type: ignore[arg-type]
        raw_elapsed_seconds=float(elapsed),
        compute_metadata=resumed_metadata,
        checkpoint_path=str(requested_source),
        connectivity_audit=connectivity,
        relation_reference=reference,
        family_provenance=configuration["family_provenance"],  # type: ignore[arg-type]
    )


def resume_garcia_local_relation_from_csr_checkpoint(
    path: str | Path,
    *,
    expected_family: GarciaMixedDyadicFamily | None = None,
    expected_family_provenance: Mapping[str, object] | None = None,
    expected_run_configuration: Mapping[str, object],
    max_vertices: int,
    max_relation_edges: int,
    max_csr_payload_bytes: int,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
    reference_base_samples_per_stride: int = 17,
    reference_handle_samples: int = 9,
    reference_max_step: float = 0.005,
) -> Any:
    """Resume only reference/candidate audits from a strict CSR bundle."""

    stage = read_garcia_csr_provenance_checkpoint(
        path,
        expected_family=expected_family,
        expected_family_provenance=expected_family_provenance,
        expected_run_configuration=expected_run_configuration,
        max_vertices=max_vertices,
        max_relation_edges=max_relation_edges,
        max_csr_payload_bytes=max_csr_payload_bytes,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )
    return _finalize_garcia_raw_relation(
        stage,
        reference_base_samples_per_stride=reference_base_samples_per_stride,
        reference_handle_samples=reference_handle_samples,
        reference_max_step=reference_max_step,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )


__all__ = [
    "CSR_BUNDLE_SCHEMA",
    "CSR_CONFIGURATION_SCHEMA",
    "CSR_PROVENANCE_SCHEMA",
    "CSR_REFERENCE_SCHEMA",
    "CSR_RELATION_REVISION",
    "complete_garcia_csr_bundle",
    "garcia_csr_configuration",
    "load_csr_reference",
    "read_garcia_csr_provenance_checkpoint",
    "resume_garcia_local_relation_from_csr_checkpoint",
    "write_and_load_garcia_map_graph_csr",
    "write_garcia_csr_provenance_checkpoint",
    "validate_garcia_csr_bundle",
]
