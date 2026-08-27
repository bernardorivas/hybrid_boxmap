"""Lazy CSR relation resolution for persisted Garcia derived audits.

The large MapGraph adjacency lives in the atomic Garcia CSR bundle.  A small
derived audit refers to that bundle by a sibling directory name and immutable
manifest digests.  This module resolves that reference without recreating
``cell.image`` lists or Python edge containers.

The caller must supply the exact expected run configuration and fresh resource
caps.  Values copied out of the artifact are never treated as the expectation
for their own validation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .garcia_passive_walker_csr import (
    CSR_PROVENANCE_SCHEMA,
    load_csr_reference,
    validate_garcia_csr_bundle,
)
from .garcia_passive_walker_local import _GARCIA_SAMPLE_PROVENANCE_POLICY


DERIVED_BUNDLE_REFERENCE_SCHEMA: Final = (
    "garcia-walker-derived-csr-bundle-reference-v1"
)
_REFERENCE_KEYS: Final = frozenset(
    {
        "schema",
        "path",
        "bundle_fingerprint",
        "relation_fingerprint",
        "relation_configuration_sha256",
        "provenance_sha256",
    }
)
_PROVENANCE_HEADER_KEYS: Final = frozenset(
    {
        "record",
        "schema",
        "checkpoint_scope",
        "checkpoint_complete",
        "audit_complete",
        "scientific_result_accepted",
        "whole_cell_outer_enclosure_certified",
        "sample_failure_provenance_validation",
        "configuration",
        "relation_csr",
        "compute_metadata",
        "raw_elapsed_seconds",
    }
)
_SHARED_CONFIGURATION_KEYS: Final = (
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
)
_LEGACY_PROVENANCE_CONFIGURATION_KEYS: Final = frozenset(
    {"schema", "relation_scope", *_SHARED_CONFIGURATION_KEYS}
)


__all__ = [
    "DERIVED_BUNDLE_REFERENCE_SCHEMA",
    "GarciaCSRDerivedArtifact",
    "attach_garcia_csr_derived_artifact_fingerprint",
    "garcia_csr_bundle_reference",
    "load_garcia_csr_derived_artifact",
    "validate_garcia_csr_derived_artifact_fingerprint",
]


def _exact_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a nonnegative exact integer")
    try:
        result = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{label} must be a nonnegative exact integer") from None
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return int(result)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not permitted")


def _strict_json_loads(encoded: bytes) -> object:
    try:
        return json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("persisted JSON is unreadable") from None


def _normalize_configuration(configuration: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(configuration, Mapping):
        raise TypeError("expected_configuration must be a mapping")
    try:
        encoded = json.dumps(
            dict(configuration),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        result = _strict_json_loads(encoded)
    except (TypeError, ValueError):
        raise TypeError(
            "expected_configuration must contain finite JSON-compatible values"
        ) from None
    if not isinstance(result, dict):  # pragma: no cover - dict above is decisive
        raise TypeError("expected_configuration must normalize to an object")
    return result


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_garcia_csr_derived_artifact_fingerprint(
    payload: dict[str, object],
) -> dict[str, object]:
    """Attach the canonical content fingerprint used by the strict loader."""

    if not isinstance(payload, dict):
        raise TypeError("derived artifact payload must be a mutable dictionary")
    if "fingerprint" in payload:
        raise ValueError("derived artifact already has a fingerprint")
    artifact_schema = payload.get("schema")
    if not isinstance(artifact_schema, str) or not artifact_schema:
        raise ValueError("derived artifact must have a nonempty schema")
    payload["fingerprint"] = {
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
    }
    return payload


def validate_garcia_csr_derived_artifact_fingerprint(
    payload: Mapping[str, object],
) -> str:
    """Validate and return a derived artifact's canonical content digest."""

    if not isinstance(payload, Mapping):
        raise TypeError("derived artifact payload must be a mapping")
    fingerprint = payload.get("fingerprint")
    required = {"algorithm", "sha256"}
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != required:
        raise ValueError("derived artifact fingerprint is not canonical")
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected = _canonical_sha256(unsigned)
    if (
        fingerprint["algorithm"] != "sha256"
        or not _is_sha256(fingerprint["sha256"])
        or fingerprint["sha256"] != expected
    ):
        raise ValueError("derived artifact fingerprint does not match its content")
    return expected


def _read_bounded_json_artifact(path: Path, *, max_bytes: int) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("derived artifact must be a regular file")
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            encoded = stream.read(max_bytes + 1)
            if len(encoded) > max_bytes:
                raise ValueError("derived artifact exceeds its explicit byte cap")
    except (OSError, EOFError, gzip.BadGzipFile):
        raise ValueError("derived artifact is unreadable") from None
    payload = _strict_json_loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("derived artifact must contain one JSON object")
    return payload


def _read_provenance_header(path: Path, *, max_bytes: int) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("CSR provenance must be a regular file")
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            encoded = stream.readline(max_bytes + 1)
    except (OSError, EOFError, gzip.BadGzipFile):
        raise ValueError("CSR provenance header is unreadable") from None
    if not encoded or len(encoded) > max_bytes or not encoded.endswith(b"\n"):
        raise ValueError("CSR provenance header is absent or exceeds its byte cap")
    header = _strict_json_loads(encoded)
    if not isinstance(header, dict):
        raise ValueError("CSR provenance header must be an object")
    if set(header) != _PROVENANCE_HEADER_KEYS:
        raise ValueError("CSR provenance header fields are not canonical")
    if (
        header["record"] != "header"
        or header["schema"] != CSR_PROVENANCE_SCHEMA
        or header["checkpoint_scope"] != "post_mapgraph_pre_audit"
        or header["checkpoint_complete"] is not False
        or header["audit_complete"] is not False
        or header["scientific_result_accepted"] is not False
        or header["whole_cell_outer_enclosure_certified"] is not False
        or header["sample_failure_provenance_validation"]
        != _GARCIA_SAMPLE_PROVENANCE_POLICY
    ):
        raise ValueError("CSR provenance header is malformed or mis-scoped")
    return header


def _strict_reference(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != _REFERENCE_KEYS:
        raise ValueError("derived artifact relation_bundle reference is not canonical")
    reference = dict(raw)
    if reference["schema"] != DERIVED_BUNDLE_REFERENCE_SCHEMA:
        raise ValueError("derived artifact uses an unsupported bundle reference")
    name = reference["path"]
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
    ):
        raise ValueError("derived bundle path must name one sibling directory")
    for key in (
        "bundle_fingerprint",
        "relation_fingerprint",
        "relation_configuration_sha256",
        "provenance_sha256",
    ):
        if not _is_sha256(reference[key]):
            raise ValueError(f"derived bundle {key} is not a lowercase SHA-256 digest")
    return reference


def _bundle_reference_fields(
    artifact_path: Path,
    bundle_path: Path,
    *,
    max_provenance_header_bytes: int,
) -> tuple[dict[str, object], dict[str, object], Path]:
    if (
        not bundle_path.name
        or bundle_path.name in {".", ".."}
        or Path(bundle_path.name).name != bundle_path.name
    ):
        raise ValueError("CSR bundle must be one named sibling directory")
    if bundle_path.parent.resolve() != artifact_path.parent.resolve():
        raise ValueError("derived artifact and CSR bundle must be siblings")
    provenance, manifest = validate_garcia_csr_bundle(bundle_path)
    header = _read_provenance_header(
        provenance,
        max_bytes=max_provenance_header_bytes,
    )
    relation_reference = header["relation_csr"]
    if not isinstance(relation_reference, Mapping):
        raise ValueError("CSR provenance lacks a canonical relation reference")
    fields = {
        "schema": DERIVED_BUNDLE_REFERENCE_SCHEMA,
        "path": bundle_path.name,
        "bundle_fingerprint": manifest["fingerprint"],
        "relation_fingerprint": manifest["relation_fingerprint"],
        "relation_configuration_sha256": manifest[
            "relation_configuration_sha256"
        ],
        "provenance_sha256": manifest["provenance_sha256"],
    }
    if (
        relation_reference.get("fingerprint") != fields["relation_fingerprint"]
        or relation_reference.get("configuration_sha256")
        != fields["relation_configuration_sha256"]
    ):
        raise ValueError("CSR bundle manifest disagrees with provenance header")
    return fields, header, provenance


def garcia_csr_bundle_reference(
    derived_artifact_path: str | Path,
    bundle_path: str | Path,
    *,
    max_provenance_header_bytes: int,
) -> dict[str, object]:
    """Build the canonical six-field reference for one sibling CSR bundle."""

    header_cap = _exact_nonnegative_integer(
        max_provenance_header_bytes,
        label="max_provenance_header_bytes",
    )
    artifact = Path(derived_artifact_path)
    bundle = Path(bundle_path)
    fields, _header, _provenance = _bundle_reference_fields(
        artifact,
        bundle,
        max_provenance_header_bytes=header_cap,
    )
    return fields


@dataclass(frozen=True)
class GarciaCSRDerivedArtifact:
    """A small persisted audit paired with its strict mmap relation."""

    path: Path
    payload: Mapping[str, object]
    bundle_path: Path
    bundle_reference: Mapping[str, object]
    relation_reference: Mapping[str, object]
    relation: Any
    canonical_sha256: str


def load_garcia_csr_derived_artifact(
    path: str | Path,
    *,
    expected_artifact_schema: str,
    expected_configuration: Mapping[str, object],
    max_vertices: int,
    max_relation_edges: int,
    max_csr_payload_bytes: int,
    max_derived_artifact_bytes: int,
    max_provenance_header_bytes: int,
) -> GarciaCSRDerivedArtifact:
    """Load a derived audit and its exact relation without expanding edges.

    All caps and the expected run configuration come from the caller.  The
    bundle path is restricted to one sibling directory, every manifest digest
    is cross-checked, and the returned relation is CMGDB's read-only mmap CSR
    mapping.
    """

    if not isinstance(expected_artifact_schema, str) or not expected_artifact_schema:
        raise TypeError("expected_artifact_schema must be a nonempty string")
    vertex_cap = _exact_nonnegative_integer(max_vertices, label="max_vertices")
    edge_cap = _exact_nonnegative_integer(
        max_relation_edges,
        label="max_relation_edges",
    )
    payload_cap = _exact_nonnegative_integer(
        max_csr_payload_bytes,
        label="max_csr_payload_bytes",
    )
    artifact_cap = _exact_nonnegative_integer(
        max_derived_artifact_bytes,
        label="max_derived_artifact_bytes",
    )
    header_cap = _exact_nonnegative_integer(
        max_provenance_header_bytes,
        label="max_provenance_header_bytes",
    )
    normalized_expected = _normalize_configuration(expected_configuration)

    artifact_path = Path(path)
    payload = _read_bounded_json_artifact(artifact_path, max_bytes=artifact_cap)
    if payload.get("schema") != expected_artifact_schema:
        raise ValueError("derived artifact schema differs from this run")
    canonical_sha256 = validate_garcia_csr_derived_artifact_fingerprint(payload)
    reference = _strict_reference(payload.get("relation_bundle"))
    bundle_path = artifact_path.parent / str(reference["path"])
    observed, header, provenance_path = _bundle_reference_fields(
        artifact_path,
        bundle_path,
        max_provenance_header_bytes=header_cap,
    )
    if reference != observed:
        raise ValueError("derived artifact bundle reference disagrees with the bundle")

    relation_reference = header["relation_csr"]
    if not isinstance(relation_reference, Mapping):  # checked above, keeps typing local
        raise ValueError("CSR provenance relation reference is malformed")
    stored_configuration = relation_reference.get("configuration")
    if not isinstance(stored_configuration, Mapping):
        raise ValueError("CSR relation configuration is malformed")
    if dict(stored_configuration) != normalized_expected:
        raise ValueError("CSR relation configuration differs from this run")
    provenance_configuration = header.get("configuration")
    if not isinstance(provenance_configuration, Mapping):
        raise ValueError("CSR provenance configuration is malformed")
    has_coordinate_system = "coordinate_system" in normalized_expected
    if has_coordinate_system:
        coordinate_system = normalized_expected["coordinate_system"]
        if coordinate_system not in {"physical", "guard_aligned"}:
            raise ValueError("expected coordinate_system is unsupported")
    elif normalized_expected.get("physical_model_revision") != (
        "garcia-physical-event-reset-v2"
    ):
        raise ValueError(
            "only the legacy physical configuration may omit coordinate_system"
        )
    shared_keys = _SHARED_CONFIGURATION_KEYS + (
        ("coordinate_system",) if has_coordinate_system else ()
    )
    provenance_keys = _LEGACY_PROVENANCE_CONFIGURATION_KEYS | (
        {"coordinate_system"} if has_coordinate_system else set()
    )
    if (
        set(provenance_configuration) != provenance_keys
        or provenance_configuration["schema"] != CSR_PROVENANCE_SCHEMA
        or provenance_configuration["relation_scope"]
        != "post_mapgraph_pre_audit_checkpoint"
    ):
        raise ValueError("CSR provenance configuration is not canonical")
    for key in shared_keys:
        if (
            key not in normalized_expected
            or provenance_configuration[key] != normalized_expected[key]
        ):
            raise ValueError(
                f"CSR provenance and expected configurations differ at {key}"
            )

    relation = load_csr_reference(
        provenance_path,
        relation_reference,
        max_vertices=vertex_cap,
        max_edges=edge_cap,
        max_payload_bytes=payload_cap,
    )
    return GarciaCSRDerivedArtifact(
        path=artifact_path,
        payload=payload,
        bundle_path=bundle_path,
        bundle_reference=reference,
        relation_reference=dict(relation_reference),
        relation=relation,
        canonical_sha256=canonical_sha256,
    )
