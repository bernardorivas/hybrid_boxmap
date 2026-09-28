#!/usr/bin/env python3
"""Audit the persisted depth-20 Garcia relation after strict source masking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    read_garcia_csr_provenance_checkpoint,
    validate_garcia_csr_bundle,
)
from hybrid_dynamics.examples.garcia_passive_walker_csr_derived import (
    attach_garcia_csr_derived_artifact_fingerprint,
    garcia_csr_bundle_reference,
)
from hybrid_dynamics.examples.garcia_passive_walker_strict_restriction import (
    STRICT_SOURCE_RESTRICTION_SCHEMA,
    audit_garcia_strict_source_restriction,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


DEFAULT_BUNDLE_FINGERPRINT = (
    "2d5a88134ff2bc426efce0ff9747b45391b7d6fa26c11dfcb621370891bccb18"
)
DEFAULT_BUNDLE = Path(
    "data/garcia_passive_walker_atlas/"
    "tube_relation_bundle_tau050_depth20_geom8b7625700202"
)
DEFAULT_OUTPUT = Path(
    "data/garcia_passive_walker_atlas/"
    "tube_strict_source_restriction_tau050_depth20_geom8b7625700202.json"
)
_PROVENANCE_HEADER_CAP = 4 * 1024 * 1024


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-vertices", type=int, default=50_000)
    parser.add_argument("--max-relation-edges", type=int, default=50_000_000)
    parser.add_argument("--max-spatial-adjacencies", type=int, default=5_000_000)
    parser.add_argument("--max-csr-payload-bytes", type=int, default=2**30)
    parser.add_argument("--max-relation-storage-bytes", type=int, default=2**30)
    parser.add_argument("--max-adjacency-storage-bytes", type=int, default=2**30)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    for label in (
        "max_vertices",
        "max_relation_edges",
        "max_spatial_adjacencies",
        "max_csr_payload_bytes",
        "max_relation_storage_bytes",
        "max_adjacency_storage_bytes",
    ):
        if getattr(args, label) < 0:
            raise SystemExit(f"--{label.replace('_', '-')} must be nonnegative")
    if args.output.parent.resolve() != args.bundle.parent.resolve():
        raise SystemExit("diagnostic output and authoritative bundle must be siblings")
    _provenance, manifest = validate_garcia_csr_bundle(args.bundle)
    if manifest["fingerprint"] != DEFAULT_BUNDLE_FINGERPRINT:
        raise SystemExit("bundle fingerprint differs from the pinned depth-20 run")

    expected_run = {
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
    started = time.perf_counter()
    stage = read_garcia_csr_provenance_checkpoint(
        args.bundle,
        expected_run_configuration=expected_run,
        max_vertices=args.max_vertices,
        max_relation_edges=args.max_relation_edges,
        max_csr_payload_bytes=args.max_csr_payload_bytes,
        max_relation_storage_bytes=args.max_relation_storage_bytes,
        max_undirected_adjacencies=args.max_spatial_adjacencies,
        max_adjacency_storage_bytes=args.max_adjacency_storage_bytes,
    )
    audit = audit_garcia_strict_source_restriction(
        stage,
        max_relation_edges=args.max_relation_edges,
        max_relation_storage_bytes=args.max_relation_storage_bytes,
        max_undirected_adjacencies=args.max_spatial_adjacencies,
        max_adjacency_storage_bytes=args.max_adjacency_storage_bytes,
    )
    payload = {
        "schema": STRICT_SOURCE_RESTRICTION_SCHEMA,
        "relation_bundle": garcia_csr_bundle_reference(
            args.output,
            args.bundle,
            max_provenance_header_bytes=_PROVENANCE_HEADER_CAP,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "no_ODE_or_box_map_evaluation": True,
        "authoritative_source_relation_mutated": False,
        "diagnostic_only": True,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
        "finite_relation_conley_index_computed": False,
        **audit.to_dict(),
    }
    payload = attach_garcia_csr_derived_artifact_fingerprint(payload)
    atomic_write_json(args.output, payload, durable=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "restricted_sources": len(audit.relation.restricted_sources),
                "retained_sources": len(audit.relation)
                - len(audit.relation.restricted_sources),
                "restricted_edges": audit.restricted_connectivity.relation_edges,
                "morse_nodes": len(audit.recurrent_components),
                "reference_candidate": audit.selected_morse_node,
                "pair_defined": audit.pair_defined,
                "scientific_result_accepted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
