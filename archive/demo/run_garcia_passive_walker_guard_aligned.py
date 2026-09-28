#!/usr/bin/env python3
"""Compute one predeclared guard-aligned Garcia local relation."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
import warnings
from dataclasses import replace
from pathlib import Path

from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    resume_garcia_local_relation_from_csr_checkpoint,
)
from hybrid_dynamics.examples.garcia_passive_walker_csr_derived import (
    attach_garcia_csr_derived_artifact_fingerprint,
    garcia_csr_bundle_reference,
)
from hybrid_dynamics.examples.garcia_passive_walker_guard_aligned import (
    GUARD_ALIGNED_FAMILY_ALGORITHM,
    build_guard_aligned_garcia_tube,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaRefinementSizeLimitExceeded,
    GarciaRelationSizeLimitExceeded,
    compute_garcia_local_relation,
    estimated_callback_point_evaluations,
    exit_aware_index_pair_refinement,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
)
from hybrid_dynamics.examples.garcia_passive_walker_tube import (
    audit_garcia_orbit_tube_candidate_boundary,
    garcia_orbit_tube_acceptance_blockers,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


_HEADER_CAP = 4 * 1024 * 1024


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-depth", type=int, choices=(4, 5), required=True)
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--precompute-workers", type=int, default=8)
    parser.add_argument("--max-cells", type=int, default=80_000)
    parser.add_argument("--max-point-evaluations", type=int, default=7_000_000)
    parser.add_argument("--max-relation-edges", type=int, default=75_000_000)
    parser.add_argument("--max-spatial-adjacencies", type=int, default=10_000_000)
    parser.add_argument("--max-csr-payload-gib", type=float, default=1.0)
    parser.add_argument("--max-native-cache-gib", type=float, default=1.5)
    parser.add_argument("--max-connectivity-memory-gib", type=float, default=1.5)
    parser.add_argument("--max-provenance-and-csr-gib", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/garcia_passive_walker_guard_aligned"),
    )
    return parser.parse_args()


def _status(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _read_geometry(path: Path) -> dict[str, object]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("guard-aligned geometry artifact is not an object")
    return result


def _family_reference(path: Path, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "garcia-guard-aligned-family-reference-v1",
        "path": path.name,
        "fingerprint": payload["fingerprint"]["sha256"],
        "family_cells_sha256": payload["family_cells_sha256"],
        "attachment_carriers_sha256": payload["attachment_carriers_sha256"],
        "algorithm": GUARD_ALIGNED_FAMILY_ALGORITHM,
        "coordinate_system": "guard_aligned",
        "legacy_bundle_fingerprint": payload["legacy_negative_diagnostic"][
            "bundle_fingerprint"
        ],
    }


def main() -> int:
    args = _arguments()
    if args.compute and args.resume_bundle is not None:
        raise SystemExit("choose either --compute or --resume-bundle")
    for name in (
        "max_csr_payload_gib",
        "max_native_cache_gib",
        "max_connectivity_memory_gib",
        "max_provenance_and_csr_gib",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    construction = build_guard_aligned_garcia_tube(axis_depth=args.axis_depth)
    geometry_payload = construction.to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.output_dir / (
        f"geometry_tau050_depth{4 * args.axis_depth}_r00625.json.gz"
    )
    if geometry_path.exists():
        canonical_geometry_payload = json.loads(
            json.dumps(geometry_payload, sort_keys=True, allow_nan=False)
        )
        if _read_geometry(geometry_path) != canonical_geometry_payload:
            raise SystemExit("existing geometry differs from the predeclared rebuild")
    else:
        construction.write(geometry_path)
    family_reference = _family_reference(geometry_path, geometry_payload)
    logical_samples = estimated_callback_point_evaluations(construction.family, 3)
    _status(
        "geometry_validated",
        axis_depth=args.axis_depth,
        cells=len(construction.family.cells),
        chart_counts=construction.family.chart_counts(),
        logical_samples=logical_samples,
        fingerprint=geometry_payload["fingerprint"]["sha256"],
        same_chart_boundary_cells=len(construction.same_chart_open_boundary_cells),
        reverse_open_boundary_cells=len(
            construction.reverse_open_base_boundary_cells
        ),
    )
    if len(construction.family.cells) > args.max_cells:
        raise SystemExit("predeclared family exceeds --max-cells")
    if logical_samples > args.max_point_evaluations:
        raise SystemExit("predeclared family exceeds --max-point-evaluations")
    if not args.compute and args.resume_bundle is None:
        _status(
            "geometry_only_stop",
            scientific_result_accepted=False,
            reason="source-box dynamics require explicit --compute",
        )
        return 0

    tag = geometry_payload["fingerprint"]["sha256"][:12]
    bundle_path = args.output_dir / (
        f"relation_bundle_tau050_depth{4 * args.axis_depth}_geom{tag}"
    )
    summary_path = args.output_dir / (
        f"relation_audit_tau050_depth{4 * args.axis_depth}_geom{tag}.json"
    )
    if args.resume_bundle is not None:
        bundle_path = args.resume_bundle
        summary_path = bundle_path.parent / summary_path.name
    csr_bytes = int(args.max_csr_payload_gib * 1024**3)
    native_bytes = int(args.max_native_cache_gib * 1024**3)
    adjacency_bytes = int(args.max_connectivity_memory_gib * 1024**3)
    relation_bytes = int(args.max_provenance_and_csr_gib * 1024**3)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if args.resume_bundle is not None:
                run = resume_garcia_local_relation_from_csr_checkpoint(
                    args.resume_bundle,
                    expected_family=construction.family,
                    expected_family_provenance=family_reference,
                    expected_run_configuration={
                        "coordinate_system": "guard_aligned",
                        "t_star": 0.5,
                        "gamma": DEFAULT_GAMMA,
                        "guard_delta": DEFAULT_GUARD_DELTA,
                        "transversality_eta": DEFAULT_TRANSVERSALITY_ETA,
                        "samples_per_axis": 3,
                        "padding_cells": 1.0,
                        "max_step": 0.02,
                        "max_jumps": 20,
                        "require_domain_path": True,
                    },
                    max_vertices=args.max_cells,
                    max_relation_edges=args.max_relation_edges,
                    max_csr_payload_bytes=csr_bytes,
                    max_relation_storage_bytes=relation_bytes,
                    max_undirected_adjacencies=args.max_spatial_adjacencies,
                    max_adjacency_storage_bytes=adjacency_bytes,
                )
            else:
                run = compute_garcia_local_relation(
                    construction.family,
                    coordinate_system="guard_aligned",
                    t_star=0.5,
                    samples_per_axis=3,
                    padding_cells=1.0,
                    max_step=0.02,
                    max_jumps=20,
                    precompute_workers=args.precompute_workers,
                    csr_bundle_path=bundle_path,
                    family_provenance=family_reference,
                    max_relation_edges=args.max_relation_edges,
                    max_csr_payload_bytes=csr_bytes,
                    max_native_cache_bytes=native_bytes,
                    max_relation_storage_bytes=relation_bytes,
                    max_undirected_adjacencies=args.max_spatial_adjacencies,
                    max_adjacency_storage_bytes=adjacency_bytes,
                )
    except (GarciaRelationSizeLimitExceeded, MemoryError, RuntimeError) as error:
        _status(
            "hard_cap_stop",
            error=type(error).__name__,
            message=str(error),
            bundle_preserved=bundle_path.exists(),
            scientific_result_accepted=False,
        )
        return 3

    try:
        _, support = exit_aware_index_pair_refinement(
            run,
            target_axis_depth=args.axis_depth,
            max_output_cells=args.max_cells,
        )
    except GarciaRefinementSizeLimitExceeded as error:
        boundary = audit_garcia_orbit_tube_candidate_boundary(
            construction,
            run.dyadic_cells,
            run.candidate.s_cells,
            run.candidate.touches_nonglued_ambient_boundary,
        )
        blockers = ["support_expansion_exceeds_cell_cap"]
        if run.connectivity_audit.disconnected_sources:
            blockers.append("quotient_disconnected_represented_image_in_active_family")
        if (
            run.candidate.reference_labels_recovered
            != run.candidate.reference_labels_total
        ):
            blockers.append("full_known_gait_not_in_reference_recurrent_SCC")
        if not boundary.passed:
            blockers.append("S_touches_local_open_boundary")
        cap_summary: dict[str, object] = {
            "schema": "garcia-guard-aligned-local-relation-audit-v1",
            "coordinate_system": "guard_aligned",
            "axis_depth": args.axis_depth,
            "total_depth": 4 * args.axis_depth,
            "geometry": geometry_path.name,
            "geometry_fingerprint": geometry_payload["fingerprint"]["sha256"],
            "family_provenance": family_reference,
            "relation_bundle": garcia_csr_bundle_reference(
                summary_path,
                bundle_path,
                max_provenance_header_bytes=_HEADER_CAP,
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "cells": len(run.cells),
            "relation_edges": run.connectivity_audit.relation_edges,
            "spatial_undirected_adjacencies": (
                run.connectivity_audit.total_undirected_adjacencies
            ),
            "morse_nodes": int(run.morse_graph.num_vertices()),
            "morse_edges": [list(edge) for edge in sorted(run.morse_graph.edges())],
            "candidate": run.candidate.to_dict(),
            "support_audit": {
                "schema": "garcia-open-index-pair-support-cap-stop-v1",
                "support_saturated": False,
                "required_active_cell_lower_bound": error.lower_bound,
                "active_cell_cap": error.limit,
                "additional_cells_lower_bound": max(
                    0, error.lower_bound - len(run.cells)
                ),
                "exact_support_family_materialized": False,
                "scientific_result_accepted": False,
            },
            "local_boundary_audit": boundary.to_dict(),
            "global_connected_image_audit": {
                "represented_nonempty_images": (
                    run.connectivity_audit.relation_nonempty_images
                ),
                "disconnected_sources": list(
                    run.connectivity_audit.disconnected_sources
                ),
                "passed": not run.connectivity_audit.disconnected_sources,
            },
            "acceptance_blockers_before_persistence": sorted(set(blockers)),
            "local_acceptance_gates_passed": False,
            "refinement_persistence_checked": False,
            "finite_relation_conley_index_computed": False,
            "scientific_result_accepted": False,
            "whole_cell_outer_enclosure_certified": False,
            "continuous_system_conley_index_certified": False,
        }
        atomic_write_json(
            summary_path,
            attach_garcia_csr_derived_artifact_fingerprint(cap_summary),
            durable=True,
        )
        _status(
            "support_cap_stop",
            summary=str(summary_path),
            bundle=str(bundle_path),
            lower_bound=error.lower_bound,
            cap=error.limit,
            scientific_result_accepted=False,
        )
        return 3
    run = replace(run, support_saturated=support.support_saturated, support_audit=support)
    boundary = audit_garcia_orbit_tube_candidate_boundary(
        construction,  # shared boundary contract is structural
        run.dyadic_cells,
        run.candidate.s_cells,
        run.candidate.touches_nonglued_ambient_boundary,
    )
    blockers = garcia_orbit_tube_acceptance_blockers(
        run,
        support,
        same_chart_boundary_in_s=set(boundary.same_chart_witnesses_in_s),
        reverse_seam_boundary_in_s=set(boundary.reverse_seam_witnesses_in_s),
    )
    if run.connectivity_audit.disconnected_sources:
        blockers.append("quotient_disconnected_represented_image_in_active_family")
    if run.candidate.reference_labels_recovered != run.candidate.reference_labels_total:
        blockers.append("full_known_gait_not_in_reference_recurrent_SCC")
    blockers = sorted(set(blockers))
    local_gates_passed = bool(
        run.candidate.discovery_gates_passed
        and support.support_saturated
        and not blockers
        and boundary.passed
    )
    summary: dict[str, object] = {
        "schema": "garcia-guard-aligned-local-relation-audit-v1",
        "coordinate_system": "guard_aligned",
        "axis_depth": args.axis_depth,
        "total_depth": 4 * args.axis_depth,
        "geometry": geometry_path.name,
        "geometry_fingerprint": geometry_payload["fingerprint"]["sha256"],
        "family_provenance": family_reference,
        "relation_bundle": garcia_csr_bundle_reference(
            summary_path,
            bundle_path,
            max_provenance_header_bytes=_HEADER_CAP,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "cells": len(run.cells),
        "relation_edges": run.connectivity_audit.relation_edges,
        "spatial_undirected_adjacencies": (
            run.connectivity_audit.total_undirected_adjacencies
        ),
        "morse_nodes": int(run.morse_graph.num_vertices()),
        "morse_edges": [list(edge) for edge in sorted(run.morse_graph.edges())],
        "candidate": run.candidate.to_dict(),
        "support_audit": support.to_dict(),
        "local_boundary_audit": boundary.to_dict(),
        "global_connected_image_audit": {
            "represented_nonempty_images": run.connectivity_audit.relation_nonempty_images,
            "disconnected_sources": list(run.connectivity_audit.disconnected_sources),
            "passed": not run.connectivity_audit.disconnected_sources,
        },
        "acceptance_blockers_before_persistence": blockers,
        "local_acceptance_gates_passed": local_gates_passed,
        "refinement_persistence_checked": False,
        "finite_relation_conley_index_computed": False,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "continuous_system_conley_index_certified": False,
    }
    atomic_write_json(
        summary_path,
        attach_garcia_csr_derived_artifact_fingerprint(summary),
        durable=True,
    )
    _status(
        "completed",
        summary=str(summary_path),
        bundle=str(bundle_path),
        cells=len(run.cells),
        relation_edges=run.connectivity_audit.relation_edges,
        morse_nodes=int(run.morse_graph.num_vertices()),
        blockers=blockers,
        local_acceptance_gates_passed=local_gates_passed,
        scientific_result_accepted=False,
    )
    return 0 if local_gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
