#!/usr/bin/env python3
"""Validate or compute the fixed-radius Garcia walker orbit tube relation."""

from __future__ import annotations

import argparse
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
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaRelationSizeLimitExceeded,
    GarciaRefinementSizeLimitExceeded,
    compute_garcia_local_relation,
    estimated_callback_point_evaluations,
    exit_aware_index_pair_refinement,
    garcia_failure_reason_is_explicit_domain_exit,
    garcia_non_domain_failure_sources,
)
from hybrid_dynamics.examples.garcia_passive_walker_tube import (
    DEFAULT_TUBE_SHA256,
    audit_garcia_orbit_tube_candidate_boundary,
    build_dense_garcia_orbit_tube,
    garcia_orbit_tube_acceptance_blockers,
    read_garcia_orbit_tube,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


_DERIVED_PROVENANCE_HEADER_CAP = 4 * 1024 * 1024


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path(
            "data/garcia_passive_walker_atlas/"
            "tube_geometry_tau050_depth20_r00625.json.gz"
        ),
    )
    parser.add_argument(
        "--refresh-geometry",
        action="store_true",
        help="rerun only the inexpensive dense reference integration",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="explicitly authorize the expensive source-box/MapGraph computation",
    )
    parser.add_argument(
        "--resume-provenance",
        type=Path,
        help="strict CSR-provenance checkpoint; does not rerun source-box dynamics",
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--samples-per-axis", type=int, default=3)
    parser.add_argument("--padding-cells", type=float, default=1.0)
    parser.add_argument("--precompute-workers", type=int, default=8)
    parser.add_argument("--max-step", type=float, default=0.02)
    parser.add_argument("--max-jumps", type=int, default=20)
    parser.add_argument("--max-cells", type=int, default=50_000)
    parser.add_argument("--max-point-evaluations", type=int, default=5_000_000)
    parser.add_argument("--max-relation-edges", type=int, default=50_000_000)
    parser.add_argument("--max-spatial-adjacencies", type=int, default=5_000_000)
    parser.add_argument("--max-csr-payload-gib", type=float, default=1.0)
    parser.add_argument("--max-native-cache-gib", type=float, default=1.0)
    parser.add_argument("--max-provenance-and-csr-gib", type=float, default=1.0)
    parser.add_argument("--max-connectivity-memory-gib", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/garcia_passive_walker_atlas"),
    )
    return parser.parse_args()


def _status(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _family_reference(geometry: Path, construction) -> dict[str, object]:
    payload = construction.to_dict()
    return {
        "schema": "garcia-orbit-tube-family-reference-v1",
        "path": geometry.name,
        "fingerprint": payload["fingerprint"]["sha256"],
        "family_cells_sha256": payload["family_cells_sha256"],
        "attachment_carriers_sha256": payload["attachment_carriers"]["sha256"],
        "attachment_policy": payload["attachment_policy"],
        "reverse_extra_seam_cofaces_are_open": True,
    }


def main() -> int:
    args = _arguments()
    if args.compute and args.resume_provenance is not None:
        raise SystemExit("choose either --compute or --resume-provenance")
    if args.refresh_geometry and args.resume_provenance is not None:
        raise SystemExit("resume requires the already fingerprinted geometry")
    declared_configuration = {
        "tau": (args.tau, 0.5),
        "samples_per_axis": (args.samples_per_axis, 3),
        "padding_cells": (args.padding_cells, 1.0),
        "max_step": (args.max_step, 0.02),
        "max_jumps": (args.max_jumps, 20),
    }
    for label, (actual, expected) in declared_configuration.items():
        if actual != expected:
            raise SystemExit(
                f"this pinned flagship runner requires {label}={expected!r}"
            )
    for name in (
        "max_csr_payload_gib",
        "max_native_cache_gib",
        "max_provenance_and_csr_gib",
        "max_connectivity_memory_gib",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    if args.refresh_geometry:
        construction = build_dense_garcia_orbit_tube()
        construction.write(args.geometry)
    construction = read_garcia_orbit_tube(
        args.geometry,
        expected_fingerprint=DEFAULT_TUBE_SHA256,
    )
    geometry_payload = construction.to_dict()
    family_reference = _family_reference(args.geometry, construction)
    logical_samples = estimated_callback_point_evaluations(
        construction.family, args.samples_per_axis
    )
    preflight = {
        "cells": len(construction.family.cells),
        "chart_counts": construction.family.chart_counts(),
        "logical_samples": logical_samples,
        "geometry_fingerprint": geometry_payload["fingerprint"]["sha256"],
        "active_quotient_incidences": construction.active_quotient_incidences,
        "reverse_open_quotient_incidences": (
            construction.reverse_open_quotient_incidences
        ),
        "same_chart_closed_one_ring_boundary_cells": len(
            construction.same_chart_open_boundary_cells
        ),
        "reverse_open_base_boundary_cells": len(
            construction.reverse_open_base_boundary_cells
        ),
    }
    _status("geometry_validated", **preflight)
    if len(construction.family.cells) > args.max_cells:
        raise SystemExit("tube exceeds --max-cells")
    if logical_samples > args.max_point_evaluations:
        raise SystemExit("tube exceeds --max-point-evaluations")
    if not args.compute and args.resume_provenance is None:
        _status(
            "geometry_only_stop",
            reason="source-box dynamics require explicit --compute",
            scientific_result_accepted=False,
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = geometry_payload["fingerprint"]["sha256"][:12]
    raw_path = args.output_dir / f"tube_relation_bundle_tau050_depth20_geom{tag}"
    summary_path = args.output_dir / f"tube_relation_audit_tau050_depth20_geom{tag}.json"
    if args.resume_provenance is not None:
        raw_path = args.resume_provenance
        summary_path = raw_path.parent / summary_path.name
    csr_bytes = int(args.max_csr_payload_gib * 1024**3)
    native_bytes = int(args.max_native_cache_gib * 1024**3)
    relation_bytes = int(args.max_provenance_and_csr_gib * 1024**3)
    adjacency_bytes = int(args.max_connectivity_memory_gib * 1024**3)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if args.resume_provenance is not None:
                run = resume_garcia_local_relation_from_csr_checkpoint(
                    args.resume_provenance,
                    expected_family=construction.family,
                    expected_family_provenance=family_reference,
                    expected_run_configuration={
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
                    t_star=args.tau,
                    samples_per_axis=args.samples_per_axis,
                    padding_cells=args.padding_cells,
                    max_step=args.max_step,
                    max_jumps=args.max_jumps,
                    precompute_workers=args.precompute_workers,
                    csr_bundle_path=raw_path,
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
            raw_provenance_preserved=raw_path.exists(),
            csr_preserved=(raw_path / "relation.csr").exists(),
            scientific_result_accepted=False,
        )
        return 3
    try:
        _, support_audit = exit_aware_index_pair_refinement(
            run,
            target_axis_depth=construction.axis_depth,
            max_output_cells=args.max_cells,
        )
    except GarciaRefinementSizeLimitExceeded as error:
        cap_summary = {
            "schema": "garcia-walker-orbit-tube-audit-v1",
            "geometry": str(args.geometry),
            "relation_bundle": garcia_csr_bundle_reference(
                summary_path,
                raw_path,
                max_provenance_header_bytes=_DERIVED_PROVENANCE_HEADER_CAP,
            ),
            "support_expansion_lower_bound": error.lower_bound,
            "max_cells": error.limit,
            "acceptance_blockers": ["support_expansion_exceeds_cell_cap"],
            "scientific_result_accepted": False,
            "continuous_system_conley_index_certified": False,
            "whole_cell_outer_enclosure_certified": False,
        }
        atomic_write_json(
            summary_path,
            attach_garcia_csr_derived_artifact_fingerprint(cap_summary),
            durable=True,
        )
        _status(
            "support_audit_cap_stop",
            summary=str(summary_path),
            observed=error.lower_bound,
            limit=error.limit,
            scientific_result_accepted=False,
        )
        return 3
    run = replace(
        run,
        support_saturated=support_audit.support_saturated,
        support_audit=support_audit,
    )
    boundary_audit = audit_garcia_orbit_tube_candidate_boundary(
        construction,
        run.dyadic_cells,
        run.candidate.s_cells,
        run.candidate.touches_nonglued_ambient_boundary,
    )
    blockers = garcia_orbit_tube_acceptance_blockers(
        run,
        support_audit,
        same_chart_boundary_in_s=set(boundary_audit.same_chart_witnesses_in_s),
        reverse_seam_boundary_in_s=set(boundary_audit.reverse_seam_witnesses_in_s),
    )
    accepted = bool(
        run.candidate.discovery_gates_passed
        and support_audit.support_saturated
        and not blockers
    )
    summary = {
        "schema": "garcia-walker-orbit-tube-audit-v1",
        "geometry": str(args.geometry),
        "family_provenance": family_reference,
        "relation_bundle": garcia_csr_bundle_reference(
            summary_path,
            raw_path,
            max_provenance_header_bytes=_DERIVED_PROVENANCE_HEADER_CAP,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "cells": len(run.cells),
        "relation_edges": run.connectivity_audit.relation_edges,
        "spatial_undirected_adjacencies": (
            run.connectivity_audit.total_undirected_adjacencies
        ),
        "morse_nodes": int(run.morse_graph.num_vertices()),
        "candidate": run.candidate.to_dict(),
        "support_audit": support_audit.to_dict(),
        "local_boundary_audit": boundary_audit.to_dict(),
        "exit_set_A_policy_audit": {
            "policy": (
                "explicit domain-exit failures and disconnected or empty images "
                "in L=A are allowed only when F(A) intersection N is contained "
                "in A; unresolved stages and non-domain failures block"
            ),
            "explicit_domain_exit_failure_sources": sorted(
                source
                for source in run.candidate.a_cells
                if any(
                    count
                    and garcia_failure_reason_is_explicit_domain_exit(reason)
                    for reason, count in run.source_provenance[source].failure_reasons
                )
            ),
            "non_domain_failure_sources": sorted(
                garcia_non_domain_failure_sources(
                    run.source_provenance, run.candidate.a_cells
                )
            ),
            "unresolved_stage_sources": sorted(
                source
                for source in run.candidate.a_cells
                if run.source_provenance[source].unresolved_stage_edges
            ),
            "disconnected_sources_allowed_as_exits": sorted(
                run.candidate.disconnected_sources_in_a
            ),
            "empty_sources_allowed_as_exits": sorted(
                source
                for source in run.candidate.a_cells
                if not run.relation[source]
            ),
            "finite_pair_second_condition_violations": sorted(
                run.candidate.pair_second_condition_violations
            ),
        },
        "acceptance_blockers": blockers,
        "scientific_result_accepted": accepted,
        "continuous_system_conley_index_certified": False,
        "whole_cell_outer_enclosure_certified": False,
    }
    atomic_write_json(
        summary_path,
        attach_garcia_csr_derived_artifact_fingerprint(summary),
        durable=True,
    )
    _status(
        "completed",
        summary=str(summary_path),
        raw_provenance_bundle=str(raw_path),
        csr=str(raw_path / "relation.csr"),
        derived_audit=str(summary_path),
        relation_edges=run.connectivity_audit.relation_edges,
        morse_nodes=int(run.morse_graph.num_vertices()),
        acceptance_blockers=blockers,
        scientific_result_accepted=accepted,
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
