#!/usr/bin/env python3
"""Compute exit-aware local Garcia walker relations with mixed-depth refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaExitAwareSupportAudit,
    GarciaLocalRelationRun,
    GarciaRefinementSizeLimitExceeded,
    GarciaRelationSizeLimitExceeded,
    compute_garcia_local_relation,
    estimated_callback_point_evaluations,
    exit_aware_index_pair_refinement,
    full_garcia_dyadic_family,
    persisted_refinement_size_bounds,
    read_garcia_local_relation,
    resume_garcia_local_relation_from_raw_checkpoint,
    safe_locator_refinement,
    safe_locator_refinement_from_payload,
)
from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    DEFAULT_BASE_BOUNDS,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-depths",
        type=int,
        nargs="+",
        default=[3],
        help=(
            "per-axis dyadic depths, in increasing order; 3,4,5 correspond "
            "to CMGDB total depths 12,16,20"
        ),
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--samples-per-axis", type=int, default=3)
    parser.add_argument("--padding-cells", type=float, default=1.0)
    parser.add_argument(
        "--precompute-workers",
        type=int,
        default=0,
        help="local processes used only to precompute deterministic endpoints",
    )
    parser.add_argument(
        "--collar-fraction",
        type=float,
        default=0.0625,
        help="physical collar width as a fraction of each chart-axis span",
    )
    parser.add_argument("--max-refinement-rounds", type=int, default=12)
    parser.add_argument(
        "--stop-after-initial-audit",
        action="store_true",
        help=(
            "persist round 0 and its exit-aware (N,L) audit without starting "
            "a support-expansion recomputation"
        ),
    )
    parser.add_argument("--max-cells", type=int, default=500_000)
    parser.add_argument("--max-point-evaluations", type=int, default=50_000_000)
    parser.add_argument(
        "--max-relation-edges",
        type=int,
        default=50_000_000,
        help=(
            "hard cap checked after the atomic raw checkpoint and before "
            "connectivity/candidate audits"
        ),
    )
    parser.add_argument(
        "--max-spatial-adjacencies",
        type=int,
        default=20_000_000,
        help="hard cap on undirected sparse grid-plus-quotient adjacencies",
    )
    parser.add_argument(
        "--max-raw-relation-memory-gib",
        type=float,
        default=8.0,
        help=(
            "conservative cap for the stored relation plus per-source provenance; "
            "checked while reading a checkpoint and immediately after a fresh one"
        ),
    )
    parser.add_argument(
        "--max-connectivity-memory-gib",
        type=float,
        default=1.0,
        help=(
            "conservative peak-memory cap for sparse adjacency construction; "
            "the raw relation has its own independent cap"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "complete relation JSON(.gz) from the immediately preceding depth; "
            "the first requested depth is refined from this payload"
        ),
    )
    parser.add_argument(
        "--resume-raw-checkpoint",
        type=Path,
        help=(
            "completed post-MapGraph JSON-lines checkpoint for the first run; "
            "skips the expensive box-map/MapGraph ODE evaluation but reruns "
            "the small reference simulation and every derived audit"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="persist refinement-size bounds without materializing or evaluating it",
    )
    parser.add_argument(
        "--preflight-collar-fractions",
        type=float,
        nargs="+",
        default=[0.0625, 0.125],
        help="fixed physical collars recorded by --preflight-only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/garcia_passive_walker_atlas"),
    )
    parser.add_argument(
        "--require-discovery-gates",
        action="store_true",
        help="return nonzero if the final candidate fails a local discovery gate",
    )
    return parser.parse_args()


def _status(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _relation_name(tau: float, axis_depth: int, round_index: int) -> str:
    tau_tag = f"{int(round(100 * tau)):03d}"
    depth = 4 * axis_depth
    suffix = "" if round_index == 0 else f"_round{round_index}"
    return f"local_relation_tau{tau_tag}_depth{depth}{suffix}.json.gz"


def _raw_checkpoint_name(
    tau: float,
    axis_depth: int,
    round_index: int,
    family,
    args: argparse.Namespace,
) -> str:
    tau_tag = f"{int(round(100 * tau)):03d}"
    depth = 4 * axis_depth
    suffix = "" if round_index == 0 else f"_round{round_index}"
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "tau": float(tau),
                "samples_per_axis": int(args.samples_per_axis),
                "padding_cells": float(args.padding_cells),
                "gamma": DEFAULT_GAMMA,
                "guard_delta": DEFAULT_GUARD_DELTA,
                "transversality_eta": DEFAULT_TRANSVERSALITY_ETA,
                "max_step": 0.02,
                "max_jumps": 20,
                "family_role": family.role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for chart_id, cell_depth, coordinates in family.tagged_cells():
        digest.update(bytes((int(chart_id), int(cell_depth))))
        digest.update(
            ",".join(str(int(value)) for value in coordinates).encode("ascii")
        )
        digest.update(b";")
    return (
        f"raw_relation_checkpoint_tau{tau_tag}_depth{depth}{suffix}_"
        f"cfg{digest.hexdigest()[:12]}.jsonl.gz"
    )


def _run_checked(
    family,
    args: argparse.Namespace,
    *,
    axis_depth: int,
    round_index: int,
) -> GarciaLocalRelationRun | None:
    point_evaluations = estimated_callback_point_evaluations(
        family,
        args.samples_per_axis,
    )
    _status(
        "preflight",
        axis_depth=axis_depth,
        total_depth=4 * axis_depth,
        round=round_index,
        cells=len(family.cells),
        estimated_callback_point_evaluations=point_evaluations,
        max_cells=args.max_cells,
        max_point_evaluations=args.max_point_evaluations,
        max_relation_edges=args.max_relation_edges,
        max_raw_relation_memory_gib=args.max_raw_relation_memory_gib,
        max_spatial_adjacencies=args.max_spatial_adjacencies,
        max_connectivity_memory_gib=args.max_connectivity_memory_gib,
    )
    if len(family.cells) > args.max_cells or point_evaluations > args.max_point_evaluations:
        _status(
            "cost_cap",
            axis_depth=axis_depth,
            round=round_index,
            reason="requested family exceeds an explicit computation cap",
        )
        return None

    started = time.perf_counter()
    raw_checkpoint = args.output_dir / _raw_checkpoint_name(
        args.tau, axis_depth, round_index, family, args
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if args.resume_raw_checkpoint is not None:
                if round_index != 0:
                    raise SystemExit(
                        "--resume-raw-checkpoint can only supply the first requested run"
                    )
                run = resume_garcia_local_relation_from_raw_checkpoint(
                    args.resume_raw_checkpoint,
                    expected_family=family,
                    t_star=args.tau,
                    samples_per_axis=args.samples_per_axis,
                    padding_cells=args.padding_cells,
                    gamma=DEFAULT_GAMMA,
                    guard_delta=DEFAULT_GUARD_DELTA,
                    transversality_eta=DEFAULT_TRANSVERSALITY_ETA,
                    max_step=0.02,
                    max_jumps=20,
                    max_relation_edges=args.max_relation_edges,
                    max_relation_storage_bytes=int(
                        args.max_raw_relation_memory_gib * 1024**3
                    ),
                    max_undirected_adjacencies=args.max_spatial_adjacencies,
                    max_adjacency_storage_bytes=int(
                        args.max_connectivity_memory_gib * 1024**3
                    ),
                )
                raw_checkpoint = Path(args.resume_raw_checkpoint)
            else:
                run = compute_garcia_local_relation(
                    family,
                    t_star=args.tau,
                    samples_per_axis=args.samples_per_axis,
                    padding_cells=args.padding_cells,
                    precompute_workers=args.precompute_workers,
                    raw_checkpoint_path=raw_checkpoint,
                    max_relation_edges=args.max_relation_edges,
                    max_relation_storage_bytes=int(
                        args.max_raw_relation_memory_gib * 1024**3
                    ),
                    max_undirected_adjacencies=args.max_spatial_adjacencies,
                    max_adjacency_storage_bytes=int(
                        args.max_connectivity_memory_gib * 1024**3
                    ),
                )
    except GarciaRelationSizeLimitExceeded as error:
        _status(
            "post_mapgraph_audit_cap",
            axis_depth=axis_depth,
            round=round_index,
            kind=error.kind,
            observed=error.observed,
            limit=error.limit,
            raw_checkpoint=str(raw_checkpoint),
            raw_checkpoint_preserved=raw_checkpoint.exists(),
            scientific_result_accepted=False,
        )
        return None
    output = args.output_dir / _relation_name(args.tau, axis_depth, round_index)
    run.write(output)
    candidate = run.candidate
    _status(
        "completed",
        axis_depth=axis_depth,
        total_depth=4 * axis_depth,
        round=round_index,
        cells=len(family.cells),
        wall_seconds=time.perf_counter() - started,
        relation_file=str(output),
        raw_checkpoint=str(raw_checkpoint),
        relation_edges=(
            None
            if run.connectivity_audit is None
            else run.connectivity_audit.relation_edges
        ),
        estimated_raw_relation_storage_bytes=(
            None
            if run.connectivity_audit is None
            else run.connectivity_audit.estimated_raw_relation_storage_bytes
        ),
        spatial_undirected_adjacencies=(
            None
            if run.connectivity_audit is None
            else run.connectivity_audit.total_undirected_adjacencies
        ),
        estimated_peak_adjacency_storage_bytes=(
            None
            if run.connectivity_audit is None
            else run.connectivity_audit.estimated_peak_adjacency_storage_bytes
        ),
        connectivity_algorithm=(
            None
            if run.connectivity_audit is None
            else run.connectivity_audit.algorithm
        ),
        morse_nodes=int(run.morse_graph.num_vertices()),
        reference_node=candidate.morse_node,
        S=len(candidate.s_cells),
        X=len(candidate.x_cells),
        A=len(candidate.a_cells),
        failed_sources_in_S=len(candidate.failed_sources_in_s),
        open_exit_sources_in_S=len(candidate.open_exit_sources_in_s),
        unresolved_stage_sources_in_S=len(
            candidate.unresolved_stage_sources_in_s
        ),
        reference_endpoint_misses=candidate.reference_endpoint_misses,
        reference_evaluation_failures=candidate.reference_evaluation_failures,
        disconnected_sources_in_S=len(candidate.disconnected_sources_in_s),
        missing_in_domain_sources_in_S=len(
            candidate.missing_in_domain_sources_in_s
        ),
        missing_in_domain_sources_in_X=len(candidate.missing_in_domain_sources_in_x),
        boundary_sources_in_S=len(candidate.touches_nonglued_ambient_boundary),
        discovery_gates_passed=candidate.discovery_gates_passed,
        precompute_workers=args.precompute_workers,
    )
    return run


def _validate_default_resume_configuration(payload: dict[str, object]) -> None:
    """Reject a valid artifact whose physical setup this CLI cannot reproduce."""

    metadata = payload["metadata"]
    charts = payload["charts"]
    expected_scalars = {
        "gamma": DEFAULT_GAMMA,
        "guard_delta": DEFAULT_GUARD_DELTA,
        "transversality_eta": DEFAULT_TRANSVERSALITY_ETA,
        "max_step": 0.02,
    }
    for key, expected in expected_scalars.items():
        if abs(float(metadata[key]) - float(expected)) > 1.0e-13:
            raise SystemExit(
                f"--resume-from uses nondefault {key}; this runner refuses to "
                "silently switch physical models"
            )
    if int(metadata["max_jumps"]) != 20:
        raise SystemExit(
            "--resume-from uses nondefault max_jumps; this runner refuses to "
            "silently switch physical models"
        )
    stored_bounds = charts["base"]["bounds"]
    if any(
        abs(float(actual) - float(expected)) > 1.0e-13
        for actual_interval, expected_interval in zip(
            stored_bounds, DEFAULT_BASE_BOUNDS, strict=True
        )
        for actual, expected in zip(
            actual_interval, expected_interval, strict=True
        )
    ):
        raise SystemExit(
            "--resume-from uses nondefault base bounds; this runner refuses to "
            "silently switch physical models"
        )


def _persist_support_status(
    run: GarciaLocalRelationRun,
    args: argparse.Namespace,
    *,
    axis_depth: int,
    round_index: int,
    support_saturated: bool,
    support_audit: GarciaExitAwareSupportAudit,
) -> GarciaLocalRelationRun:
    updated = replace(
        run,
        support_saturated=support_saturated,
        support_audit=support_audit,
    )
    updated.write(
        args.output_dir / _relation_name(args.tau, axis_depth, round_index)
    )
    return updated


def main() -> int:
    args = parse_args()
    depths = args.axis_depths
    if not depths or any(depth < 1 for depth in depths):
        raise SystemExit("--axis-depths must contain positive integers")
    if depths != sorted(set(depths)):
        raise SystemExit("--axis-depths must be strictly increasing")
    if args.max_refinement_rounds < 1:
        raise SystemExit("--max-refinement-rounds must be positive")
    if args.precompute_workers < 0:
        raise SystemExit("--precompute-workers must be non-negative")
    if args.max_relation_edges < 1 or args.max_spatial_adjacencies < 1:
        raise SystemExit("relation and spatial-adjacency caps must be positive")
    if (
        not math.isfinite(args.max_raw_relation_memory_gib)
        or args.max_raw_relation_memory_gib <= 0.0
    ):
        raise SystemExit("--max-raw-relation-memory-gib must be finite and positive")
    if (
        not math.isfinite(args.max_connectivity_memory_gib)
        or args.max_connectivity_memory_gib <= 0.0
    ):
        raise SystemExit("--max-connectivity-memory-gib must be finite and positive")
    if args.preflight_only and args.resume_raw_checkpoint is not None:
        raise SystemExit("--preflight-only cannot use --resume-raw-checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        if args.resume_from is None:
            raise SystemExit("--preflight-only requires --resume-from")
        payload = read_garcia_local_relation(args.resume_from)
        _validate_default_resume_configuration(payload)
        if abs(float(payload["metadata"]["t_star"]) - args.tau) > 1.0e-13:
            raise SystemExit("--resume-from tau differs from --tau")
        candidate = payload["candidate"]
        entries: list[dict[str, object]] = []
        for target_depth in depths:
            bounds = persisted_refinement_size_bounds(
                payload,
                target_axis_depth=target_depth,
                selector="safe_locator",
                retain_unselected=False,
            )
            for fraction in args.preflight_collar_fractions:
                if fraction <= 0.0:
                    raise SystemExit("preflight collar fractions must be positive")
                lower_cells = int(bounds["candidate_only_cell_lower_bound"])
                lower_points = int(
                    bounds["candidate_only_logical_sample_lower_bound"]
                )
                started = time.perf_counter()
                effective_cell_cap = min(
                    args.max_cells,
                    args.max_point_evaluations // args.samples_per_axis**4,
                )
                try:
                    materialized = safe_locator_refinement_from_payload(
                        payload,
                        target_axis_depth=target_depth,
                        collar_fraction=fraction,
                        include_target_support=False,
                        retain_unselected=False,
                        max_output_cells=effective_cell_cap,
                    )
                except GarciaRefinementSizeLimitExceeded as error:
                    entries.append(
                        {
                            **bounds,
                            "collar_fraction": fraction,
                            "materialized_relation_saturated_cells": None,
                            "materialized_logical_sample_points": None,
                            "materialized_chart_counts": None,
                            "materialized_min_axis_depth": None,
                            "materialized_max_axis_depth": None,
                            "materialization_seconds": time.perf_counter() - started,
                            "materialization_stage": (
                                "local open N0: safe locator + all gait trace cells + "
                                "physical collar + quotient seam closure"
                            ),
                            "materialization_includes_coarse_F_support": False,
                            "construction_cell_lower_bound_at_stop": error.lower_bound,
                            "run_recommended": False,
                            "blockers": ["construction_cell_cap_exceeded"],
                        }
                    )
                    continue
                actual_cells = len(materialized.cells)
                actual_points = estimated_callback_point_evaluations(
                    materialized,
                    args.samples_per_axis,
                )
                blockers = []
                if actual_cells > args.max_cells:
                    blockers.append("materialized_refinement_cells_exceed_cap")
                if actual_points > args.max_point_evaluations:
                    blockers.append("materialized_refinement_samples_exceed_cap")
                entries.append(
                    {
                        **bounds,
                        "collar_fraction": fraction,
                        "materialized_relation_saturated_cells": actual_cells,
                        "materialized_logical_sample_points": actual_points,
                        "materialized_chart_counts": materialized.chart_counts(),
                        "materialized_min_axis_depth": materialized.min_axis_depth,
                        "materialized_max_axis_depth": materialized.max_axis_depth,
                        "materialization_seconds": time.perf_counter() - started,
                        "materialization_stage": (
                            "local open N0: safe locator + all gait trace cells + "
                            "physical collar + quotient seam closure"
                        ),
                        "materialization_includes_coarse_F_support": False,
                        "run_recommended": not blockers,
                        "blockers": blockers,
                    }
                )
        preflight = {
            "schema": "garcia-walker-local-refinement-preflight-v1",
            "source_relation": str(args.resume_from),
            "source_discovery_gates_passed": bool(
                candidate["discovery_gates_passed"]
            ),
            "max_cells": args.max_cells,
            "max_point_evaluations": args.max_point_evaluations,
            "entries": entries,
            "no_dynamics_evaluated": True,
        }
        tau_tag = f"{int(round(100 * args.tau)):03d}"
        output = (
            args.output_dir / f"local_refinement_preflight_tau{tau_tag}.json"
        )
        output.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _status("preflight_summary", path=str(output), entries=len(entries))
        return 0

    if args.resume_from is None:
        family = full_garcia_dyadic_family(depths[0])
    else:
        resume_payload = read_garcia_local_relation(args.resume_from)
        _validate_default_resume_configuration(resume_payload)
        metadata = resume_payload["metadata"]
        if abs(float(metadata["t_star"]) - args.tau) > 1.0e-13:
            raise SystemExit("--resume-from tau differs from --tau")
        if int(metadata["samples_per_axis"]) != args.samples_per_axis:
            raise SystemExit("--resume-from samples-per-axis differs from this run")
        if abs(float(metadata["padding_cells"]) - args.padding_cells) > 1.0e-13:
            raise SystemExit("--resume-from padding-cells differs from this run")
        source_axis_depth = int(metadata["max_axis_depth"])
        if depths[0] <= source_axis_depth:
            raise SystemExit(
                "the first --axis-depths value must exceed the resume relation depth"
            )
        locator = resume_payload.get("refinement_locator")
        if not isinstance(locator, dict) or not locator.get("selected_component"):
            tau_tag = f"{int(round(100 * args.tau)):03d}"
            negative = {
                "schema": "garcia-walker-local-saturation-summary-v1",
                "tau": args.tau,
                "resume_from": str(args.resume_from),
                "requested_axis_depths": depths,
                "completed": [],
                "terminal_pair_failure": True,
                "negative_stop": "no_safe_refinement_locator",
                "continuous_system_conley_index_certified": False,
                "legacy_cemetery_used": False,
            }
            summary_path = args.output_dir / f"local_saturation_tau{tau_tag}.json"
            summary_path.write_text(
                json.dumps(negative, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _status(
                "no_safe_refinement_locator",
                source=str(args.resume_from),
                summary=str(summary_path),
            )
            return 2
        try:
            family = safe_locator_refinement_from_payload(
                resume_payload,
                target_axis_depth=depths[0],
                collar_fraction=args.collar_fraction,
                include_target_support=False,
                retain_unselected=False,
                max_output_cells=min(
                    args.max_cells,
                    args.max_point_evaluations // args.samples_per_axis**4,
                ),
            )
        except GarciaRefinementSizeLimitExceeded as error:
            _status(
                "cost_cap",
                axis_depth=depths[0],
                reason="refinement construction exceeded its explicit cap",
                construction_cell_lower_bound_at_stop=error.lower_bound,
                max_cells=error.limit,
            )
            return 3
        _status(
            "resumed",
            source=str(args.resume_from),
            source_axis_depth=source_axis_depth,
            target_axis_depth=depths[0],
            cells=len(family.cells),
        )
    run = _run_checked(family, args, axis_depth=depths[0], round_index=0)
    if run is None:
        return 3
    # A raw checkpoint always corresponds to exactly one active family/run.
    # Later refinement depths and saturation rounds must compute their own raw
    # relation and checkpoint under their deterministic per-round filename.
    args.resume_raw_checkpoint = None
    completed: list[dict[str, object]] = []
    cost_capped = False
    saturation_capped = False
    terminal_pair_failure = False
    initial_audit_stop = False
    support_saturated_by_depth: dict[str, bool | None] = {}

    if args.resume_from is None:
        completed.append(
            {
                "axis_depth": depths[0],
                "round": 0,
                "cells": len(family.cells),
                "elapsed_seconds": run.elapsed_seconds,
                "raw_checkpoint": run.raw_checkpoint_path,
                "relation_edges": run.connectivity_audit.relation_edges,
                "spatial_undirected_adjacencies": (
                    run.connectivity_audit.total_undirected_adjacencies
                ),
                "support_saturated": None,
                "discovery_gates_passed": run.candidate.discovery_gates_passed,
            }
        )
        support_saturated_by_depth[str(depths[0])] = None
        remaining_depths = depths[1:]
        resumed_first_run = None
    else:
        remaining_depths = depths
        resumed_first_run = run

    for target_depth in remaining_depths:
        if resumed_first_run is not None:
            current_run = resumed_first_run
            resumed_first_run = None
            round_index = 0
        else:
            if not run.refinement_locator.selected_component:
                _status(
                    "no_safe_refinement_locator",
                    source_axis_depth=run.family.max_axis_depth,
                    target_axis_depth=target_depth,
                )
                terminal_pair_failure = True
                support_saturated_by_depth[str(target_depth)] = False
                break
            try:
                family = safe_locator_refinement(
                    run,
                    target_axis_depth=target_depth,
                    collar_fraction=args.collar_fraction,
                    include_target_support=False,
                    retain_unselected=False,
                    max_output_cells=min(
                        args.max_cells,
                        args.max_point_evaluations // args.samples_per_axis**4,
                    ),
                )
            except GarciaRefinementSizeLimitExceeded as error:
                _status(
                    "cost_cap",
                    axis_depth=target_depth,
                    reason="refinement construction exceeded its explicit cap",
                    construction_cell_lower_bound_at_stop=error.lower_bound,
                    max_cells=error.limit,
                )
                cost_capped = True
                support_saturated_by_depth[str(target_depth)] = False
                break
            round_index = 0
            current_run = _run_checked(
                family,
                args,
                axis_depth=target_depth,
                round_index=round_index,
            )
            if current_run is None:
                cost_capped = True
                support_saturated_by_depth[str(target_depth)] = False
                break

        while True:
            run = current_run
            if run.candidate.morse_node is None or not run.candidate.x_cells:
                completed.append(
                    {
                        "axis_depth": target_depth,
                        "round": round_index,
                        "cells": len(family.cells),
                        "elapsed_seconds": run.elapsed_seconds,
                        "raw_checkpoint": run.raw_checkpoint_path,
                        "support_saturated": False,
                        "support_audit": None,
                        "discovery_gates_passed": False,
                        "negative_stop": "no_reference_selected_candidate",
                    }
                )
                support_saturated_by_depth[str(target_depth)] = False
                _status(
                    "no_reference_candidate",
                    axis_depth=target_depth,
                    round=round_index,
                    cells=len(family.cells),
                )
                terminal_pair_failure = True
                break
            try:
                next_family, support_audit = exit_aware_index_pair_refinement(
                    run,
                    target_axis_depth=target_depth,
                    max_output_cells=min(
                        args.max_cells,
                        args.max_point_evaluations // args.samples_per_axis**4,
                    ),
                )
            except GarciaRefinementSizeLimitExceeded as error:
                _status(
                    "cost_cap",
                    axis_depth=target_depth,
                    round=round_index,
                    reason="exit-aware support expansion exceeded its explicit cap",
                    construction_cell_lower_bound_at_stop=error.lower_bound,
                    max_cells=error.limit,
                )
                cost_capped = True
                support_saturated_by_depth[str(target_depth)] = False
                break
            support_saturated = next_family.cells == family.cells
            run = _persist_support_status(
                run,
                args,
                axis_depth=target_depth,
                round_index=round_index,
                support_saturated=support_saturated,
                support_audit=support_audit,
            )
            completed.append(
                {
                    "axis_depth": target_depth,
                    "round": round_index,
                    "cells": len(family.cells),
                    "elapsed_seconds": run.elapsed_seconds,
                    "raw_checkpoint": run.raw_checkpoint_path,
                    "relation_edges": run.connectivity_audit.relation_edges,
                    "spatial_undirected_adjacencies": (
                        run.connectivity_audit.total_undirected_adjacencies
                    ),
                    "support_saturated": support_saturated,
                    "support_audit": support_audit.to_dict(),
                    "discovery_gates_passed": run.candidate.discovery_gates_passed,
                }
            )
            support_saturated_by_depth[str(target_depth)] = support_saturated
            if support_audit.terminal_blockers:
                _status(
                    "terminal_pair_failure",
                    axis_depth=target_depth,
                    round=round_index,
                    cells=len(family.cells),
                    blockers=list(support_audit.terminal_blockers),
                    missing_support_cells=len(support_audit.added_cells),
                )
                terminal_pair_failure = True
                break
            if support_saturated:
                _status(
                    "saturated",
                    axis_depth=target_depth,
                    round=round_index,
                    cells=len(family.cells),
                )
                break
            if args.stop_after_initial_audit and round_index == 0:
                _status(
                    "initial_audit_stop",
                    axis_depth=target_depth,
                    cells=len(family.cells),
                    support_saturated=support_saturated,
                    terminal_blockers=[],
                    missing_support_cells=len(support_audit.added_cells),
                )
                initial_audit_stop = True
                break
            if round_index + 1 >= args.max_refinement_rounds:
                _status(
                    "support_saturation_cap",
                    axis_depth=target_depth,
                    round=round_index,
                    cells=len(family.cells),
                    expanded_cells=len(next_family.cells),
                )
                saturation_capped = True
                break
            family = next_family
            round_index += 1
            next_run = _run_checked(
                family,
                args,
                axis_depth=target_depth,
                round_index=round_index,
            )
            if next_run is None:
                cost_capped = True
                support_saturated_by_depth[str(target_depth)] = False
                break
            current_run = next_run
        if (
            cost_capped
            or saturation_capped
            or terminal_pair_failure
            or initial_audit_stop
        ):
            break

    summary = {
        "schema": "garcia-walker-local-saturation-summary-v1",
        "tau": args.tau,
        "samples_per_axis": args.samples_per_axis,
        "padding_cells": args.padding_cells,
        "precompute_workers": args.precompute_workers,
        "max_relation_edges": args.max_relation_edges,
        "max_raw_relation_memory_gib": args.max_raw_relation_memory_gib,
        "max_spatial_adjacencies": args.max_spatial_adjacencies,
        "max_connectivity_memory_gib": args.max_connectivity_memory_gib,
        "collar_fraction": args.collar_fraction,
        "refinement_selector": "garcia-safe-refinement-locator-v1",
        "refinement_selector_is_acceptance_certificate": False,
        "acceptance_uses_complete_unpruned_relation": True,
        "requested_axis_depths": depths,
        "resume_from": None if args.resume_from is None else str(args.resume_from),
        "completed": completed,
        "support_saturated_by_axis_depth": support_saturated_by_depth,
        "all_requested_refinement_depths_saturated": bool(
            remaining_depths
            and not cost_capped
            and not saturation_capped
            and not terminal_pair_failure
            and not initial_audit_stop
            and all(
                support_saturated_by_depth.get(str(depth)) is True
                for depth in remaining_depths
            )
        ),
        "cost_cap_reached": cost_capped,
        "support_saturation_cap_reached": saturation_capped,
        "terminal_pair_failure": terminal_pair_failure,
        "stopped_after_initial_audit": initial_audit_stop,
        "final_discovery_gates_passed": run.candidate.discovery_gates_passed,
        "final_candidate": {
            "morse_node": run.candidate.morse_node,
            "S": len(run.candidate.s_cells),
            "X": len(run.candidate.x_cells),
            "A": len(run.candidate.a_cells),
            "reference_endpoint_misses": run.candidate.reference_endpoint_misses,
            "reference_evaluation_failures": (
                run.candidate.reference_evaluation_failures
            ),
            "failed_sources_in_S": len(run.candidate.failed_sources_in_s),
            "open_exit_sources_in_S": len(run.candidate.open_exit_sources_in_s),
            "unresolved_stage_sources_in_S": len(
                run.candidate.unresolved_stage_sources_in_s
            ),
            "disconnected_sources_in_S": len(
                run.candidate.disconnected_sources_in_s
            ),
            "missing_in_domain_sources_in_S": len(
                run.candidate.missing_in_domain_sources_in_s
            ),
            "missing_in_domain_sources_in_X": len(
                run.candidate.missing_in_domain_sources_in_x
            ),
            "boundary_sources_in_S": len(
                run.candidate.touches_nonglued_ambient_boundary
            ),
        },
        "continuous_system_conley_index_certified": False,
        "legacy_cemetery_used": False,
    }
    tau_tag = f"{int(round(100 * args.tau)):03d}"
    summary_path = args.output_dir / f"local_saturation_tau{tau_tag}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _status(
        "summary",
        path=str(summary_path),
        cost_cap_reached=cost_capped,
        support_saturation_cap_reached=saturation_capped,
        terminal_pair_failure=terminal_pair_failure,
        stopped_after_initial_audit=initial_audit_stop,
    )
    if cost_capped or saturation_capped:
        return 3
    if terminal_pair_failure:
        return 2
    if args.require_discovery_gates and not run.candidate.discovery_gates_passed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
