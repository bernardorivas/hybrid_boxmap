#!/usr/bin/env python3
"""Run the separately versioned scientific-clock neuron Atlas protocol.

This protocol was fixed from the independently computed suspension period
(about 148.85), not from a Morse-node count.  It preserves the depth/sampling
geometry of the near-identity plumbing study and runs all four combinations
of clocks 20/40 and total depths 12/14.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    AtlasNeuronCell,
    DEFAULT_MAX_STEP,
    audit_spiking_neuron_sparse_incidence_parity,
    authenticate_spiking_neuron_provenance,
    build_spiking_neuron_atlas_model,
    compute_spiking_neuron_atlas_acceptance,
    plot_spiking_neuron_acceptance,
    validate_spiking_neuron_provenance,
    write_spiking_neuron_csr_checkpoint,
    write_spiking_neuron_provenance,
)
from hybrid_dynamics.examples.spiking_neuron_precompute import (
    NeuronEndpointPrecomputeConfig,
    atlas_source_boxes,
    precompute_spiking_neuron_endpoints,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


SCIENTIFIC_PROTOCOL_REVISION = "spiking-neuron-scientific-clock-v1"
FROZEN_STAGES = (
    (20.0, 12, 3, 10),
    (20.0, 14, 3, 10),
    (40.0, 12, 3, 20),
    (40.0, 14, 3, 20),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spiking_neuron_atlas/scientific_clock_v1"),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--only",
        choices=[
            f"t{int(t_star)}d{depth}s{samples}"
            for t_star, depth, samples, _max_jumps in FROZEN_STAGES
        ],
        action="append",
    )
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def _stage_name(t_star: float, depth: int, samples: int) -> str:
    return f"tau{int(round(100*t_star)):04d}_depth{depth}_samples{samples}"


def _run_stage(
    output_dir: Path,
    *,
    t_star: float,
    depth: int,
    samples: int,
    max_jumps: int,
    workers: int,
) -> dict[str, object]:
    name = _stage_name(t_star, depth, samples)
    stage_dir = output_dir / name
    if stage_dir.exists():
        raise FileExistsError(f"refusing to overwrite stage directory {stage_dir}")
    stage_dir.mkdir(parents=True)
    setup = build_spiking_neuron_atlas_model(
        total_depth=depth,
        t_star=t_star,
        samples_per_axis=samples,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
        max_jumps=max_jumps,
    )
    precompute = precompute_spiking_neuron_endpoints(
        atlas_source_boxes(setup),
        NeuronEndpointPrecomputeConfig(
            t_star=t_star,
            samples_per_axis=samples,
            padding_cells=1.0,
            max_step=DEFAULT_MAX_STEP,
            max_jumps=max_jumps,
        ),
        workers=workers,
    )
    result = compute_spiking_neuron_atlas_acceptance(
        total_depth=depth,
        samples_per_axis=samples,
        t_star=t_star,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
        max_jumps=max_jumps,
        precomputed_entries=precompute.entries,
        endpoint_precompute=precompute.metadata(),
        protocol_revision=SCIENTIFIC_PROTOCOL_REVISION,
    )
    csr = write_spiking_neuron_csr_checkpoint(result, stage_dir / "relation.csr")
    result = replace(result, relation_csr=csr)
    plot_spiking_neuron_acceptance(result, stage_dir / "morse_support.png")
    provenance_path = write_spiking_neuron_provenance(
        result, stage_dir / "provenance.jsonl.gz"
    )
    provenance = validate_spiking_neuron_provenance(
        provenance_path,
        expected_relation_csr_fingerprint=str(csr["fingerprint"]),
    )
    summary = result.summary()
    summary["scientific_protocol_revision"] = SCIENTIFIC_PROTOCOL_REVISION
    summary["artifacts"] = {
        "relation_csr": "relation.csr",
        "source_provenance": "provenance.jsonl.gz",
        "chart_aware_plot": "morse_support.png",
    }
    summary["diagnostic_source_provenance"] = provenance
    atomic_write_json(stage_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = _arguments()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    stages = FROZEN_STAGES
    if args.only:
        selected = set(args.only)
        stages = tuple(
            stage
            for stage in FROZEN_STAGES
            if f"t{int(stage[0])}d{stage[1]}s{stage[2]}" in selected
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for t_star, depth, samples, max_jumps in stages:
        name = _stage_name(t_star, depth, samples)
        summary_path = args.output_dir / name / "summary.json"
        if summary_path.exists() and args.allow_existing:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            relation = summary.get("relation_csr")
            if isinstance(relation, dict) and isinstance(
                relation.get("fingerprint"), str
            ):
                summary["diagnostic_source_provenance"] = (
                    authenticate_spiking_neuron_provenance(
                        summary_path.parent / "provenance.jsonl.gz",
                        relation_csr_fingerprint=relation["fingerprint"],
                    )
                )
            if "sparse_incidence_parity" not in summary:
                setup = build_spiking_neuron_atlas_model(
                    total_depth=depth,
                    t_star=t_star,
                    samples_per_axis=samples,
                    max_jumps=max_jumps,
                )
                atlas = setup.model.phaseSpace()
                cells = tuple(
                    AtlasNeuronCell(
                        index,
                        int(atlas.cell(index).chart_id),
                        tuple(float(value) for value in atlas.cell(index).bounds),
                    )
                    for index in range(int(atlas.size()))
                )
                summary["sparse_incidence_parity"] = (
                    audit_spiking_neuron_sparse_incidence_parity(
                        cells, setup.charts, axis_depth=depth // 2
                    ).to_dict()
                )
                summary["connectivity_algorithm"] = (
                    "quadratic_geometry_with_exact_sparse_parity"
                )
                summary.setdefault("artifact_migrations", []).append(
                    {
                        "kind": "authenticated_no_dynamics_sparse_incidence_parity",
                        "relation_csr_fingerprint": (
                            relation.get("fingerprint")
                            if isinstance(relation, dict)
                            else None
                        ),
                        "dynamics_recomputed": False,
                    }
                )
            summary["relation_stage_completed"] = bool(
                isinstance(relation, dict)
                and isinstance(relation.get("fingerprint"), str)
            )
            summary["acceptance_audit_completed"] = bool(
                summary["relation_stage_completed"]
                and isinstance(summary.get("sparse_incidence_parity"), dict)
                and summary["sparse_incidence_parity"].get("passed") is True
                and isinstance(summary.get("connectivity_algorithm"), str)
            )
            atomic_write_json(summary_path, summary)
            summaries[name] = summary
            continue
        print(json.dumps({"event": "stage_started", "stage": name}), flush=True)
        try:
            summary = _run_stage(
                args.output_dir,
                t_star=t_star,
                depth=depth,
                samples=samples,
                max_jumps=max_jumps,
                workers=args.workers,
            )
        except (MemoryError, RuntimeError) as error:
            summary = {
                "schema": "spiking-neuron-scientific-stage-failure-v1",
                "scientific_protocol_revision": SCIENTIFIC_PROTOCOL_REVISION,
                "stage": name,
                "error": type(error).__name__,
                "message": str(error),
                "stage_gates_passed": False,
                "reference_nodes_containing_complete_cycle": [],
                "scientific_result_accepted": False,
            }
            atomic_write_json(
                args.output_dir / name / "stage_failure.json", summary
            )
        summaries[name] = summary
        print(
            json.dumps(
                {
                    "event": "stage_completed",
                    "stage": name,
                    "morse_nodes": summary.get("morse_nodes"),
                    "stage_gates_passed": summary["stage_gates_passed"],
                }
            ),
            flush=True,
        )
    all_selected_relation_stages_complete = len(summaries) == len(stages)
    frozen_required_names = {
        _stage_name(t_star, depth, samples)
        for t_star, depth, samples, _max_jumps in FROZEN_STAGES
    }
    all_frozen_relation_stages_complete = set(summaries) == frozen_required_names
    acceptance_audited_names = {
        name
        for name, summary in summaries.items()
        if bool(summary.get("acceptance_audit_completed", False))
    }
    selected_names = {
        _stage_name(t_star, depth, samples)
        for t_star, depth, samples, _max_jumps in stages
    }
    all_selected_acceptance_audits_complete = (
        acceptance_audited_names == selected_names
    )
    all_frozen_acceptance_audits_complete = bool(
        all_frozen_relation_stages_complete
        and acceptance_audited_names == frozen_required_names
    )
    reference_persistent = all(
        bool(summary["reference_nodes_containing_complete_cycle"])
        for summary in summaries.values()
    )
    all_gates = all(bool(summary["stage_gates_passed"]) for summary in summaries.values())
    protocol = {
        "schema": "spiking-neuron-scientific-clock-protocol-v1",
        "scientific_protocol_revision": SCIENTIFIC_PROTOCOL_REVISION,
        "protocol_amendment": {
            "reason": (
                "the frozen t*=0.5 relation is a near-identity plumbing test; "
                "scientific clocks 20 and 40 were predeclared from the independent "
                "suspension period 148.8545 before their Morse counts were computed"
            ),
            "near_identity_results_preserved": "../protocol_summary.json",
            "stage_selection_depends_on_scc_count": False,
        },
        "frozen_configuration": {
            "stages": [
                {
                    "t_star": t_star,
                    "total_depth": depth,
                    "samples_per_axis": samples,
                    "padding_cells": 1.0,
                    "max_step": DEFAULT_MAX_STEP,
                    "max_jumps": max_jumps,
                }
                for t_star, depth, samples, max_jumps in FROZEN_STAGES
            ],
            "depth16_excluded_before_scientific_results": True,
            "depth16_exclusion_reason": (
                "fixed cost preflight: 34,832 sources and 140,570 unique 3x3 "
                "endpoints, each requiring 20 or 40 units at max_step=0.02; "
                "depths 12/14 already provide a predeclared refinement check"
            ),
        },
        "requested_stages": [
            _stage_name(t_star, depth, samples)
            for t_star, depth, samples, _max_jumps in stages
        ],
        "completed_stages": sorted(summaries),
        "acceptance_audited_stages": sorted(acceptance_audited_names),
        "all_selected_relation_stages_completed": (
            all_selected_relation_stages_complete
        ),
        "all_frozen_relation_stages_completed": (
            all_frozen_relation_stages_complete
        ),
        "all_selected_acceptance_audits_completed": (
            all_selected_acceptance_audits_complete
        ),
        "all_frozen_acceptance_audits_completed": (
            all_frozen_acceptance_audits_complete
        ),
        "all_selected_stages_completed": all_selected_acceptance_audits_complete,
        "all_frozen_stages_completed": all_frozen_acceptance_audits_complete,
        "every_stage_contains_complete_reference_cycle": reference_persistent,
        "every_stage_passed_local_gates": all_gates,
        "scientific_clock_persistence_passed": bool(
            all_frozen_acceptance_audits_complete
            and reference_persistent
            and all_gates
        ),
        "stages": summaries,
        "analytic_conley_label_attached": False,
        "finite_relation_conley_index_computed": False,
        "whole_cell_outer_enclosure_certified": False,
    }
    atomic_write_json(args.output_dir / "protocol_summary.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)
    return 0 if protocol["scientific_clock_persistence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
