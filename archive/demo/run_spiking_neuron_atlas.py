#!/usr/bin/env python3
"""Run the frozen four-stage spiking-neuron Atlas protocol.

The stage list is fixed independently of any Morse-node count:

* total depths 12, 14, and 16 with 3x3 tensor sampling; and
* the predeclared depth-12 5x5 sampling sensitivity run.

Every feasible stage is persisted even when an acceptance gate fails.
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
    DEFAULT_T_STAR,
    FROZEN_SAMPLING_SENSITIVITY,
    FROZEN_TOTAL_DEPTHS,
    PROTOCOL_REVISION,
    authenticate_spiking_neuron_provenance,
    audit_spiking_neuron_sparse_incidence_parity,
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


FROZEN_STAGES = tuple((depth, 3) for depth in FROZEN_TOTAL_DEPTHS) + (
    FROZEN_SAMPLING_SENSITIVITY,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spiking_neuron_atlas"),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--only",
        choices=[f"d{depth}s{samples}" for depth, samples in FROZEN_STAGES],
        action="append",
        help="run selected frozen stages; may be repeated (default: all)",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="skip stages whose complete summary already exists",
    )
    return parser.parse_args()


def _stage_name(depth: int, samples: int) -> str:
    return f"tau050_depth{depth}_samples{samples}"


def _run_stage(
    output_dir: Path,
    *,
    depth: int,
    samples: int,
    workers: int,
) -> dict[str, object]:
    name = _stage_name(depth, samples)
    stage_dir = output_dir / name
    if stage_dir.exists():
        raise FileExistsError(f"refusing to overwrite stage directory {stage_dir}")
    stage_dir.mkdir(parents=True)
    setup = build_spiking_neuron_atlas_model(
        total_depth=depth,
        t_star=DEFAULT_T_STAR,
        samples_per_axis=samples,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
    )
    precompute = precompute_spiking_neuron_endpoints(
        atlas_source_boxes(setup),
        NeuronEndpointPrecomputeConfig(
            t_star=DEFAULT_T_STAR,
            samples_per_axis=samples,
            padding_cells=1.0,
            max_step=DEFAULT_MAX_STEP,
            max_jumps=4,
        ),
        workers=workers,
    )
    result = compute_spiking_neuron_atlas_acceptance(
        total_depth=depth,
        samples_per_axis=samples,
        t_star=DEFAULT_T_STAR,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
        precomputed_entries=precompute.entries,
        endpoint_precompute=precompute.metadata(),
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
    summary["artifacts"] = {
        "relation_csr": "relation.csr",
        "source_provenance": "provenance.jsonl.gz",
        "chart_aware_plot": "morse_support.png",
    }
    summary["diagnostic_source_provenance"] = provenance
    atomic_write_json(stage_dir / "summary.json", summary)
    return summary


def _protocol_summary(
    summaries: dict[str, dict[str, object]],
    *,
    requested: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    completed = tuple(
        (depth, samples)
        for depth, samples in requested
        if _stage_name(depth, samples) in summaries
    )
    frozen_required_names = {
        _stage_name(depth, samples) for depth, samples in FROZEN_STAGES
    }
    all_selected_relation_stages_completed = completed == requested
    all_frozen_relation_stages_completed = set(summaries) == frozen_required_names
    acceptance_audited = tuple(
        (depth, samples)
        for depth, samples in requested
        if bool(
            summaries.get(_stage_name(depth, samples), {}).get(
                "acceptance_audit_completed", False
            )
        )
    )
    all_selected_acceptance_audits_completed = acceptance_audited == requested
    all_frozen_acceptance_audits_completed = bool(
        all_frozen_relation_stages_completed
        and all(
            bool(summary.get("acceptance_audit_completed", False))
            for summary in summaries.values()
        )
    )
    every_stage_has_reference_cycle = all(
        bool(summary["reference_nodes_containing_complete_cycle"])
        for summary in summaries.values()
    )
    every_stage_passed = all(
        bool(summary["stage_gates_passed"]) for summary in summaries.values()
    )
    return {
        "schema": "spiking-neuron-atlas-frozen-protocol-v1",
        "protocol_revision": PROTOCOL_REVISION,
        "frozen_configuration": {
            "t_star": DEFAULT_T_STAR,
            "padding_cells": 1.0,
            "max_step": DEFAULT_MAX_STEP,
            "primary_stages": [
                {"total_depth": depth, "samples_per_axis": 3}
                for depth in FROZEN_TOTAL_DEPTHS
            ],
            "sampling_sensitivity": {
                "total_depth": FROZEN_SAMPLING_SENSITIVITY[0],
                "samples_per_axis": FROZEN_SAMPLING_SENSITIVITY[1],
            },
            "stage_selection_depends_on_scc_count": False,
        },
        "requested_stages": [
            _stage_name(depth, samples) for depth, samples in requested
        ],
        "completed_stages": [
            _stage_name(depth, samples) for depth, samples in completed
        ],
        "acceptance_audited_stages": [
            _stage_name(depth, samples) for depth, samples in acceptance_audited
        ],
        "all_selected_relation_stages_completed": (
            all_selected_relation_stages_completed
        ),
        "all_frozen_relation_stages_completed": (
            all_frozen_relation_stages_completed
        ),
        "all_selected_acceptance_audits_completed": (
            all_selected_acceptance_audits_completed
        ),
        "all_frozen_acceptance_audits_completed": (
            all_frozen_acceptance_audits_completed
        ),
        # Backward-compatible aliases now mean the complete acceptance audit,
        # not merely the existence of a relation checkpoint.
        "all_selected_stages_completed": all_selected_acceptance_audits_completed,
        "all_frozen_stages_completed": all_frozen_acceptance_audits_completed,
        "every_stage_contains_complete_reference_cycle": (
            every_stage_has_reference_cycle
        ),
        "every_stage_passed_local_gates": every_stage_passed,
        "predeclared_refinement_and_sampling_persistence_passed": bool(
            all_frozen_acceptance_audits_completed
            and every_stage_has_reference_cycle
            and every_stage_passed
        ),
        "stages": summaries,
        "analytic_conley_label_attached": False,
        "finite_relation_conley_index_computed": False,
        "whole_cell_outer_enclosure_certified": False,
    }


def main() -> int:
    args = _arguments()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    requested = FROZEN_STAGES
    if args.only:
        selected = set(args.only)
        requested = tuple(
            stage
            for stage in FROZEN_STAGES
            if f"d{stage[0]}s{stage[1]}" in selected
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for depth, samples in requested:
        name = _stage_name(depth, samples)
        summary_path = args.output_dir / name / "summary.json"
        if summary_path.exists() and args.allow_existing:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            relation = summary.get("relation_csr")
            if isinstance(relation, dict) and isinstance(
                relation.get("fingerprint"), str
            ):
                provenance = authenticate_spiking_neuron_provenance(
                    summary_path.parent / "provenance.jsonl.gz",
                    relation_csr_fingerprint=relation["fingerprint"],
                )
                summary["diagnostic_source_provenance"] = provenance
            if "sparse_incidence_parity" not in summary:
                setup = build_spiking_neuron_atlas_model(
                    total_depth=depth,
                    samples_per_axis=samples,
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
        summary = _run_stage(
            args.output_dir,
            depth=depth,
            samples=samples,
            workers=args.workers,
        )
        summaries[name] = summary
        print(
            json.dumps(
                {
                    "event": "stage_completed",
                    "stage": name,
                    "morse_nodes": summary["morse_nodes"],
                    "stage_gates_passed": summary["stage_gates_passed"],
                }
            ),
            flush=True,
        )
    protocol = _protocol_summary(summaries, requested=requested)
    atomic_write_json(args.output_dir / "protocol_summary.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True), flush=True)
    return 0 if protocol["predeclared_refinement_and_sampling_persistence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
