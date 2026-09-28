#!/usr/bin/env python3
"""Run the reviewed t*=20 adaptive neuron relation with terminal carriers.

This is a new, non-overwriting protocol.  It reconstructs the exact mixed
family and tensor sample set from the relation-bound preflight, then enables
only the provenance-bearing single-handle terminal-image carrier.  No t*=40
stage is part of this protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    DEFAULT_MAX_STEP,
    authenticate_spiking_neuron_provenance,
    build_spiking_neuron_atlas_model,
    compute_spiking_neuron_atlas_acceptance,
    load_spiking_neuron_adaptive_preflight,
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
from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    SINGLE_HANDLE_BRIDGE_ASSUMPTIONS,
)


PROTOCOL_REVISION = "spiking-neuron-adaptive-terminal-bridge-t20-v1"
EXPECTED_REFINED_PARENTS = 122
EXPECTED_ACTIVE_CELLS = 9_074
EXPECTED_LOGICAL_POINTS = 81_666
EXPECTED_UNIQUE_POINTS = 37_030
EXPECTED_FAMILY_FINGERPRINT = (
    "45917b973ccf7e10eb891ad27276529377db5f9de961dedbb62468936d445706"
)
EXPECTED_SAMPLE_FINGERPRINT = (
    "f8c7344cf9403fad8a7fa91fc92340daf0fb5732ed604ffe008c7ea5dae2d774"
)
BRIDGE_MAX_BISECTIONS = 12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_preflight_v1/t20/preflight.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_v1/t20"
        ),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--authorize-reviewed-terminal-bridge",
        action="store_true",
        help="required after the focused synthetic and single-source review",
    )
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def _verify_preflight(payload: dict[str, object], family: object) -> None:
    refinement = payload["refinement"]
    sample = payload["exact_sample_cost"]
    if not isinstance(refinement, dict) or not isinstance(sample, dict):
        raise ValueError("adaptive preflight has malformed metadata")
    checks = (
        float(payload["t_star"]) == 20.0,
        int(payload["source_total_depth"]) == 14,
        int(payload["fine_axis_depth"]) == 8,
        int(payload["samples_per_axis"]) == 3,
        float(payload["padding_cells"]) == 1.0,
        float(payload["max_step"]) == DEFAULT_MAX_STEP,
        int(payload["max_jumps"]) == 10,
        int(refinement["refined_parent_count"]) == EXPECTED_REFINED_PARENTS,
        int(refinement["adaptive_leaf_cells"]) == EXPECTED_ACTIVE_CELLS,
        getattr(family, "fingerprint") == EXPECTED_FAMILY_FINGERPRINT,
        int(sample["logical_tensor_points"]) == EXPECTED_LOGICAL_POINTS,
        int(sample["unique_tensor_points"]) == EXPECTED_UNIQUE_POINTS,
        sample["tensor_point_fingerprint"] == EXPECTED_SAMPLE_FINGERPRINT,
        payload["cost_gate_passed"] is True,
    )
    if not all(checks):
        raise ValueError("reviewed adaptive preflight configuration changed")


def _write_protocol(stage_dir: Path, summary: dict[str, object]) -> None:
    gates = bool(summary["stage_gates_passed"])
    protocol = {
        "schema": "spiking-neuron-adaptive-terminal-bridge-protocol-v1",
        "protocol_revision": PROTOCOL_REVISION,
        "selection_uses_scc_or_morse_count": False,
        "baseline_preserved": "../adaptive_primary_v1/t20/summary.json",
        "primary_stage": {
            "path": "t20/summary.json",
            "completed": True,
            "stage_gates_passed": gates,
            "relation_csr_fingerprint": summary["relation_csr"]["fingerprint"],
            "preflight_fingerprint": summary["adaptive_preflight"]["fingerprint"],
        },
        "terminal_carrier": {
            "enabled": True,
            "algorithm_revision": SINGLE_HANDLE_BRIDGE_ALGORITHM,
            "max_bisections": BRIDGE_MAX_BISECTIONS,
            "assumptions": list(SINGLE_HANDLE_BRIDGE_ASSUMPTIONS),
            "terminal_image_carrier_only": True,
            "intermediate_time_graph_edges_added": False,
        },
        "t40_status": "not_in_scope",
        "finite_relation_conley_index_computed": False,
        "conley_index_status": (
            "pending_after_passed_gates" if gates else "not_computed_failed_gates"
        ),
        "analytic_conley_label_attached": False,
        "whole_cell_outer_enclosure_certified": False,
    }
    canonical = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    protocol["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    atomic_write_json(stage_dir.parent / "protocol_summary.json", protocol)


def main() -> int:
    args = _arguments()
    if not args.authorize_reviewed_terminal_bridge:
        raise SystemExit("--authorize-reviewed-terminal-bridge is required")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and args.allow_existing:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        relation = summary["relation_csr"]
        summary["diagnostic_source_provenance"] = (
            authenticate_spiking_neuron_provenance(
                args.output_dir / "provenance.jsonl.gz",
                relation_csr_fingerprint=relation["fingerprint"],
            )
        )
        atomic_write_json(summary_path, summary)
        _write_protocol(args.output_dir, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["stage_gates_passed"] else 2
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    payload, family = load_spiking_neuron_adaptive_preflight(args.preflight)
    _verify_preflight(payload, family)
    preflight_reference = {
        "schema": "spiking-neuron-adaptive-preflight-reference-v1",
        "path": os.path.relpath(args.preflight.resolve(), args.output_dir.resolve()),
        "fingerprint": payload["fingerprint"],
        "source_relation_csr_fingerprint": payload["source_artifacts"][
            "relation_csr_fingerprint"
        ],
        "family_fingerprint": EXPECTED_FAMILY_FINGERPRINT,
        "tensor_point_fingerprint": EXPECTED_SAMPLE_FINGERPRINT,
        "refined_parents": EXPECTED_REFINED_PARENTS,
        "dynamics_launch_authorized_after_focused_bridge_review": True,
        "selection_uses_scc_or_morse_count": False,
    }

    setup = build_spiking_neuron_atlas_model(
        total_depth=16,
        t_star=20.0,
        samples_per_axis=3,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
        max_jumps=10,
        active_family=family,
        single_handle_bridge=True,
        single_handle_bridge_max_bisections=BRIDGE_MAX_BISECTIONS,
    )
    precompute = precompute_spiking_neuron_endpoints(
        atlas_source_boxes(setup),
        NeuronEndpointPrecomputeConfig(
            t_star=20.0,
            samples_per_axis=3,
            padding_cells=1.0,
            max_step=DEFAULT_MAX_STEP,
            max_jumps=10,
        ),
        workers=args.workers,
    )
    if (
        precompute.logical_tensor_points != EXPECTED_LOGICAL_POINTS
        or precompute.unique_points != EXPECTED_UNIQUE_POINTS
    ):
        raise RuntimeError("terminal-bridge run changed the frozen tensor sample set")
    result = compute_spiking_neuron_atlas_acceptance(
        total_depth=16,
        source_total_depth=14,
        samples_per_axis=3,
        t_star=20.0,
        padding_cells=1.0,
        max_step=DEFAULT_MAX_STEP,
        max_jumps=10,
        precomputed_entries=precompute.entries,
        endpoint_precompute=precompute.metadata(),
        protocol_revision=PROTOCOL_REVISION,
        active_family=family,
        adaptive_preflight=preflight_reference,
        max_relation_edges=50_000_000,
        max_native_cache_bytes=1 << 30,
        single_handle_bridge=True,
        single_handle_bridge_max_bisections=BRIDGE_MAX_BISECTIONS,
    )
    args.output_dir.mkdir(parents=True)
    csr = write_spiking_neuron_csr_checkpoint(
        result,
        args.output_dir / "relation.csr",
        max_edges=50_000_000,
        max_payload_bytes=1 << 30,
    )
    result = replace(result, relation_csr=csr)
    plot_spiking_neuron_acceptance(result, args.output_dir / "morse_support.png")
    provenance_path = write_spiking_neuron_provenance(
        result, args.output_dir / "provenance.jsonl.gz"
    )
    provenance = validate_spiking_neuron_provenance(
        provenance_path,
        expected_relation_csr_fingerprint=str(csr["fingerprint"]),
    )
    summary = result.summary()
    summary["adaptive_terminal_bridge_protocol_revision"] = PROTOCOL_REVISION
    summary["artifacts"] = {
        "relation_csr": "relation.csr",
        "source_provenance": "provenance.jsonl.gz",
        "chart_aware_plot": "morse_support.png",
    }
    summary["diagnostic_source_provenance"] = provenance
    summary["t40_status"] = "not_in_scope"
    atomic_write_json(summary_path, summary)
    _write_protocol(args.output_dir, summary)
    print(
        json.dumps(
            {
                "event": "adaptive_terminal_bridge_completed",
                "active_cells": summary["active_cells"],
                "relation_edges": summary["relation_edges"],
                "morse_nodes": summary["morse_nodes"],
                "reference_nodes": summary[
                    "reference_nodes_containing_complete_cycle"
                ],
                "raw_unresolved_sources": summary["raw_unresolved_sources"],
                "residual_unresolved_sources": summary["unresolved_sources"],
                "bridge_sources": summary["single_handle_bridge_sources"],
                "disconnected_images": summary["disconnected_images"],
                "reference_problem_sources": summary[
                    "reference_recurrent_problem_sources"
                ],
                "stage_gates_passed": summary["stage_gates_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["stage_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
