#!/usr/bin/env python3
"""Run the predeclared 5x5 sensitivity on the exact accepted neuron family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    DEFAULT_MAX_STEP,
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
from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


PROTOCOL_REVISION = "spiking-neuron-adaptive-terminal-bridge-samples5-t20-v1"
EXPECTED_PREFLIGHT_FINGERPRINT = (
    # Recomputed 2026-09-03 (previously f22fc029...); see the relation note below.
    "bbbd20468b30dd7d9d382dcfa70af468fc66f95eed5c7535b7647d3b4a94835b"
)
EXPECTED_ADAPTIVE_PREFLIGHT_FINGERPRINT = (
    # Recomputed 2026-09-03 (previously 0d35774f...); only source_artifacts changed.
    "4554dbbd8d31fa45df16e85486f508c03cc9ed0a87413635f3bb1b4506faabe1"
)
EXPECTED_FAMILY_FINGERPRINT = (
    "45917b973ccf7e10eb891ad27276529377db5f9de961dedbb62468936d445706"
)
EXPECTED_PRIMARY_RELATION_FINGERPRINT = (
    # Recomputed 2026-09-03.  The relation arrays (offsets/targets sha256) are
    # unchanged from the 2026-08-23 run (44ea6ef4...); the CSR configuration
    # metadata gained adaptive_preflight_fingerprint, which shifts the hash.
    "6a00731ee973c00cb906f5a454d9638028c1044e9c09cdcc3be2fc404c980661"
)
EXPECTED_LOGICAL_POINTS = 226_850
EXPECTED_UNIQUE_POINTS = 146_650
EXPECTED_SAMPLE_FINGERPRINT = (
    "7376b101e06dcaa729914252d51519c742bff762b379b38c318df42d84bdf269"
)
EXPECTED_ACTIVE_CELLS = 9_074
EXPECTED_REFINED_PARENTS = 122
BRIDGE_MAX_BISECTIONS = 12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_samples5_v1/preflight.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_samples5_v1/t20"
        ),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--authorize-predeclared-sensitivity",
        action="store_true",
        help="required explicit authorization for the frozen samples5 run",
    )
    return parser.parse_args()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_preflight(path: Path) -> tuple[dict[str, object], object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "spiking-neuron-terminal-bridge-samples5-preflight-v1"
    ):
        raise ValueError("unknown samples5 sensitivity preflight")
    claimed = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if claimed != _fingerprint(unsigned) or claimed != EXPECTED_PREFLIGHT_FINGERPRINT:
        raise ValueError("samples5 sensitivity preflight fingerprint changed")
    configuration = payload["configuration"]
    sample = payload["exact_sample_cost"]
    frozen = payload["frozen_inputs"]
    checks = (
        payload["dynamics_evaluated"] is False,
        payload["selection_uses_scc_or_morse_count"] is False,
        payload["cost_gate_passed"] is True,
        payload["launch_authorized"] is True,
        float(configuration["t_star"]) == 20.0,
        int(configuration["samples_per_axis"]) == 5,
        float(configuration["padding_cells"]) == 1.0,
        float(configuration["max_step"]) == DEFAULT_MAX_STEP,
        int(configuration["max_jumps"]) == 10,
        configuration["single_handle_bridge_algorithm_revision"]
        == SINGLE_HANDLE_BRIDGE_ALGORITHM,
        int(configuration["single_handle_bridge_max_bisections"])
        == BRIDGE_MAX_BISECTIONS,
        int(sample["logical_tensor_points"]) == EXPECTED_LOGICAL_POINTS,
        int(sample["unique_tensor_points"]) == EXPECTED_UNIQUE_POINTS,
        sample["tensor_point_fingerprint"] == EXPECTED_SAMPLE_FINGERPRINT,
        frozen["family_fingerprint"] == EXPECTED_FAMILY_FINGERPRINT,
        int(frozen["active_cells"]) == EXPECTED_ACTIVE_CELLS,
        int(frozen["refined_parents"]) == EXPECTED_REFINED_PARENTS,
        frozen["primary_samples3_relation_fingerprint"]
        == EXPECTED_PRIMARY_RELATION_FINGERPRINT,
    )
    if not all(checks):
        raise ValueError("predeclared samples5 sensitivity configuration changed")
    adaptive_path = (
        path.parent / str(frozen["adaptive_preflight_path"])
    ).resolve()
    adaptive_payload, family = load_spiking_neuron_adaptive_preflight(adaptive_path)
    if adaptive_payload["fingerprint"] != EXPECTED_ADAPTIVE_PREFLIGHT_FINGERPRINT:
        raise ValueError("the relation-bound adaptive family preflight changed")
    if family.fingerprint != EXPECTED_FAMILY_FINGERPRINT:
        raise ValueError("the exact mixed family changed")
    return payload, family


def _update_protocol(
    output_dir: Path,
    summary: dict[str, object],
) -> None:
    protocol_path = output_dir.parent.parent / (
        "adaptive_terminal_bridge_v1/protocol_summary.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.pop("fingerprint", None)
    protocol["samples5_sensitivity"] = {
        "status": (
            "relation_passed_conley_pending"
            if summary["stage_gates_passed"]
            else "relation_failed_gates"
        ),
        "path": os.path.relpath(
            (output_dir / "summary.json").resolve(),
            protocol_path.parent.resolve(),
        ),
        "preflight_path": os.path.relpath(
            (output_dir.parent / "preflight.json").resolve(),
            protocol_path.parent.resolve(),
        ),
        "preflight_fingerprint": EXPECTED_PREFLIGHT_FINGERPRINT,
        "relation_csr_fingerprint": summary["relation_csr"]["fingerprint"],
        "same_mixed_family": (
            summary["family_fingerprint"] == EXPECTED_FAMILY_FINGERPRINT
        ),
        "stage_gates_passed": summary["stage_gates_passed"],
        "finite_relation_conley_index_computed": False,
        "paper_acceptance_ready": False,
    }
    protocol["fingerprint"] = _fingerprint(protocol)
    atomic_write_json(protocol_path, protocol)


def main() -> int:
    args = _arguments()
    if not args.authorize_predeclared_sensitivity:
        raise SystemExit("--authorize-predeclared-sensitivity is required")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    preflight, family = _load_preflight(args.preflight)
    preflight_reference = {
        "schema": "spiking-neuron-samples5-preflight-reference-v1",
        "path": os.path.relpath(args.preflight.resolve(), args.output_dir.resolve()),
        "fingerprint": preflight["fingerprint"],
        "adaptive_preflight_fingerprint": (
            EXPECTED_ADAPTIVE_PREFLIGHT_FINGERPRINT
        ),
        "family_fingerprint": EXPECTED_FAMILY_FINGERPRINT,
        "tensor_point_fingerprint": EXPECTED_SAMPLE_FINGERPRINT,
        "primary_samples3_relation_fingerprint": (
            EXPECTED_PRIMARY_RELATION_FINGERPRINT
        ),
        "selection_uses_scc_or_morse_count": False,
    }

    setup = build_spiking_neuron_atlas_model(
        total_depth=16,
        t_star=20.0,
        samples_per_axis=5,
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
            samples_per_axis=5,
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
        raise RuntimeError("samples5 precompute changed its frozen tensor-node cost")

    result = compute_spiking_neuron_atlas_acceptance(
        total_depth=16,
        source_total_depth=14,
        samples_per_axis=5,
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
    summary["sampling_sensitivity_protocol_revision"] = PROTOCOL_REVISION
    summary["sampling_sensitivity_preflight"] = preflight_reference
    summary["primary_samples3_relation_fingerprint"] = (
        EXPECTED_PRIMARY_RELATION_FINGERPRINT
    )
    summary["diagnostic_source_provenance"] = provenance
    summary["artifacts"] = {
        "relation_csr": "relation.csr",
        "source_provenance": "provenance.jsonl.gz",
        "chart_aware_plot": "morse_support.png",
    }
    summary["finite_relation_conley_index_computed"] = False
    summary["paper_acceptance_ready"] = False
    atomic_write_json(args.output_dir / "summary.json", summary)
    _update_protocol(args.output_dir, summary)
    print(
        json.dumps(
            {
                "event": "spiking_neuron_samples5_sensitivity_completed",
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
