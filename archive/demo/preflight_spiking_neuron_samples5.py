#!/usr/bin/env python3
"""Freeze the exact mixed-family 5x5 neuron sensitivity without dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    DEFAULT_MAX_STEP,
    load_spiking_neuron_adaptive_preflight,
    spiking_neuron_tensor_sample_cost,
)
from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    SINGLE_HANDLE_BRIDGE_ASSUMPTIONS,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


PROTOCOL_REVISION = "spiking-neuron-adaptive-terminal-bridge-samples5-t20-v1"
EXPECTED_FAMILY_FINGERPRINT = (
    "45917b973ccf7e10eb891ad27276529377db5f9de961dedbb62468936d445706"
)
EXPECTED_PRIMARY_RELATION_FINGERPRINT = (
    # Recomputed 2026-09-03.  The relation arrays (offsets/targets sha256) are
    # unchanged from the 2026-08-23 run (44ea6ef4...); the CSR configuration
    # metadata gained adaptive_preflight_fingerprint, which shifts the hash.
    "6a00731ee973c00cb906f5a454d9638028c1044e9c09cdcc3be2fc404c980661"
)
EXPECTED_PRIMARY_CONLEY_FINGERPRINT = (
    # Recomputed 2026-09-03 (previously 07f5dd2a...): same homology, induced
    # maps, nerve, pair, and carrier; the summary now embeds the index record
    # in its chain-checkpoint reference and binds to the recomputed CSR hash.
    "2fc5d7869b41c7fb1afcc8c0b1074ec0d94a0486261744cb5f5e366ef69b74fb"
)
BRIDGE_MAX_BISECTIONS = 12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-preflight",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_preflight_v1/t20/preflight.json"
        ),
    )
    parser.add_argument(
        "--primary-stage",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_v1/t20"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_samples5_v1/preflight.json"
        ),
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


def main() -> int:
    args = _arguments()
    adaptive_payload, family = load_spiking_neuron_adaptive_preflight(
        args.adaptive_preflight
    )
    primary = json.loads(
        (args.primary_stage / "summary.json").read_text(encoding="utf-8")
    )
    conley = json.loads(
        (args.primary_stage / "conley_v2/finite_relation_conley.json").read_text(
            encoding="utf-8"
        )
    )
    if family.fingerprint != EXPECTED_FAMILY_FINGERPRINT:
        raise ValueError("the exact mixed adaptive family changed")
    if primary["relation_csr"]["fingerprint"] != (
        EXPECTED_PRIMARY_RELATION_FINGERPRINT
    ):
        raise ValueError("the accepted samples3 primary relation changed")
    if conley["fingerprint"] != EXPECTED_PRIMARY_CONLEY_FINGERPRINT:
        raise ValueError("the accepted samples3 finite index changed")
    if not primary["stage_gates_passed"] or not conley[
        "finite_relation_conley_index_computed"
    ]:
        raise ValueError("samples5 sensitivity requires accepted samples3 inputs")

    sample_cost = spiking_neuron_tensor_sample_cost(
        family,
        samples_per_axis=5,
    )
    baseline_precompute = primary["endpoint_precompute"]
    projected_seconds = (
        float(baseline_precompute["elapsed_seconds"])
        * int(sample_cost["unique_tensor_points"])
        / int(baseline_precompute["unique_points"])
    )
    payload = {
        "schema": "spiking-neuron-terminal-bridge-samples5-preflight-v1",
        "protocol_revision": PROTOCOL_REVISION,
        "dynamics_evaluated": False,
        "selection_uses_scc_or_morse_count": False,
        "family_selected_before_sensitivity_results": True,
        "scientific_role": "fixed_sampling_sensitivity_of_accepted_t20_mixed_family",
        "configuration": {
            "t_star": 20.0,
            "samples_per_axis": 5,
            "padding_cells": 1.0,
            "max_step": DEFAULT_MAX_STEP,
            "max_jumps": 10,
            "total_depth": 16,
            "source_total_depth": 14,
            "single_handle_bridge_enabled": True,
            "single_handle_bridge_algorithm_revision": (
                SINGLE_HANDLE_BRIDGE_ALGORITHM
            ),
            "single_handle_bridge_max_bisections": BRIDGE_MAX_BISECTIONS,
            "terminal_carrier_assumptions": list(SINGLE_HANDLE_BRIDGE_ASSUMPTIONS),
            "intermediate_time_graph_edges_added": False,
        },
        "frozen_inputs": {
            "adaptive_preflight_path": os.path.relpath(
                args.adaptive_preflight.resolve(), args.output.parent.resolve()
            ),
            "adaptive_preflight_fingerprint": adaptive_payload["fingerprint"],
            "family_fingerprint": family.fingerprint,
            "active_cells": len(family.cells),
            "refined_parents": len(family.refined_parents),
            "primary_samples3_relation_fingerprint": (
                EXPECTED_PRIMARY_RELATION_FINGERPRINT
            ),
            "primary_samples3_conley_fingerprint": (
                EXPECTED_PRIMARY_CONLEY_FINGERPRINT
            ),
            "primary_samples3_reference_support_fingerprint": conley[
                "reference_support"
            ]["fingerprint"],
        },
        "exact_sample_cost": sample_cost,
        "fixed_caps": {
            "max_relation_edges": 50_000_000,
            "max_native_cache_bytes": 1 << 30,
            "max_csr_payload_bytes": 1 << 30,
            "max_unique_endpoint_points": 200_000,
            "max_horizon_over_step_units": 200_000_000,
        },
        "cost_gate_passed": bool(
            int(sample_cost["unique_tensor_points"]) <= 200_000
            and int(sample_cost["unique_tensor_points"]) * 20.0 / DEFAULT_MAX_STEP
            <= 200_000_000
        ),
        "timing_projection_not_a_guarantee_seconds": projected_seconds,
        "acceptance_rule": {
            "relation_stage_must_pass_every_existing_gate": True,
            "complete_reference_cycle_must_lie_in_recurrent_component": True,
            "finite_relation_index_must_be_computed_on_standard_pair": True,
            "reference_support_and_shift_class_must_match_samples3_primary": True,
            "no_expected_polynomial_gate": True,
            "no_analytic_fallback": True,
        },
        "launch_authorized": True,
    }
    payload["fingerprint"] = _fingerprint(payload)
    atomic_write_json(args.output, payload, refuse_existing=True)

    protocol_path = args.primary_stage.parent / "protocol_summary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.pop("fingerprint", None)
    protocol["samples5_sensitivity"] = {
        "status": "predeclared_authorized_not_run",
        "preflight_path": os.path.relpath(
            args.output.resolve(), protocol_path.parent.resolve()
        ),
        "preflight_fingerprint": payload["fingerprint"],
        "same_mixed_family_required": True,
        "paper_acceptance_ready": False,
    }
    protocol["fingerprint"] = _fingerprint(protocol)
    atomic_write_json(protocol_path, protocol, refuse_existing=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
