#!/usr/bin/env python3
"""Compute the honest GF(5) finite-relation index of the accepted neuron stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_conley import (
    CONLEY_PROTOCOL_REVISION,
    compute_spiking_neuron_finite_relation_conley,
    validate_spiking_neuron_conley_summary,
    write_spiking_neuron_conley_checkpoint,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/"
            "adaptive_terminal_bridge_v1/t20"
        ),
    )
    parser.add_argument(
        "--morse-node",
        type=int,
        default=None,
        help="defaults to the unique node containing the complete reference cycle",
    )
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _with_fingerprint(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("fingerprint", None)
    result["fingerprint"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _update_acceptance_and_protocol(
    stage: Path,
    conley_reference: dict[str, object],
    conley_summary: dict[str, object],
) -> None:
    acceptance_path = stage / "summary.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["finite_relation_conley_index_computed"] = True
    acceptance["finite_relation_conley"] = {
        "schema": "spiking-neuron-finite-relation-conley-reference-v2",
        "path": "conley_v2/finite_relation_conley.json",
        "fingerprint": conley_summary["fingerprint"],
        "chain_checkpoint": conley_reference,
        "shift_class": conley_summary["finite_relation_shift_class"],
        "continuous_system_conley_index_certified": False,
        "analytic_conley_label_attached": False,
    }
    atomic_write_json(acceptance_path, acceptance, binary=True)

    local_protocol = stage.parent / "protocol_summary.json"
    protocol_path = (
        local_protocol
        if local_protocol.exists()
        else stage.parent.parent
        / "adaptive_terminal_bridge_v1/protocol_summary.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if "sampling_sensitivity_protocol_revision" in acceptance:
        sensitivity = dict(protocol["samples5_sensitivity"])
        sensitivity["status"] = "relation_and_conley_completed_comparison_pending"
        sensitivity["finite_relation_conley_index_computed"] = True
        sensitivity["finite_relation_conley"] = acceptance[
            "finite_relation_conley"
        ]
        sensitivity["paper_acceptance_ready"] = False
        protocol["samples5_sensitivity"] = sensitivity
    else:
        protocol["finite_relation_conley_index_computed"] = True
        protocol["conley_index_status"] = "finite_relation_shift_class_computed"
        protocol["finite_relation_conley"] = acceptance[
            "finite_relation_conley"
        ]
    protocol["continuous_system_conley_index_certified"] = False
    protocol["analytic_conley_label_attached"] = False
    protocol.setdefault(
        "samples5_sensitivity",
        {
            "status": "predeclared_pending",
            "same_mixed_family_required": True,
            "paper_acceptance_ready": False,
        },
    )
    atomic_write_json(
        protocol_path, _with_fingerprint(protocol), binary=True
    )


def main() -> int:
    args = _arguments()
    output = args.stage_dir / "conley_v2"
    summary_path = output / "finite_relation_conley.json"
    checkpoint_path = output / "chain_checkpoint.json.gz"
    if summary_path.exists():
        if not args.validate_existing:
            raise FileExistsError(f"refusing to overwrite {summary_path}")
        validation = validate_spiking_neuron_conley_summary(
            summary_path,
            checkpoint_path=checkpoint_path,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["strict_existing_validation"] = validation
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    acceptance = json.loads(
        (args.stage_dir / "summary.json").read_text(encoding="utf-8")
    )
    morse_node = args.morse_node
    if morse_node is None:
        reference_nodes = tuple(
            int(node)
            for node in acceptance["reference_nodes_containing_complete_cycle"]
        )
        if len(reference_nodes) != 1:
            raise ValueError(
                "Conley runner requires one explicit --morse-node when the "
                "reference cycle is represented by zero or multiple nodes"
            )
        morse_node = reference_nodes[0]
    started = time.perf_counter()
    result = compute_spiking_neuron_finite_relation_conley(
        args.stage_dir,
        morse_node=morse_node,
    )
    computed_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_reference = write_spiking_neuron_conley_checkpoint(
        result, checkpoint_path
    )
    summary = result.summary()
    summary["conley_computation_seconds"] = computed_seconds
    summary["chain_checkpoint"] = checkpoint_reference
    summary["protocol_revision"] = CONLEY_PROTOCOL_REVISION
    summary = _with_fingerprint(summary)
    atomic_write_json(summary_path, summary, binary=True)
    _update_acceptance_and_protocol(
        args.stage_dir,
        checkpoint_reference,
        summary,
    )
    print(
        json.dumps(
            {
                "event": "spiking_neuron_finite_relation_conley_completed",
                "pair": summary["top_cell_pair"],
                "cells_by_dimension": summary["quotient_nerve"][
                    "cells_by_dimension"
                ],
                "homology_dimensions": summary["finite_relation_conley_index"][
                    "homology_dimensions"
                ],
                "shift_class": summary["finite_relation_shift_class"],
                "finite_relation_conley_index_computed": summary[
                    "finite_relation_conley_index_computed"
                ],
                "continuous_system_conley_index_certified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
