#!/usr/bin/env python3
"""Bind and compare the independently computed neuron samples3/samples5 indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hybrid_dynamics.examples.spiking_neuron_conley import (
    validate_spiking_neuron_conley_summary,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


COMPARISON_REVISION = "spiking-neuron-samples3-samples5-persistence-v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-dir",
        type=Path,
        default=Path("data/spiking_neuron_atlas/scientific_clock_v1"),
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


def _relation_difference(first: Path, second: Path) -> dict[str, object]:
    first_offsets = np.load(first / "offsets.npy", mmap_mode="r")
    first_targets = np.load(first / "targets.npy", mmap_mode="r")
    second_offsets = np.load(second / "offsets.npy", mmap_mode="r")
    second_targets = np.load(second / "targets.npy", mmap_mode="r")
    if len(first_offsets) != len(second_offsets):
        raise ValueError("sensitivity relations have different vertex counts")
    changed: list[int] = []
    added = 0
    removed = 0
    for source in range(len(first_offsets) - 1):
        first_row = first_targets[first_offsets[source] : first_offsets[source + 1]]
        second_row = second_targets[
            second_offsets[source] : second_offsets[source + 1]
        ]
        if np.array_equal(first_row, second_row):
            continue
        changed.append(source)
        first_set = set(int(value) for value in first_row)
        second_set = set(int(value) for value in second_row)
        added += len(second_set - first_set)
        removed += len(first_set - second_set)
    return {
        "changed_source_rows": len(changed),
        "changed_source_indices": changed,
        "samples5_edges_added": added,
        "samples5_edges_removed": removed,
    }


def _support_without_binding(summary: dict[str, object]) -> dict[str, object]:
    support = dict(summary["reference_support"])
    support.pop("fingerprint", None)
    support.pop("relation_csr_fingerprint", None)
    return support


def main() -> int:
    args = _arguments()
    primary_stage = args.scientific_dir / "adaptive_terminal_bridge_v1/t20"
    sensitivity_stage = (
        args.scientific_dir / "adaptive_terminal_bridge_samples5_v1/t20"
    )
    primary_conley_path = primary_stage / "conley_v2/finite_relation_conley.json"
    sensitivity_conley_path = (
        sensitivity_stage / "conley_v2/finite_relation_conley.json"
    )
    primary_validation = validate_spiking_neuron_conley_summary(
        primary_conley_path
    )
    sensitivity_validation = validate_spiking_neuron_conley_summary(
        sensitivity_conley_path
    )
    primary_conley = json.loads(primary_conley_path.read_text(encoding="utf-8"))
    sensitivity_conley = json.loads(
        sensitivity_conley_path.read_text(encoding="utf-8")
    )
    primary = json.loads((primary_stage / "summary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads(
        (sensitivity_stage / "summary.json").read_text(encoding="utf-8")
    )

    checks = {
        "both_relation_stages_passed": bool(
            primary["stage_gates_passed"] and sensitivity["stage_gates_passed"]
        ),
        "same_exact_mixed_family": (
            primary["family_fingerprint"] == sensitivity["family_fingerprint"]
        ),
        "same_standard_top_cell_pair": (
            primary_conley["pair_fingerprint"]
            == sensitivity_conley["pair_fingerprint"]
        ),
        "same_reset_quotient_nerve": (
            primary_conley["nerve_fingerprint"]
            == sensitivity_conley["nerve_fingerprint"]
        ),
        "same_relation_carrier": (
            primary_conley["carrier_fingerprint"]
            == sensitivity_conley["carrier_fingerprint"]
        ),
        "same_reference_source_support": (
            _support_without_binding(primary_conley)
            == _support_without_binding(sensitivity_conley)
        ),
        "same_full_finite_relation_index": (
            primary_conley["finite_relation_conley_index"]
            == sensitivity_conley["finite_relation_conley_index"]
        ),
        "same_shift_class": (
            primary_conley["finite_relation_shift_class"]
            == sensitivity_conley["finite_relation_shift_class"]
        ),
        "neither_claims_continuous_certification": (
            primary_conley["continuous_system_conley_index_certified"] is False
            and sensitivity_conley["continuous_system_conley_index_certified"]
            is False
        ),
    }
    relation_difference = _relation_difference(
        primary_stage / "relation.csr",
        sensitivity_stage / "relation.csr",
    )
    payload = {
        "schema": "spiking-neuron-sampling-persistence-comparison-v1",
        "comparison_revision": COMPARISON_REVISION,
        "selection_uses_scc_or_morse_count": False,
        "primary_samples3": {
            "relation_csr_fingerprint": primary["relation_csr"]["fingerprint"],
            "relation_edges": primary["relation_edges"],
            "raw_unresolved_sources": primary["raw_unresolved_sources"],
            "terminal_bridge_sources": primary["single_handle_bridge_sources"],
            "conley_summary_fingerprint": primary_conley["fingerprint"],
            "chain_checkpoint_fingerprint": primary_conley["chain_checkpoint"][
                "fingerprint"
            ],
            "strict_validation": primary_validation,
        },
        "sensitivity_samples5": {
            "relation_csr_fingerprint": sensitivity["relation_csr"][
                "fingerprint"
            ],
            "relation_edges": sensitivity["relation_edges"],
            "raw_unresolved_sources": sensitivity["raw_unresolved_sources"],
            "terminal_bridge_sources": sensitivity[
                "single_handle_bridge_sources"
            ],
            "conley_summary_fingerprint": sensitivity_conley["fingerprint"],
            "chain_checkpoint_fingerprint": sensitivity_conley[
                "chain_checkpoint"
            ]["fingerprint"],
            "strict_validation": sensitivity_validation,
        },
        "relation_difference": relation_difference,
        "checks": checks,
        "homology_dimensions": sensitivity_conley[
            "finite_relation_conley_index"
        ]["homology_dimensions"],
        "induced_maps": sensitivity_conley["finite_relation_conley_index"][
            "induced_maps"
        ],
        "finite_relation_shift_class": sensitivity_conley[
            "finite_relation_shift_class"
        ],
        "sampling_persistence_passed": all(checks.values()),
        "continuous_system_conley_index_certified": False,
        "analytic_conley_label_attached": False,
        "numerically_rigorous_outer_approximation_claimed": False,
    }
    if not payload["sampling_persistence_passed"]:
        raise ValueError("samples3/samples5 finite results do not persist")
    payload["fingerprint"] = _fingerprint(payload)
    output = args.scientific_dir / (
        "adaptive_terminal_bridge_samples5_v1/comparison.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    atomic_write_json(output, payload)

    sensitivity["sampling_persistence_comparison"] = {
        "path": "../comparison.json",
        "fingerprint": payload["fingerprint"],
        "passed": True,
    }
    sensitivity["paper_acceptance_ready"] = True
    atomic_write_json(sensitivity_stage / "summary.json", sensitivity)

    protocol_path = args.scientific_dir / (
        "adaptive_terminal_bridge_v1/protocol_summary.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.pop("fingerprint", None)
    protocol["samples5_sensitivity"].update(
        {
            "status": "relation_conley_and_comparison_passed",
            "comparison_path": "../adaptive_terminal_bridge_samples5_v1/comparison.json",
            "comparison_fingerprint": payload["fingerprint"],
            "sampling_persistence_passed": True,
            "paper_acceptance_ready": True,
        }
    )
    protocol["fingerprint"] = _fingerprint(protocol)
    atomic_write_json(protocol_path, protocol)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
