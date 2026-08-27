#!/usr/bin/env python3
"""Build relation-bound adaptive neuron families without integrating dynamics.

The rule is fixed by the depth-14 source diagnostics, never by an SCC count:
refine unresolved or quotient-disconnected-image sources, one closed
same-chart ring, and their exact positive-length seam cofaces to axis depth 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hybrid_dynamics.examples.spiking_neuron_atlas import (
    ADAPTIVE_PREFLIGHT_REVISION,
    build_spiking_neuron_adaptive_preflight,
)
from hybrid_dynamics.src.io_utils import atomic_write_json


FROZEN_CLOCKS = (20, 40)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-dir",
        type=Path,
        default=Path("data/spiking_neuron_atlas/scientific_clock_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/spiking_neuron_atlas/scientific_clock_v1/adaptive_preflight_v1"
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


def main() -> int:
    args = _arguments()
    stages: dict[str, dict[str, object]] = {}
    for clock in FROZEN_CLOCKS:
        source = args.scientific_dir / f"tau{100*clock:04d}_depth14_samples3"
        result = build_spiking_neuron_adaptive_preflight(
            source / "summary.json",
            source / "provenance.jsonl.gz",
        )
        payload = result.to_dict()
        payload["source_paths"] = {
            "summary": f"../../tau{100*clock:04d}_depth14_samples3/summary.json",
            "provenance": (
                f"../../tau{100*clock:04d}_depth14_samples3/provenance.jsonl.gz"
            ),
            "relation_csr": f"../../tau{100*clock:04d}_depth14_samples3/relation.csr",
        }
        payload["scientific_use_gate"] = (
            {
                "status": "primary_candidate_requires_review",
                "reason": (
                    "t*=20 is the predeclared primary scientific clock; this "
                    "preflight reports costs before any adaptive dynamics launch"
                ),
            }
            if clock == 20
            else {
                "status": "pending_t20_adaptive_outcome",
                "reason": (
                    "t*=40 is a sensitivity clock and is launched only if the "
                    "primary adaptive relation makes the error-driven refinement "
                    "scientifically informative"
                ),
            }
        )
        payload["fingerprint"] = hashlib.sha256(_canonical(payload)).hexdigest()
        atomic_write_json(
            args.output_dir / f"t{clock}" / "preflight.json", payload
        )
        stages[f"t{clock}"] = payload

    protocol = {
        "schema": "spiking-neuron-adaptive-preflight-protocol-v1",
        "preflight_revision": ADAPTIVE_PREFLIGHT_REVISION,
        "dynamics_evaluated": False,
        "selection_uses_scc_or_morse_count": False,
        "both_source_clocks_preflighted": set(stages) == {"t20", "t40"},
        "adaptive_dynamics_launch_authorized": False,
        "stages": stages,
    }
    protocol["fingerprint"] = hashlib.sha256(_canonical(protocol)).hexdigest()
    atomic_write_json(args.output_dir / "protocol_summary.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
