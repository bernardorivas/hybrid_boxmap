#!/usr/bin/env python3
"""Run the rimless-wheel fixed-time suspension through native CMGDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_dynamics.examples.rimless_wheel_atlas import (
    compute_rimless_wheel_atlas_acceptance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--samples-per-axis", type=int, default=3)
    parser.add_argument("--padding-cells", type=float, default=1.0)
    parser.add_argument("--reference-base-samples", type=int, default=241)
    parser.add_argument("--reference-handle-samples", type=int, default=121)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-local-window-gates",
        action="store_true",
        help=(
            "exit nonzero unless the nonempty local relation has the two-node "
            "direct saddle-to-gait result, connected image values and gait "
            "support, covered reference-gait probes, and no skipped event "
            "stages; empty sources and callback sample failures are reported "
            "but do not fail this local-window gate"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_rimless_wheel_atlas_acceptance(
        depth=args.depth,
        t_star=args.tau,
        samples_per_axis=args.samples_per_axis,
        padding_cells=args.padding_cells,
        reference_base_samples=args.reference_base_samples,
        reference_handle_samples=args.reference_handle_samples,
    )
    summary = result.summary()
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if not args.require_local_window_gates:
        return 0
    passed = (
        result.reference_covered
        and result.image_values_connected
        and result.gait_base_connected
        and result.box_map_diagnostics.unresolved_stage_edges == 0
        and result.morse_graph.num_vertices() == 2
        and result.saddle_to_gait_direct
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
