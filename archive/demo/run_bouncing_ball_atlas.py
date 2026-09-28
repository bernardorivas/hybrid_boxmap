#!/usr/bin/env python3
"""Run the bouncing-ball fixed-time suspension through native CMGDB."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from hybrid_dynamics.examples.bouncing_ball_atlas import (
    DEFAULT_T_STAR,
    compute_bouncing_ball_atlas_acceptance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        action="append",
        type=int,
        help="even Atlas depth; repeat to run a refinement comparison",
    )
    parser.add_argument(
        "--tau",
        action="append",
        type=float,
        help="fixed suspension time; repeat to screen several values",
    )
    parser.add_argument("--samples-per-axis", type=int, default=3)
    parser.add_argument("--padding-cells", type=float, default=1.0)
    parser.add_argument("--reference-speed-samples", type=int, default=25)
    parser.add_argument("--reference-base-samples", type=int, default=17)
    parser.add_argument("--reference-handle-samples", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-local-window-gates",
        action="store_true",
        help=(
            "exit nonzero unless every run has one Morse node, covered analytic "
            "shrinking-bounce probes, connected nonempty image values and Zeno "
            "support, and no skipped event stages; domain exits are disclosed but "
            "do not turn this into a global self-map claim"
        ),
    )
    parser.add_argument(
        "--require-stable-two-depths",
        action="store_true",
        help=(
            "also require at least two distinct depths for every requested tau "
            "and passage of the local-window gates at all those depths"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    depths = sorted(set(args.depth or [8]))
    taus = sorted(set(args.tau or [DEFAULT_T_STAR]))
    runs = []
    results = []
    for tau in taus:
        for depth in depths:
            started = time.perf_counter()
            result = compute_bouncing_ball_atlas_acceptance(
                depth=depth,
                t_star=tau,
                samples_per_axis=args.samples_per_axis,
                padding_cells=args.padding_cells,
                reference_speed_samples=args.reference_speed_samples,
                reference_base_samples=args.reference_base_samples,
                reference_handle_samples=args.reference_handle_samples,
            )
            summary = result.summary()
            summary["elapsed_seconds"] = time.perf_counter() - started
            runs.append(summary)
            results.append(result)

    stability = {}
    for tau in taus:
        selected = [result for result in results if result.t_star == tau]
        stability[str(tau)] = {
            "depths": [result.depth for result in selected],
            "at_least_two_depths": len({result.depth for result in selected}) >= 2,
            "one_morse_node_at_all_depths": all(
                result.one_morse_node for result in selected
            ),
            "local_window_gates_at_all_depths": all(
                result.local_window_gates_passed for result in selected
            ),
        }

    payload = {
        "model": "bouncing_ball_fixed_time_suspension_atlas",
        "computed_at": date.today().isoformat(),
        "shared_configuration": {
            "base_chart": {
                "coordinates": ["h", "v"],
                "bounds": [[0.0, 2.0], [-5.0, 5.0]],
            },
            "handle_chart": {
                "coordinates": ["v_guard", "s"],
                "bounds": [[-5.0, 0.0], [0.0, 1.0]],
            },
            "gravity": 9.81,
            "coefficient_of_restitution": 0.8,
            "source_samples_per_axis": args.samples_per_axis,
            "padding_cells": args.padding_cells,
            "require_domain_path": True,
            "max_step": 0.02,
            "reference_family": (
                "closed-form rebound arcs, both reset seams, exact rest fiber, "
                "and logarithmically shrinking pre-impact speeds"
            ),
        },
        "runs": runs,
        "stability_by_tau": stability,
        "scope": {
            "relation_scope": "local-window computation on the nonempty relation",
            "whole_cell_outer_enclosure_certified": False,
            "conley_index_computed": False,
            "global_attractor_lattice_interpretation": False,
            "note": (
                "Finite analytic probes, event-stage checks, and quotient "
                "connectivity are falsification gates. Failed samples and empty "
                "images are retained as explicit evidence of exits from the "
                "rectangular chart window, not silently converted into dynamics."
            ),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.require_local_window_gates and not all(
        result.local_window_gates_passed for result in results
    ):
        return 2
    if args.require_stable_two_depths and not all(
        record["at_least_two_depths"]
        and record["local_window_gates_at_all_depths"]
        for record in stability.values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
