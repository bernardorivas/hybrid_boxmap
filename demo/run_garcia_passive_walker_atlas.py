#!/usr/bin/env python3
"""Run the coarse Garcia walker suspension plumbing through native CMGDB."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    DEFAULT_T_STAR,
    compute_garcia_passive_walker_atlas_acceptance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Atlas depth, currently restricted to a positive multiple of four",
    )
    parser.add_argument("--tau", type=float, default=DEFAULT_T_STAR)
    parser.add_argument("--samples-per-axis", type=int, default=3)
    parser.add_argument("--padding-cells", type=float, default=1.0)
    parser.add_argument("--reference-base-samples-per-stride", type=int, default=17)
    parser.add_argument("--reference-handle-samples", type=int, default=9)
    parser.add_argument("--reference-max-step", type=float, default=0.005)
    parser.add_argument(
        "--active-stencil-radius",
        type=int,
        help=(
            "use the native selected-dyadic Atlas at depth/4 per axis, with "
            "this positive Chebyshev halo around the stored two-stride gait"
        ),
    )
    parser.add_argument("--active-base-samples-per-stride", type=int)
    parser.add_argument("--active-handle-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-plumbing-gates",
        action="store_true",
        help=(
            "exit nonzero unless stored-gait endpoints are covered, every "
            "nonempty image is quotient-connected, and no sample adjacency "
            "skips an event stage; this does not require or imply a complete "
            "self-map, refined Morse recovery, or whole-cell enclosure"
        ),
    )
    parser.add_argument(
        "--require-complete-sampled-self-map",
        action="store_true",
        help=(
            "also exit nonzero when any grid source has an empty image or any "
            "callback sample fails; the current coarse local box is expected "
            "to fail this stronger check"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = compute_garcia_passive_walker_atlas_acceptance(
            depth=args.depth,
            t_star=args.tau,
            samples_per_axis=args.samples_per_axis,
            padding_cells=args.padding_cells,
            reference_base_samples_per_stride=(
                args.reference_base_samples_per_stride
            ),
            reference_handle_samples=args.reference_handle_samples,
            reference_max_step=args.reference_max_step,
            active_stencil_radius=args.active_stencil_radius,
            active_base_samples_per_stride=args.active_base_samples_per_stride,
            active_handle_samples=args.active_handle_samples,
        )
    payload = {
        "model": "garcia_passive_walker_fixed_time_suspension_atlas",
        "elapsed_seconds": time.perf_counter() - started,
        "result": result.summary(),
        "scope_note": (
            "The active run uses a disclosed positive-radius union of full "
            "dyadic boxes around both continuous stride arcs and both handle "
            "traversals; it is not an orbit-only graph. Active-boundary exits "
            "are reported explicitly. Passing finite gates does not certify a "
            "whole-cell outer relation or a scientific Morse decomposition."
            if args.active_stencil_radius is not None
            else "This is a coarse full-rectangle plumbing run. Passing finite "
            "gates does not certify a whole-cell outer relation or a scientific "
            "Morse decomposition."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.require_plumbing_gates and not result.plumbing_gates_passed:
        return 2
    if (
        args.require_complete_sampled_self_map
        and not result.sampled_self_map_complete
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
