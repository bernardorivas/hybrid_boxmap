#!/usr/bin/env python3
"""Build an exit-aware local pair from a persisted walker Atlas relation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hybrid_dynamics.examples.walker_local_index import (
    ExitAwareLocalIndexInput,
    construct_exit_aware_local_index_pair,
    load_exit_aware_relation,
    merge_relation_inputs,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "relation",
        type=Path,
        help=(
            "persisted physical-conley or Garcia walker open-exit "
            ".json.gz relation cache"
        ),
    )
    parser.add_argument(
        "specification",
        type=Path,
        nargs="?",
        help=(
            "optional exit-aware-local-index-input-v1 JSON file; walker "
            "relations can supply their embedded candidate and exit flags"
        ),
    )
    parser.add_argument(
        "--morse-node",
        type=int,
        help="override the specification or embedded gait Morse node",
    )
    parser.add_argument(
        "--collar-layers",
        type=int,
        help="override the specification collar width (default without a spec: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="certificate JSON path; written for both pass and fail results",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=1.0e-12,
        help="closed-box contact tolerance (default: 1e-12)",
    )
    return parser.parse_args(argv)


def _check_expected_relation(
    snapshot: object,
    specification: ExitAwareLocalIndexInput,
) -> None:
    expected = specification.metadata.get("expected_relation")
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise ValueError("metadata.expected_relation must be an object")
    comparisons = {
        "model": getattr(snapshot, "model"),
        "depth": getattr(snapshot, "depth"),
        "t_star": getattr(snapshot, "t_star"),
    }
    mismatches = {}
    for key, actual in comparisons.items():
        if key not in expected:
            continue
        expected_value = expected[key]
        matches = (
            abs(float(expected_value) - float(actual)) <= 1.0e-12
            if key == "t_star"
            else expected_value == actual
        )
        if not matches:
            mismatches[key] = {"expected": expected_value, "actual": actual}
    if mismatches:
        raise ValueError(
            "local-index specification does not match its relation cache: "
            f"{mismatches!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    loaded_relation = load_exit_aware_relation(args.relation)
    snapshot = loaded_relation.snapshot
    if args.specification is None:
        morse_node = (
            args.morse_node
            if args.morse_node is not None
            else loaded_relation.embedded_morse_node
        )
        if morse_node is None:
            raise ValueError(
                "a Morse node is required when the relation has no embedded "
                "gait candidate"
            )
        external_specification = ExitAwareLocalIndexInput(
            morse_node=morse_node,
            collar_layers=(
                1 if args.collar_layers is None else args.collar_layers
            ),
            open_exit_flags={},
        )
    else:
        external_specification = ExitAwareLocalIndexInput.read_json(
            args.specification
        )
        if args.morse_node is not None or args.collar_layers is not None:
            external_specification = ExitAwareLocalIndexInput(
                morse_node=(
                    external_specification.morse_node
                    if args.morse_node is None
                    else args.morse_node
                ),
                collar_layers=(
                    external_specification.collar_layers
                    if args.collar_layers is None
                    else args.collar_layers
                ),
                open_exit_flags=external_specification.open_exit_flags,
                allowed_cells=external_specification.allowed_cells,
                quotient_neighbor_pairs=(
                    external_specification.quotient_neighbor_pairs
                ),
                metadata=external_specification.metadata,
            )
    specification = merge_relation_inputs(
        external_specification,
        loaded_relation,
    )
    _check_expected_relation(snapshot, specification)
    certificate = construct_exit_aware_local_index_pair(
        snapshot,
        specification,
        persistence_schema=loaded_relation.persistence_schema,
        relation_sha256=sha256_file(args.relation),
        tolerance=args.contact_tolerance,
    )
    certificate.write_json(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": certificate.model,
                "depth": certificate.depth,
                "morse_node": certificate.morse_node,
                "S_cells": len(certificate.s_cells),
                "N_cells": len(certificate.n_cells),
                "L_cells": len(certificate.l_cells),
                "finite_relation_pair_certified": (
                    certificate.finite_relation_pair_certified
                ),
                "continuous_system_index_pair_certified": False,
            },
            sort_keys=True,
        )
    )
    return 0 if certificate.finite_relation_pair_certified else 2


if __name__ == "__main__":
    raise SystemExit(main())
