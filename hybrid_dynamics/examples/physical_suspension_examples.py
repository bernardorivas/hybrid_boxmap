"""Run the three physical fixed-time suspension examples with one interface."""

from __future__ import annotations

import argparse
import json

from ._fixed_time_suspension import FixedTimeSuspensionPipelineResult
from .bouncing_ball_suspension import build_bouncing_ball_suspension_pipeline
from .garcia_passive_walker_suspension import (
    build_garcia_passive_walker_suspension_pipeline,
)
from .rimless_wheel_suspension import build_rimless_wheel_suspension_pipeline


def run_all_physical_suspension_examples(
    *,
    t_star: float = 0.5,
    handle_slabs: int = 8,
    compute_cmgdb: bool = False,
) -> tuple[FixedTimeSuspensionPipelineResult, ...]:
    """Return ball, wheel, and walker results in implementation-plan order."""

    return (
        build_bouncing_ball_suspension_pipeline(
            t_star=t_star,
            handle_slabs=handle_slabs,
            compute_cmgdb=compute_cmgdb,
        ),
        build_rimless_wheel_suspension_pipeline(
            t_star=t_star,
            handle_slabs=handle_slabs,
            compute_cmgdb=compute_cmgdb,
        ),
        build_garcia_passive_walker_suspension_pipeline(
            t_star=t_star,
            handle_slabs=handle_slabs,
            compute_cmgdb=compute_cmgdb,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the three physical fixed-time suspension examples",
    )
    parser.add_argument("--t-star", type=float, default=0.5)
    parser.add_argument("--handle-slabs", type=int, default=8)
    parser.add_argument(
        "--compute-cmgdb",
        action="store_true",
        help="run the optional generalized CMGDB bridge on each local orbit skeleton",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            [
                result.summary()
                for result in run_all_physical_suspension_examples(
                    t_star=arguments.t_star,
                    handle_slabs=arguments.handle_slabs,
                    compute_cmgdb=arguments.compute_cmgdb,
                )
            ],
            indent=2,
            sort_keys=True,
        ),
    )


__all__ = ["run_all_physical_suspension_examples"]
