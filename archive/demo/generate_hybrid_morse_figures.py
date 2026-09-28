"""Generate paper figures from cellwise fixed-time suspension relations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hybrid_dynamics import (
    CMGDB_MORSE_PALETTE,
    PlotHybridMorseSets,
    SCIENTIFIC_MORSE_PALETTE,
    hybrid_morse_components,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.examples.physical_suspension_grid import (
    build_bouncing_ball_suspension_grid,
    build_garcia_passive_walker_suspension_grid,
    build_rimless_wheel_suspension_grid,
)


_OUTPUT_NAMES = {
    "bouncing_ball_zeno_suspension": "hybrid-morse-bouncing-ball",
    "rimless_wheel_walking_suspension": "hybrid-morse-rimless-wheel",
    "garcia_passive_walker_period_two_suspension": "hybrid-morse-passive-walker",
}

_PRODUCTION_SPECS = (
    (
        "bouncing_ball_zeno_suspension",
        build_bouncing_ball_suspension_grid,
        {"subdivisions": (64, 256), "handle_slabs": 16, "padding_cells": 1.0},
    ),
    (
        "rimless_wheel_walking_suspension",
        build_rimless_wheel_suspension_grid,
        {"subdivisions": (201, 201), "handle_slabs": 16, "padding_cells": 1.0},
    ),
    (
        "garcia_passive_walker_period_two_suspension",
        build_garcia_passive_walker_suspension_grid,
        {
            "subdivisions": (16, 16, 16, 16),
            "handle_slabs": 16,
            "padding_cells": 1.0,
            "active_tube_radius": 1,
        },
    ),
)

_QUICK_SPECS = (
    (
        "bouncing_ball_zeno_suspension",
        build_bouncing_ball_suspension_grid,
        {"subdivisions": (12, 48), "handle_slabs": 8, "padding_cells": 1.0},
    ),
    (
        "rimless_wheel_walking_suspension",
        build_rimless_wheel_suspension_grid,
        {"subdivisions": (32, 32), "handle_slabs": 8, "padding_cells": 1.0},
    ),
    (
        "garcia_passive_walker_period_two_suspension",
        build_garcia_passive_walker_suspension_grid,
        {
            "subdivisions": (4, 4, 4, 4),
            "handle_slabs": 8,
            "padding_cells": 1.0,
            "active_tube_radius": 1,
        },
    ),
)


def _cache_name(model_name: str, parameters: dict[str, object]) -> str:
    subdivisions = "x".join(str(value) for value in parameters["subdivisions"])
    active_suffix = (
        f"-tube{parameters['active_tube_radius']}"
        if "active_tube_radius" in parameters
        else ""
    )
    return (
        f"{model_name}-{subdivisions}-s{parameters['handle_slabs']}"
        f"-p{parameters['padding_cells']:g}{active_suffix}.pkl"
    )


def _progress(model_name: str):
    last: dict[str, int] = {}

    def report(stage: str, completed: int, total: int) -> None:
        percentage = int(100 * completed / max(total, 1))
        bucket = percentage // 10
        if last.get(stage) == bucket and completed != total:
            return
        last[stage] = bucket
        print(f"{model_name}: {stage} {percentage}%", flush=True)

    return report


def generate_hybrid_morse_figures(
    output_dir: Path,
    *,
    cache_dir: Path,
    recompute: bool = False,
    quick: bool = False,
) -> tuple[Path, ...]:
    """Generate matching PDF/PNG figures and a computation summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    summaries = []
    for model_name, builder, parameters in (_QUICK_SPECS if quick else _PRODUCTION_SPECS):
        cache_path = cache_dir / _cache_name(model_name, parameters)
        if cache_path.is_file() and not recompute:
            with cache_path.open("rb") as stream:
                result = pickle.load(stream)
        else:
            result = builder(
                **parameters,
                progress_callback=_progress(model_name),
            )
            with cache_path.open("wb") as stream:
                pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        is_walker = result.model_name.startswith("garcia_")
        is_wheel = result.model_name.startswith("rimless_")
        physical_component_count = sum(
            not all(
                node in result.suspension_complex_ingredients.failure_cells
                for node in component
            )
            for component in result.recurrent_sccs
        )
        palette = (
            CMGDB_MORSE_PALETTE
            if physical_component_count > len(SCIENTIFIC_MORSE_PALETTE)
            else SCIENTIFIC_MORSE_PALETTE
        )
        plot = PlotHybridMorseSets(
            result,
            clist=palette,
            show_transient=False,
            show_status_note=False,
            show_legend=False,
            show_panel_titles=False,
            show_reset_attachments=False,
            show_cemetery=False,
            show_grid=False,
            show_morse_graph=True,
            base_view="domain",
            fig_w=10.6 if is_walker else (10.0 if is_wheel else 7.8),
            fig_h=4.8 if is_wheel else 2.9,
            dpi=300,
        )
        all_components = hybrid_morse_components(result, palette=palette)
        summary = result.summary()
        summary["recurrent_components"] = [
            {
                "index": component.index,
                "kind": "cemetery" if component.is_cemetery else "physical",
                "displayed": not component.is_cemetery,
                "cells": len(component.nodes),
                "base_cells": len(component.base_cells),
                "phase_cells": len(component.phase_cells),
            }
            for component in all_components
        ]
        summary["displayed_hasse_edges"] = [
            [int(source), int(target)]
            for source, target in sorted(plot.morse_graph.edges())
        ]
        try:
            outputs.extend(
                save_hybrid_morse_figure(
                    plot,
                    output_dir / _OUTPUT_NAMES[result.model_name],
                    formats=("pdf", "png"),
                    dpi=300,
                ),
            )
        finally:
            plt.close(plot.figure)
        summaries.append(summary)
    summary_path = output_dir / "hybrid-morse-computation-summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    outputs.append(summary_path)
    return tuple(outputs)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate hybrid Morse-set figures for the paper",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "paper" / "figures",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "code" / "data" / "fixed_time_suspension_grid",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    for output in generate_hybrid_morse_figures(
        arguments.output_dir,
        cache_dir=arguments.cache_dir,
        recompute=arguments.recompute,
        quick=arguments.quick,
    ):
        print(output)


if __name__ == "__main__":
    main()
