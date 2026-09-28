"""Generate parameter-study alternatives without replacing paper baselines.

The figures in ``paper/figures/alternatives`` are exploratory views of the
finite corner-sampled, padded suspension relations.  In particular, the
rimless-wheel ``principal`` view displays the analytically identified saddle
and walking-gait components while computing their order through the *full*
relation.  It is not presented as the complete SCC graph.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
from hybrid_dynamics.examples.rimless_wheel_suspension import (
    walking_fixed_point_speed,
)
from hybrid_dynamics.src.grid import Grid
from hybrid_dynamics.src.sampled_suspension import BaseCell


Builder = Callable[..., object]


def _progress(label: str):
    last: dict[str, int] = {}

    def report(stage: str, completed: int, total: int) -> None:
        percentage = int(100 * completed / max(total, 1))
        bucket = percentage // 10
        if last.get(stage) == bucket and completed != total:
            return
        last[stage] = bucket
        print(f"{label}: {stage} {percentage}%", flush=True)

    return report


def _cached_result(
    cache_dir: Path,
    cache_name: str,
    builder: Builder,
    parameters: dict[str, object],
    *,
    recompute: bool,
) -> object:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_name}.pkl"
    if cache_path.is_file() and not recompute:
        with cache_path.open("rb") as stream:
            return pickle.load(stream)
    result = builder(
        **parameters,
        progress_callback=_progress(cache_name),
    )
    with cache_path.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _component_records(result: object) -> list[dict[str, object]]:
    return [
        {
            "index": component.index,
            "kind": "cemetery" if component.is_cemetery else "physical",
            "cells": len(component.nodes),
            "base_cells": len(component.base_cells),
            "phase_cells": len(component.phase_cells),
        }
        for component in hybrid_morse_components(
            result,
            palette=CMGDB_MORSE_PALETTE,
        )
    ]


def _point_component_indices(
    result: object,
    point: Sequence[float],
) -> tuple[int, ...]:
    ingredients = result.suspension_complex_ingredients
    grid = Grid(
        bounds=[list(pair) for pair in ingredients.base_bounds],
        subdivisions=list(ingredients.base_subdivisions),
    )
    incident = {
        BaseCell(int(index))
        for index in grid.find_boxes_containing_point(np.asarray(point, dtype=float))
    }
    return tuple(
        index
        for index, component in enumerate(result.recurrent_sccs)
        if any(cell in component for cell in incident)
    )


def _largest_component(result: object, indices: Iterable[int], role: str) -> int:
    candidates = tuple(int(index) for index in indices)
    if not candidates:
        raise RuntimeError(f"no recurrent component contains the {role} seed")
    return max(candidates, key=lambda index: len(result.recurrent_sccs[index]))


def _save_plot(
    result: object,
    stem: Path,
    *,
    morse_nodes: Iterable[int] | None = None,
    palette: Sequence[str] = SCIENTIFIC_MORSE_PALETTE,
    show_handles: bool = True,
    conley_indices: dict[int, tuple[str, ...]] | None = None,
    fig_w: float = 7.8,
    fig_h: float = 3.0,
) -> tuple[Path, ...]:
    plot = PlotHybridMorseSets(
        result,
        morse_nodes=morse_nodes,
        clist=palette,
        show_transient=False,
        show_status_note=False,
        show_legend=False,
        show_panel_titles=False,
        show_reset_attachments=False,
        show_cemetery=False,
        show_grid=False,
        show_handles=show_handles,
        show_morse_graph=True,
        conley_indices=conley_indices,
        base_view="domain",
        fig_w=fig_w,
        fig_h=fig_h,
        dpi=300,
    )
    try:
        outputs = save_hybrid_morse_figure(
            plot,
            stem,
            formats=("pdf", "png"),
            dpi=300,
        )
        edges = [
            [int(source), int(target)]
            for source, target in sorted(plot.morse_graph.edges())
        ]
    finally:
        plt.close(plot.figure)
    metadata_path = stem.with_suffix(".json")
    metadata = result.summary()
    metadata["recurrent_components"] = _component_records(result)
    metadata["displayed_components"] = [
        component.index for component in plot.components
    ]
    metadata["displayed_hasse_edges"] = edges
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return (*outputs, metadata_path)


def generate_alternatives(
    output_dir: Path,
    *,
    cache_dir: Path,
    models: Iterable[str],
    recompute: bool = False,
) -> tuple[Path, ...]:
    """Generate selected alternative plots and machine-readable provenance."""

    selected = frozenset(models)
    unknown = selected - {"ball", "wheel", "walker"}
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if "ball" in selected:
        parameters: dict[str, object] = {
            "subdivisions": (64, 256),
            "t_star": 0.8,
            "handle_slabs": 16,
            "padding_cells": 1.0,
        }
        result = _cached_result(
            cache_dir,
            "ball-tau080-grid64x256-s16-p1",
            build_bouncing_ball_suspension_grid,
            parameters,
            recompute=recompute,
        )
        physical = tuple(
            component.index
            for component in hybrid_morse_components(result)
            if not component.is_cemetery
        )
        if len(physical) != 1:
            raise RuntimeError(
                "the recorded ball alternative no longer has one physical SCC: "
                f"found {len(physical)}"
            )
        stem = output_dir / "hybrid-morse-bouncing-ball-tau080-grid64x256-s16-p1"
        saved = _save_plot(result, stem, morse_nodes=physical)
        metadata = json.loads(saved[-1].read_text(encoding="utf-8"))
        metadata["selection_status"] = (
            "one physical SCC at two tested spatial resolutions for S=16 and "
            "padding=1; not stable under every tested phase/padding variation"
        )
        saved[-1].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        outputs.extend(saved)

        long_parameters: dict[str, object] = {
            "subdivisions": (64, 256),
            "t_star": 1.5,
            "handle_slabs": 16,
            "padding_cells": 1.0,
        }
        long_result = _cached_result(
            cache_dir,
            "ball-tau150-grid64x256-s16-p1",
            build_bouncing_ball_suspension_grid,
            long_parameters,
            recompute=recompute,
        )
        long_physical = tuple(
            component.index
            for component in hybrid_morse_components(long_result)
            if not component.is_cemetery
        )
        if len(long_physical) != 1:
            raise RuntimeError(
                "the long-clock ball alternative no longer has one physical SCC: "
                f"found {len(long_physical)}"
            )
        long_stem = (
            output_dir / "hybrid-morse-bouncing-ball-tau150-grid64x256-s16-p1"
        )
        long_saved = _save_plot(
            long_result,
            long_stem,
            morse_nodes=long_physical,
        )
        long_metadata = json.loads(long_saved[-1].read_text(encoding="utf-8"))
        long_metadata["selection_status"] = (
            "one physical SCC at 48x192 and 64x256 for S=16 and padding=1"
        )
        long_saved[-1].write_text(
            json.dumps(long_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(long_saved)

        analytic_ball_index = {long_physical[0]: ("x-1", "x-1", "0")}
        analytic_stem = (
            output_dir
            / "hybrid-morse-bouncing-ball-tau150-grid64x256-s16-p1-base-analytic-index"
        )
        analytic_saved = _save_plot(
            long_result,
            analytic_stem,
            morse_nodes=long_physical,
            show_handles=False,
            conley_indices=analytic_ball_index,
            fig_w=6.6,
            fig_h=3.0,
        )
        analytic_metadata = json.loads(
            analytic_saved[-1].read_text(encoding="utf-8")
        )
        analytic_metadata["conley_annotations"] = {
            str(index): list(value) for index, value in analytic_ball_index.items()
        }
        analytic_metadata["conley_annotation_provenance"] = (
            "analytic fixed-time suspension benchmark for the attracting Zeno "
            "orbit; not computed from this finite grid relation"
        )
        analytic_metadata["physical_grid_conley_index_computed"] = False
        analytic_saved[-1].write_text(
            json.dumps(analytic_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(analytic_saved)

    if "wheel" in selected:
        parameters = {
            "subdivisions": (96, 96),
            "t_star": 0.9,
            "handle_slabs": 16,
            "padding_cells": 1.0,
        }
        result = _cached_result(
            cache_dir,
            "wheel-tau090-grid96x96-s16-p1",
            build_rimless_wheel_suspension_grid,
            parameters,
            recompute=recompute,
        )
        full_stem = output_dir / "hybrid-morse-rimless-wheel-tau090-grid96x96-s16-p1-full"
        full_saved = _save_plot(
            result,
            full_stem,
            palette=CMGDB_MORSE_PALETTE,
            fig_w=9.8,
            fig_h=4.4,
        )
        full_metadata = json.loads(full_saved[-1].read_text(encoding="utf-8"))
        full_metadata["selection_status"] = (
            "complete recurrent-SCC view; includes small sampled self-loop "
            "components near the slow saddle/stable-manifold region"
        )
        full_saved[-1].write_text(
            json.dumps(full_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(full_saved)

        alpha = 0.4
        gamma = 0.2
        gait_seed = (
            gamma - alpha,
            walking_fixed_point_speed(alpha=alpha, gamma=gamma),
        )
        saddle_seed = (0.0, 0.0)
        gait_index = _largest_component(
            result,
            _point_component_indices(result, gait_seed),
            "walking-gait",
        )
        saddle_index = _largest_component(
            result,
            _point_component_indices(result, saddle_seed),
            "saddle",
        )
        if gait_index == saddle_index:
            raise RuntimeError("the sampled relation does not separate gait and saddle")
        principal_stem = (
            output_dir
            / "hybrid-morse-rimless-wheel-tau090-grid96x96-s16-p1-principal"
        )
        principal_saved = _save_plot(
            result,
            principal_stem,
            morse_nodes=(gait_index, saddle_index),
            fig_w=7.8,
            fig_h=3.0,
        )
        principal_metadata = json.loads(
            principal_saved[-1].read_text(encoding="utf-8")
        )
        principal_metadata["principal_selection"] = {
            "gait_component": gait_index,
            "gait_seed": list(gait_seed),
            "saddle_component": saddle_index,
            "saddle_seed": list(saddle_seed),
            "status": (
                "display-only principal-component view; omitted recurrent SCCs "
                "remain in the full relation, and displayed reachability is "
                "computed through that full relation"
            ),
        }
        principal_saved[-1].write_text(
            json.dumps(principal_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(principal_saved)

        long_parameters = {
            "subdivisions": (96, 96),
            "t_star": 2.0,
            "handle_slabs": 16,
            "padding_cells": 1.0,
        }
        long_result = _cached_result(
            cache_dir,
            "wheel-tau200-grid96x96-s16-p1",
            build_rimless_wheel_suspension_grid,
            long_parameters,
            recompute=recompute,
        )
        long_alpha = 0.4
        long_gamma = 0.2
        long_gait_seed = (
            long_gamma - long_alpha,
            walking_fixed_point_speed(alpha=long_alpha, gamma=long_gamma),
        )
        long_saddle_seed = (0.0, 0.0)
        long_gait_index = _largest_component(
            long_result,
            _point_component_indices(long_result, long_gait_seed),
            "long-clock walking-gait",
        )
        long_saddle_index = _largest_component(
            long_result,
            _point_component_indices(long_result, long_saddle_seed),
            "long-clock saddle",
        )
        long_physical = tuple(
            component.index
            for component in hybrid_morse_components(long_result)
            if not component.is_cemetery
        )
        if set(long_physical) != {long_gait_index, long_saddle_index}:
            raise RuntimeError(
                "the long-clock wheel alternative does not have exactly the "
                "analytic gait and saddle as its physical recurrent SCCs"
            )
        long_stem = (
            output_dir / "hybrid-morse-rimless-wheel-tau200-grid96x96-s16-p1"
        )
        long_saved = _save_plot(
            long_result,
            long_stem,
            morse_nodes=long_physical,
            fig_w=7.8,
            fig_h=3.0,
        )
        long_metadata = json.loads(long_saved[-1].read_text(encoding="utf-8"))
        if long_metadata["displayed_hasse_edges"] != [
            [long_saddle_index, long_gait_index]
        ]:
            raise RuntimeError(
                "the long-clock wheel alternative does not have saddle-to-gait order"
            )
        long_metadata["analytic_components"] = {
            "gait_component": long_gait_index,
            "gait_seed": list(long_gait_seed),
            "saddle_component": long_saddle_index,
            "saddle_seed": list(long_saddle_seed),
        }
        long_metadata["selection_status"] = (
            "complete physical recurrent-SCC graph; the same two components and "
            "order occur at 64x64 and 96x96 for S=16 and padding=1"
        )
        long_saved[-1].write_text(
            json.dumps(long_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(long_saved)

        analytic_wheel_indices = {
            long_gait_index: ("x-1", "x-1", "0"),
            long_saddle_index: ("0", "x-1", "0"),
        }
        analytic_stem = (
            output_dir
            / "hybrid-morse-rimless-wheel-tau200-grid96x96-s16-p1-base-analytic-index"
        )
        analytic_saved = _save_plot(
            long_result,
            analytic_stem,
            morse_nodes=long_physical,
            show_handles=False,
            conley_indices=analytic_wheel_indices,
            fig_w=6.6,
            fig_h=3.0,
        )
        analytic_metadata = json.loads(
            analytic_saved[-1].read_text(encoding="utf-8")
        )
        analytic_metadata["conley_annotations"] = {
            str(index): list(value)
            for index, value in analytic_wheel_indices.items()
        }
        analytic_metadata["conley_annotation_provenance"] = (
            "analytic fixed-time suspension benchmarks for the attracting gait "
            "and one-unstable-direction saddle; not computed from this finite "
            "grid relation"
        )
        analytic_metadata["physical_grid_conley_index_computed"] = False
        analytic_saved[-1].write_text(
            json.dumps(analytic_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(analytic_saved)

    if "walker" in selected:
        parameters = {
            "subdivisions": (20, 20, 20, 20),
            "t_star": 0.5,
            "handle_slabs": 16,
            "padding_cells": 1.0,
            "active_tube_radius": 1,
        }
        result = _cached_result(
            cache_dir,
            "walker-tau050-grid20x20x20x20-s16-p1-tube1",
            build_garcia_passive_walker_suspension_grid,
            parameters,
            recompute=recompute,
        )
        stem = (
            output_dir
            / "hybrid-morse-passive-walker-tau050-grid20x20x20x20-s16-p1-tube1"
        )
        saved = _save_plot(result, stem, fig_w=10.6, fig_h=3.0)
        metadata = json.loads(saved[-1].read_text(encoding="utf-8"))
        metadata["selection_status"] = (
            "sharper orbit-seeded local active-tube visualization; refinement "
            "improves rasterization but still produces one dominant physical SCC"
        )
        saved[-1].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        outputs.extend(saved)

        base_stem = (
            output_dir
            / "hybrid-morse-passive-walker-tau050-grid20x20x20x20-s16-p1-tube1-base"
        )
        base_saved = _save_plot(
            result,
            base_stem,
            show_handles=False,
            fig_w=8.8,
            fig_h=3.0,
        )
        base_metadata = json.loads(base_saved[-1].read_text(encoding="utf-8"))
        base_metadata["physical_grid_conley_index_computed"] = False
        base_metadata["conley_annotation_provenance"] = (
            "none; the physical-grid passive-walker index remains pending"
        )
        base_saved[-1].write_text(
            json.dumps(base_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.extend(base_saved)

    return tuple(outputs)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate separate hybrid Morse parameter-study figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "paper" / "figures" / "alternatives",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            project_root / "code" / "data" / "fixed_time_suspension_alternatives"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ball", "wheel", "walker"),
        default=("ball", "wheel", "walker"),
    )
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    for output in generate_alternatives(
        arguments.output_dir,
        cache_dir=arguments.cache_dir,
        models=arguments.models,
        recompute=arguments.recompute,
    ):
        print(output)


if __name__ == "__main__":
    main()
