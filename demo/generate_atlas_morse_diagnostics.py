#!/usr/bin/env python3
"""Generate chart-aware diagnostics from accepted native CMGDB Atlas runs.

The first run stores the exact output of ``morse_set_chart_boxes`` in a small
JSON cache. Later figure builds read that cache and do not reevaluate the
hybrid dynamics. The script verifies node, edge, and chart-box counts against
the independently recorded acceptance report before writing a figure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hybrid_dynamics import (
    AtlasMorseBox,
    AtlasMorsePlotData,
    PlotHybridMorseSets,
    extract_atlas_morse_plot_data,
    load_atlas_finite_relation_index_annotations,
    load_atlas_morse_plot_data,
    save_atlas_morse_plot_data,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.examples.bouncing_ball_atlas import (
    build_bouncing_ball_atlas_model,
)
from hybrid_dynamics.examples.rimless_wheel_atlas import (
    build_rimless_wheel_atlas_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
PAPER_OUTPUT = PROJECT_ROOT / "paper" / "figures" / "atlas-diagnostics"


@dataclass(frozen=True)
class DiagnosticConfiguration:
    name: str
    depth: int
    tau: float
    acceptance_report: Path
    finite_relation_audit: Path
    cache_path: Path
    output_stem: Path
    base_labels: tuple[str, ...]
    handle_labels: tuple[str, ...]
    base_view: str
    setup_builder: Callable[..., object]


CONFIGURATIONS = {
    "rimless-wheel": DiagnosticConfiguration(
        name="rimless-wheel",
        depth=12,
        tau=2.0,
        acceptance_report=(
            CODE_ROOT
            / "data"
            / "rimless_wheel_atlas"
            / "acceptance_tau200_depth10_depth12.json"
        ),
        finite_relation_audit=(
            CODE_ROOT / "data" / "physical_conley" / "physical_conley_audit_wheel.json"
        ),
        cache_path=(
            CODE_ROOT
            / "data"
            / "rimless_wheel_atlas"
            / "morse_plot_tau200_depth12.json"
        ),
        output_stem=PAPER_OUTPUT / "hybrid-morse-rimless-wheel-atlas-tau200-depth12",
        base_labels=(r"$\theta$", r"$\dot\theta$"),
        handle_labels=(r"$\dot\theta_G$", r"$s$"),
        base_view="support",
        setup_builder=build_rimless_wheel_atlas_model,
    ),
    "bouncing-ball": DiagnosticConfiguration(
        name="bouncing-ball",
        depth=10,
        tau=1.5,
        acceptance_report=(
            CODE_ROOT
            / "data"
            / "bouncing_ball_atlas"
            / "acceptance_tau150_tau200_depth8_depth10.json"
        ),
        finite_relation_audit=(
            CODE_ROOT / "data" / "physical_conley" / "physical_conley_audit_ball.json"
        ),
        cache_path=(
            CODE_ROOT
            / "data"
            / "bouncing_ball_atlas"
            / "morse_plot_tau150_depth10.json"
        ),
        output_stem=PAPER_OUTPUT / "hybrid-morse-bouncing-ball-atlas-tau150-depth10",
        base_labels=(r"$h$", r"$v$"),
        handle_labels=(r"$v_G$", r"$s$"),
        base_view="domain",
        setup_builder=build_bouncing_ball_atlas_model,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="*",
        choices=tuple(CONFIGURATIONS),
        help="accepted model(s); the default generates both",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="reevaluate the CMGDB Atlas box map even when a plot cache exists",
    )
    parser.add_argument(
        "--include-handles",
        action="store_true",
        help="add the intrinsic guard-coordinate/phase chart panel",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "png"),
        choices=("pdf", "png", "svg"),
    )
    return parser.parse_args()


def _accepted_run(configuration: DiagnosticConfiguration) -> dict[str, object]:
    payload = json.loads(configuration.acceptance_report.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        depth = int(run.get("atlas_depth", run.get("depth", -1)))
        tau = float(
            run.get("t_star", payload["shared_configuration"].get("t_star", -1))
        )
        if depth == configuration.depth and tau == configuration.tau:
            return run
    raise RuntimeError(
        f"accepted run tau={configuration.tau}, depth={configuration.depth} "
        f"is absent from {configuration.acceptance_report}"
    )


def _expected_signature(
    configuration: DiagnosticConfiguration,
    run: dict[str, object],
) -> tuple[int, tuple[tuple[int, int], ...], dict[int, dict[str, int]]]:
    if configuration.name == "rimless-wheel":
        graph = run["morse_graph"]
        return (
            int(graph["nodes"]),
            tuple(tuple(int(value) for value in edge) for edge in graph["edges"]),
            {
                int(node): {key: int(value) for key, value in counts.items()}
                for node, counts in graph["morse_set_cell_counts"].items()
            },
        )
    return (
        int(run["morse_nodes"]),
        tuple(tuple(int(value) for value in edge) for edge in run["morse_edges"]),
        {
            int(node): {key: int(value) for key, value in counts.items()}
            for node, counts in run["morse_set_cell_counts"].items()
        },
    )


def _validate_against_acceptance(
    configuration: DiagnosticConfiguration,
    data: AtlasMorsePlotData,
    run: dict[str, object],
) -> None:
    expected_nodes, expected_edges, expected_counts = _expected_signature(
        configuration,
        run,
    )
    if len(data.nodes) != expected_nodes:
        raise RuntimeError(
            f"fresh {configuration.name} graph has {len(data.nodes)} Morse nodes; "
            f"accepted report has {expected_nodes}"
        )
    if set(data.edges) != set(expected_edges):
        raise RuntimeError(
            f"fresh {configuration.name} Morse edges {data.edges!r} differ from "
            f"accepted edges {expected_edges!r}"
        )
    for node in data.nodes:
        actual = {
            "base": sum(box.chart_id == data.base_chart_id for box in node.boxes),
            "handle": sum(box.chart_id == data.handle_chart_id for box in node.boxes),
            "total": len(node.boxes),
        }
        if actual != expected_counts.get(node.index):
            raise RuntimeError(
                f"fresh {configuration.name} M({node.index}) chart-box counts "
                f"{actual!r} differ from accepted {expected_counts.get(node.index)!r}"
            )


def _boxes_intersect(
    first: AtlasMorseBox,
    second: AtlasMorseBox,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return all(
        left_lower <= right_upper + tolerance
        and right_lower <= left_upper + tolerance
        for left_lower, left_upper, right_lower, right_upper in zip(
            first.lower,
            first.upper,
            second.lower,
            second.upper,
        )
    )


def _closed_box_component_sizes(boxes: Sequence[AtlasMorseBox]) -> tuple[int, ...]:
    remaining = set(boxes)
    sizes = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            adjacent = {
                candidate
                for candidate in remaining
                if _boxes_intersect(current, candidate)
            }
            remaining.difference_update(adjacent)
            component.update(adjacent)
            stack.extend(adjacent)
        sizes.append(len(component))
    return tuple(sorted(sizes, reverse=True))


def _validate_plotted_base_support(
    configuration: DiagnosticConfiguration,
    data: AtlasMorsePlotData,
) -> dict[str, list[int]]:
    component_sizes: dict[str, list[int]] = {}
    for node in data.nodes:
        boxes = tuple(
            box for box in node.boxes if box.chart_id == data.base_chart_id
        )
        sizes = _closed_box_component_sizes(boxes)
        component_sizes[str(node.index)] = list(sizes)
    if configuration.name == "rimless-wheel":
        # The accepted report identifies M(0) as the gait set. Its actual
        # closed base boxes, not merely their plotted 2-D centers, must form
        # one connected union before the figure is written.
        if len(component_sizes.get("0", [])) != 1:
            raise RuntimeError(
                "refusing to plot: rimless-wheel gait base support is disconnected "
                f"with component sizes {component_sizes.get('0')}"
            )
    return component_sizes


def _compute_cache(
    configuration: DiagnosticConfiguration,
    run: dict[str, object],
) -> AtlasMorsePlotData:
    import CMGDB

    setup = configuration.setup_builder(
        depth=configuration.depth,
        t_star=configuration.tau,
        samples_per_axis=3,
        padding_cells=1.0,
    )
    morse_graph, _map_graph = CMGDB.ComputeMorseGraph(setup.model)
    data = extract_atlas_morse_plot_data(
        morse_graph,
        setup.charts,
        metadata={
            "model": configuration.name,
            "depth": configuration.depth,
            "t_star": configuration.tau,
            "relation_scope": "nonempty local-window sampled Atlas relation",
            "source_samples_per_axis": 3,
            "padding_cells": 1.0,
            "whole_cell_outer_enclosure_certified": False,
            "global_attractor_lattice_interpretation": False,
            "finite_relation_index_annotations_stored_separately": True,
            "continuous_system_conley_index_certified": False,
            "acceptance_report": str(
                configuration.acceptance_report.relative_to(PROJECT_ROOT)
            ),
        },
    )
    _validate_against_acceptance(configuration, data, run)
    component_sizes = _validate_plotted_base_support(configuration, data)
    data = AtlasMorsePlotData(
        base_chart_id=data.base_chart_id,
        handle_chart_id=data.handle_chart_id,
        base_bounds=data.base_bounds,
        handle_bounds=data.handle_bounds,
        nodes=data.nodes,
        edges=data.edges,
        metadata={**data.metadata, "base_box_component_sizes": component_sizes},
    )
    save_atlas_morse_plot_data(data, configuration.cache_path)
    return data


def _load_or_compute(
    configuration: DiagnosticConfiguration,
    *,
    recompute: bool,
) -> tuple[AtlasMorsePlotData, dict[str, object]]:
    run = _accepted_run(configuration)
    if configuration.cache_path.is_file() and not recompute:
        data = load_atlas_morse_plot_data(configuration.cache_path)
        _validate_against_acceptance(configuration, data, run)
        _validate_plotted_base_support(configuration, data)
        return data, run
    return _compute_cache(configuration, run), run


def _write_manifest(records: list[dict[str, object]]) -> None:
    PAPER_OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "Atlas diagnostics included in the manuscript with explicit scope",
        "method": (
            "Each colored rectangle is read from the native CMGDB Atlas "
            "MorseGraph.morse_set_chart_boxes output. The adjacent directed "
            "graph is the CMGDB Morse order. Node tuples are loaded only from "
            "the persisted physical finite-relation nerve/carrier audits."
        ),
        "limitations": {
            "relation_scope": "nonempty local-window sampled Atlas relation",
            "whole_cell_outer_enclosure_certified": False,
            "global_attractor_lattice_interpretation": False,
            "finite_relation_conley_index_computed": True,
            "finite_relation_coefficient_field": "GF(5)",
            "continuous_system_conley_index_certified": False,
            "analytic_or_hci_label_fallback_used": False,
            "note": (
                "The displayed tuples are shift classes of the finite reset-"
                "quotient relation, computed on an actual-Atlas-box nerve with "
                "a validated acyclic carrier. They are not certified indices of "
                "the continuous fixed-time suspension map because a whole-cell "
                "outer enclosure has not been proved. Empty source images are "
                "retained as exits from the chart window."
            ),
        },
        "figures": records,
    }
    (PAPER_OUTPUT / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    requested = args.model or list(CONFIGURATIONS)
    records = []
    for name in requested:
        configuration = CONFIGURATIONS[name]
        data, run = _load_or_compute(configuration, recompute=args.recompute)
        finite_annotations = load_atlas_finite_relation_index_annotations(
            configuration.finite_relation_audit,
            data,
        )
        plot = PlotHybridMorseSets(
            data,
            finite_relation_annotations=finite_annotations,
            axis_labels=configuration.base_labels,
            handle_axis_labels=configuration.handle_labels,
            show_handles=args.include_handles,
            show_morse_graph=True,
            show_legend=False,
            show_panel_titles=False,
            show_component_sizes=False,
            show_status_note=False,
            base_view=configuration.base_view,
            fig_h=3.4,
        )
        try:
            outputs = save_hybrid_morse_figure(
                plot,
                configuration.output_stem,
                formats=args.formats,
                dpi=400,
            )
        finally:
            plt.close(plot.figure)
        records.append(
            {
                "model": name,
                "depth": configuration.depth,
                "t_star": configuration.tau,
                "plot_cache": str(configuration.cache_path.relative_to(PROJECT_ROOT)),
                "acceptance_report": str(
                    configuration.acceptance_report.relative_to(PROJECT_ROOT)
                ),
                "finite_relation_audit": str(
                    configuration.finite_relation_audit.relative_to(PROJECT_ROOT)
                ),
                "outputs": [str(path.relative_to(PROJECT_ROOT)) for path in outputs],
                "morse_nodes": list(data.vertex_ids),
                "morse_edges": [list(edge) for edge in data.edges],
                "morse_set_cell_counts": {
                    str(node.index): len(node.boxes) for node in data.nodes
                },
                "base_box_component_sizes": data.metadata.get(
                    "base_box_component_sizes"
                ),
                "accepted_empty_source_images": int(
                    run["empty_images"]
                    if name == "bouncing-ball"
                    else run["acceptance_gates"]["empty_source_images"]
                ),
                "finite_relation_conley_index": {
                    "computed": True,
                    "coefficient_field": 5,
                    "result_scope": finite_annotations.result_scope,
                    "shift_classes": {
                        str(node): list(entries)
                        for node, entries in finite_annotations.shift_classes.items()
                    },
                },
                "continuous_system_conley_index_certified": False,
            }
        )
    _write_manifest(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
