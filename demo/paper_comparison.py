#!/usr/bin/env python3
"""Render the coarse ball and matched wheel panels from recorded Morse atoms.

No dynamics or indices are recomputed and the input records remain unchanged.
The ball has an explicit zoom; the wheel comparison uses common base and saddle
zoom limits. Individual panels have no heading apart from the zoom letter A.
Use ``--titles`` to add parameter headings to the combined notebook figures.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from demo.replot_paper import rebuild_problem_and_grid, recorded_morse_sets
from hybrid_dynamics.examples.paper_figures import (
    _panel_dpi, _paper_plot_options, conley_index_labels,
    paper_index_labels, paper_morse_colors,
)
from hybrid_dynamics.src.atlas_morse_plot import (
    AtlasDetailZoom, AtlasFiniteRelationIndexAnnotations,
    PANEL_ZOOM_MARGINS, PANEL_ZOOM_SIZE, ZOOM_FIGURE_STYLE,
    _chart_boxes, _detail_zooms, _draw_chart_projection, _draw_zoom_panel,
    _figure_with_axes, _mark_zoom_window, _projected_bounds, _small_set_nodes,
    plot_atlas_hybrid_morse_panels,
)
from hybrid_dynamics.src.hybrid_morse_plot import (
    _draw_morse_graph, save_hybrid_morse_figure,
)
from hybrid_dynamics.src.suspension_grid_plot import (
    morse_figure_selection, suspension_grid_morse_sets_plot_data,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _union_limits(windows):
    return tuple((min(w[k][0] for w in windows), max(w[k][1] for w in windows))
                 for k in (0, 1))


def _prepare(record: Path, example: str, dpi: int):
    path = Path(record).resolve()
    digest = _digest(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["example"] != example:
        raise ValueError(f"Expected {example}: {path}")
    problem, grid = rebuild_problem_and_grid(summary, path)
    morse_sets = recorded_morse_sets(summary, path, grid)
    edges, conley = summary["morse_graph"]["edges"], summary["conley"]
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, edges)
    del problem, grid, morse_sets
    gc.collect()
    colors = paper_morse_colors(len(data.nodes), conley)
    labels, blocked, _ = paper_index_labels(conley)
    labels, _ = conley_index_labels(labels)
    fields = {int(entry["coefficient_field"]) for entry in conley}
    if len(fields) != 1:
        raise ValueError(f"Expected one coefficient field: {path}")
    variants = {}
    for variant in ("nontrivial", "all"):
        selection = morse_figure_selection(len(data.nodes), edges, conley, variant)
        annotations = AtlasFiniteRelationIndexAnnotations(
            shift_classes={node: labels[node] for node in selection.shown if node in labels},
            coefficient_field=next(iter(fields)),
            result_scope="finite_reset_quotient_relation", audit_path=path,
        )
        options = _paper_plot_options(
            data, example=example, shown=selection.shown,
            blocked={node: blocked[node] for node in selection.blocked},
            annotations=annotations, colors=colors,
        )
        options.update(detail_zooms=False, show_handles=False)
        panels = plot_atlas_hybrid_morse_panels(data, **options, dpi=dpi)
        if tuple(sorted(panels.morse_graph.edges)) != selection.order:
            raise AssertionError("Drawn Morse order differs from the record")
        variants[variant] = dict(panels=panels, selection=selection, annotations=annotations)
    panels = variants["nontrivial"]["panels"]
    limits = (panels.axes["base"].get_xlim(), panels.axes["base"].get_ylim())
    if example == "rimless-wheel":
        targets = tuple(int(entry["morse_node"]) for entry in conley
                        if entry["computed"] and entry.get("shift_class", [])[:3]
                        == ["0", "x-1", "0"])
        if len(targets) != 1:
            raise ValueError(f"Expected one saddle for the wheel zoom; found {targets}")
        candidates = _detail_zooms(
            panels.components, chart_kind="base", projection=(0, 1),
            x_limits=limits[0], y_limits=limits[1],
        )
        windows = [(x, y) for nodes, x, y in candidates if set(nodes).intersection(targets)]
    else:
        targets = tuple(variants["nontrivial"]["selection"].shown)
        windows = []
    if not windows:
        # Coarse cells may be visible without an automatic detail panel.
        bounds = np.concatenate([
            _projected_bounds(_chart_boxes(component, "base"), (0, 1))
            for component in panels.components if component.index in targets
        ])
        lo, hi = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
        pad = np.maximum(0.3 * (hi - lo), 3 * np.array(summary["grid"]["base_cell_widths"]))
        windows = [tuple((float(lo[k] - pad[k]), float(hi[k] + pad[k])) for k in (0, 1))]
    return dict(path=path, digest=digest, summary=summary, data=data, variants=variants,
                colors=colors, labels=labels, blocked=blocked, limits=limits,
                windows=windows, targets=targets)


def _base(axis, run, limits, zoom, small):
    _draw_chart_projection(
        axis, run["variants"]["nontrivial"]["panels"].components,
        chart_kind="base", projection=(0, 1), limits=limits,
        labels=_axis_labels(run), show_grid=False, show_panel_title=False,
        spines_below_cells=True, outlined=small,
    )
    _mark_zoom_window(axis, zoom)


def _axis_labels(run):
    return ((r"$h$", r"$v$") if run["summary"]["example"] == "bouncing-ball"
            else (r"$\theta$", r"$\dot\theta$"))


def _graph(axis, run, variant="nontrivial", font_size=12):
    item = run["variants"][variant]
    _draw_morse_graph(
        axis, item["panels"].morse_graph, item["panels"].components,
        conley_indices=item["annotations"].shift_classes, show_component_sizes=False,
        show_title=False, blocked_index_nodes=run["blocked"], font_size=font_size,
    )


def _save(figure, stem: Path, dpi: int):
    return [str(path.resolve()) for path in save_hybrid_morse_figure(
        figure, stem, formats=("pdf", "png"), dpi=_panel_dpi(figure, dpi),
    )]


def _draw_run(run, directory, limits, zoom_limits, axes, dpi):
    directory.mkdir(parents=True, exist_ok=True)
    panels = run["variants"]["nontrivial"]["panels"]
    zoom = AtlasDetailZoom("A", "base", (0, 1), *zoom_limits, run["targets"])
    small = _small_set_nodes(panels.components, chart_kind="base", projection=(0, 1),
                             x_limits=limits[0], y_limits=limits[1])
    panels.axes["base"].clear()
    _base(panels.axes["base"], run, limits, zoom, small)
    zoom_figure, zoom_axis = _figure_with_axes(PANEL_ZOOM_SIZE, PANEL_ZOOM_MARGINS, dpi)
    try:
        _draw_zoom_panel(zoom_axis, zoom, panels.components, outlined=small,
                         style=ZOOM_FIGURE_STYLE, labels=_axis_labels(run))
        selected = {
            "nontrivial-base": panels.figures["base"],
            "nontrivial-zoom-A": zoom_figure,
            "nontrivial-graph": panels.figures["graph"],
            "graph": run["variants"]["all"]["panels"].figures["graph"],
        }
        files = {name: _save(figure, directory / f"{run['path'].stem}-{name}", dpi)
                 for name, figure in selected.items()}
        _base(axes[0], run, limits, zoom, small)
        _draw_zoom_panel(axes[1], zoom, panels.components, outlined=small,
                         style=ZOOM_FIGURE_STYLE, labels=_axis_labels(run))
        _graph(axes[2], run)
        if (panels.axes["base"].get_xlim(), panels.axes["base"].get_ylim()) != limits:
            raise AssertionError("Base axes differ from the requested limits")
        if (zoom_axis.get_xlim(), zoom_axis.get_ylim()) != zoom_limits:
            raise AssertionError("Zoom axes differ from the requested limits")
    finally:
        plt.close(zoom_figure)
    return dict(
        record=str(run["path"]), sha256=run["digest"],
        **{key: run["summary"][key] for key in ("base_cells_per_axis", "phase_cells", "tau", "level")},
        level_offset=run["summary"].get("level_offset", 0),
        original_base_limits=run["limits"], original_zoom_windows=run["windows"],
        base_limits=limits, zoom_A=zoom.to_dict(), colors=run["colors"],
        selections={name: item["selection"].to_dict() for name, item in run["variants"].items()},
        panel_files=files,
    )


def _finish(manifest, runs, directory, name):
    for run in runs:
        if _digest(run["path"]) != run["digest"]:
            raise AssertionError(f"Input record changed: {run['path']}")
    manifest.update(script="demo/paper_comparison.py", records_modified=False,
                    dynamics_recomputed=False,
                    zoom_rule="Union of automatic zooms, with fallback to the target cell bounds padded by max(30 percent of span, three cells)")
    path = directory / name
    manifest["manifest_path"] = str(path)

    def relative(value):
        if isinstance(value, dict):
            return {key: relative(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [relative(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute():
            return os.path.relpath(value, directory)
        return value

    path.write_text(json.dumps(relative(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _close(runs):
    for run in runs:
        for item in run["variants"].values():
            item["panels"].close()


def _render_single(record, output_dir, example, dpi, titles):
    directory = Path(output_dir).resolve()
    run = _prepare(record, example, dpi)
    figure, axes = plt.subplots(1, 3, figsize=(10.3, 3.8),
                               gridspec_kw={"width_ratios": [1.15, 1.0, 0.8]})
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.17,
                           top=0.78 if titles else 0.90, wspace=0.55)
    try:
        manifest = _draw_run(run, directory, run["limits"], _union_limits(run["windows"]), axes, dpi)
        if titles:
            figure.suptitle(f"{example.replace('-', ' ').capitalize()}: "
                            f"{manifest['base_cells_per_axis']} × {manifest['base_cells_per_axis']} base · "
                            f"{manifest['phase_cells']} phase cells · τ = {manifest['tau']}", fontsize=14)
        manifest["comparison_files"] = _save(figure, directory / f"{example}-panels", dpi)
        return _finish(manifest, [run], directory, f"{example}-panels.json")
    finally:
        plt.close(figure)
        _close([run])


def render_ball(record: Path, output_dir: Path, *, dpi: int = 300, titles: bool = False) -> dict:
    """Draw the recorded ball with an explicit zoom; return absolute output paths."""
    return _render_single(record, output_dir, "bouncing-ball", dpi, titles)


def render_wheel(record: Path, output_dir: Path, *, dpi: int = 300, titles: bool = False) -> dict:
    """Draw one recorded wheel, including its saddle zoom and both Morse graphs."""
    return _render_single(record, output_dir, "rimless-wheel", dpi, titles)


def render_wheel_comparison(coarse_record: Path, fine_record: Path, output_dir: Path,
                            *, dpi: int = 300, titles: bool = False) -> dict:
    """Draw wheel records at common limits; return panels, composites, and provenance.

    ``comparison_files`` and ``full_graph_comparison_files`` list PDF then PNG;
    ``records`` contains coarse and fine entries, each with ``panel_files``.
    The saved manifest uses paths relative to its directory for portability.
    """
    directory = Path(output_dir).resolve()
    runs = []
    figures = []
    try:
        for path in (coarse_record, fine_record):
            runs.append(_prepare(path, "rimless-wheel", dpi))
        if runs[0]["data"].base_bounds != runs[1]["data"].base_bounds:
            raise ValueError("The recorded physical windows differ")
        limits = _union_limits([run["limits"] for run in runs])
        zoom_limits = _union_limits([window for run in runs for window in run["windows"]])
        comparison, axes = plt.subplots(3, 2, figsize=(9.3, 10),
                                        gridspec_kw={"height_ratios": [1.2, 0.85, 0.85]})
        figures.append(comparison)
        comparison.subplots_adjust(left=0.095, right=0.98, bottom=0.04,
                                   top=0.93 if titles else 0.98, hspace=0.40, wspace=0.34)
        graphs, graph_axes = plt.subplots(1, 2, figsize=(9.0, 11.5))
        figures.append(graphs)
        graphs.subplots_adjust(left=0.025, right=0.975, bottom=0.015,
                               top=0.92 if titles else 0.985, wspace=0.12)
        manifest = dict(common_base_limits=limits, common_zoom_A_limits=zoom_limits, records=[])
        graph_limits = []
        for column, (role, run) in enumerate(zip(("coarse", "fine"), runs)):
            item = _draw_run(run, directory / role, limits, zoom_limits, axes[:, column], dpi)
            item["role"] = role
            manifest["records"].append(item)
            _graph(graph_axes[column], run, "all", font_size=9)
            graph_limits.append((graph_axes[column].get_xlim(), graph_axes[column].get_ylim()))
            if titles:
                title = f"{role.capitalize()}: {item['base_cells_per_axis']} × {item['base_cells_per_axis']}"
                axes[0, column].set_title(f"{title} base\n{item['phase_cells']} phase cells", fontsize=13, pad=12)
                graph_axes[column].set_title(f"{title}\n{len(run['data'].nodes)} Morse sets", fontsize=14, pad=18)
        span_x = max(x[1] - x[0] for x, y in graph_limits)
        span_y = max(y[1] - y[0] for x, y in graph_limits)
        for axis, (x, y) in zip(graph_axes, graph_limits):
            center = sum(x) / 2
            axis.set_xlim(center - span_x / 2, center + span_x / 2)
            axis.set_ylim(y[0], y[0] + span_y)
        manifest["comparison_files"] = _save(comparison, directory / "wheel-coarse-fine-comparison", dpi)
        manifest["full_graph_comparison_files"] = _save(graphs, directory / "wheel-full-graphs-comparison", dpi)
        return _finish(manifest, runs, directory, "matching-axes.json")
    finally:
        for figure in figures:
            plt.close(figure)
        _close(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("bouncing-ball", "rimless-wheel"))
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--titles", action="store_true")
    args = parser.parse_args()
    options = dict(dpi=args.dpi, titles=args.titles)
    if args.example == "bouncing-ball" and len(args.records) == 1:
        result = render_ball(args.records[0], args.output_dir, **options)
    elif args.example == "rimless-wheel" and len(args.records) == 1:
        result = render_wheel(args.records[0], args.output_dir, **options)
    elif args.example == "rimless-wheel" and len(args.records) == 2:
        result = render_wheel_comparison(*args.records, args.output_dir, **options)
    else:
        parser.error("Use one ball record, or one wheel record or a coarse/fine pair")
    print(json.dumps({key: result[key] for key in ("manifest_path", "comparison_files")}, indent=2))


if __name__ == "__main__":
    main()
