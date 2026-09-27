"""Figures of the paper-grid runs, shared by the runner and the replot script.

:func:`write_paper_grid_figures` draws the Morse sets and the Morse graph of a
run on ``Xi_n`` in the figure variants of :mod:`suspension_grid_plot`:
``all`` (every Morse node) and ``nontrivial`` (Morse nodes with a computed
trivial finite-relation index hidden, nodes without a label kept and marked
with the dimensions of the relative homology of their pair).  The
``all`` figure is written to ``<stem>.pdf``/``.png`` and the ``nontrivial``
figure to ``<stem>-nontrivial.pdf``/``.png``.

Each figure has the base chart (the base readout ``d_n^{-1}(M)`` of each Morse
set) and the Morse graph, in the same colors.  The handle chart (handle pieces,
guard coordinate against the phase ``s`` in ``[0, 1]``) is added only when a
shown Morse set has no base cell, since the base chart would not show it.  A Morse set too
small to see in a chart panel gets a zoom panel (``A``, ``B``, ...) next to
it, unless it is too spread for a zoom that magnifies at least about twice;
the window is outlined and labeled in the panel, and the zoom shows the
other Morse sets in the window faded.  Every cell is drawn at its true extent,
with no symbol; the cells of a set too small to see in a chart panel are also
outlined by a thin line in the color of the set, in the panel and in its
zooms, so cells smaller than a point are still seen.

Each panel is also drawn as its own figure (:func:`draw_paper_grid_panels`)
and written next to the figure of its variant, as ``<variant stem>-base``,
``<variant stem>-zoom-A``, ... (one per zoom), ``<variant stem>-graph``, and
``<variant stem>-handle`` when the handle chart is drawn, each as PDF and PNG,
where ``<variant stem>`` is ``<stem>`` or ``<stem>-nontrivial``.

The colors are those of :func:`paper_grid_morse_colors`: a Morse set whose
index is computed and trivial is gray, and Morse set ``M(i)`` otherwise takes
color ``i`` of CMGDB's default palette, so a set has the same color in both
variants and in every panel of a run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.figure import Figure

from ..src.atlas_morse_plot import (
    AtlasFiniteRelationIndexAnnotations,
    AtlasHybridMorsePlot,
    AtlasMorsePanelFigures,
    AtlasMorsePlotData,
    plot_atlas_hybrid_morse_panels,
    plot_atlas_hybrid_morse_sets,
)
from ..src.hybrid_morse_plot import CMGDB_MORSE_PALETTE, save_hybrid_morse_figure
from ..src.suspension_grid import SuspensionGrid
from ..src.suspension_grid_plot import (
    FIGURE_VARIANTS,
    index_status,
    morse_figure_selection,
    suspension_grid_morse_sets_plot_data,
)
from .paper_grid_examples import paper_grid_example


#: Name of the palette of the Morse sets, as recorded in the JSON summary.
PAPER_GRID_PALETTE_NAME = "CMGDB default"

#: The default Morse-set palette of CMGDB's plotting functions.
PAPER_GRID_PALETTE: tuple[str, ...] = CMGDB_MORSE_PALETTE

#: The grays of :data:`PAPER_GRID_PALETTE`.  A Morse set with a nontrivial
#: label, or without a label, skips them, so gray always means trivial index.
PALETTE_GRAYS = frozenset({"#7f7f7f", "#636363", "#c7c7c7"})

#: Color of a Morse set whose finite-relation index is computed and trivial.
TRIVIAL_INDEX_COLOR = "#BBBBBB"

COLOR_RULE = (
    "Morse nodes whose finite-relation index is computed and trivial are gray; "
    "Morse node i with a nontrivial label, or without a label, takes palette color "
    "i as in CMGDB (the next color that is not gray, when color i is gray), so a "
    "node has the same color in every figure and panel of the run"
)


def paper_grid_morse_colors(
    n_nodes: int,
    conley: Sequence[Mapping[str, Any]] = (),
) -> dict[int, str]:
    """Color of each Morse node of a run, the same in every figure and panel.

    ``conley`` holds the index records of the run (possibly none).  A node
    whose index is computed and trivial is :data:`TRIVIAL_INDEX_COLOR`.  Any
    other node ``i`` takes color ``i`` of :data:`PAPER_GRID_PALETTE`, as in
    CMGDB, or the next color that is not one of :data:`PALETTE_GRAYS`.
    """

    colors = dict.fromkeys(range(int(n_nodes)), TRIVIAL_INDEX_COLOR)
    for node in _palette_nodes(n_nodes, conley):
        colors[node] = _palette_color(node)
    return colors


def _palette_color(node: int) -> str:
    """Palette color ``node``, or the next color that is not gray."""

    count = len(PAPER_GRID_PALETTE)
    for step in range(count):
        color = PAPER_GRID_PALETTE[(node + step) % count]
        if color.lower() not in PALETTE_GRAYS:
            return color
    raise ValueError("the palette has no color that is not gray")


def _palette_nodes(n_nodes: int, conley: Sequence[Mapping[str, Any]]) -> list[int]:
    """The Morse nodes that take a palette color: all but the trivial ones."""

    trivial = {
        int(entry["morse_node"]) for entry in conley if index_status(entry) == "trivial"
    }
    return [node for node in range(int(n_nodes)) if node not in trivial]


def paper_grid_color_record(
    n_nodes: int,
    conley: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The ``colors`` record of a figure variant in the JSON summary.

    It names the palette and lists the color of every Morse node of the run
    (a node hidden in the variant is not drawn).  When a node that takes a
    palette color is numbered past the end of the palette, ``palette_cycled``
    is true and ``palette_cycled_note`` says from which node the colors repeat.
    """

    colors = paper_grid_morse_colors(n_nodes, conley)
    beyond = [
        node for node in _palette_nodes(n_nodes, conley) if node >= len(PAPER_GRID_PALETTE)
    ]
    record: dict[str, Any] = {
        "palette": PAPER_GRID_PALETTE_NAME,
        "palette_colors": list(PAPER_GRID_PALETTE),
        "trivial_index_color": TRIVIAL_INDEX_COLOR,
        "rule": COLOR_RULE,
        "morse_node_colors": [
            {"morse_node": node, "color": color} for node, color in colors.items()
        ],
        "palette_cycled": bool(beyond),
    }
    if beyond:
        record["palette_cycled_note"] = (
            f"the palette has {len(PAPER_GRID_PALETTE)} colors; the colors repeat from "
            f"Morse node {beyond[0]} on"
        )
    return record


#: Fraction of the chart span added on each side of a chart shown whole, so
#: cells on the window boundary (the ball's Zeno cap at ``h = 0``, the
#: oscillator's Zeno point on the wall) are not drawn under the axis lines.
FRAME_MARGIN = 0.02

#: Axis labels of the base and handle charts and the base view of each example
#: (a variant of an example is drawn in the style of the example).
PAPER_GRID_FIGURE_STYLE: dict[str, dict[str, Any]] = {
    "bouncing-ball": {"labels": (r"$h$", r"$v$"), "handle": (r"$v_G$", r"$s$"), "view": "domain"},
    "rimless-wheel": {
        "labels": (r"$\theta$", r"$\dot\theta$"),
        "handle": (r"$\dot\theta_G$", r"$s$"),
        "view": "support",
    },
    "spiking-neuron": {"labels": (r"$v$", r"$u$"), "handle": (r"$u_G$", r"$s$"), "view": "support"},
    "impact-vdp-duffing": {
        "labels": (r"$x$", r"$v$"),
        "handle": (r"$v_G$", r"$s$"),
        "view": "domain",
    },
}


#: Largest side, in pixels, of the PNG of a panel figure.  A Morse graph with
#: hundreds of nodes is many inches wide at its natural size (the 332 nodes
#: of the neuron at ``tau = 1`` take 16 x 27 inches); its PNG is written at a
#: lower resolution so that it keeps within this size.
PANEL_MAX_PIXELS = 8000


def _panel_dpi(figure: Figure, dpi: int) -> int:
    """Resolution of a panel figure: ``dpi``, lowered for a very large figure.

    The side is counted with the padding of the tight bounding box the
    files are saved with.
    """

    side = max(figure.get_size_inches()) + 2.0 * float(plt.rcParams["savefig.pad_inches"])
    return max(1, min(int(dpi), int(PANEL_MAX_PIXELS / side)))


def figure_variant_stem(stem: str | Path, variant: str) -> Path:
    """Output stem of a figure variant (``all`` keeps the run's stem)."""

    if variant not in FIGURE_VARIANTS:
        raise ValueError(f"unknown figure variant {variant!r}")
    stem = Path(stem)
    return stem if variant == "all" else stem.with_name(f"{stem.name}-{variant}")


def blocked_index_line(entry: Mapping[str, Any]) -> str:
    """Second line of the Morse-graph node of a Morse set without a label.

    When the relative homology of its pair is known, the line lists its
    dimensions from degree ``0`` up, as ``dim H (0, 1, 1, 0)``; otherwise it
    is ``blocked``.
    """

    if entry.get("homology_computed"):
        dimensions = ", ".join(str(int(value)) for value in entry["homology_dimensions"])
        return f"dim H ({dimensions})"
    return "blocked"


def _paper_grid_plot_options(
    plot_data: AtlasMorsePlotData,
    *,
    example: str,
    shown: Sequence[int],
    blocked: Sequence[int] | Mapping[int, str],
    annotations: AtlasFiniteRelationIndexAnnotations | None,
    colors: Mapping[int, str] | None,
) -> dict[str, Any]:
    """Options shared by the figure of a variant and its panel figures.

    Without ``colors``, every node takes a palette color, as when the run
    has no index records.
    """

    style = PAPER_GRID_FIGURE_STYLE[paper_grid_example(example)]
    shown_set = {int(node) for node in shown}
    show_handles = any(
        not any(box.chart_id == plot_data.base_chart_id for box in node.boxes)
        for node in plot_data.nodes
        if int(node.index) in shown_set
    )
    if colors is None:
        colors = paper_grid_morse_colors(len(plot_data.nodes))
    return {
        "clist": dict(colors),
        "morse_nodes": shown,
        "finite_relation_annotations": annotations,
        "blocked_index_nodes": blocked,
        "axis_labels": style["labels"],
        "handle_axis_labels": style["handle"],
        "show_handles": show_handles,
        "show_morse_graph": True,
        "show_component_sizes": False,
        "base_view": style["view"],
        "handle_view": "domain",
        "frame_margin": FRAME_MARGIN,
        "detail_zooms": True,
    }


def draw_paper_grid_figure(
    plot_data: AtlasMorsePlotData,
    *,
    example: str,
    shown: Sequence[int],
    blocked: Sequence[int] | Mapping[int, str] = (),
    annotations: AtlasFiniteRelationIndexAnnotations | None = None,
    colors: Mapping[int, str] | None = None,
) -> AtlasHybridMorsePlot:
    """Draw the figure of the Morse nodes ``shown``: base chart and Morse graph.

    ``example`` is the name of an example or a variant.  The base chart shows
    the base readout ``d_n^{-1}(M)`` of each Morse set in the view of
    :data:`PAPER_GRID_FIGURE_STYLE`, widened by :data:`FRAME_MARGIN` when
    shown whole, and a Morse set too small to see gets a zoom panel.  The
    handle chart (guard coordinate against the phase ``s``) is drawn only
    when a shown Morse set has no base cell, since such a set would
    otherwise not appear in the figure.  ``colors`` maps each Morse node to
    its color (see :func:`paper_grid_morse_colors`); without it, every node
    takes a palette color, as when the run has no index records.
    """

    return plot_atlas_hybrid_morse_sets(
        plot_data,
        **_paper_grid_plot_options(
            plot_data,
            example=example,
            shown=shown,
            blocked=blocked,
            annotations=annotations,
            colors=colors,
        ),
        show_legend=False,
        show_panel_titles=False,
        show_status_note=False,
        fig_h=3.4,
    )


def draw_paper_grid_panels(
    plot_data: AtlasMorsePlotData,
    *,
    example: str,
    shown: Sequence[int],
    blocked: Sequence[int] | Mapping[int, str] = (),
    annotations: AtlasFiniteRelationIndexAnnotations | None = None,
    colors: Mapping[int, str] | None = None,
    dpi: int = 300,
) -> AtlasMorsePanelFigures:
    """Draw each panel of :func:`draw_paper_grid_figure` as its own figure.

    The panels are ``base``, ``zoom-A``, ``zoom-B``, ..., ``handle`` (when
    the figure has the handle chart), and ``graph``, with the same colors,
    cells, and zoom windows as the figure (see
    :func:`plot_atlas_hybrid_morse_panels` for their sizes).
    """

    return plot_atlas_hybrid_morse_panels(
        plot_data,
        **_paper_grid_plot_options(
            plot_data,
            example=example,
            shown=shown,
            blocked=blocked,
            annotations=annotations,
            colors=colors,
        ),
        dpi=dpi,
    )


def write_paper_grid_figures(
    grid: SuspensionGrid,
    morse_sets: Sequence[npt.ArrayLike],
    edges: Sequence[Sequence[int]],
    conley: Sequence[Mapping[str, Any]],
    *,
    example: str,
    tau: float,
    level: int,
    output_stem: str | Path,
    audit_path: str | Path,
    variants: Sequence[str] = ("all",),
    dpi: int = 400,
    display_path: Callable[[Path], str] = str,
) -> dict[str, Any]:
    """Draw and save the requested figure variants of one run.

    ``conley`` holds the index records of the run (the ``conley`` list of its
    JSON summary).  Returns ``{"figures": [...], "figure_variants": {...}}``,
    where each variant records its files, the shown and hidden nodes (with
    the reason), the blocked nodes it marks, the order it draws, its zoom
    panels (label, chart, window, and the Morse nodes they are drawn for),
    under ``panel_files`` the files of each panel drawn as its own figure
    (``base``, ``zoom-A``, ..., ``handle``, ``graph``), and under ``colors``
    the palette and the color of every Morse node
    (:func:`paper_grid_color_record`).  ``figures`` lists the files of each
    variant, then those of its panels.  Summaries written before cells were
    outlined also have a ``marked_in_panel`` list (sets whose cells were
    marked by squares), summaries written before the panel figures have no
    ``panel_files``, and summaries written before the color records have no
    ``colors``; a replot replaces the record.
    """

    plot_data = suspension_grid_morse_sets_plot_data(
        grid,
        morse_sets,
        edges,
        metadata={"model": example, "t_star": tau, "level": level},
    )
    colors = paper_grid_morse_colors(len(morse_sets), conley)
    labels = {
        int(entry["morse_node"]): tuple(str(value) for value in entry["shift_class"])
        for entry in conley
        if entry["computed"]
    }
    blocked_lines = {
        int(entry["morse_node"]): blocked_index_line(entry)
        for entry in conley
        if not entry["computed"]
    }
    figures: list[str] = []
    records: dict[str, Any] = {}
    for variant in dict.fromkeys(variants):
        selection = morse_figure_selection(len(morse_sets), edges, conley, variant)
        record = selection.to_dict()
        record["colors"] = paper_grid_color_record(len(morse_sets), conley)
        if not selection.shown:
            record["files"] = []
            record["panel_files"] = {}
            record["not_written"] = "every Morse node is hidden"
            records[variant] = record
            continue
        shown_labels = {node: labels[node] for node in selection.shown if node in labels}
        annotations = (
            AtlasFiniteRelationIndexAnnotations(
                shift_classes=shown_labels,
                coefficient_field=5,
                result_scope="finite_reset_quotient_relation",
                audit_path=Path(audit_path),
            )
            if shown_labels
            else None
        )
        options = {
            "example": example,
            "shown": selection.shown,
            "blocked": {node: blocked_lines[node] for node in selection.blocked},
            "annotations": annotations,
            "colors": colors,
        }
        shown_colors = {node: colors[node] for node in selection.shown}
        variant_stem = figure_variant_stem(output_stem, variant)
        plot = draw_paper_grid_figure(plot_data, **options)
        try:
            drawn = tuple(sorted((int(p), int(q)) for p, q in plot.morse_graph.edges))
            if drawn != selection.order:
                raise AssertionError(
                    f"drawn Morse order {drawn!r} differs from the selection {selection.order!r}"
                )
            if {part.index: part.color for part in plot.components} != shown_colors:
                raise AssertionError("the figure has other colors than the color record")
            outputs = save_hybrid_morse_figure(
                plot, variant_stem, formats=("pdf", "png"), dpi=dpi
            )
        finally:
            plt.close(plot.figure)
        panels = draw_paper_grid_panels(plot_data, **options, dpi=dpi)
        try:
            if panels.zooms != plot.zooms:
                raise AssertionError("the panel figures have other zooms than the figure")
            if {part.index: part.color for part in panels.components} != shown_colors:
                raise AssertionError("the panel figures have other colors than the color record")
            panel_outputs = {
                name: save_hybrid_morse_figure(
                    figure,
                    variant_stem.with_name(f"{variant_stem.name}-{name}"),
                    formats=("pdf", "png"),
                    dpi=_panel_dpi(figure, dpi),
                )
                for name, figure in panels.figures.items()
            }
        finally:
            panels.close()
        record["zooms"] = [zoom.to_dict() for zoom in plot.zooms]
        record["files"] = [display_path(path) for path in outputs]
        record["panel_files"] = {
            name: [display_path(path) for path in paths] for name, paths in panel_outputs.items()
        }
        figures.extend(record["files"])
        for paths in record["panel_files"].values():
            figures.extend(paths)
        records[variant] = record
    return {"figures": figures, "figure_variants": records}


__all__ = [
    "COLOR_RULE",
    "FRAME_MARGIN",
    "PAPER_GRID_FIGURE_STYLE",
    "PAPER_GRID_PALETTE",
    "PAPER_GRID_PALETTE_NAME",
    "PANEL_MAX_PIXELS",
    "TRIVIAL_INDEX_COLOR",
    "draw_paper_grid_figure",
    "draw_paper_grid_panels",
    "figure_variant_stem",
    "paper_grid_color_record",
    "paper_grid_morse_colors",
    "write_paper_grid_figures",
]
