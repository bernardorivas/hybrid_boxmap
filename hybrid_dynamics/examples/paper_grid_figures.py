"""Figures of the paper-grid runs, shared by the runner and the replot script.

:func:`write_paper_grid_figures` draws the Morse sets and the Morse graph of a
run on ``Xi_n`` in the figure variants of :mod:`suspension_grid_plot`:
``all`` (every Morse node) and ``nontrivial`` (Morse nodes with a computed
trivial finite-relation index hidden, nodes without a label kept and marked
with the dimensions of the relative homology of their pair).  The
``all`` figure is written to ``<stem>.pdf``/``.png`` and the ``nontrivial``
figure to ``<stem>-nontrivial.pdf``/``.png``.

Each figure has the base chart (the base readout ``d_n^{-1}(M)`` of each Morse
set), the handle chart (its handle pieces, guard coordinate against the phase
``s`` in ``[0, 1]``), and the Morse graph, in the same colors.  A Morse set too
small to see in a chart panel gets a zoom panel (``A``, ``B``, ...) next to
it; its window is outlined and labeled in the panel, and the zoom shows the
other Morse sets in the window faded.  Every cell is drawn at its true extent,
with no symbol; the cells of a set too small to see in a chart panel are also
outlined by a thin line in the color of the set, in the panel and in its
zooms, so cells smaller than a point are still seen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy.typing as npt

from ..src.atlas_morse_plot import (
    AtlasFiniteRelationIndexAnnotations,
    AtlasHybridMorsePlot,
    AtlasMorsePlotData,
    plot_atlas_hybrid_morse_sets,
)
from ..src.hybrid_morse_plot import CMGDB_MORSE_PALETTE, save_hybrid_morse_figure
from ..src.suspension_grid import SuspensionGrid
from ..src.suspension_grid_plot import (
    FIGURE_VARIANTS,
    morse_figure_selection,
    suspension_grid_morse_sets_plot_data,
)
from .paper_grid_examples import paper_grid_example


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


def draw_paper_grid_figure(
    plot_data: AtlasMorsePlotData,
    *,
    example: str,
    shown: Sequence[int],
    blocked: Sequence[int] | Mapping[int, str] = (),
    annotations: AtlasFiniteRelationIndexAnnotations | None = None,
) -> AtlasHybridMorsePlot:
    """Draw the figure of the Morse nodes ``shown``: base chart, handle chart, graph.

    ``example`` is the name of an example or a variant.  The base chart uses
    the view of :data:`PAPER_GRID_FIGURE_STYLE`; the handle chart shows the
    whole guard interval against the phase ``s`` in ``[0, 1]``.  Both are
    widened by :data:`FRAME_MARGIN` when shown whole, and a Morse set too
    small to see in a chart panel gets a zoom panel.
    """

    style = PAPER_GRID_FIGURE_STYLE[paper_grid_example(example)]
    return plot_atlas_hybrid_morse_sets(
        plot_data,
        clist=CMGDB_MORSE_PALETTE,
        morse_nodes=shown,
        finite_relation_annotations=annotations,
        blocked_index_nodes=blocked,
        axis_labels=style["labels"],
        handle_axis_labels=style["handle"],
        show_handles=True,
        show_morse_graph=True,
        show_legend=False,
        show_panel_titles=False,
        show_component_sizes=False,
        show_status_note=False,
        base_view=style["view"],
        handle_view="domain",
        frame_margin=FRAME_MARGIN,
        detail_zooms=True,
        fig_h=3.4,
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
    the reason), the blocked nodes it marks, the order it draws, and its
    zoom panels (label, chart, window, and the Morse nodes they are drawn
    for).  Summaries written before cells were outlined also have a
    ``marked_in_panel`` list (sets whose cells were marked by squares); a
    replot replaces the record and drops it.
    """

    plot_data = suspension_grid_morse_sets_plot_data(
        grid,
        morse_sets,
        edges,
        metadata={"model": example, "t_star": tau, "level": level},
    )
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
        if not selection.shown:
            record["files"] = []
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
        plot = draw_paper_grid_figure(
            plot_data,
            example=example,
            shown=selection.shown,
            blocked={node: blocked_lines[node] for node in selection.blocked},
            annotations=annotations,
        )
        try:
            drawn = tuple(sorted((int(p), int(q)) for p, q in plot.morse_graph.edges))
            if drawn != selection.order:
                raise AssertionError(
                    f"drawn Morse order {drawn!r} differs from the selection {selection.order!r}"
                )
            outputs = save_hybrid_morse_figure(
                plot, figure_variant_stem(output_stem, variant), formats=("pdf", "png"), dpi=dpi
            )
        finally:
            plt.close(plot.figure)
        record["zooms"] = [zoom.to_dict() for zoom in plot.zooms]
        record["files"] = [display_path(path) for path in outputs]
        figures.extend(record["files"])
        records[variant] = record
    return {"figures": figures, "figure_variants": records}


__all__ = [
    "FRAME_MARGIN",
    "PAPER_GRID_FIGURE_STYLE",
    "draw_paper_grid_figure",
    "figure_variant_stem",
    "write_paper_grid_figures",
]
