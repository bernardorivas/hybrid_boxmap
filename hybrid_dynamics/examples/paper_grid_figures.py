"""Figures of the paper-grid runs, shared by the runner and the replot script.

:func:`write_paper_grid_figures` draws the Morse sets and the Morse graph of a
run on ``Xi_n`` in the figure variants of :mod:`suspension_grid_plot`:
``all`` (every Morse node) and ``nontrivial`` (Morse nodes with a computed
trivial finite-relation index hidden, blocked nodes kept and marked).  The
``all`` figure is written to ``<stem>.pdf``/``.png`` and the ``nontrivial``
figure to ``<stem>-nontrivial.pdf``/``.png``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy.typing as npt

from ..src.atlas_morse_plot import AtlasFiniteRelationIndexAnnotations
from ..src.hybrid_morse_plot import PlotHybridMorseSets, save_hybrid_morse_figure
from ..src.suspension_grid import SuspensionGrid
from ..src.suspension_grid_plot import (
    FIGURE_VARIANTS,
    morse_figure_selection,
    suspension_grid_morse_sets_plot_data,
)


#: Axis labels and base view of the figure of each example.
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
    the reason), the blocked nodes it marks, and the order it draws.
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
    style = PAPER_GRID_FIGURE_STYLE[example]
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
        plot = PlotHybridMorseSets(
            plot_data,
            morse_nodes=selection.shown,
            finite_relation_annotations=annotations,
            blocked_index_nodes=selection.blocked,
            axis_labels=style["labels"],
            handle_axis_labels=style["handle"],
            show_handles=False,
            show_morse_graph=True,
            show_legend=False,
            show_panel_titles=False,
            show_component_sizes=False,
            show_status_note=False,
            base_view=style["view"],
            fig_h=3.4,
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
        record["files"] = [display_path(path) for path in outputs]
        figures.extend(record["files"])
        records[variant] = record
    return {"figures": figures, "figure_variants": records}


__all__ = [
    "PAPER_GRID_FIGURE_STYLE",
    "figure_variant_stem",
    "write_paper_grid_figures",
]
