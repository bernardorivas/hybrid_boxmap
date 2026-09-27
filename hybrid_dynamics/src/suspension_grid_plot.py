"""Plot data for Morse sets on the paper suspension grid.

The figures reuse the Atlas plotter of :mod:`atlas_morse_plot`, so they have
the layout and palette of the earlier manuscript figures.  Each Morse node is
represented by the closed base cells of its base readout ``d_n^{-1}(M)`` and,
in the handle chart ``(u, s)``, by the handle pieces of its atoms.
"""

from __future__ import annotations

from collections.abc import Mapping

from .atlas_morse_plot import AtlasMorseBox, AtlasMorseNode, AtlasMorsePlotData
from .suspension_grid import SuspensionGrid
from .suspension_grid_relation import SuspensionMorseGraph


BASE_CHART = 0
HANDLE_CHART = 1


def suspension_grid_morse_plot_data(
    grid: SuspensionGrid,
    morse_graph: SuspensionMorseGraph,
    *,
    metadata: Mapping[str, object] | None = None,
) -> AtlasMorsePlotData:
    """Base readout and handle pieces of every Morse set, with the order."""

    nodes = []
    for index, morse_set in enumerate(morse_graph.morse_sets):
        base_cells = grid.base_readout(morse_set)
        boxes = [
            AtlasMorseBox(BASE_CHART, tuple(float(value) for value in bounds))
            for bounds in grid.base_bounds(base_cells)
        ]
        pieces = [piece for atom in morse_set for piece in grid.atom(int(atom)).tolist()]
        handle = [piece for piece in pieces if piece >= grid.n_base]
        boxes.extend(
            AtlasMorseBox(HANDLE_CHART, tuple(float(value) for value in bounds))
            for bounds in grid.handle_bounds(handle)
        )
        nodes.append(AtlasMorseNode(index=index, boxes=tuple(sorted(boxes))))
    return AtlasMorsePlotData(
        base_chart_id=BASE_CHART,
        handle_chart_id=HANDLE_CHART,
        base_bounds=tuple(tuple(interval) for interval in grid.window.ambient_bounds),
        handle_bounds=((float(grid.guard.u_bounds[0]), float(grid.guard.u_bounds[1])), (0.0, 1.0)),
        nodes=tuple(nodes),
        edges=tuple(morse_graph.edges),
        metadata=dict(metadata or {}),
    )


__all__ = ["suspension_grid_morse_plot_data"]
