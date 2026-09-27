"""Plot data for Morse sets on the paper suspension grid.

The figures reuse the Atlas plotter of :mod:`atlas_morse_plot`, so they have
the layout and palette of the earlier manuscript figures.  Each Morse node is
represented by the closed base cells of its base readout ``d_n^{-1}(M)`` and,
in the handle chart ``(u, s)``, by the handle pieces of its atoms.

Two figure variants are defined.  ``all`` shows every Morse node.
``nontrivial`` hides the Morse nodes whose finite-relation index was computed
and is trivial (every homology dimension zero); a node whose index is blocked
is never hidden.  The remaining nodes keep their numbers and colors, and the
order drawn between them is reachability in the full Morse graph (paths may
pass through hidden nodes), transitively reduced.

The Morse sets are stored in a run's JSON summary as strings of atom ranges
(:func:`encode_index_ranges`), so the figures can be redrawn from the JSON
and the grid, which is rebuilt deterministically from the example and level.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

from .atlas_morse_plot import AtlasMorseBox, AtlasMorseNode, AtlasMorsePlotData
from .suspension_grid import SuspensionGrid
from .suspension_grid_relation import SuspensionMorseGraph


BASE_CHART = 0
HANDLE_CHART = 1

#: Figure variants: every Morse node, or the nodes whose index is not trivial.
FIGURE_VARIANTS = ("all", "nontrivial")

TRIVIAL_INDEX_REASON = (
    "finite-relation index computed and trivial: every homology dimension is zero over GF(5)"
)
RANGE_ENCODING = (
    "space-separated nonnegative integers in increasing order; 'a-b' is the closed range a..b"
)


def suspension_grid_morse_sets_plot_data(
    grid: SuspensionGrid,
    morse_sets: Sequence[npt.ArrayLike],
    edges: Iterable[Sequence[int]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> AtlasMorsePlotData:
    """Base readout and handle pieces of the Morse sets (atom arrays), with the order."""

    nodes = []
    for index, raw_set in enumerate(morse_sets):
        morse_set = np.asarray(raw_set, dtype=np.int64)
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
        edges=tuple((int(source), int(target)) for source, target in edges),
        metadata=dict(metadata or {}),
    )


def suspension_grid_morse_plot_data(
    grid: SuspensionGrid,
    morse_graph: SuspensionMorseGraph,
    *,
    metadata: Mapping[str, object] | None = None,
) -> AtlasMorsePlotData:
    """Base readout and handle pieces of every Morse set, with the order."""

    return suspension_grid_morse_sets_plot_data(
        grid, morse_graph.morse_sets, morse_graph.edges, metadata=metadata
    )


# ---------------------------------------------------------------------------
# Figure variants
# ---------------------------------------------------------------------------


def index_status(entry: Mapping[str, Any]) -> str:
    """``"blocked"``, ``"trivial"``, or ``"nontrivial"`` for one index record.

    ``entry`` is a record of :meth:`SuspensionGridConleyResult.to_dict` (the
    ``conley`` list of a run's JSON summary).  An index is trivial when it was
    computed and every homology dimension of its shift class is zero.
    """

    if not entry["computed"]:
        return "blocked"
    return "nontrivial" if any(int(value) for value in entry["homology_dimensions"]) else "trivial"


def restricted_morse_order(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    shown: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    """Order among ``shown`` nodes: reachability in the full graph, reduced.

    ``(p, q)`` is returned when ``q`` is reachable from ``p`` in the graph
    with vertices ``0, ..., n_nodes - 1`` and the given edges (paths may pass
    through nodes that are not shown), and no shown node lies strictly
    between them.
    """

    full = nx.DiGraph()
    full.add_nodes_from(range(int(n_nodes)))
    full.add_edges_from((int(source), int(target)) for source, target in edges)
    selected = sorted({int(node) for node in shown})
    unknown = [node for node in selected if node not in full]
    if unknown:
        raise ValueError(f"unknown Morse nodes: {unknown!r}")
    order = nx.DiGraph()
    order.add_nodes_from(selected)
    selected_set = set(selected)
    for source in selected:
        order.add_edges_from(
            (source, target)
            for target in nx.descendants(full, source)
            if target in selected_set
        )
    reduced = nx.transitive_reduction(order) if order.nodes else order
    return tuple(sorted((int(source), int(target)) for source, target in reduced.edges))


@dataclass(frozen=True)
class MorseFigureSelection:
    """The Morse nodes drawn in one figure variant, and why the others are not."""

    variant: str
    shown: tuple[int, ...]
    hidden: tuple[tuple[int, str], ...]
    blocked: tuple[int, ...]
    order: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "rule": (
                "every Morse node"
                if self.variant == "all"
                else (
                    "Morse nodes whose finite-relation index is trivial are hidden; "
                    "nodes with a blocked index are kept and marked 'blocked'; the "
                    "order shown is reachability in the full Morse graph, transitively "
                    "reduced"
                )
            ),
            "shown_nodes": list(self.shown),
            "hidden_nodes": [
                {"morse_node": node, "reason": reason} for node, reason in self.hidden
            ],
            "blocked_nodes_shown": list(self.blocked),
            "order_shown": [list(edge) for edge in self.order],
        }


def morse_figure_selection(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    conley: Sequence[Mapping[str, Any]],
    variant: str,
) -> MorseFigureSelection:
    """Select the Morse nodes of a figure variant from the index records.

    ``conley`` holds the index records of the run (possibly empty for
    ``variant="all"``).  Blocked nodes are always shown; ``nontrivial`` needs
    a record for every node and hides exactly the trivial ones.
    """

    if variant not in FIGURE_VARIANTS:
        raise ValueError(f"unknown figure variant {variant!r}; expected one of {FIGURE_VARIANTS}")
    edges = tuple((int(source), int(target)) for source, target in edges)
    status: dict[int, str] = {}
    for entry in conley:
        node = int(entry["morse_node"])
        if not 0 <= node < n_nodes:
            raise ValueError(f"index record for unknown Morse node {node}")
        if node in status:
            raise ValueError(f"two index records for Morse node {node}")
        status[node] = index_status(entry)
    if variant == "nontrivial":
        missing = sorted(set(range(n_nodes)).difference(status))
        if missing:
            raise ValueError(
                "the nontrivial figure variant needs an index record for every "
                f"Morse node; missing {missing!r}"
            )
        hidden = tuple(
            (node, TRIVIAL_INDEX_REASON) for node in range(n_nodes) if status[node] == "trivial"
        )
    else:
        hidden = ()
    hidden_nodes = {node for node, _reason in hidden}
    shown = tuple(node for node in range(n_nodes) if node not in hidden_nodes)
    blocked = tuple(node for node in shown if status.get(node) == "blocked")
    return MorseFigureSelection(
        variant=variant,
        shown=shown,
        hidden=hidden,
        blocked=blocked,
        order=restricted_morse_order(n_nodes, edges, shown),
    )


# ---------------------------------------------------------------------------
# Morse sets in the JSON summary
# ---------------------------------------------------------------------------


def encode_index_ranges(values: npt.ArrayLike) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2 5 7-8"`` (see :data:`RANGE_ENCODING`)."""

    array = np.unique(np.asarray(values, dtype=np.int64))
    if array.size and array[0] < 0:
        raise ValueError("range encoding needs nonnegative integers")
    if not array.size:
        return ""
    breaks = np.flatnonzero(np.diff(array) != 1)
    starts = np.r_[array[0], array[breaks + 1]]
    stops = np.r_[array[breaks], array[-1]]
    return " ".join(
        str(start) if start == stop else f"{start}-{stop}"
        for start, stop in zip(starts.tolist(), stops.tolist())
    )


def decode_index_ranges(text: str) -> npt.NDArray[np.int64]:
    """Inverse of :func:`encode_index_ranges`."""

    parts: list[npt.NDArray[np.int64]] = []
    for token in text.split():
        start, _, stop = token.partition("-")
        first = int(start)
        last = int(stop) if stop else first
        if first < 0 or last < first:
            raise ValueError(f"malformed index range {token!r}")
        parts.append(np.arange(first, last + 1, dtype=np.int64))
    result = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if result.size > 1 and np.any(np.diff(result) <= 0):
        raise ValueError("index ranges must be increasing and disjoint")
    return result


__all__ = [
    "FIGURE_VARIANTS",
    "MorseFigureSelection",
    "RANGE_ENCODING",
    "TRIVIAL_INDEX_REASON",
    "decode_index_ranges",
    "encode_index_ranges",
    "index_status",
    "morse_figure_selection",
    "restricted_morse_order",
    "suspension_grid_morse_plot_data",
    "suspension_grid_morse_sets_plot_data",
]
