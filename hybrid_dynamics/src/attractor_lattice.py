"""The lattice of down-sets of a Morse order and its Hasse diagram.

By Birkhoff's representation theorem, the attractor lattice of a Morse
decomposition with Morse order ``P`` (a finite poset) is the lattice ``O(P)``
of down-sets of ``P`` ordered by inclusion, and its join-irreducible elements
are the principal down-sets ``↓p``, one for each element ``p`` of ``P``.
Every down-set ``D`` is the join of the ``↓p`` for the elements ``p`` maximal
in ``D``, and one down-set covers another exactly when it has one more
element.

The order is given as in a Morse graph: an edge ``(p, q)`` means that ``q``
is reachable from ``p``, so ``q`` lies below ``p`` (attractors are at the
bottom).  A down-set contains, with each element, every element reachable
from it.

:func:`down_set_lattice` computes ``O(P)`` and :func:`draw_lattice_hasse_diagram`
draws its Hasse diagram in the style of the Morse graph (Graphviz ``dot``
layout, ellipses labeled in inches), bottom element at the bottom.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .hybrid_morse_plot import (
    _draw_morse_node_label,
    _ellipse_edge_point,
    _morse_graph_positions,
    _morse_label_color,
    _morse_node_size,
)


#: Largest number of down-sets :func:`down_set_lattice` enumerates.  An
#: antichain of ``k`` elements has ``2**k`` down-sets.
MAX_LATTICE_ELEMENTS = 512

#: Fill of a lattice element drawn without a color (not join-irreducible).
LATTICE_ELEMENT_FILL = "#ffffff"

#: The join symbol ``∨`` of a label is drawn as the math text ``\vee`` of
#: this font set (the thin TeX glyph, which reads as an operator; the ``∨`` of
#: DejaVu Sans reads as the letter v).
JOIN_SYMBOL_MATH_FONTSET = "cm"


class LatticeTooLargeError(ValueError):
    """The poset has more down-sets than the limit."""


@dataclass(frozen=True)
class DownSetLattice:
    """The down-sets of a finite poset, ordered by inclusion.

    ``poset`` lists the elements of the poset in increasing order and
    ``order`` its covering pairs ``(p, q)``, ``q`` below ``p``.  ``elements``
    lists the down-sets by size, then by their sorted members, so element
    ``0`` is the empty down-set (the bottom) and the last element is the
    whole poset (the top).  ``covers`` lists the covering pairs ``(i, j)`` of
    the lattice: ``elements[j]`` covers ``elements[i]``, which it contains
    with one more element.
    """

    poset: tuple[int, ...]
    order: tuple[tuple[int, int], ...]
    elements: tuple[frozenset[int], ...]
    covers: tuple[tuple[int, int], ...]
    _index: Mapping[frozenset[int], int] = field(repr=False, compare=False)
    _below: Mapping[int, frozenset[int]] = field(repr=False, compare=False)

    @property
    def bottom(self) -> int:
        return 0

    @property
    def top(self) -> int:
        return len(self.elements) - 1

    def lower_covers(self, element: int) -> tuple[int, ...]:
        """The elements that ``element`` covers."""

        return tuple(lower for lower, upper in self.covers if upper == int(element))

    @property
    def join_irreducibles(self) -> tuple[int, ...]:
        """The elements with exactly one lower cover, in the order of ``elements``."""

        counts = dict.fromkeys(range(len(self.elements)), 0)
        for _lower, upper in self.covers:
            counts[upper] += 1
        return tuple(element for element, count in counts.items() if count == 1)

    def principal(self, node: int) -> int:
        """The element ``↓node``: ``node`` and every poset element below it."""

        return self._index[self._below[int(node)] | {int(node)}]

    def index(self, down_set: Iterable[int]) -> int:
        """The element of a down-set given by its members."""

        return self._index[frozenset(int(node) for node in down_set)]

    def maximal(self, element: int) -> tuple[int, ...]:
        """The poset elements maximal in ``elements[element]``, increasing.

        The element is the join of the principal down-sets of these.
        """

        members = self.elements[int(element)]
        return tuple(
            node
            for node in sorted(members)
            if not any(node in self._below[other] for other in members)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "poset": list(self.poset),
            "order": [list(pair) for pair in self.order],
            "elements": [sorted(members) for members in self.elements],
            "covers": [list(pair) for pair in self.covers],
            "join_irreducibles": list(self.join_irreducibles),
        }


def down_set_lattice(
    poset: Iterable[int],
    order: Iterable[Sequence[int]],
    *,
    max_elements: int = MAX_LATTICE_ELEMENTS,
) -> DownSetLattice:
    """The lattice of down-sets of a finite poset, with its covering pairs.

    ``order`` holds pairs ``(p, q)`` with ``q`` below ``p``; the order is
    their reflexive and transitive closure, which must be antisymmetric.
    The down-sets are generated from the empty one by adding, one at a time,
    an element whose lower covers are already in the set; these additions are
    exactly the covering pairs of the lattice.  Raises
    :class:`LatticeTooLargeError` when there are more than ``max_elements``
    down-sets.
    """

    nodes = sorted({int(node) for node in poset})
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for pair in order:
        upper, lower = (int(value) for value in pair)
        if upper not in graph or lower not in graph:
            raise ValueError(f"order pair {(upper, lower)!r} has an element outside the poset")
        graph.add_edge(upper, lower)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the order relation has a cycle")
    reduced = nx.transitive_reduction(graph) if nodes else graph
    lower_covers = {node: frozenset(reduced.successors(node)) for node in nodes}
    below = {node: frozenset(nx.descendants(graph, node)) for node in nodes}

    found = {frozenset()}
    frontier = [frozenset()]
    pairs: set[tuple[frozenset[int], frozenset[int]]] = set()
    while frontier:
        following = []
        for down_set in frontier:
            for node in nodes:
                if node in down_set or not lower_covers[node] <= down_set:
                    continue
                larger = down_set | {node}
                pairs.add((down_set, larger))
                if larger not in found:
                    found.add(larger)
                    following.append(larger)
                    if len(found) > int(max_elements):
                        raise LatticeTooLargeError(
                            f"the poset of {len(nodes)} elements has more than "
                            f"{int(max_elements)} down-sets"
                        )
        frontier = following

    elements = tuple(sorted(found, key=lambda members: (len(members), sorted(members))))
    index = {members: position for position, members in enumerate(elements)}
    return DownSetLattice(
        poset=tuple(nodes),
        order=tuple(sorted((int(p), int(q)) for p, q in reduced.edges)),
        elements=elements,
        covers=tuple(sorted((index[lower], index[upper]) for lower, upper in pairs)),
        _index=index,
        _below=below,
    )


def cover_crossings(
    positions: Mapping[int, tuple[float, float]],
    covers: Iterable[Sequence[int]],
) -> int:
    """The number of pairs of covering pairs whose lines cross.

    Each covering pair ``(i, j)`` is the segment between the centers
    ``positions[i]`` and ``positions[j]``.  Two segments with a common end
    do not count, and two segments count when each meets the line through
    the other at a point other than its ends.
    """

    def side(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    segments = [(int(lower), int(upper)) for lower, upper in covers]
    count = 0
    for position, (a, b) in enumerate(segments):
        for c, d in segments[position + 1 :]:
            if {a, b} & {c, d}:
                continue
            p, q, r, s = (positions[node] for node in (a, b, c, d))
            if side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0:
                count += 1
    return count


def draw_lattice_hasse_diagram(
    lattice: DownSetLattice,
    *,
    labels: Mapping[int, str],
    colors: Mapping[int, str] | None = None,
    dashed: Collection[int] = (),
    font_size: float = 7.0,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Draw the Hasse diagram of ``lattice`` as its own figure.

    ``labels`` gives the text of every element (one line per row; a join
    symbol ``∨`` is drawn as the TeX ``\vee``, see
    :data:`JOIN_SYMBOL_MATH_FONTSET`), ``colors`` the fill of some elements
    (the others are :data:`LATTICE_ELEMENT_FILL`), and ``dashed`` the elements
    with a dashed outline.  Each element is an ellipse around its label,
    placed by the Graphviz layout of the Morse graph with the top element at
    the top (the nodes of a rank ordered to reduce crossings; the elements
    covering the bottom are placed from left to right in the order of
    ``elements``, that is, of their poset elements, unless this layout has
    more crossings, see :func:`cover_crossings`), and each covering pair is
    a line.  As for the Morse graph drawn as its own figure,
    the layout is in inches, with labels of ``font_size`` points, and the
    figure has the size of the layout, so the labels print at that size when
    the figure is shown at its natural size.
    """

    colors = dict(colors or {})
    missing = sorted(set(range(len(lattice.elements))).difference(labels))
    if missing:
        raise ValueError(f"no label for the lattice elements {missing!r}")
    dashed = frozenset(int(element) for element in dashed)
    drawn = {element: str(labels[element]).replace("∨", r"$\vee$") for element in labels}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(lattice.elements)))
    # Graphviz places the source of an edge above its target.
    graph.add_edges_from((upper, lower) for lower, upper in lattice.covers)
    with plt.rc_context({"mathtext.fontset": JOIN_SYMBOL_MATH_FONTSET}):
        sizes = {
            element: _morse_node_size(drawn[element], font_size) for element in graph.nodes
        }
    positions = _morse_graph_positions(
        graph, sizes, keep_edge_order=False, in_ordered=(lattice.bottom,)
    )
    free = _morse_graph_positions(graph, sizes, keep_edge_order=False)
    if cover_crossings(free, lattice.covers) < cover_crossings(positions, lattice.covers):
        positions = free

    figure = plt.figure(figsize=(1.0, 1.0), dpi=dpi)
    try:
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        for lower, upper in lattice.covers:
            start = _ellipse_edge_point(positions[upper], positions[lower], sizes[upper])
            end = _ellipse_edge_point(positions[lower], positions[upper], sizes[lower])
            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                color="#202020",
                linewidth=0.8,
                solid_capstyle="butt",
                zorder=1,
            )
        for element in sorted(graph.nodes):
            fill = colors.get(element, LATTICE_ELEMENT_FILL)
            width, height = sizes[element]
            axis.add_patch(
                patches.Ellipse(
                    positions[element],
                    width=width,
                    height=height,
                    facecolor=fill,
                    edgecolor="#202020",
                    linewidth=1.0 if element in dashed else 0.8,
                    linestyle=(0, (2.2, 1.4)) if element in dashed else "solid",
                    zorder=2,
                )
            )
            with plt.rc_context({"mathtext.fontset": JOIN_SYMBOL_MATH_FONTSET}):
                _draw_morse_node_label(
                    axis,
                    positions[element],
                    drawn[element],
                    font_size,
                    color=_morse_label_color(fill),
                    text=str(labels[element]),
                )
        x_lower = min(positions[node][0] - sizes[node][0] / 2.0 for node in positions)
        x_upper = max(positions[node][0] + sizes[node][0] / 2.0 for node in positions)
        y_lower = min(positions[node][1] - sizes[node][1] / 2.0 for node in positions)
        y_upper = max(positions[node][1] + sizes[node][1] / 2.0 for node in positions)
        axis.set_xlim(x_lower - 0.12, x_upper + 0.12)
        axis.set_ylim(y_lower - 0.12, y_upper + 0.12)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        figure.set_size_inches(x_upper - x_lower + 0.24, y_upper - y_lower + 0.24)
    except BaseException:
        plt.close(figure)
        raise
    return figure, axis


__all__ = [
    "JOIN_SYMBOL_MATH_FONTSET",
    "LATTICE_ELEMENT_FILL",
    "MAX_LATTICE_ELEMENTS",
    "DownSetLattice",
    "LatticeTooLargeError",
    "cover_crossings",
    "down_set_lattice",
    "draw_lattice_hasse_diagram",
]
