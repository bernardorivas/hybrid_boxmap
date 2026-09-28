"""The lattice of down-sets of a Morse order and the attractor lattice figure."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pytest  # noqa: E402
from matplotlib import patches  # noqa: E402

from hybrid_dynamics.examples.paper_grid_figures import (  # noqa: E402
    ATTRACTOR_LATTICE_RULE,
    attractor_lattice_labels,
    paper_grid_attractor_lattice,
    write_attractor_lattice_figure,
)
from hybrid_dynamics.src.attractor_lattice import (  # noqa: E402
    LATTICE_ELEMENT_FILL,
    LatticeTooLargeError,
    down_set_lattice,
    draw_lattice_hasse_diagram,
)
from hybrid_dynamics.src.hybrid_morse_plot import morse_graph_node_labels  # noqa: E402


CODE_ROOT = Path(__file__).resolve().parents[2]
OSCILLATOR_RUN = (
    CODE_ROOT
    / "figures"
    / "paper_grid"
    / "paper-grid-impact-vdp-duffing-beta076-tau050-level7-base2048-corners-gap-refined.json"
)

# The poset N: 0 and 1 below 2, 1 below 3 (an edge (p, q) puts q below p,
# as in a Morse graph).
N_POSET = (0, 1, 2, 3)
N_ORDER = ((2, 0), (2, 1), (3, 1))


def _brute_force_down_sets(poset, order) -> set[frozenset[int]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(poset)
    graph.add_edges_from(order)
    return {
        frozenset(subset)
        for size in range(len(poset) + 1)
        for subset in itertools.combinations(poset, size)
        if all(nx.descendants(graph, node) <= set(subset) for node in subset)
    }


def _brute_force_covers(elements) -> set[tuple[frozenset[int], frozenset[int]]]:
    return {
        (lower, upper)
        for lower in elements
        for upper in elements
        if lower < upper and not any(lower < middle < upper for middle in elements)
    }


def _brute_force_join_irreducibles(elements) -> set[frozenset[int]]:
    """Nonzero elements that are not the join (union) of two smaller elements."""

    return {
        element
        for element in elements
        if element
        and not any(
            first | second == element
            for first in elements
            for second in elements
            if first < element and second < element
        )
    }


@pytest.mark.parametrize(
    ("poset", "order", "count"),
    [
        (N_POSET, N_ORDER, 8),
        # A chain of four elements, given with a transitive pair.
        ((0, 1, 2, 3), ((1, 0), (2, 1), (3, 2), (3, 0)), 5),
        # An antichain of three elements: the Boolean lattice of 8 subsets.
        ((0, 1, 2), (), 8),
        # The restricted Morse order of the oscillator at beta = 0.76, tau = 0.5.
        ((0, 1, 2, 11, 21), ((11, 0), (11, 1), (21, 2), (21, 11)), 11),
    ],
)
def test_down_set_lattice_of_a_small_poset(poset, order, count):
    lattice = down_set_lattice(poset, order)
    elements = set(lattice.elements)
    assert len(lattice.elements) == len(elements) == count
    assert elements == _brute_force_down_sets(poset, order)
    # Bottom (the empty down-set) first, top (the whole poset) last.
    assert lattice.elements[lattice.bottom] == frozenset()
    assert lattice.elements[lattice.top] == frozenset(poset)
    # The Hasse edges are exactly the covering pairs of the inclusion order.
    covers = {(lattice.elements[lower], lattice.elements[upper]) for lower, upper in lattice.covers}
    assert len(covers) == len(lattice.covers)
    assert covers == _brute_force_covers(elements)
    # The join-irreducible elements are exactly the principal down-sets.
    join_irreducibles = {lattice.elements[element] for element in lattice.join_irreducibles}
    assert join_irreducibles == _brute_force_join_irreducibles(elements)
    graph = nx.DiGraph(list(order))
    graph.add_nodes_from(poset)
    principal = {frozenset(nx.descendants(graph, node) | {node}) for node in poset}
    assert join_irreducibles == principal
    assert {lattice.principal(node) for node in poset} == set(lattice.join_irreducibles)
    assert len(lattice.join_irreducibles) == len(poset)
    # Every element is the join (union) of the principal down-sets of its
    # maximal elements, and no maximal element lies below another member.
    for element, members in enumerate(lattice.elements):
        maximal = lattice.maximal(element)
        principals = [lattice.elements[lattice.principal(node)] for node in maximal]
        assert frozenset().union(*principals) == members
        assert not any(
            node in nx.descendants(graph, other) for node in maximal for other in members
        )


def test_down_set_lattice_of_the_n_poset():
    lattice = down_set_lattice(N_POSET, N_ORDER)
    assert [sorted(members) for members in lattice.elements] == [
        [],
        [0],
        [1],
        [0, 1],
        [1, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 2, 3],
    ]
    assert lattice.order == ((2, 0), (2, 1), (3, 1))
    assert lattice.join_irreducibles == (1, 2, 4, 5)
    assert lattice.maximal(lattice.index([0, 1, 3])) == (0, 3)
    assert lattice.lower_covers(lattice.top) == (lattice.index([0, 1, 2]), lattice.index([0, 1, 3]))
    assert lattice.to_dict()["join_irreducibles"] == [1, 2, 4, 5]


def test_down_set_lattice_rejects_cycles_and_large_lattices():
    with pytest.raises(ValueError, match="cycle"):
        down_set_lattice((0, 1), ((0, 1), (1, 0)))
    with pytest.raises(ValueError, match="outside the poset"):
        down_set_lattice((0, 1), ((2, 0),))
    with pytest.raises(LatticeTooLargeError):
        down_set_lattice(range(10), (), max_elements=512)
    assert len(down_set_lattice(range(9), ()).elements) == 512
    empty = down_set_lattice((), ())
    assert empty.elements == (frozenset(),) and empty.covers == ()


def _record(node: int, shift_class: tuple[str, ...], computed: bool = True) -> dict[str, object]:
    return {
        "morse_node": node,
        "computed": computed,
        "shift_class": list(shift_class) if computed else [],
        "homology_dimensions": [int(entry != "0") for entry in shift_class] if computed else [],
        "blocker": "" if computed else "ValueError: carrier image is not acyclic over GF(5)",
    }


def test_the_oscillator_run_has_11_elements_and_5_join_irreducibles():
    summary = json.loads(OSCILLATOR_RUN.read_text(encoding="utf-8"))
    lattice, selection = paper_grid_attractor_lattice(
        len(summary["morse_graph"]["nodes"]), summary["morse_graph"]["edges"], summary["conley"]
    )
    assert selection.shown == (0, 1, 2, 11, 21) and selection.blocked == ()
    assert lattice.order == ((11, 0), (11, 1), (21, 2), (21, 11))
    assert len(lattice.elements) == 11
    assert len(lattice.join_irreducibles) == 5
    assert [sorted(lattice.elements[element]) for element in lattice.join_irreducibles] == [
        [0],
        [1],
        [2],
        [0, 1, 11],
        [0, 1, 2, 11, 21],
    ]
    labels = attractor_lattice_labels(lattice)
    assert labels[lattice.bottom] == "0"
    assert labels[lattice.top] == "↓M(21)"
    assert labels[lattice.index([0, 1, 2, 11])] == "↓M(2) ∨ ↓M(11)"
    assert labels[lattice.index([0, 1, 2])] == "↓M(0) ∨ ↓M(1) ∨ ↓M(2)"
    assert {labels[element] for element in lattice.join_irreducibles} == {
        f"↓M({node})" for node in (0, 1, 2, 11, 21)
    }


def test_the_hasse_diagram_draws_every_element_and_cover():
    lattice = down_set_lattice(N_POSET, N_ORDER)
    labels = {element: f"e{element}" for element in range(len(lattice.elements))}
    labels[lattice.index([0, 1, 3])] = "↓M(0) ∨ ↓M(3)"
    colors = {lattice.principal(node): "#1f77b4" for node in N_POSET}
    dashed = {lattice.principal(3)}
    figure, axis = draw_lattice_hasse_diagram(lattice, labels=labels, colors=colors, dashed=dashed)
    try:
        ellipses = [patch for patch in axis.patches if isinstance(patch, patches.Ellipse)]
        assert len(ellipses) == len(lattice.elements)
        assert len(axis.lines) == len(lattice.covers)
        # The labels are recorded as given; the join symbol is drawn as math text.
        assert sorted(morse_graph_node_labels(axis)) == sorted(labels.values())
        centers = [ellipse.center for ellipse in ellipses]
        heights = [center[1] for center in centers]
        assert heights[lattice.bottom] == min(heights) and heights[lattice.top] == max(heights)
        for lower, upper in lattice.covers:
            assert heights[lower] < heights[upper]
        fills = [matplotlib.colors.to_hex(ellipse.get_facecolor()) for ellipse in ellipses]
        colored = [element for element, fill in enumerate(fills) if fill == "#1f77b4"]
        assert colored == sorted(colors)
        assert all(
            fill == LATTICE_ELEMENT_FILL
            for element, fill in enumerate(fills)
            if element not in colors
        )
        styles = [ellipse.get_linestyle() for ellipse in ellipses]
        assert [element for element, style in enumerate(styles) if style != "solid"] == sorted(
            dashed
        )
        # The figure has the size of the layout: one unit is one inch.
        layout = (
            float(axis.get_xlim()[1] - axis.get_xlim()[0]),
            float(axis.get_ylim()[1] - axis.get_ylim()[0]),
        )
        assert tuple(figure.get_size_inches()) == pytest.approx(layout)
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="no label"):
        draw_lattice_hasse_diagram(lattice, labels={0: "0"})


def test_the_attractor_lattice_figure_and_its_record(tmp_path):
    # Morse graph 3 -> 2 -> 1 -> 0 and 3 -> 0; node 1 is trivial and node 3
    # has no label, so the poset is 0 < 2 < 3 through the hidden node 1.
    edges = ((1, 0), (2, 1), (3, 2), (3, 0))
    conley = [
        _record(0, ("x-1", "0", "0", "0")),
        _record(1, ("0", "0", "0", "0")),
        _record(2, ("0", "x-1", "0", "0")),
        _record(3, (), computed=False),
    ]
    stem = tmp_path / "paper-grid-bouncing-ball-tau050-level2-corners"
    record = write_attractor_lattice_figure(4, edges, conley, output_stem=stem, dpi=60)
    assert [Path(path).name for path in record["files"]] == [
        f"{stem.name}-attractor-lattice.pdf",
        f"{stem.name}-attractor-lattice.png",
    ]
    assert all(Path(path).is_file() for path in record["files"])
    assert record["rule"] == ATTRACTOR_LATTICE_RULE
    assert record["morse_nodes"] == [0, 2, 3]
    assert record["morse_order"] == [[2, 0], [3, 2]]
    assert record["blocked_nodes"] == [3]
    assert record["element_count"] == 4
    labels = [entry["label"] for entry in record["elements"]]
    assert labels == ["0", "↓M(0)", "↓M(2)", "↓M(3)"]
    assert [entry["morse_nodes"] for entry in record["elements"]] == [[], [0], [0, 2], [0, 2, 3]]
    assert record["covers"] == [[0, 1], [1, 2], [2, 3]]
    assert record["join_irreducibles"] == [
        {"element": 1, "morse_node": 0, "label": "↓M(0)"},
        {"element": 2, "morse_node": 2, "label": "↓M(2)"},
        {"element": 3, "morse_node": 3, "label": "↓M(3)"},
    ]
    assert [entry["join_irreducible"] for entry in record["elements"]] == [False, True, True, True]
    json.dumps(record)

    # Nothing is drawn without an index record for every node, or when every
    # index is trivial.
    missing = write_attractor_lattice_figure(4, edges, conley[:3], output_stem=stem)
    assert missing["files"] == [] and "[3]" in missing["not_written"]
    trivial = [_record(node, ("0", "0", "0")) for node in range(2)]
    none = write_attractor_lattice_figure(2, ((1, 0),), trivial, output_stem=stem)
    assert none["files"] == [] and none["not_written"] == "every Morse node has a trivial index"
    many = [_record(node, ("x-1", "0", "0")) for node in range(10)]
    large = write_attractor_lattice_figure(10, (), many, output_stem=stem)
    assert large["files"] == [] and "more than 512 down-sets" in large["not_written"]
