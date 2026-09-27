"""Plot recurrent components of a fixed-time hybrid suspension relation.

``CMGDB.PlotMorseSets`` draws recurrent boxes in a projection of the state
space.  A suspension relation needs one extra piece of information: some
vertices are handle-phase cells and therefore do not live in the base state
space.  :func:`PlotHybridMorseSets` keeps those roles separate:

* base cells are drawn as projected grid boxes;
* reset endpoints are indicated schematically in the base projection; and
* the implicit phase cells are drawn in a dedicated ``s in [0, 1]`` panel.

The public function also dispatches native CMGDB Atlas results to the
chart-aware implementation in :mod:`atlas_morse_plot`.  That path reads
``morse_set_chart_boxes`` directly and leaves CMGDB in charge of SCCs and the
Morse order. Neither path certifies that a sampled relation is an outer
enclosure or that a Morse set is an index pair.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Collection, Hashable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from .sampled_suspension import BaseCell, PhaseCell

if TYPE_CHECKING:
    from .atlas_morse_plot import (
        AtlasFiniteRelationIndexAnnotations,
        AtlasHybridMorsePlot,
    )
    from .cmgdb_suspension_boxmap import SuspensionAtlasCharts


#: Second line of a Morse-graph node whose index computation was blocked.
BLOCKED_INDEX_LABEL = "blocked"

# The first entries agree with CMGDB.PlotMorseSets/PlotMorseGraph.  Keeping a
# local immutable copy avoids making hybrid_dynamics depend on an installed
# CMGDB Python extension merely to prepare a plot.
CMGDB_MORSE_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#e6550d",
    "#31a354",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#80b1d3",
    "#ffffb3",
    "#fccde5",
    "#b3de69",
    "#fdae6b",
    "#6a3d9a",
    "#c49c94",
    "#fb8072",
    "#dbdb8d",
    "#bc80bd",
    "#ffed6f",
    "#637939",
    "#c5b0d5",
    "#636363",
    "#c7c7c7",
    "#8dd3c7",
    "#b15928",
    "#e8cb32",
    "#9e9ac8",
    "#74c476",
    "#ff7f0e",
    "#9edae5",
    "#90d743",
    "#e7969c",
    "#17becf",
    "#7b4173",
    "#8ca252",
    "#ad494a",
    "#8c6d31",
    "#a55194",
    "#00cc49",
)

# Colorblind-safe palette used by the paper-ready latent-dynamics figures.  It
# remains opt-in so existing CMGDB node/color assignments are not silently
# changed for callers that rely on the historical palette.
SCIENTIFIC_MORSE_PALETTE: tuple[str, ...] = (
    "#FFB000",
    "#DC267F",
    "#648FFF",
    "#FE6100",
    "#785EF0",
    "#008080",
    "#FCC2E8",
)


_MODEL_TITLES = MappingProxyType(
    {
        "bouncing_ball_zeno_suspension": "Bouncing ball (Zeno suspension)",
        "rimless_wheel_walking_suspension": "Rimless wheel",
        "garcia_passive_walker_period_two_suspension": "Passive walker",
    },
)

_MODEL_STATE_LABELS = MappingProxyType(
    {
        "bouncing_ball_zeno_suspension": (r"$h$", r"$v$"),
        "rimless_wheel_walking_suspension": (
            r"$\theta$",
            r"$\dot\theta$",
        ),
        "garcia_passive_walker_period_two_suspension": (
            r"$\theta$",
            r"$\dot\theta$",
            r"$\phi$",
            r"$\dot\phi$",
        ),
    },
)


@dataclass(frozen=True)
class HybridMorseComponent:
    """Presentation data for one recurrent SCC/Morse set."""

    index: int
    label: str
    color: str
    nodes: frozenset[Hashable]
    base_cells: tuple[BaseCell, ...]
    phase_cells: tuple[PhaseCell, ...]
    cemetery_cells: tuple[Hashable, ...]

    @property
    def is_cemetery(self) -> bool:
        """Whether this component consists entirely of failure cells."""

        return bool(self.cemetery_cells) and len(self.cemetery_cells) == len(self.nodes)


@dataclass(frozen=True)
class HybridMorsePlot:
    """Handles and metadata returned by :func:`PlotHybridMorseSets`."""

    figure: Figure
    projection_axes: tuple[Axes, ...]
    handle_axis: Axes | None
    morse_graph_axis: Axes | None
    morse_graph: nx.DiGraph
    projections: tuple[tuple[int, int], ...]
    components: tuple[HybridMorseComponent, ...]


def _canonical_node_key(node: Hashable) -> tuple[object, ...]:
    if isinstance(node, BaseCell):
        return ("base", node.index)
    if isinstance(node, PhaseCell):
        return ("phase", node.guard_cell, node.slab)
    return (type(node).__qualname__, repr(node))


def _selected_indices(
    component_count: int,
    morse_nodes: Iterable[int] | None,
) -> tuple[int, ...]:
    if morse_nodes is None:
        return tuple(range(component_count))
    indices = tuple(int(index) for index in morse_nodes)
    if len(indices) != len(set(indices)):
        raise ValueError("morse_nodes must not contain duplicate indices")
    if any(index < 0 or index >= component_count for index in indices):
        raise IndexError("morse_nodes contains an out-of-range component index")
    return indices


def hybrid_morse_components(
    result: object,
    *,
    morse_nodes: Iterable[int] | None = None,
    palette: Sequence[str] = CMGDB_MORSE_PALETTE,
    cemetery_color: str = "#8f8f8f",
) -> tuple[HybridMorseComponent, ...]:
    """Extract deterministic component labels, colors, and cell kinds.

    Component indices are the indices already assigned by
    ``result.recurrent_sccs``.  Thus filtering does not renormalize colors:
    ``M(3)`` remains palette color 3 when it is plotted by itself, matching the
    convention used by CMGDB's plotting functions.
    """

    if not palette:
        raise ValueError("palette must contain at least one color")
    recurrent_sccs = tuple(getattr(result, "recurrent_sccs"))
    selected = _selected_indices(len(recurrent_sccs), morse_nodes)
    failure_cells = frozenset(
        getattr(
            getattr(result, "suspension_complex_ingredients"),
            "failure_cells",
            (),
        ),
    )
    components = []
    for index in selected:
        nodes = frozenset(recurrent_sccs[index])
        base_cells = tuple(
            sorted(
                (node for node in nodes if isinstance(node, BaseCell)),
                key=lambda cell: cell.index,
            ),
        )
        phase_cells = tuple(
            sorted(
                (node for node in nodes if isinstance(node, PhaseCell)),
                key=lambda cell: (cell.guard_cell, cell.slab),
            ),
        )
        cemetery_cells = tuple(
            sorted(
                (
                    node
                    for node in nodes
                    if node in failure_cells
                    or type(node).__qualname__ == "CemeteryCell"
                ),
                key=_canonical_node_key,
            ),
        )
        component = HybridMorseComponent(
            index=index,
            label=f"M({index})",
            color=str(palette[index % len(palette)]),
            nodes=nodes,
            base_cells=base_cells,
            phase_cells=phase_cells,
            cemetery_cells=cemetery_cells,
        )
        if component.is_cemetery:
            component = HybridMorseComponent(
                index=component.index,
                label=component.label,
                color=cemetery_color,
                nodes=component.nodes,
                base_cells=component.base_cells,
                phase_cells=component.phase_cells,
                cemetery_cells=component.cemetery_cells,
            )
        components.append(component)
    return tuple(components)


def _normalize_projections(
    dimension: int,
    proj_dims: Sequence[int] | Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    if dimension <= 0:
        raise ValueError("the base grid must have positive dimension")
    if proj_dims is None:
        if dimension == 1:
            return ((0, 0),)
        if dimension == 4:
            return ((0, 1), (2, 3))
        return ((0, 1),)

    raw = tuple(proj_dims)
    if len(raw) == 2 and all(isinstance(value, (int, np.integer)) for value in raw):
        normalized = ((int(raw[0]), int(raw[1])),)
    else:
        normalized = tuple(tuple(int(value) for value in pair) for pair in raw)
        if any(len(pair) != 2 for pair in normalized):
            raise ValueError("each projection must contain exactly two dimensions")
    if not normalized:
        raise ValueError("at least one projection is required")
    if any(min(pair) < 0 or max(pair) >= dimension for pair in normalized):
        raise IndexError("a projection dimension is outside the base grid")
    if dimension > 1 and any(first == second for first, second in normalized):
        raise ValueError("projection dimensions must be distinct")
    return normalized


def _box_bounds(
    index: int,
    bounds: np.ndarray,
    subdivisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(np.prod(subdivisions))
    if index < 0 or index >= total:
        raise IndexError(f"base-cell index {index} is outside the declared grid")
    coordinates = np.asarray(np.unravel_index(index, tuple(subdivisions)), dtype=int)
    widths = (bounds[:, 1] - bounds[:, 0]) / subdivisions
    lower = bounds[:, 0] + coordinates * widths
    return lower, lower + widths


def _component_lookup(
    components: Sequence[HybridMorseComponent],
) -> dict[Hashable, HybridMorseComponent]:
    return {
        node: component
        for component in components
        for node in component.nodes
    }


def _validate_recurrent_components(result: object) -> None:
    """Check the plotted SCCs against the materialized/reconstructed graph.

    Physical pipeline results additionally certify that expanding their
    base-only phase descriptors reproduces the materialized relation.  In that
    case this graph check also validates the plotted partition against the
    implicit reconstruction, without turning its transient SCCs into Morse
    sets.
    """

    if getattr(result, "descriptor_expansion_matches_relation", True) is False:
        raise ValueError(
            "phase-descriptor expansion does not match the materialized relation",
        )
    graph = getattr(result, "relation_graph")
    expected = set()
    for raw_component in nx.strongly_connected_components(graph):
        component = frozenset(raw_component)
        if len(component) > 1:
            expected.add(component)
            continue
        node = next(iter(component))
        if graph.has_edge(node, node):
            expected.add(component)
    supplied = {frozenset(component) for component in getattr(result, "recurrent_sccs")}
    if supplied != expected:
        raise ValueError(
            "recurrent_sccs does not equal the recurrent SCC partition of "
            "relation_graph",
        )


def hybrid_morse_hasse(
    result: object,
    components: Sequence[HybridMorseComponent] | None = None,
) -> nx.DiGraph:
    """Return the transitively reduced Morse order on displayed components.

    Reachability is computed in the full condensation graph, so a Morse-order
    relation may pass through transient SCCs.  This matches CMGDB's convention
    and avoids the common mistake of restricting the relation graph to its
    recurrent vertices before computing the order.
    """

    graph = getattr(result, "relation_graph")
    displayed = tuple(
        hybrid_morse_components(result) if components is None else components
    )
    order = nx.DiGraph()
    order.add_nodes_from(component.index for component in displayed)
    if len(displayed) < 2:
        return order

    condensation = nx.condensation(graph)
    node_to_condensation = condensation.graph["mapping"]
    condensation_nodes = {
        component.index: node_to_condensation[next(iter(component.nodes))]
        for component in displayed
    }
    for source in displayed:
        source_node = condensation_nodes[source.index]
        reachable = nx.descendants(condensation, source_node)
        for target in displayed:
            if (
                source.index != target.index
                and condensation_nodes[target.index] in reachable
            ):
                order.add_edge(source.index, target.index)
    return nx.transitive_reduction(order)


def _state_labels(result: object, dimension: int) -> tuple[str, ...]:
    model_name = str(getattr(result, "model_name", ""))
    configured = _MODEL_STATE_LABELS.get(model_name)
    if configured is not None and len(configured) == dimension:
        return configured
    return tuple(rf"$x_{{{index + 1}}}$" for index in range(dimension))


def _draw_projected_box(
    axis: Axes,
    lower: np.ndarray,
    upper: np.ndarray,
    projection: tuple[int, int],
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    zorder: float,
) -> None:
    first, second = projection
    if first == second:
        # CMGDB similarly introduces a harmless display coordinate for 1-D
        # boxes.  It has no dynamical meaning.
        rectangle = patches.Rectangle(
            (lower[first], 0.0),
            upper[first] - lower[first],
            1.0,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
    else:
        rectangle = patches.Rectangle(
            (lower[first], lower[second]),
            upper[first] - lower[first],
            upper[second] - lower[second],
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
    axis.add_patch(rectangle)


def _draw_base_projection(
    axis: Axes,
    result: object,
    projection: tuple[int, int],
    components: Sequence[HybridMorseComponent],
    bounds: np.ndarray,
    subdivisions: np.ndarray,
    state_labels: Sequence[str],
    *,
    show_transient: bool,
    show_reset_attachments: bool,
    base_view: str,
    show_title: bool,
    show_grid: bool,
) -> None:
    selected_lookup = _component_lookup(components)
    graph = getattr(result, "relation_graph")
    graph_base_cells = {
        node for node in graph.nodes if isinstance(node, BaseCell)
    }
    selected_base_cells = {
        cell for component in components for cell in component.base_cells
    }

    if show_transient:
        for cell in sorted(graph_base_cells - selected_base_cells, key=lambda item: item.index):
            lower, upper = _box_bounds(cell.index, bounds, subdivisions)
            _draw_projected_box(
                axis,
                lower,
                upper,
                projection,
                facecolor="#d9d9d9",
                edgecolor="#bdbdbd",
                alpha=0.28,
                linewidth=0.35,
                zorder=1,
            )

    for component in components:
        # Projection can identify distinct high-dimensional boxes.  Drawing a
        # projected rectangle once per component avoids opacity depending on
        # the number of hidden-coordinate cells.
        seen_bounds: set[tuple[float, ...]] = set()
        for cell in component.base_cells:
            lower, upper = _box_bounds(cell.index, bounds, subdivisions)
            key = (
                float(lower[projection[0]]),
                float(upper[projection[0]]),
                float(lower[projection[1]]),
                float(upper[projection[1]]),
            )
            if key in seen_bounds:
                continue
            seen_bounds.add(key)
            _draw_projected_box(
                axis,
                lower,
                upper,
                projection,
                facecolor=component.color,
                edgecolor="none",
                alpha=0.88,
                linewidth=0.0,
                zorder=3,
            )

    if show_reset_attachments:
        for handle in getattr(
            getattr(result, "suspension_complex_ingredients"),
            "handles",
            (),
        ):
            guard = np.asarray(handle.guard_state, dtype=float)
            reset = np.asarray(handle.reset_state, dtype=float)
            first, second = projection
            guard_xy = (guard[first], 0.5 if first == second else guard[second])
            reset_xy = (reset[first], 0.5 if first == second else reset[second])
            phase_components = {
                selected_lookup[cell]
                for cell in handle.phase_cells
                if cell in selected_lookup
            }
            handle_color = (
                next(iter(phase_components)).color
                if len(phase_components) == 1
                else "#525252"
            )
            axis.add_patch(
                patches.FancyArrowPatch(
                    guard_xy,
                    reset_xy,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.16",
                    mutation_scale=8,
                    linestyle=(0, (3, 2)),
                    linewidth=0.9,
                    color=handle_color,
                    alpha=0.72,
                    zorder=5,
                ),
            )
            axis.scatter(
                [guard_xy[0]],
                [guard_xy[1]],
                marker="^",
                s=24,
                facecolor="white",
                edgecolor=handle_color,
                linewidth=0.9,
                zorder=6,
            )
            axis.scatter(
                [reset_xy[0]],
                [reset_xy[1]],
                marker="o",
                s=20,
                facecolor=handle_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )

    first, second = projection
    visible_cells = (
        graph_base_cells if show_transient else selected_base_cells
    )
    if base_view == "support" and visible_cells:
        visible_bounds = [
            _box_bounds(cell.index, bounds, subdivisions)
            for cell in visible_cells
        ]
        x_min = min(lower[first] for lower, _ in visible_bounds)
        x_max = max(upper[first] for _, upper in visible_bounds)
        x_margin = max(0.08 * (x_max - x_min), 0.01 * np.ptp(bounds[first]))
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
    else:
        axis.set_xlim(bounds[first])
    if first == second:
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("display strip")
        axis.set_yticks([])
    else:
        if base_view == "support" and visible_cells:
            y_min = min(lower[second] for lower, _ in visible_bounds)
            y_max = max(upper[second] for _, upper in visible_bounds)
            y_margin = max(0.08 * (y_max - y_min), 0.01 * np.ptp(bounds[second]))
            axis.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            axis.set_ylim(bounds[second])
        axis.set_ylabel(state_labels[second])
    axis.set_xlabel(state_labels[first])
    if show_title:
        axis.set_title(f"base projection {state_labels[first]}, {state_labels[second]}")
    if show_grid:
        axis.grid(color="#eeeeee", linewidth=0.45, zorder=0)
    else:
        axis.grid(False)
    axis.set_axisbelow(True)


def _draw_dense_handle_panel(
    axis: Axes,
    result: object,
    components: Sequence[HybridMorseComponent],
    handles: Sequence[object],
    bounds: np.ndarray,
    subdivisions: np.ndarray,
    state_labels: Sequence[str],
    *,
    show_title: bool,
) -> None:
    """Draw many guard cells as a projected cylinder grid, not as lanes."""

    lookup = _component_lookup(components)
    guard_states = np.asarray([handle.guard_state for handle in handles], dtype=float)
    domain_widths = np.maximum(bounds[:, 1] - bounds[:, 0], np.finfo(float).eps)
    normalized_span = np.ptp(guard_states, axis=0) / domain_widths
    coordinate = int(np.argmax(normalized_span))
    slab_count = max(int(getattr(result, "handle_slabs", 0)), 1)

    displayed_y: list[tuple[float, float]] = []
    for component in components:
        rectangles: list[patches.Rectangle] = []
        for handle in handles:
            lower, upper = _box_bounds(
                handle.guard_cell.index,
                bounds,
                subdivisions,
            )
            for phase_cell in handle.phase_cells:
                if lookup.get(phase_cell) is not component:
                    continue
                rectangles.append(
                    patches.Rectangle(
                        (phase_cell.slab / slab_count, lower[coordinate]),
                        1.0 / slab_count,
                        upper[coordinate] - lower[coordinate],
                    )
                )
                displayed_y.append((float(lower[coordinate]), float(upper[coordinate])))
        if rectangles:
            collection = PatchCollection(
                rectangles,
                facecolor=component.color,
                edgecolor="none",
                alpha=0.9,
                rasterized=len(rectangles) > 1500,
                zorder=2,
            )
            axis.add_collection(collection)

    axis.set_xlim(0.0, 1.0)
    if displayed_y:
        y_min = min(lower for lower, _ in displayed_y)
        y_max = max(upper for _, upper in displayed_y)
        margin = max(0.04 * (y_max - y_min), 0.01 * domain_widths[coordinate])
        axis.set_ylim(y_min - margin, y_max + margin)
    else:
        axis.set_ylim(bounds[coordinate])
    axis.set_xlabel(r"$s$")
    axis.set_ylabel(state_labels[coordinate])
    axis.set_xticks((0.0, 0.5, 1.0))
    if show_title:
        axis.set_title("reset-handle cells")
    axis.grid(False)


def _draw_handle_panel(
    axis: Axes,
    result: object,
    components: Sequence[HybridMorseComponent],
    bounds: np.ndarray,
    subdivisions: np.ndarray,
    state_labels: Sequence[str],
    *,
    show_cemetery: bool,
    show_title: bool,
) -> None:
    lookup = _component_lookup(components)
    handles = tuple(
        getattr(
            getattr(result, "suspension_complex_ingredients"),
            "handles",
            (),
        ),
    )
    cemetery_components = tuple(
        component for component in components if component.cemetery_cells
    )
    if len(handles) > 12:
        _draw_dense_handle_panel(
            axis,
            result,
            components,
            handles,
            bounds,
            subdivisions,
            state_labels,
            show_title=show_title,
        )
        return
    row_count = len(handles) + (1 if show_cemetery and cemetery_components else 0)

    for row, handle in enumerate(handles):
        phase_cells = tuple(sorted(handle.phase_cells, key=lambda cell: cell.slab))
        slab_count = max(int(getattr(result, "handle_slabs", 0)), len(phase_cells), 1)
        by_slab = {cell.slab: cell for cell in phase_cells}
        for slab in range(slab_count):
            cell = by_slab.get(slab)
            component = lookup.get(cell)
            axis.add_patch(
                patches.Rectangle(
                    (slab / slab_count, row - 0.18),
                    1.0 / slab_count,
                    0.36,
                    facecolor=component.color if component else "#d9d9d9",
                    edgecolor="white",
                    linewidth=0.65,
                    alpha=0.84 if component else 0.45,
                    zorder=2,
                ),
            )
        axis.scatter(
            [0.0],
            [row],
            marker="^",
            s=30,
            facecolor="white",
            edgecolor="#303030",
            linewidth=0.9,
            zorder=4,
        )
        axis.scatter(
            [1.0],
            [row],
            marker="o",
            s=25,
            facecolor="#303030",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.text(
            -0.035,
            row,
            f"$H_{{{row}}}$",
            ha="right",
            va="center",
            fontsize=8,
        )

    if show_cemetery and cemetery_components:
        cemetery_row = len(handles)
        for offset, component in enumerate(cemetery_components):
            x_position = 0.5 + 0.12 * (
                offset - (len(cemetery_components) - 1) / 2.0
            )
            axis.scatter(
                [x_position],
                [cemetery_row],
                marker="X",
                s=65,
                facecolor=component.color,
                edgecolor="#202020",
                linewidth=0.8,
                zorder=5,
            )
            axis.annotate(
                component.label,
                (x_position, cemetery_row),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
            )
        axis.text(
            -0.035,
            cemetery_row,
            "cemetery",
            ha="right",
            va="center",
            fontsize=8,
        )

    axis.annotate(
        "",
        xy=(0.96, -0.48),
        xytext=(0.04, -0.48),
        arrowprops={"arrowstyle": "->", "color": "#525252", "linewidth": 0.8},
        annotation_clip=False,
    )
    axis.set_xlim(-0.08, 1.04)
    axis.set_ylim(-0.65, max(row_count - 0.35, 0.65))
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_xticklabels(("guard", r"$s$", "reset"))
    axis.set_yticks([])
    if show_title:
        axis.set_title("implicit reset handles")
    for spine_name in ("left", "right", "top"):
        axis.spines[spine_name].set_visible(False)
    axis.grid(False)


def _format_index_label(value: object) -> str:
    entries = (
        value
        if isinstance(value, Sequence) and not isinstance(value, str)
        else (value,)
    )
    return "(" + ", ".join(str(entry) for entry in entries) + ")"


def _morse_graph_positions(
    graph: nx.DiGraph,
    node_sizes: Mapping[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Place repellers above attractors using deterministic Hasse ranks.

    ``node_sizes`` is expressed in Graphviz inches.  Passing the presentation
    size into ``dot`` makes its rank spacing account for long Conley-index
    annotations rather than laying out every vertex as an unlabeled default
    node and allowing the rendered ellipses to overlap afterwards.
    """

    if not graph.nodes:
        return {}
    dot = shutil.which("dot")
    if dot is not None and len(graph.nodes) > 1:
        statements = [
            "digraph MorseOrder {",
            'graph [rankdir=TB, ranksep="0.30", nodesep="0.22", ordering=out];',
        ]
        statements.extend(
            (
                f'"{int(node)}" [shape=ellipse, fixedsize=true, '
                f'width="{node_sizes[int(node)][0]:.6g}", '
                f'height="{node_sizes[int(node)][1]:.6g}"];'
            )
            for node in sorted(graph.nodes)
        )
        statements.extend(
            f'"{int(source)}" -> "{int(target)}";'
            for source, target in sorted(graph.edges)
        )
        statements.append("}")
        try:
            completed = subprocess.run(
                (dot, "-Tplain"),
                input="\n".join(statements),
                text=True,
                capture_output=True,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            positions: dict[int, tuple[float, float]] = {}
            for line in completed.stdout.splitlines():
                fields = shlex.split(line)
                if len(fields) >= 4 and fields[0] == "node":
                    positions[int(fields[1])] = (float(fields[2]), float(fields[3]))
            if set(positions) == {int(node) for node in graph.nodes}:
                return positions

    rank: dict[int, int] = {int(node): 0 for node in graph.nodes}
    for node in nx.topological_sort(graph):
        for successor in graph.successors(node):
            rank[int(successor)] = max(rank[int(successor)], rank[int(node)] + 1)
    max_rank = max(rank.values(), default=0)
    layers: dict[int, list[int]] = {}
    for node, node_rank in rank.items():
        layers.setdefault(node_rank, []).append(node)
    positions: dict[int, tuple[float, float]] = {}
    layer_heights = {
        node_rank: max(node_sizes[node][1] for node in layer_nodes)
        for node_rank, layer_nodes in layers.items()
    }
    rank_y: dict[int, float] = {}
    current_y = 0.0
    for node_rank in range(max_rank, -1, -1):
        if node_rank not in layers:
            continue
        if rank_y:
            previous_rank = next(reversed(rank_y))
            current_y += (
                layer_heights[previous_rank] / 2.0
                + 0.50
                + layer_heights[node_rank] / 2.0
            )
        rank_y[node_rank] = current_y
    for node_rank, layer_nodes in sorted(layers.items()):
        ordered = sorted(layer_nodes)
        total_width = sum(node_sizes[node][0] for node in ordered)
        total_width += 0.35 * max(0, len(ordered) - 1)
        x_cursor = -total_width / 2.0
        x_positions = []
        for node in ordered:
            width = node_sizes[node][0]
            x_positions.append(x_cursor + width / 2.0)
            x_cursor += width + 0.35
        y_position = rank_y[node_rank]
        positions.update(
            (node, (float(x_position), y_position))
            for node, x_position in zip(ordered, x_positions)
        )
    return positions


MORSE_LABEL_LINE_SPACING = 1.2
"""Distance between label baselines, in multiples of the font size."""


def _morse_node_size(label: str, font_size: float) -> tuple[float, float]:
    """Return the ellipse size, in inches, that contains a node label.

    The label is measured at ``font_size`` points.  The ellipse has semi-axes
    ``sqrt(2)`` times the half-sides of the padded text box, so the corners of
    the box lie on the ellipse and the whole label lies inside it.
    """

    lines = label.splitlines() or [""]
    text_width = max(
        TextPath((0.0, 0.0), line, size=font_size).get_extents().width
        if line
        else 0.0
        for line in lines
    )
    text_height = len(lines) * MORSE_LABEL_LINE_SPACING * font_size
    padding = 0.35 * font_size
    width = np.sqrt(2.0) * (text_width + padding) / 72.0
    height = np.sqrt(2.0) * (text_height + padding) / 72.0
    return float(max(width, 0.76 * font_size / 7.2)), float(height)


class MorseNodeLabel(patches.PathPatch):
    """The label of one Morse-graph node, drawn as a path.

    ``text`` is the label as a string, one line per row of the drawing.
    """

    def __init__(self, path: MplPath, text: str, **kwargs: object) -> None:
        super().__init__(path, **kwargs)
        self.text = text


def morse_graph_node_labels(axis: Axes) -> tuple[str, ...]:
    """Return the node labels drawn on a Morse-graph axis."""

    return tuple(
        patch.text for patch in axis.patches if isinstance(patch, MorseNodeLabel)
    )


def _draw_morse_node_label(
    axis: Axes,
    center: tuple[float, float],
    label: str,
    font_size: float,
) -> None:
    """Draw a node label as a path in data coordinates.

    The graph layout is in inches, and the label is drawn at ``font_size``
    points in the same units, so it scales with its ellipse however large the
    graph and however small the axis.  A text artist would keep its point size
    and leave the ellipse when the axis shrinks the layout.
    """

    lines = label.splitlines()
    size = font_size / 72.0
    line_step = MORSE_LABEL_LINE_SPACING * size
    first_center = center[1] + line_step * (len(lines) - 1) / 2.0
    line_paths = []
    for row, line in enumerate(lines):
        if not line:
            continue
        path = TextPath((0.0, 0.0), line, size=size)
        extents = path.get_extents()
        baseline = first_center - row * line_step - 0.30 * size
        line_paths.append(
            path.transformed(
                Affine2D().translate(
                    center[0] - (extents.x0 + extents.width / 2.0),
                    baseline,
                )
            )
        )
    if not line_paths:
        return
    axis.add_patch(
        MorseNodeLabel(
            MplPath.make_compound_path(*line_paths),
            label,
            facecolor="#111111",
            edgecolor="none",
            zorder=3,
        ),
    )


def _ellipse_edge_point(
    center: tuple[float, float],
    toward: tuple[float, float],
    size: tuple[float, float],
) -> tuple[float, float]:
    """Intersect the center-to-center edge with an axis-aligned ellipse."""

    delta_x = float(toward[0] - center[0])
    delta_y = float(toward[1] - center[1])
    radius_x = max(float(size[0]) / 2.0, np.finfo(float).eps)
    radius_y = max(float(size[1]) / 2.0, np.finfo(float).eps)
    denominator = np.hypot(delta_x / radius_x, delta_y / radius_y)
    if denominator <= np.finfo(float).eps:
        return center
    return (
        float(center[0] + delta_x / denominator),
        float(center[1] + delta_y / denominator),
    )


def _draw_morse_graph(
    axis: Axes,
    graph: nx.DiGraph,
    components: Sequence[HybridMorseComponent],
    *,
    conley_indices: Mapping[int, object] | None,
    show_component_sizes: bool,
    show_title: bool,
    graph_title: str = "Conley--Morse graph",
    blocked_index_nodes: Collection[int] | Mapping[int, str] = (),
    font_size: float | None = None,
) -> None:
    """Draw a compact CMGDB-style Conley--Morse Hasse diagram.

    Nodes in ``blocked_index_nodes`` (an index computation was attempted and
    returned a blocker) have a dashed outline.  Their second line is
    ``blocked``, or the text given for the node when ``blocked_index_nodes``
    is a mapping.

    The layout is in inches, with labels of ``font_size`` points (by default
    7.2, 6.7, or 6.1 as the graph has up to 4, up to 10, or more nodes), so
    an axis of the size of its limits in inches shows the labels at that
    size.
    """

    by_index = {component.index: component for component in components}
    blocked_lines = (
        {int(node): str(text) for node, text in blocked_index_nodes.items()}
        if isinstance(blocked_index_nodes, Mapping)
        else {int(node): BLOCKED_INDEX_LABEL for node in blocked_index_nodes}
    )
    blocked = frozenset(blocked_lines)
    if conley_indices is not None:
        labeled_and_blocked = sorted(blocked.intersection(conley_indices))
        if labeled_and_blocked:
            raise ValueError(
                "Morse nodes cannot carry an index label and be blocked: "
                f"{labeled_and_blocked!r}"
            )
    node_count = len(graph.nodes)
    if font_size is not None:
        font_size = float(font_size)
    elif node_count <= 4:
        font_size = 7.2
    elif node_count <= 10:
        font_size = 6.7
    else:
        font_size = 6.1
    labels: dict[int, str] = {}
    for node in sorted(graph.nodes):
        component = by_index[int(node)]
        lines = [component.label]
        if conley_indices is not None and component.index in conley_indices:
            lines.append(_format_index_label(conley_indices[component.index]))
        elif component.index in blocked:
            lines.append(blocked_lines[component.index])
        elif show_component_sizes:
            lines.append(f"{len(component.nodes)} cells")
        labels[int(node)] = "\n".join(lines)
    node_sizes = {
        node: _morse_node_size(label, font_size)
        for node, label in labels.items()
    }
    positions = _morse_graph_positions(graph, node_sizes)
    for source, target in graph.edges:
        source_index = int(source)
        target_index = int(target)
        source_center = positions[source_index]
        target_center = positions[target_index]
        start = _ellipse_edge_point(
            source_center,
            target_center,
            node_sizes[source_index],
        )
        end = _ellipse_edge_point(
            target_center,
            source_center,
            node_sizes[target_index],
        )
        axis.add_patch(
            patches.FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color="#202020",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            ),
        )
    for node in sorted(graph.nodes):
        component = by_index[int(node)]
        x_position, y_position = positions[int(node)]
        width, height = node_sizes[int(node)]
        axis.add_patch(
            patches.Ellipse(
                (x_position, y_position),
                width=width,
                height=height,
                facecolor=component.color,
                edgecolor="#202020",
                linewidth=1.0 if component.index in blocked else 0.8,
                linestyle=(0, (2.2, 1.4)) if component.index in blocked else "solid",
                zorder=2,
            ),
        )
        _draw_morse_node_label(
            axis,
            (x_position, y_position),
            labels[int(node)],
            font_size,
        )

    if positions:
        x_lower = min(
            positions[node][0] - node_sizes[node][0] / 2.0
            for node in positions
        )
        x_upper = max(
            positions[node][0] + node_sizes[node][0] / 2.0
            for node in positions
        )
        y_lower = min(
            positions[node][1] - node_sizes[node][1] / 2.0
            for node in positions
        )
        y_upper = max(
            positions[node][1] + node_sizes[node][1] / 2.0
            for node in positions
        )
        axis.set_xlim(x_lower - 0.12, x_upper + 0.12)
        axis.set_ylim(y_lower - 0.12, y_upper + 0.12)
    else:
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(-0.5, 0.5)
    if show_title:
        axis.set_title(graph_title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _legend_handles(
    result: object,
    components: Sequence[HybridMorseComponent],
    *,
    show_transient: bool,
    show_reset_attachments: bool,
) -> list[patches.Patch]:
    handles: list[patches.Patch] = []
    for component in components:
        suffix = (
            "cemetery"
            if component.is_cemetery
            else f"{len(component.base_cells)} base, {len(component.phase_cells)} phase"
        )
        handles.append(
            patches.Patch(
                facecolor=component.color,
                edgecolor="#202020",
                label=f"{component.label}: {suffix}",
            ),
        )
    selected_nodes = {node for component in components for node in component.nodes}
    has_transient_base_support = any(
        isinstance(node, BaseCell) and node not in selected_nodes
        for node in getattr(result, "relation_graph").nodes
    )
    if show_transient and has_transient_base_support:
        handles.append(
            patches.Patch(
                facecolor="#d9d9d9",
                edgecolor="#bdbdbd",
                alpha=0.45,
                label="sampled transient support",
            ),
        )
    if show_reset_attachments:
        handles.append(
            patches.Patch(
                facecolor="none",
                edgecolor="#525252",
                linestyle=(0, (3, 2)),
                label="reset attachment (schematic)",
            ),
        )
    return handles


def PlotHybridMorseSets(
    result: object,
    morse_nodes: Iterable[int] | None = None,
    proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    *,
    clist: Sequence[str] | None = None,
    axis_labels: Sequence[str] | None = None,
    atlas_charts: "SuspensionAtlasCharts | None" = None,
    handle_proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    handle_axis_labels: Sequence[str] | None = None,
    show_transient: bool = False,
    show_handles: bool | None = None,
    show_morse_graph: bool = True,
    show_reset_attachments: bool = False,
    show_cemetery: bool = False,
    show_grid: bool = False,
    validate_components: bool = True,
    show_status_note: bool = False,
    show_legend: bool = False,
    show_panel_titles: bool = False,
    conley_indices: Mapping[int, object] | None = None,
    finite_relation_annotations: "AtlasFiniteRelationIndexAnnotations | None" = None,
    blocked_index_nodes: Collection[int] | Mapping[int, str] = (),
    show_component_sizes: bool = False,
    base_view: str = "support",
    title: str | None = None,
    fig_w: float | None = None,
    fig_h: float = 3.6,
    fig_fname: str | Path | None = None,
    dpi: int = 300,
) -> "HybridMorsePlot | AtlasHybridMorsePlot":
    """Plot recurrent SCCs in base projections plus implicit handle phase.

    Args:
        result: A legacy ``FixedTimeSuspensionPipelineResult``-compatible
            object, a native Atlas-backed CMGDB ``MorseGraph``, an acceptance
            object exposing one as ``.morse_graph``, or a cached
            ``AtlasMorsePlotData`` value. Native results also require
            ``atlas_charts``.
        morse_nodes: Optional recurrent-component indices to display.
        proj_dims: One pair, such as ``(0, 1)``, or several pairs, such as
            ``((0, 1), (2, 3))``.  Four-dimensional results default to the two
            canonical coordinate pairs; other dimensions default to ``(0, 1)``.
        clist: Stable component palette.  Defaults to CMGDB's palette.
        axis_labels: Optional labels for every base-state coordinate.
        show_transient: Draw sampled nonrecurrent base support in light gray.
            The default is false, matching ``CMGDB.PlotMorseSets`` semantics.
        atlas_charts: Chart coordinates for native CMGDB Atlas output. Omit
            this when ``result`` is cached ``AtlasMorsePlotData``.
        handle_proj_dims: Projection(s) for an Atlas handle chart.
        handle_axis_labels: Labels for intrinsic handle coordinates.
        show_handles: Add a handle panel. The default is true for the legacy
            relation and false for the native Atlas path.
        show_morse_graph: Add the transitively reduced Morse-order diagram.
        show_reset_attachments: Mark guard/reset attachments in base panels.
        show_cemetery: Mark recurrent compactification/failure cells.
        show_grid: Add a light Cartesian grid to base panels.
        validate_components: Check the supplied recurrent components against
            the materialized relation and, when present, its descriptor-expansion
            certificate.
        show_status_note: Add a small sampled-relation qualification below the
            panels.  It is omitted in paper figures whose caption carries the
            qualification.
        show_legend: Add the component/reset legend.  The default omits it
            because component identities are repeated in the Morse graph.
        show_panel_titles: Add descriptive titles above individual panels.
        conley_indices: Optional genuine degreewise Conley-index annotations
            keyed by component index.  A value such as ``("x-1", "0")`` is
            shown below ``M(i)`` as ``(x-1, 0)``.  No label is synthesized when
            this is absent.
        finite_relation_annotations: Certificate-backed degreewise shift classes
            for the finite sampled Atlas relation.  Load these from a physical
            nerve/carrier audit with
            ``load_atlas_finite_relation_index_annotations``.  They are labeled
            separately from a continuous-system Conley index.
        blocked_index_nodes: Morse nodes whose index computation returned a
            blocker instead of a label.  They are drawn with the line
            ``blocked`` and a dashed outline in the Morse graph.
        show_component_sizes: Put cell counts below graph node labels when no
            Conley-index annotation is supplied.
        base_view: ``"support"`` zooms to the displayed sampled base boxes,
            like ``CMGDB.PlotMorseSets``.  ``"domain"`` shows declared bounds.
        title: Optional figure title.  No title is added by default.
        fig_w, fig_h: Figure dimensions in inches.
        fig_fname: Optional one-file output path.
        dpi: Raster resolution and saved-figure resolution.

    Returns:
        A :class:`HybridMorsePlot` containing the figure, axes, and stable
        component presentation data.  The function never calls ``show()``.
    """

    # Native CMGDB Atlas results are routed before touching the legacy result
    # contract.  This path consumes the chart tags returned by
    # morse_set_chart_boxes and never reconstructs SCCs from a Python graph.
    from .atlas_morse_plot import AtlasMorsePlotData, plot_atlas_hybrid_morse_sets

    atlas_graph = getattr(result, "morse_graph", result)
    is_atlas_result = (
        isinstance(result, AtlasMorsePlotData)
        or atlas_charts is not None
        or callable(getattr(atlas_graph, "morse_set_chart_boxes", None))
    )
    if is_atlas_result:
        if show_transient:
            raise ValueError(
                "show_transient is unavailable for Atlas MorseGraph input: "
                "morse_set_chart_boxes contains recurrent boxes only"
            )
        if show_reset_attachments:
            raise ValueError(
                "show_reset_attachments is a legacy schematic and is not "
                "drawn over native Atlas boxes"
            )
        if show_cemetery:
            raise ValueError(
                "the current Atlas computation has no cemetery chart to plot"
            )
        if conley_indices is not None:
            raise ValueError(
                "Conley-index labels are unavailable for the current "
                "AtlasModel result"
            )
        return plot_atlas_hybrid_morse_sets(
            result,
            atlas_charts=atlas_charts,
            finite_relation_annotations=finite_relation_annotations,
            blocked_index_nodes=blocked_index_nodes,
            morse_nodes=morse_nodes,
            proj_dims=proj_dims,
            handle_proj_dims=handle_proj_dims,
            clist=CMGDB_MORSE_PALETTE if clist is None else clist,
            axis_labels=axis_labels,
            handle_axis_labels=handle_axis_labels,
            show_handles=False if show_handles is None else show_handles,
            show_morse_graph=show_morse_graph,
            show_grid=show_grid,
            show_status_note=show_status_note,
            show_legend=show_legend,
            show_panel_titles=show_panel_titles,
            show_component_sizes=show_component_sizes,
            base_view=base_view,
            title=title,
            fig_w=fig_w,
            fig_h=fig_h,
            fig_fname=fig_fname,
            dpi=dpi,
        )

    if finite_relation_annotations is not None:
        raise ValueError(
            "finite_relation_annotations are available only for native Atlas output"
        )

    if atlas_charts is not None or handle_proj_dims is not None or handle_axis_labels is not None:
        raise ValueError("Atlas-only plotting arguments require an Atlas result")
    if validate_components:
        _validate_recurrent_components(result)
    if base_view not in {"support", "domain"}:
        raise ValueError("base_view must be 'support' or 'domain'")
    ingredients = getattr(result, "suspension_complex_ingredients")
    bounds = np.asarray(ingredients.base_bounds, dtype=float)
    subdivisions = np.asarray(ingredients.base_subdivisions, dtype=int)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("base_bounds must have shape (dimension, 2)")
    if subdivisions.shape != (bounds.shape[0],) or np.any(subdivisions <= 0):
        raise ValueError("base_subdivisions must be positive and match base_bounds")
    projections = _normalize_projections(bounds.shape[0], proj_dims)
    labels = tuple(axis_labels) if axis_labels is not None else _state_labels(
        result,
        bounds.shape[0],
    )
    if len(labels) != bounds.shape[0]:
        raise ValueError("axis_labels must provide one label per base dimension")
    components = hybrid_morse_components(
        result,
        morse_nodes=morse_nodes,
        palette=CMGDB_MORSE_PALETTE if clist is None else clist,
    )
    if not show_cemetery:
        components = tuple(
            component for component in components if not component.is_cemetery
        )

    morse_graph = hybrid_morse_hasse(result, components)
    display_handles = (True if show_handles is None else show_handles) and any(
        component.phase_cells for component in components
    )
    panel_count = len(projections) + int(display_handles) + int(show_morse_graph)
    if fig_w is None:
        fig_w = 3.25 * panel_count
    width_ratios = (
        [1.0] * len(projections)
        + ([0.82] if display_handles else [])
        + (
            [
                1.35
                if len(components) > 15
                else (1.0 if len(components) > 8 else 0.62)
            ]
            if show_morse_graph
            else []
        )
    )
    figure, raw_axes = plt.subplots(
        1,
        panel_count,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = tuple(raw_axes[0])
    projection_axes = axes[: len(projections)]
    for axis, projection in zip(projection_axes, projections):
        _draw_base_projection(
            axis,
            result,
            projection,
            components,
            bounds,
            subdivisions,
            labels,
            show_transient=show_transient,
            show_reset_attachments=show_reset_attachments,
            base_view=base_view,
            show_title=show_panel_titles,
            show_grid=show_grid,
        )

    next_axis = len(projections)
    handle_axis = axes[next_axis] if display_handles else None
    if handle_axis is not None:
        _draw_handle_panel(
            handle_axis,
            result,
            components,
            bounds,
            subdivisions,
            labels,
            show_cemetery=show_cemetery,
            show_title=show_panel_titles,
        )
        next_axis += 1

    morse_graph_axis = axes[next_axis] if show_morse_graph else None
    if morse_graph_axis is not None:
        _draw_morse_graph(
            morse_graph_axis,
            morse_graph,
            components,
            conley_indices=conley_indices,
            show_component_sizes=show_component_sizes,
            show_title=show_panel_titles,
            blocked_index_nodes=blocked_index_nodes,
        )

    if title is not None:
        figure.suptitle(title)
    legend = _legend_handles(
        result,
        components,
        show_transient=show_transient,
        show_reset_attachments=show_reset_attachments,
    )
    if show_legend and legend:
        figure.legend(
            handles=legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=min(3, len(legend)),
            frameon=False,
            fontsize=7.5,
        )
    if show_status_note:
        figure.text(
            0.995,
            0.002,
            rf"sampled fixed-time relation, $t_*={float(getattr(result, 't_star')):g}$",
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#666666",
        )
    top = 0.82 if title is not None else 0.96
    bottom = 0.24 if show_legend else (0.15 if show_status_note else 0.13)
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=top,
        bottom=bottom,
        wspace=0.30,
    )

    plot = HybridMorsePlot(
        figure=figure,
        projection_axes=projection_axes,
        handle_axis=handle_axis,
        morse_graph_axis=morse_graph_axis,
        morse_graph=morse_graph,
        projections=projections,
        components=components,
    )
    if fig_fname is not None:
        output = Path(fig_fname)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return plot


def plot_hybrid_morse_sets(
    *args: object,
    **kwargs: object,
) -> "HybridMorsePlot | AtlasHybridMorsePlot":
    """PEP-8 alias for :func:`PlotHybridMorseSets`."""

    return PlotHybridMorseSets(*args, **kwargs)


def save_hybrid_morse_figure(
    plot: "HybridMorsePlot | AtlasHybridMorsePlot | Figure",
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Save matching paper-ready vector and raster versions of a plot.

    Both legacy and native Atlas plot results are accepted. ``output_stem``
    should normally have no suffix. Supported formats are PDF, SVG, and PNG.
    Directories are created as needed, and the exact output paths are returned
    for reproducible paper build scripts.
    """

    figure = getattr(plot, "figure", plot)
    if not isinstance(figure, Figure):
        raise TypeError("plot must expose a matplotlib Figure as .figure")
    normalized_formats = tuple(str(value).lower().lstrip(".") for value in formats)
    if not normalized_formats:
        raise ValueError("formats must contain at least one output format")
    if len(normalized_formats) != len(set(normalized_formats)):
        raise ValueError("formats must not contain duplicates")
    unsupported = set(normalized_formats) - {"pdf", "svg", "png"}
    if unsupported:
        raise ValueError(f"unsupported figure formats: {sorted(unsupported)!r}")

    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for format_name in normalized_formats:
        output = stem.with_suffix(f".{format_name}")
        figure.savefig(
            output,
            format=format_name,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(output)
    return tuple(outputs)


__all__ = [
    "CMGDB_MORSE_PALETTE",
    "SCIENTIFIC_MORSE_PALETTE",
    "HybridMorseComponent",
    "HybridMorsePlot",
    "hybrid_morse_components",
    "hybrid_morse_hasse",
    "PlotHybridMorseSets",
    "plot_hybrid_morse_sets",
    "save_hybrid_morse_figure",
]
