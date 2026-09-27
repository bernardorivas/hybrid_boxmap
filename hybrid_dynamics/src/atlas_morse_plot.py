"""Chart-aware plotting for native CMGDB suspension-Atlas Morse graphs.

The plotting input is the actual output of ``CMGDB.ComputeMorseGraph``: Morse
vertices, Morse-order edges, and the tagged rectangles returned by
``MorseGraph.morse_set_chart_boxes``.  No SCC decomposition is repeated here,
and no Conley-index annotation is inferred from a Morse set.

The compact :class:`AtlasMorsePlotData` representation is deliberately JSON
serializable.  An expensive physical box-map computation can therefore be run
once, audited separately, and its exact chart-tagged Morse boxes reused for
figures without resampling the dynamics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure

from .cmgdb_suspension_boxmap import SuspensionAtlasCharts


ATLAS_MORSE_PLOT_SCHEMA = "hybrid-atlas-morse-plot-v1"
PHYSICAL_CONLEY_FINITE_RELATION_AUDIT_SCHEMA = (
    "physical-conley-finite-relation-audit-v2"
)


@dataclass(frozen=True, order=True)
class AtlasMorseBox:
    """One closed, chart-tagged rectangle in a CMGDB Morse set."""

    chart_id: int
    bounds: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.bounds) // 2

    @property
    def lower(self) -> tuple[float, ...]:
        return self.bounds[: self.dimension]

    @property
    def upper(self) -> tuple[float, ...]:
        return self.bounds[self.dimension :]


@dataclass(frozen=True)
class AtlasMorseNode:
    """The exact tagged boxes reported for one CMGDB Morse vertex."""

    index: int
    boxes: tuple[AtlasMorseBox, ...]


@dataclass(frozen=True)
class AtlasMorsePlotData:
    """Portable plotting projection of an Atlas-backed CMGDB Morse graph."""

    base_chart_id: int
    handle_chart_id: int
    base_bounds: tuple[tuple[float, float], ...]
    handle_bounds: tuple[tuple[float, float], ...]
    nodes: tuple[AtlasMorseNode, ...]
    edges: tuple[tuple[int, int], ...]
    metadata: Mapping[str, object]

    @property
    def vertex_ids(self) -> tuple[int, ...]:
        return tuple(node.index for node in self.nodes)


@dataclass(frozen=True)
class AtlasFiniteRelationIndexAnnotations:
    """Certificate-backed finite-relation labels for an Atlas Morse graph.

    These labels are deliberately distinct from a Conley index certified for
    the continuous fixed-time suspension map.  Instances returned by
    :func:`load_atlas_finite_relation_index_annotations` have passed the
    reset-quotient nerve, relative-pair, acyclic-carrier, and chain-map gates
    recorded by the physical audit.
    """

    shift_classes: Mapping[int, tuple[str, ...]]
    coefficient_field: int
    result_scope: str
    audit_path: Path
    continuous_system_conley_index_certified: bool = False

    def __post_init__(self) -> None:
        normalized = {
            int(node): tuple(str(entry) for entry in entries)
            for node, entries in self.shift_classes.items()
        }
        if not normalized or any(not entries for entries in normalized.values()):
            raise ValueError("finite-relation annotations need a nonempty shift class")
        if int(self.coefficient_field) != 5:
            raise ValueError(
                "Atlas finite-relation annotations currently require GF(5)"
            )
        if self.result_scope != "finite_reset_quotient_relation":
            raise ValueError("unsupported finite-relation result scope")
        if self.continuous_system_conley_index_certified:
            raise ValueError(
                "finite sampled-relation annotations must not claim continuous-system "
                "certification"
            )
        object.__setattr__(self, "shift_classes", MappingProxyType(normalized))
        object.__setattr__(self, "coefficient_field", 5)
        object.__setattr__(self, "audit_path", Path(self.audit_path))


@dataclass(frozen=True)
class AtlasHybridMorseComponent:
    """Presentation data for one native CMGDB Atlas Morse set."""

    index: int
    label: str
    color: str
    boxes: tuple[AtlasMorseBox, ...]
    base_boxes: tuple[AtlasMorseBox, ...]
    handle_boxes: tuple[AtlasMorseBox, ...]

    @property
    def nodes(self) -> frozenset[AtlasMorseBox]:
        """Compatibility view used by the shared Morse-graph renderer."""

        return frozenset(self.boxes)


@dataclass(frozen=True)
class AtlasDetailZoom:
    """A zoom panel on Morse sets too small to see in their chart panel.

    ``morse_nodes`` are the Morse sets the zoom is drawn for; the zoom also
    shows, faded, the cells of every other drawn Morse set in its window
    ``x_limits x y_limits`` (chart coordinates of ``projection``).
    """

    label: str
    chart: str
    projection: tuple[int, int]
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    morse_nodes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "chart": self.chart,
            "projection": list(self.projection),
            "x_limits": list(self.x_limits),
            "y_limits": list(self.y_limits),
            "morse_nodes": list(self.morse_nodes),
        }


@dataclass(frozen=True)
class AtlasHybridMorsePlot:
    """Handles and metadata returned by the Atlas plotting path."""

    figure: Figure
    projection_axes: tuple[Axes, ...]
    handle_axes: tuple[Axes, ...]
    morse_graph_axis: Axes | None
    morse_graph: nx.DiGraph
    projections: tuple[tuple[int, int], ...]
    handle_projections: tuple[tuple[int, int], ...]
    components: tuple[AtlasHybridMorseComponent, ...]
    data: AtlasMorsePlotData
    zoom_axes: tuple[Axes, ...] = ()
    zooms: tuple[AtlasDetailZoom, ...] = ()

    @property
    def handle_axis(self) -> Axes | None:
        """First handle projection, for compatibility with legacy plots."""

        return self.handle_axes[0] if self.handle_axes else None


def _normalized_intervals(
    intervals: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(f"{name}[{index}] must contain lower and upper bounds")
        lower, upper = (float(interval[0]), float(interval[1]))
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{name}[{index}] must be finite")
        if lower > upper:
            raise ValueError(f"{name}[{index}] has lower bound greater than upper")
        result.append((lower, upper))
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return tuple(result)


def _normalized_box(
    chart_id: int,
    raw_bounds: Sequence[float],
    *,
    expected_dimension: int,
) -> AtlasMorseBox:
    bounds = tuple(float(value) for value in raw_bounds)
    if len(bounds) != 2 * expected_dimension:
        raise ValueError(
            f"chart {chart_id} box has {len(bounds)} bounds; "
            f"expected {2 * expected_dimension}"
        )
    if not all(np.isfinite(value) for value in bounds):
        raise ValueError(f"chart {chart_id} box has a non-finite bound")
    lower = bounds[:expected_dimension]
    upper = bounds[expected_dimension:]
    if any(left > right for left, right in zip(lower, upper)):
        raise ValueError(f"chart {chart_id} box has lower bound greater than upper")
    return AtlasMorseBox(chart_id=int(chart_id), bounds=bounds)


def extract_atlas_morse_plot_data(
    source: object,
    charts: SuspensionAtlasCharts,
    *,
    metadata: Mapping[str, object] | None = None,
) -> AtlasMorsePlotData:
    """Extract tagged Morse boxes and the CMGDB Morse order.

    ``source`` may be a native CMGDB ``MorseGraph`` or an acceptance object
    exposing it as ``source.morse_graph``.  The extraction calls
    ``morse_set_chart_boxes`` directly; ordinary untagged ``morse_set_boxes``
    are intentionally not accepted because they erase chart identity.
    """

    morse_graph = getattr(source, "morse_graph", source)
    required = ("vertices", "edges", "morse_set_chart_boxes")
    if any(not callable(getattr(morse_graph, name, None)) for name in required):
        raise TypeError(
            "source must be an Atlas-backed CMGDB MorseGraph (or expose one "
            "as .morse_graph)"
        )

    base_bounds = _normalized_intervals(charts.base_bounds, name="base_bounds")
    handle_bounds = _normalized_intervals(
        charts.handle_bounds,
        name="handle_bounds",
    )
    chart_dimensions = {
        int(charts.base_chart_id): len(base_bounds),
        int(charts.handle_chart_id): len(handle_bounds),
    }
    vertex_ids = tuple(sorted(int(vertex) for vertex in morse_graph.vertices()))
    if len(vertex_ids) != len(set(vertex_ids)):
        raise ValueError("CMGDB Morse graph contains duplicate vertex ids")

    nodes: list[AtlasMorseNode] = []
    for vertex in vertex_ids:
        boxes: list[AtlasMorseBox] = []
        for raw_chart_id, raw_bounds in morse_graph.morse_set_chart_boxes(vertex):
            chart_id = int(raw_chart_id)
            if chart_id not in chart_dimensions:
                raise ValueError(
                    f"Morse node {vertex} contains unknown Atlas chart {chart_id}"
                )
            boxes.append(
                _normalized_box(
                    chart_id,
                    raw_bounds,
                    expected_dimension=chart_dimensions[chart_id],
                )
            )
        boxes.sort()
        nodes.append(AtlasMorseNode(index=vertex, boxes=tuple(boxes)))

    vertex_set = set(vertex_ids)
    edges = tuple(
        sorted(
            {
                (int(source_vertex), int(target_vertex))
                for source_vertex, target_vertex in morse_graph.edges()
            }
        )
    )
    unknown_edge_vertices = {
        endpoint for edge in edges for endpoint in edge if endpoint not in vertex_set
    }
    if unknown_edge_vertices:
        raise ValueError(
            "CMGDB Morse edges reference unknown vertices: "
            f"{sorted(unknown_edge_vertices)!r}"
        )
    graph = nx.DiGraph()
    graph.add_nodes_from(vertex_ids)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("CMGDB Morse order must be acyclic")

    return AtlasMorsePlotData(
        base_chart_id=int(charts.base_chart_id),
        handle_chart_id=int(charts.handle_chart_id),
        base_bounds=base_bounds,
        handle_bounds=handle_bounds,
        nodes=tuple(nodes),
        edges=edges,
        metadata=dict(metadata or {}),
    )


def atlas_morse_plot_data_payload(data: AtlasMorsePlotData) -> dict[str, object]:
    """Return the stable JSON payload for ``data``."""

    return {
        "schema": ATLAS_MORSE_PLOT_SCHEMA,
        "provenance": (
            "Direct extraction from CMGDB MorseGraph.morse_set_chart_boxes; "
            "contains no independently inferred SCCs or Conley-index labels."
        ),
        "base_chart": {
            "id": data.base_chart_id,
            "bounds": [list(interval) for interval in data.base_bounds],
        },
        "handle_chart": {
            "id": data.handle_chart_id,
            "bounds": [list(interval) for interval in data.handle_bounds],
        },
        "nodes": [
            {
                "id": node.index,
                "boxes": [
                    {"chart_id": box.chart_id, "bounds": list(box.bounds)}
                    for box in node.boxes
                ],
            }
            for node in data.nodes
        ],
        "edges": [list(edge) for edge in data.edges],
        "metadata": dict(data.metadata),
    }


def save_atlas_morse_plot_data(
    data: AtlasMorsePlotData,
    output: str | Path,
) -> Path:
    """Save exact chart-tagged plotting data as readable JSON."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(atlas_morse_plot_data_payload(data), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def load_atlas_morse_plot_data(source: str | Path) -> AtlasMorsePlotData:
    """Load and validate an :class:`AtlasMorsePlotData` JSON cache."""

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ATLAS_MORSE_PLOT_SCHEMA:
        raise ValueError(
            f"{path} is not a supported Atlas Morse plotting cache "
            f"({payload.get('schema')!r})"
        )
    base_chart = payload["base_chart"]
    handle_chart = payload["handle_chart"]
    base_bounds = _normalized_intervals(base_chart["bounds"], name="base_bounds")
    handle_bounds = _normalized_intervals(
        handle_chart["bounds"],
        name="handle_bounds",
    )
    base_chart_id = int(base_chart["id"])
    handle_chart_id = int(handle_chart["id"])
    if base_chart_id == handle_chart_id:
        raise ValueError("base and handle chart ids must be distinct")
    chart_dimensions = {
        base_chart_id: len(base_bounds),
        handle_chart_id: len(handle_bounds),
    }

    nodes: list[AtlasMorseNode] = []
    for raw_node in payload["nodes"]:
        boxes = []
        for raw_box in raw_node["boxes"]:
            chart_id = int(raw_box["chart_id"])
            if chart_id not in chart_dimensions:
                raise ValueError(f"cache contains unknown Atlas chart {chart_id}")
            boxes.append(
                _normalized_box(
                    chart_id,
                    raw_box["bounds"],
                    expected_dimension=chart_dimensions[chart_id],
                )
            )
        boxes.sort()
        nodes.append(
            AtlasMorseNode(index=int(raw_node["id"]), boxes=tuple(boxes))
        )
    nodes.sort(key=lambda node: node.index)
    vertex_ids = tuple(node.index for node in nodes)
    if len(vertex_ids) != len(set(vertex_ids)):
        raise ValueError("cache contains duplicate Morse node ids")

    edges = tuple(
        sorted({(int(edge[0]), int(edge[1])) for edge in payload["edges"]})
    )
    vertex_set = set(vertex_ids)
    if any(endpoint not in vertex_set for edge in edges for endpoint in edge):
        raise ValueError("cache contains a Morse edge with an unknown endpoint")
    graph = nx.DiGraph()
    graph.add_nodes_from(vertex_ids)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("cached Morse order must be acyclic")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata must be a JSON object")
    return AtlasMorsePlotData(
        base_chart_id=base_chart_id,
        handle_chart_id=handle_chart_id,
        base_bounds=base_bounds,
        handle_bounds=handle_bounds,
        nodes=tuple(nodes),
        edges=edges,
        metadata=dict(metadata),
    )


def _require_certificate_flags(
    record: Mapping[str, object],
    names: Sequence[str],
    *,
    context: str,
) -> None:
    failed = [name for name in names if record.get(name) is not True]
    if failed:
        raise ValueError(f"{context} is missing true certificate flags {failed!r}")


def load_atlas_finite_relation_index_annotations(
    source: str | Path,
    data: AtlasMorsePlotData,
) -> AtlasFiniteRelationIndexAnnotations:
    """Load only fully gated finite sampled-relation shift classes.

    The audit is cross-checked against the plotting cache and every finite
    topology/algebra certificate used to justify an annotation.  Continuous
    fixed-time-map certification must remain false; these labels describe the
    stored finite reset-quotient relation only.
    """

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PHYSICAL_CONLEY_FINITE_RELATION_AUDIT_SCHEMA:
        raise ValueError(f"{path} is not a supported physical Conley audit")
    if int(payload.get("certified_continuous_system_indices", -1)) != 0:
        raise ValueError(
            "finite sampled-relation annotations require the audit to report zero "
            "certified continuous-system indices"
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("physical Conley audit contains no candidates")

    required_metadata = ("model", "depth", "t_star")
    missing_metadata = [key for key in required_metadata if key not in data.metadata]
    if missing_metadata:
        raise ValueError(
            "plotting cache lacks the provenance needed to match an audit: "
            f"{missing_metadata!r}"
        )
    expected_model = data.metadata["model"]
    expected_depth = data.metadata["depth"]
    expected_tau = data.metadata["t_star"]
    shift_classes: dict[int, tuple[str, ...]] = {}
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("physical Conley audit candidate must be an object")
        candidate = raw_candidate
        name = str(candidate.get("candidate", "unnamed candidate"))
        if candidate.get("model") != expected_model:
            raise ValueError(
                f"{name} model {candidate.get('model')!r} does not match plot "
                f"model {expected_model!r}"
            )
        if int(candidate.get("depth", -1)) != int(expected_depth):
            raise ValueError(f"{name} depth does not match the plotting cache")
        if not np.isclose(
            float(candidate.get("t_star", np.nan)),
            float(expected_tau),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{name} fixed time does not match the plotting cache")
        if candidate.get("finite_relation_blockers") != []:
            raise ValueError(f"{name} has unresolved finite-relation blockers")
        if candidate.get("continuous_system_conley_index_certified") is not False:
            raise ValueError(f"{name} must not claim continuous-system certification")
        if candidate.get("whole_cell_outer_enclosure_certified") is not False:
            raise ValueError(
                f"{name} is not the expected uncertified sampled-relation result"
            )
        if candidate.get("analytic_conley_label_attached") is not False:
            raise ValueError(f"{name} uses a prohibited analytic label fallback")

        top_pair = candidate.get("top_cell_pair")
        provenance = candidate.get("relation_provenance")
        nerve = candidate.get("quotient_nerve")
        pair = candidate.get("cellular_pair")
        carrier = candidate.get("carrier_certificate")
        finite = candidate.get("finite_relation_conley_index")
        records = (top_pair, provenance, nerve, pair, carrier, finite)
        if any(not isinstance(record, dict) for record in records):
            raise ValueError(f"{name} is missing a finite-relation certificate object")
        assert isinstance(top_pair, dict)
        assert isinstance(provenance, dict)
        assert isinstance(nerve, dict)
        assert isinstance(pair, dict)
        assert isinstance(carrier, dict)
        assert isinstance(finite, dict)

        node = int(top_pair.get("morse_node", -1))
        if node not in data.vertex_ids:
            raise ValueError(f"{name} refers to unknown plotted Morse node {node}")
        if node in shift_classes:
            raise ValueError(f"physical Conley audit repeats Morse node {node}")
        _require_certificate_flags(
            top_pair,
            (
                "pair_invariance_passed",
                "combinatorial_isolation_passed",
                "s_is_strongly_connected",
            ),
            context=f"{name} top-cell pair",
        )
        for list_name in (
            "first_condition_violations",
            "second_condition_violations",
            "recurrent_components_not_s_or_a",
            "empty_s_sources",
            "unevaluated_S_sources",
            "unevaluated_X_sources",
        ):
            if top_pair.get(list_name) != []:
                raise ValueError(f"{name} has nonempty {list_name}")
        if provenance.get("X_sources_evaluated") != provenance.get(
            "X_sources_total"
        ):
            raise ValueError(f"{name} did not evaluate every source in X")
        if int(provenance.get("unresolved_event_stage_edges", -1)) != 0:
            raise ValueError(f"{name} has unresolved event-stage edges")
        if provenance.get("original_exit_edges_retained") is not True:
            raise ValueError(f"{name} did not retain original exit edges")

        _require_certificate_flags(
            nerve,
            (
                "finite_intersections_verified_contractible",
                "boundary_squared_zero_validated",
                "vertices_are_actual_atlas_boxes",
            ),
            context=f"{name} quotient nerve",
        )
        if nerve.get("analytic_orbit_skeleton_used") is not False:
            raise ValueError(f"{name} quotient nerve uses an analytic orbit skeleton")
        _require_certificate_flags(
            pair,
            (
                "P0_subset_P1",
                "carrier_preserves_pair",
                "selected_chain_map_preserves_P1",
                "selected_chain_map_preserves_P0",
            ),
            context=f"{name} cellular pair",
        )
        _require_certificate_flags(
            carrier,
            (
                "all_carrier_values_acyclic",
                "face_nesting_validated",
                "selected_chain_map_subordinate",
                "chain_equation_dF_equals_Fd",
            ),
            context=f"{name} carrier",
        )
        if carrier.get("coefficient_field") != "GF(5)":
            raise ValueError(f"{name} carrier was not checked over GF(5)")
        if carrier.get("chain_equation_failure_witnesses") != []:
            raise ValueError(f"{name} records chain-equation failure witnesses")

        validation = finite.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"{name} lacks a CMGDB validation object")
        _require_certificate_flags(
            validation,
            (
                "matrix_shapes_and_entries",
                "boundary_squared_zero",
                "chain_map_equation",
            ),
            context=f"{name} CMGDB payload",
        )
        if int(finite.get("coefficient_field", -1)) != 5:
            raise ValueError(f"{name} CMGDB result is not over GF(5)")
        if finite.get("result_scope") != "finite_reset_quotient_relation":
            raise ValueError(f"{name} has an unsupported result scope")
        if finite.get("finite_relation_algebra_validated") is not True:
            raise ValueError(f"{name} lacks the finite-algebra validation flag")
        if finite.get("continuous_system_conley_index_certified") is not False:
            raise ValueError(f"{name} CMGDB result overstates continuous certification")
        if finite.get("cell_counts") != pair.get("relative_basis_by_dimension"):
            raise ValueError(f"{name} relative basis does not match the CMGDB payload")
        if nerve.get("cells_by_dimension") != pair.get("P1_cells_by_dimension"):
            raise ValueError(f"{name} quotient nerve does not match cellular P1")

        raw_shift_class = candidate.get("finite_relation_shift_class")
        if not isinstance(raw_shift_class, list) or not raw_shift_class:
            raise ValueError(f"{name} has no finite-relation shift class")
        if raw_shift_class != finite.get("shift_class"):
            raise ValueError(f"{name} summary and CMGDB shift classes disagree")
        shift_classes[node] = tuple(str(entry) for entry in raw_shift_class)

    if set(shift_classes) != set(data.vertex_ids):
        raise ValueError(
            "finite-relation audit does not annotate exactly the plotted Morse nodes: "
            f"audit={sorted(shift_classes)!r}, plot={list(data.vertex_ids)!r}"
        )
    if int(payload.get("computed_finite_relation_indices", -1)) != len(
        shift_classes
    ):
        raise ValueError(
            "physical Conley audit computed-count does not match candidates"
        )
    return AtlasFiniteRelationIndexAnnotations(
        shift_classes=shift_classes,
        coefficient_field=5,
        result_scope="finite_reset_quotient_relation",
        audit_path=path,
        continuous_system_conley_index_certified=False,
    )


def _selected_vertex_ids(
    data: AtlasMorsePlotData,
    morse_nodes: Iterable[int] | None,
) -> tuple[int, ...]:
    available = data.vertex_ids
    if morse_nodes is None:
        return available
    selected = tuple(int(node) for node in morse_nodes)
    if len(selected) != len(set(selected)):
        raise ValueError("morse_nodes must not contain duplicates")
    unknown = set(selected) - set(available)
    if unknown:
        raise IndexError(f"unknown CMGDB Morse nodes: {sorted(unknown)!r}")
    return selected


def atlas_morse_components(
    data: AtlasMorsePlotData,
    *,
    morse_nodes: Iterable[int] | None = None,
    palette: Sequence[str],
) -> tuple[AtlasHybridMorseComponent, ...]:
    """Prepare deterministic colors and chart partitions for selected nodes."""

    if not palette:
        raise ValueError("palette must contain at least one color")
    selected = _selected_vertex_ids(data, morse_nodes)
    by_index = {node.index: node for node in data.nodes}
    components = []
    for index in selected:
        boxes = by_index[index].boxes
        components.append(
            AtlasHybridMorseComponent(
                index=index,
                label=f"M({index})",
                color=str(palette[index % len(palette)]),
                boxes=boxes,
                base_boxes=tuple(
                    box for box in boxes if box.chart_id == data.base_chart_id
                ),
                handle_boxes=tuple(
                    box for box in boxes if box.chart_id == data.handle_chart_id
                ),
            )
        )
    return tuple(components)


def atlas_morse_hasse(
    data: AtlasMorsePlotData,
    selected: Iterable[int] | None = None,
) -> nx.DiGraph:
    """Restrict the CMGDB Morse order while preserving paths through hidden nodes."""

    selected_ids = _selected_vertex_ids(data, selected)
    full = nx.DiGraph()
    full.add_nodes_from(data.vertex_ids)
    full.add_edges_from(data.edges)
    result = nx.DiGraph()
    result.add_nodes_from(selected_ids)
    for source in selected_ids:
        reachable = nx.descendants(full, source)
        result.add_edges_from(
            (source, target)
            for target in selected_ids
            if target != source and target in reachable
        )
    return nx.transitive_reduction(result) if result.nodes else result


def _normalize_projections(
    dimension: int,
    projections: Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    default: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if projections is None:
        normalized = default
    else:
        raw = tuple(projections)
        if len(raw) == 2 and all(
            isinstance(value, (int, np.integer)) for value in raw
        ):
            normalized = ((int(raw[0]), int(raw[1])),)
        else:
            normalized = tuple(tuple(int(value) for value in pair) for pair in raw)
            if any(len(pair) != 2 for pair in normalized):
                raise ValueError("each projection must contain exactly two dimensions")
    if not normalized:
        raise ValueError("at least one projection is required")
    if any(min(pair) < 0 or max(pair) >= dimension for pair in normalized):
        raise IndexError("a projection dimension is outside its Atlas chart")
    if dimension > 1 and any(first == second for first, second in normalized):
        raise ValueError("projection dimensions must be distinct")
    return normalized


def _projected_bounds(
    boxes: Sequence[AtlasMorseBox],
    projection: tuple[int, int],
) -> np.ndarray:
    """Distinct projected rectangles ``(x0, y0, x1, y1)``, sorted, as an array."""

    first, second = projection
    unique: set[tuple[float, float, float, float]] = set()
    for box in boxes:
        if first == second:
            key = (box.lower[first], 0.0, box.upper[first], 1.0)
        else:
            key = (
                box.lower[first],
                box.lower[second],
                box.upper[first],
                box.upper[second],
            )
        unique.add(tuple(float(value) for value in key))
    return np.array(sorted(unique), dtype=float).reshape(-1, 4)


def _rectangles(bounds: np.ndarray) -> tuple[patches.Rectangle, ...]:
    return tuple(
        patches.Rectangle((x_lower, y_lower), x_upper - x_lower, y_upper - y_lower)
        for x_lower, y_lower, x_upper, y_upper in bounds.tolist()
    )


def _support_limits(
    boxes: Sequence[AtlasMorseBox],
    projection: tuple[int, int],
    chart_bounds: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    first, second = projection
    if not boxes:
        return tuple(chart_bounds[first]), tuple(chart_bounds[second])
    x_lower = min(box.lower[first] for box in boxes)
    x_upper = max(box.upper[first] for box in boxes)
    if first == second:
        return _padded_interval(x_lower, x_upper, chart_bounds[first]), (0.0, 1.0)
    y_lower = min(box.lower[second] for box in boxes)
    y_upper = max(box.upper[second] for box in boxes)
    return (
        _padded_interval(x_lower, x_upper, chart_bounds[first]),
        _padded_interval(y_lower, y_upper, chart_bounds[second]),
    )


def _padded_interval(
    lower: float,
    upper: float,
    domain: Sequence[float],
) -> tuple[float, float]:
    span = max(float(upper - lower), np.finfo(float).eps)
    domain_span = max(float(domain[1] - domain[0]), np.finfo(float).eps)
    margin = max(0.06 * span, 0.008 * domain_span)
    return float(lower - margin), float(upper + margin)


def _chart_boxes(
    component: AtlasHybridMorseComponent,
    chart_kind: str,
) -> tuple[AtlasMorseBox, ...]:
    return component.base_boxes if chart_kind == "base" else component.handle_boxes


def _chart_limits(
    components: Sequence[AtlasHybridMorseComponent],
    *,
    chart_kind: str,
    projection: tuple[int, int],
    chart_bounds: Sequence[Sequence[float]],
    view: str,
    frame_margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits of a chart panel.

    ``"domain"`` shows the chart bounds widened by ``frame_margin`` times
    their span on each side, so cells on the boundary of the chart are not
    drawn under the axis lines; ``"support"`` pads the drawn cells.
    """

    first, second = projection
    if view == "domain":
        x_limits = tuple(chart_bounds[first])
        y_limits = (0.0, 1.0) if first == second else tuple(chart_bounds[second])
        if frame_margin > 0.0:
            x_limits, y_limits = (
                (lower - frame_margin * (upper - lower), upper + frame_margin * (upper - lower))
                for lower, upper in (x_limits, y_limits)
            )
        return (float(x_limits[0]), float(x_limits[1])), (
            float(y_limits[0]),
            float(y_limits[1]),
        )
    visible = [box for component in components for box in _chart_boxes(component, chart_kind)]
    return _support_limits(visible, projection, chart_bounds)


# Every cell is drawn at its true extent.  The cells of a Morse set too small
# to see in its chart panel (see DETAIL_ZOOM_MIN_AREA) are also outlined, in
# that panel and in its zooms, by a line of CELL_OUTLINE_WIDTH points in the
# color of the set, so that a cell a fraction of a point wide is still seen.
# The outline is the fill color over white, drawn opaque, so the outlines of
# neighboring cells do not darken where they overlap.  Larger sets are not
# outlined: the line would widen a thin curve of small cells.
CELL_OUTLINE_WIDTH = 0.5
# Opacity of the cells, and of the faded cells of the other sets in a zoom.
CELL_ALPHA = 0.9
FADED_CELL_ALPHA = 0.3


def _cells_in_window(
    bounds: np.ndarray,
    window: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    (x_lower, x_upper), (y_lower, y_upper) = window
    return bounds[
        (bounds[:, 2] >= x_lower)
        & (bounds[:, 0] <= x_upper)
        & (bounds[:, 3] >= y_lower)
        & (bounds[:, 1] <= y_upper)
    ]


def _draw_cells(
    axis: Axes,
    components: Sequence[AtlasHybridMorseComponent],
    *,
    chart_kind: str,
    projection: tuple[int, int],
    window: tuple[tuple[float, float], tuple[float, float]] | None = None,
    emphasized: Iterable[int] | None = None,
    outlined: Iterable[int] = (),
) -> None:
    """Draw the cells of each component in its color, at their true extent.

    With ``window``, only the cells meeting it are drawn.  With
    ``emphasized``, the other components are drawn faded, below the
    emphasized ones.  The cells of the components in ``outlined`` are also
    outlined (:data:`CELL_OUTLINE_WIDTH`).
    """

    from matplotlib.colors import to_rgb

    emphasized_nodes = None if emphasized is None else frozenset(emphasized)
    outlined_nodes = frozenset(int(node) for node in outlined)
    for component in components:
        bounds = _projected_bounds(_chart_boxes(component, chart_kind), projection)
        if window is not None and len(bounds):
            bounds = _cells_in_window(bounds, window)
        if not len(bounds):
            continue
        faded = emphasized_nodes is not None and component.index not in emphasized_nodes
        alpha = FADED_CELL_ALPHA if faded else CELL_ALPHA
        zorder = 1.8 if faded else 2.0
        rasterized = len(bounds) > 2500
        axis.add_collection(
            PatchCollection(
                _rectangles(bounds),
                facecolor=component.color,
                edgecolor="none",
                linewidth=0.0,
                alpha=alpha,
                antialiased=False,
                rasterized=rasterized,
                zorder=zorder,
            )
        )
        if component.index not in outlined_nodes:
            continue
        outline = tuple(1.0 - alpha * (1.0 - value) for value in to_rgb(component.color))
        axis.add_collection(
            PatchCollection(
                _rectangles(bounds),
                facecolor="none",
                edgecolor=outline,
                linewidth=CELL_OUTLINE_WIDTH,
                antialiased=False,
                rasterized=rasterized,
                zorder=zorder + 0.1,
            )
        )


# A Morse set is too small to see in its chart panel when its cells cover less
# than DETAIL_ZOOM_MIN_AREA bins of the DETAIL_ZOOM_BINS x DETAIL_ZOOM_BINS
# bins of the panel (area counted at most once per bin).  At the paper-figure
# size a bin is about one point, so such a set is a speck, a hairline a few
# points long, or cells scattered below the resolution of the panel.
DETAIL_ZOOM_BINS = 200
DETAIL_ZOOM_MIN_AREA = 16.0
# A zoom window spans at most this fraction of its chart panel in each
# direction.  In the paper figures a zoom panel is a quarter to 0.4 of the
# chart panel across, so it magnifies both axes at least about twice (with
# three zooms in a column, 2.3 times in x and 2.1 in y).  A small set whose
# own window would be larger (cells spread along a curve, say) gets no zoom
# and is left to the panel, where its cells are outlined; groups of small
# sets are not merged into a larger window.
DETAIL_ZOOM_MAX_WINDOW = 0.125
# Small sets closer than this (fraction of the panel) share a zoom panel.
DETAIL_ZOOM_GROUP_GAP = 0.06
# A zoom magnifies its two axes by factors that differ at most by this ratio.
DETAIL_ZOOM_MAX_ASPECT = 3.0
# At most this many zooms per chart panel, with windows that do not overlap.
DETAIL_ZOOM_MAX_PER_PANEL = 3


def _bin_coverage(
    bounds: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    bins: int,
) -> np.ndarray:
    """Area fraction of each of ``bins x bins`` panel bins covered by ``bounds``."""

    coverage = np.zeros((bins, bins), dtype=float)
    if not len(bounds):
        return coverage
    x_scale = bins / float(x_limits[1] - x_limits[0])
    y_scale = bins / float(y_limits[1] - y_limits[0])
    x0 = np.clip((bounds[:, 0] - x_limits[0]) * x_scale, 0.0, bins)
    x1 = np.clip((bounds[:, 2] - x_limits[0]) * x_scale, 0.0, bins)
    y0 = np.clip((bounds[:, 1] - y_limits[0]) * y_scale, 0.0, bins)
    y1 = np.clip((bounds[:, 3] - y_limits[0]) * y_scale, 0.0, bins)
    keep = (x1 > x0) & (y1 > y0)
    x0, x1, y0, y1 = x0[keep], x1[keep], y0[keep], y1[keep]
    i0 = np.floor(x0).astype(np.int64)
    j0 = np.floor(y0).astype(np.int64)
    ni = np.ceil(x1).astype(np.int64) - i0
    nj = np.ceil(y1).astype(np.int64) - j0
    small = (ni <= 4) & (nj <= 4)
    for di in range(4):
        for dj in range(4):
            selected = small & (di < ni) & (dj < nj)
            if not np.any(selected):
                continue
            i = i0[selected] + di
            j = j0[selected] + dj
            overlap_x = np.minimum(x1[selected], i + 1) - np.maximum(x0[selected], i)
            overlap_y = np.minimum(y1[selected], j + 1) - np.maximum(y0[selected], j)
            np.add.at(coverage, (i, j), overlap_x * overlap_y)
    for index in np.flatnonzero(~small):
        columns = np.arange(i0[index], i0[index] + ni[index])
        rows = np.arange(j0[index], j0[index] + nj[index])
        overlap_x = np.minimum(x1[index], columns + 1) - np.maximum(x0[index], columns)
        overlap_y = np.minimum(y1[index], rows + 1) - np.maximum(y0[index], rows)
        coverage[columns[0] : columns[-1] + 1, rows[0] : rows[-1] + 1] += np.outer(
            overlap_x, overlap_y
        )
    return coverage


def _too_small_to_see(
    bounds: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> bool:
    """Whether cells ``bounds`` cover less than DETAIL_ZOOM_MIN_AREA panel bins."""

    coverage = _bin_coverage(bounds, x_limits, y_limits, DETAIL_ZOOM_BINS)
    return float(np.minimum(coverage, 1.0).sum()) < DETAIL_ZOOM_MIN_AREA


def _small_set_nodes(
    components: Sequence[AtlasHybridMorseComponent],
    *,
    chart_kind: str,
    projection: tuple[int, int],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[int, ...]:
    """The Morse sets with cells in a chart panel that are too small to see there."""

    nodes = []
    for component in components:
        bounds = _projected_bounds(_chart_boxes(component, chart_kind), projection)
        if len(bounds) and _too_small_to_see(bounds, x_limits, y_limits):
            nodes.append(int(component.index))
    return tuple(nodes)


def _zoom_window_size(size: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Size of the zoom window of cells with bounding box ``size`` and cell ``cell``.

    Sizes are panel fractions, ``(..., 2)`` arrays.  The box is padded by 30%
    of its size or three cells on each side, whichever is more, and the
    window is widened so that the two axes are magnified by factors whose
    ratio is at most :data:`DETAIL_ZOOM_MAX_ASPECT`.
    """

    padded = size + 2.0 * np.maximum(0.3 * size, 3.0 * cell)
    x = np.maximum(padded[..., 0], padded[..., 1] / DETAIL_ZOOM_MAX_ASPECT)
    y = np.maximum(padded[..., 1], padded[..., 0] / DETAIL_ZOOM_MAX_ASPECT)
    return np.minimum(np.stack((x, y), axis=-1), 1.0)


def _detail_zooms(
    components: Sequence[AtlasHybridMorseComponent],
    *,
    chart_kind: str,
    projection: tuple[int, int],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> list[tuple[tuple[int, ...], tuple[float, float], tuple[float, float]]]:
    """Zoom windows ``(nodes, x_limits, y_limits)`` for the sets too small to see.

    Each window spans at most :data:`DETAIL_ZOOM_MAX_WINDOW` of the panel in
    each direction.  A small set whose own window would be larger gets no
    zoom; its cells are outlined in the panel.  Groups of small sets are
    merged, closest first, while they are closer than
    :data:`DETAIL_ZOOM_GROUP_GAP` or there are more than
    :data:`DETAIL_ZOOM_MAX_PER_PANEL` of them, but only into a window within
    the limit.  Zoom windows do not overlap, and there are at most
    :data:`DETAIL_ZOOM_MAX_PER_PANEL` of them; the groups with the most
    sets are chosen first.  A set of a group left out is drawn in a zoom
    whose window contains it, and otherwise only outlined in the panel.
    """

    if projection[0] == projection[1]:
        return []
    x_span = float(x_limits[1] - x_limits[0])
    y_span = float(y_limits[1] - y_limits[0])
    # Each small set: node, bounding box and median cell size in panel fractions.
    small_boxes: dict[int, np.ndarray] = {}
    nodes: list[list[int]] = []
    boxes: list[np.ndarray] = []
    cells: list[list[np.ndarray]] = []
    for component in components:
        bounds = _projected_bounds(_chart_boxes(component, chart_kind), projection)
        if not len(bounds) or not _too_small_to_see(bounds, x_limits, y_limits):
            continue
        fractions = np.column_stack(
            (
                (bounds[:, 0] - x_limits[0]) / x_span,
                (bounds[:, 1] - y_limits[0]) / y_span,
                (bounds[:, 2] - x_limits[0]) / x_span,
                (bounds[:, 3] - y_limits[0]) / y_span,
            )
        )
        box = np.array(
            (
                fractions[:, 0].min(),
                fractions[:, 1].min(),
                fractions[:, 2].max(),
                fractions[:, 3].max(),
            )
        )
        cell = np.median(fractions[:, 2:] - fractions[:, :2], axis=0)
        if np.max(_zoom_window_size(box[2:] - box[:2], cell)) > DETAIL_ZOOM_MAX_WINDOW:
            # Cells along a curve, or specks spread over the panel.
            continue
        small_boxes[int(component.index)] = box
        nodes.append([int(component.index)])
        boxes.append(box)
        cells.append([cell])
    if not nodes:
        return []

    def window_cell(group_cells: list[np.ndarray]) -> np.ndarray:
        return np.median(np.array(group_cells), axis=0)

    while len(nodes) > 1:
        box_array = np.array(boxes)
        # Merging is checked with the larger cell of the two groups, so the
        # window drawn, padded by the median cell, is no larger.
        cell_array = np.array([np.max(np.array(group), axis=0) for group in cells])
        lower = np.minimum(box_array[:, None, :2], box_array[None, :, :2])
        upper = np.maximum(box_array[:, None, 2:], box_array[None, :, 2:])
        merged = _zoom_window_size(
            upper - lower, np.maximum(cell_array[:, None, :], cell_array[None, :, :])
        )
        gaps = np.maximum.reduce(
            (
                np.zeros((len(nodes), len(nodes))),
                box_array[:, None, 0] - box_array[None, :, 2],
                box_array[None, :, 0] - box_array[:, None, 2],
                box_array[:, None, 1] - box_array[None, :, 3],
                box_array[None, :, 1] - box_array[:, None, 3],
            )
        )
        allowed = np.triu(merged.max(axis=-1) <= DETAIL_ZOOM_MAX_WINDOW, k=1)
        if len(nodes) <= DETAIL_ZOOM_MAX_PER_PANEL:
            allowed &= gaps < DETAIL_ZOOM_GROUP_GAP
        if not allowed.any():
            break
        # Closest pair first; ties go to the first pair in order.
        a, b = np.unravel_index(np.argmin(np.where(allowed, gaps, np.inf)), gaps.shape)
        a, b = int(a), int(b)
        nodes[a] = nodes[a] + nodes[b]
        boxes[a] = np.concatenate((lower[a, b], upper[a, b]))
        cells[a] = cells[a] + cells[b]
        del nodes[b], boxes[b], cells[b]

    # Each group: nodes and window (lower and upper corners, panel fractions).
    groups = []
    for group_nodes, box, group_cells in zip(nodes, boxes, cells):
        padded = _zoom_window_size(box[2:] - box[:2], window_cell(group_cells))
        center = np.clip(0.5 * (box[:2] + box[2:]), padded / 2.0, 1.0 - padded / 2.0)
        groups.append((sorted(group_nodes), center - padded / 2.0, center + padded / 2.0))
    # The groups with the most sets first, then the smaller windows; a window
    # that overlaps one already kept is dropped.
    groups.sort(key=lambda group: (-len(group[0]), float(np.prod(group[2] - group[1]))))
    kept: list[tuple[list[int], np.ndarray, np.ndarray]] = []
    dropped: list[int] = []
    for group in groups:
        if len(kept) < DETAIL_ZOOM_MAX_PER_PANEL and not any(
            np.all(group[1] < other[2]) and np.all(other[1] < group[2]) for other in kept
        ):
            kept.append(group)
        else:
            dropped.extend(group[0])
    # A set of a dropped group that lies in a kept window is drawn in that
    # zoom with the sets it is for.
    for node in dropped:
        box = small_boxes[node]
        for group_nodes, lower, upper in kept:
            if np.all(lower <= box[:2]) and np.all(box[2:] <= upper):
                group_nodes.append(node)
                break
    windows = [
        (
            tuple(sorted(group_nodes)),
            (
                float(x_limits[0] + lower[0] * x_span),
                float(x_limits[0] + upper[0] * x_span),
            ),
            (
                float(y_limits[0] + lower[1] * y_span),
                float(y_limits[0] + upper[1] * y_span),
            ),
        )
        for group_nodes, lower, upper in kept
    ]
    # Left to right, then top to bottom.
    windows.sort(key=lambda window: (sum(window[1]), -sum(window[2])))
    return windows


@dataclass(frozen=True)
class _ZoomStyle:
    """Ticks, lines, and labels of a zoom panel (sizes in points).

    ``axis_label_size`` is the size of the axis labels, or ``None`` for a
    zoom panel with no axis labels.
    """

    ticks: int
    tick_label_size: float
    tick_length: float
    tick_width: float
    tick_pad: float
    spine_width: float
    label_size: float
    axis_label_size: float | None = None


# A zoom panel in the column next to its chart panel, whose axis labels it
# shares, and a zoom panel drawn as its own figure (larger, so its tick labels
# can be larger too), which names its coordinates as the chart panel does.
ZOOM_COLUMN_STYLE = _ZoomStyle(
    ticks=2,
    tick_label_size=5.5,
    tick_length=2.0,
    tick_width=0.5,
    tick_pad=1.5,
    spine_width=0.6,
    label_size=7.5,
)
ZOOM_FIGURE_STYLE = _ZoomStyle(
    ticks=3,
    tick_label_size=8.0,
    tick_length=3.0,
    tick_width=0.6,
    tick_pad=2.0,
    spine_width=0.8,
    label_size=9.0,
    axis_label_size=9.0,
)


def _draw_zoom_panel(
    zoom_axis: Axes,
    zoom: AtlasDetailZoom,
    components: Sequence[AtlasHybridMorseComponent],
    outlined: Iterable[int] = (),
    style: _ZoomStyle = ZOOM_COLUMN_STYLE,
    labels: Sequence[str] = (),
) -> None:
    """Draw the cells in the window of a zoom, with its label as title.

    ``outlined`` are the Morse sets too small to see in the chart panel.
    With a ``style`` that has axis labels, the axes are labeled with the
    ``labels`` of the chart coordinates, as in the chart panel.
    """

    from matplotlib.ticker import MaxNLocator

    _draw_cells(
        zoom_axis,
        components,
        chart_kind=zoom.chart,
        projection=zoom.projection,
        window=(zoom.x_limits, zoom.y_limits),
        emphasized=zoom.morse_nodes,
        outlined=outlined,
    )
    zoom_axis.set_xlim(zoom.x_limits)
    zoom_axis.set_ylim(zoom.y_limits)
    zoom_axis.set_box_aspect(1.0)
    zoom_axis.set_anchor("W")
    zoom_axis.xaxis.set_major_locator(MaxNLocator(nbins=style.ticks, min_n_ticks=2))
    zoom_axis.yaxis.set_major_locator(MaxNLocator(nbins=style.ticks, min_n_ticks=2))
    zoom_axis.tick_params(
        labelsize=style.tick_label_size,
        length=style.tick_length,
        pad=style.tick_pad,
        width=style.tick_width,
    )
    for spine in zoom_axis.spines.values():
        spine.set_linewidth(style.spine_width)
    zoom_axis.set_title(zoom.label, loc="left", fontsize=style.label_size, pad=2.0)
    if style.axis_label_size is not None:
        # A zoom is drawn only for a projection on two coordinates.
        first, second = zoom.projection
        zoom_axis.set_xlabel(labels[first], fontsize=style.axis_label_size, labelpad=2.0)
        zoom_axis.set_ylabel(labels[second], fontsize=style.axis_label_size, labelpad=2.0)


def _mark_zoom_window(axis: Axes, zoom: AtlasDetailZoom) -> None:
    """Outline the window of a zoom on its chart panel ``axis``, with its label."""

    from matplotlib import patheffects

    (x_lower, x_upper), (y_lower, y_upper) = zoom.x_limits, zoom.y_limits
    axis.add_patch(
        patches.Rectangle(
            (x_lower, y_lower),
            x_upper - x_lower,
            y_upper - y_lower,
            fill=False,
            edgecolor="#202020",
            linewidth=0.6,
            zorder=4,
        )
    )
    panel_x = axis.get_xlim()
    panel_y = axis.get_ylim()
    right_side = (x_upper - panel_x[0]) / (panel_x[1] - panel_x[0]) < 0.85
    upper_side = (y_upper - panel_y[0]) / (panel_y[1] - panel_y[0]) < 0.9
    axis.annotate(
        zoom.label,
        xy=(x_upper if right_side else x_lower, y_upper if upper_side else y_lower),
        xytext=(2.0 if right_side else -2.0, 1.0 if upper_side else -1.0),
        textcoords="offset points",
        ha="left" if right_side else "right",
        va="bottom" if upper_side else "top",
        fontsize=7.5,
        color="#111111",
        zorder=5,
        path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")],
    )


def _draw_chart_projection(
    axis: Axes,
    components: Sequence[AtlasHybridMorseComponent],
    *,
    chart_kind: str,
    projection: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    labels: Sequence[str],
    show_grid: bool,
    show_panel_title: bool,
    spines_below_cells: bool = False,
    outlined: Iterable[int] = (),
) -> None:
    _draw_cells(
        axis,
        components,
        chart_kind=chart_kind,
        projection=projection,
        outlined=outlined,
    )
    first, second = projection
    x_limits, y_limits = limits
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    if spines_below_cells:
        for spine in axis.spines.values():
            spine.set_zorder(1.5)
    axis.set_xlabel(labels[first])
    if first == second:
        axis.set_ylabel("display strip")
        axis.set_yticks([])
    else:
        axis.set_ylabel(labels[second])
    if show_panel_title:
        axis.set_title("base chart" if chart_kind == "base" else "handle chart")
    if show_grid:
        axis.grid(color="#eeeeee", linewidth=0.45, zorder=0)
        axis.set_axisbelow(True)
    else:
        axis.grid(False)


def _status_text(data: AtlasMorsePlotData) -> str:
    scope = str(data.metadata.get("relation_scope", "sampled Atlas relation"))
    tau = data.metadata.get("t_star")
    if tau is None:
        return scope
    return rf"{scope}, $\tau={float(tau):g}$"


@dataclass(frozen=True)
class _ChartPanel:
    """One chart panel: projection, limits, zooms, and the sets too small to see."""

    chart_kind: str
    projection: tuple[int, int]
    limits: tuple[tuple[float, float], tuple[float, float]]
    labels: tuple[str, ...]
    zooms: tuple[AtlasDetailZoom, ...]
    small: tuple[int, ...]


@dataclass(frozen=True)
class _AtlasPlotLayout:
    """What the combined figure and the panel figures of an Atlas plot share."""

    data: AtlasMorsePlotData
    components: tuple[AtlasHybridMorseComponent, ...]
    order: nx.DiGraph
    projections: tuple[tuple[int, int], ...]
    handle_projections: tuple[tuple[int, int], ...]
    panels: tuple[_ChartPanel, ...]
    blocked_nodes: Mapping[int, str] | frozenset[int]

    @property
    def zooms(self) -> tuple[AtlasDetailZoom, ...]:
        return tuple(zoom for panel in self.panels for zoom in panel.zooms)


def _atlas_plot_layout(
    source: AtlasMorsePlotData | object,
    *,
    atlas_charts: SuspensionAtlasCharts | None,
    finite_relation_annotations: AtlasFiniteRelationIndexAnnotations | None,
    blocked_index_nodes: Iterable[int] | Mapping[int, str],
    morse_nodes: Iterable[int] | None,
    proj_dims: Sequence[int] | Sequence[Sequence[int]] | None,
    handle_proj_dims: Sequence[int] | Sequence[Sequence[int]] | None,
    clist: Sequence[str],
    axis_labels: Sequence[str] | None,
    handle_axis_labels: Sequence[str] | None,
    show_handles: bool,
    base_view: str,
    handle_view: str | None,
    frame_margin: float,
    detail_zooms: bool,
) -> _AtlasPlotLayout:
    """Check the options of an Atlas plot; find its chart panels and zooms."""

    if isinstance(source, AtlasMorsePlotData):
        if atlas_charts is not None:
            raise ValueError("atlas_charts must be omitted for cached plot data")
        data = source
    else:
        if atlas_charts is None:
            raise ValueError("atlas_charts is required for a native CMGDB result")
        metadata: dict[str, object] = {}
        if hasattr(source, "summary") and callable(getattr(source, "summary")):
            summary = source.summary()
            for key in ("relation_scope", "t_star", "depth", "empty_images"):
                if key in summary:
                    metadata[key] = summary[key]
        data = extract_atlas_morse_plot_data(source, atlas_charts, metadata=metadata)

    if base_view not in {"support", "domain"}:
        raise ValueError("base_view must be 'support' or 'domain'")
    if handle_view is None:
        handle_view = base_view
    if handle_view not in {"support", "domain"}:
        raise ValueError("handle_view must be 'support' or 'domain'")
    frame_margin = float(frame_margin)
    if not (np.isfinite(frame_margin) and 0.0 <= frame_margin < 0.5):
        raise ValueError("frame_margin must be in [0, 0.5)")
    base_dimension = len(data.base_bounds)
    handle_dimension = len(data.handle_bounds)
    base_default = ((0, 0),) if base_dimension == 1 else (
        ((0, 1), (2, 3)) if base_dimension == 4 else ((0, 1),)
    )
    projections = _normalize_projections(
        base_dimension,
        proj_dims,
        default=base_default,
    )
    handle_default = (
        ((0, 0),)
        if handle_dimension == 1
        else ((0, handle_dimension - 1),)
    )
    handle_projections = (
        _normalize_projections(
            handle_dimension,
            handle_proj_dims,
            default=handle_default,
        )
        if show_handles
        else ()
    )
    base_labels = (
        tuple(str(label) for label in axis_labels)
        if axis_labels is not None
        else tuple(rf"$x_{{{index + 1}}}$" for index in range(base_dimension))
    )
    if len(base_labels) != base_dimension:
        raise ValueError("axis_labels must provide one label per base coordinate")
    handle_labels = (
        tuple(str(label) for label in handle_axis_labels)
        if handle_axis_labels is not None
        else tuple(
            [rf"$g_{{{index + 1}}}$" for index in range(handle_dimension - 1)]
            + [r"$s$"]
        )
    )
    if len(handle_labels) != handle_dimension:
        raise ValueError(
            "handle_axis_labels must provide one label per handle coordinate"
        )

    components = atlas_morse_components(
        data,
        morse_nodes=morse_nodes,
        palette=clist,
    )
    if finite_relation_annotations is not None:
        unknown_annotations = set(finite_relation_annotations.shift_classes).difference(
            data.vertex_ids
        )
        if unknown_annotations:
            raise ValueError(
                "finite-relation annotations refer to unknown Morse nodes: "
                f"{sorted(unknown_annotations)!r}"
            )
    blocked_nodes = (
        {int(node): str(text) for node, text in blocked_index_nodes.items()}
        if isinstance(blocked_index_nodes, Mapping)
        else frozenset(int(node) for node in blocked_index_nodes)
    )
    unknown_blocked = set(blocked_nodes).difference(data.vertex_ids)
    if unknown_blocked:
        raise ValueError(
            f"blocked index nodes are not Morse nodes: {sorted(unknown_blocked)!r}"
        )
    order = atlas_morse_hasse(data, (component.index for component in components))

    # Chart panels in drawing order, each with its limits and zoom windows;
    # the zooms are labeled A, B, ... across the panels.
    zoom_labels = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    panels = []
    for chart_kind, chart_projections, chart_bounds, chart_labels, view in (
        ("base", projections, data.base_bounds, base_labels, base_view),
        ("handle", handle_projections, data.handle_bounds, handle_labels, handle_view),
    ):
        for projection in chart_projections:
            limits = _chart_limits(
                components,
                chart_kind=chart_kind,
                projection=projection,
                chart_bounds=chart_bounds,
                view=view,
                frame_margin=frame_margin,
            )
            windows = (
                _detail_zooms(
                    components,
                    chart_kind=chart_kind,
                    projection=projection,
                    x_limits=limits[0],
                    y_limits=limits[1],
                )
                if detail_zooms
                else []
            )
            small = _small_set_nodes(
                components,
                chart_kind=chart_kind,
                projection=projection,
                x_limits=limits[0],
                y_limits=limits[1],
            )
            zooms = tuple(
                AtlasDetailZoom(
                    label=next(zoom_labels),
                    chart=chart_kind,
                    projection=projection,
                    x_limits=x_window,
                    y_limits=y_window,
                    morse_nodes=nodes,
                )
                for nodes, x_window, y_window in windows
            )
            panels.append(
                _ChartPanel(chart_kind, projection, limits, chart_labels, zooms, small)
            )
    return _AtlasPlotLayout(
        data=data,
        components=components,
        order=order,
        projections=projections,
        handle_projections=handle_projections,
        panels=tuple(panels),
        blocked_nodes=blocked_nodes,
    )


def plot_atlas_hybrid_morse_sets(
    source: AtlasMorsePlotData | object,
    *,
    atlas_charts: SuspensionAtlasCharts | None = None,
    finite_relation_annotations: AtlasFiniteRelationIndexAnnotations | None = None,
    blocked_index_nodes: Iterable[int] | Mapping[int, str] = (),
    morse_nodes: Iterable[int] | None = None,
    proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    handle_proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    clist: Sequence[str],
    axis_labels: Sequence[str] | None = None,
    handle_axis_labels: Sequence[str] | None = None,
    show_handles: bool = False,
    show_morse_graph: bool = True,
    show_grid: bool = False,
    show_status_note: bool = False,
    show_legend: bool = False,
    show_panel_titles: bool = False,
    show_component_sizes: bool = False,
    base_view: str = "support",
    handle_view: str | None = None,
    frame_margin: float = 0.0,
    detail_zooms: bool = False,
    title: str | None = None,
    fig_w: float | None = None,
    fig_h: float = 3.6,
    fig_fname: str | Path | None = None,
    dpi: int = 300,
) -> AtlasHybridMorsePlot:
    """Plot actual CMGDB Atlas Morse boxes and the CMGDB Morse order.

    ``morse_nodes`` selects the drawn nodes; they keep their numbers and
    colors, and the drawn order is reachability through the hidden nodes,
    transitively reduced.  Nodes in ``blocked_index_nodes`` are marked as
    blocked in the Morse graph; with a mapping, the text given for a node
    replaces the word ``blocked``.

    ``base_view`` and ``handle_view`` (default: ``base_view``) are
    ``"support"`` or ``"domain"``.  ``frame_margin`` widens the ``"domain"``
    view by that fraction of the chart span on each side and draws the axis
    lines below the cells, so cells on the chart boundary stay visible.  With
    ``detail_zooms``, the Morse sets that are too small to see in a chart
    panel (see :data:`DETAIL_ZOOM_MIN_AREA`) get zoom panels, labeled ``A``,
    ``B``, ... in a column next to the panel, whose windows are outlined and
    labeled in the panel.  A window spans at most
    :data:`DETAIL_ZOOM_MAX_WINDOW` of the panel in each direction, so every
    zoom magnifies; a small set too spread for such a window gets no zoom.
    Every cell is drawn at its true extent; the cells
    of a set too small to see in its chart panel are also outlined in the
    color of the set (:data:`CELL_OUTLINE_WIDTH`), in the panel and in its
    zooms, so a cell smaller than a point is still seen.
    :func:`plot_atlas_hybrid_morse_panels` draws each panel as its own figure.
    """

    # Import at call time so the public dispatcher in hybrid_morse_plot can
    # route here without a module-import cycle.
    from .hybrid_morse_plot import _draw_morse_graph

    layout = _atlas_plot_layout(
        source,
        atlas_charts=atlas_charts,
        finite_relation_annotations=finite_relation_annotations,
        blocked_index_nodes=blocked_index_nodes,
        morse_nodes=morse_nodes,
        proj_dims=proj_dims,
        handle_proj_dims=handle_proj_dims,
        clist=clist,
        axis_labels=axis_labels,
        handle_axis_labels=handle_axis_labels,
        show_handles=show_handles,
        base_view=base_view,
        handle_view=handle_view,
        frame_margin=frame_margin,
        detail_zooms=detail_zooms,
    )
    data = layout.data
    components = layout.components

    panel_count = len(layout.panels) + int(show_morse_graph)
    if panel_count <= 0:
        raise ValueError("at least one plot panel must be enabled")
    zoom_columns = sum(1 for panel in layout.panels if panel.zooms)
    zoom_ratio = 0.40
    if fig_w is None:
        fig_w = 3.25 * panel_count + 3.25 * zoom_ratio * zoom_columns
    width_ratios: list[float] = []
    for panel in layout.panels:
        width_ratios.append(1.0 if panel.chart_kind == "base" else 0.92)
        if panel.zooms:
            width_ratios.append(zoom_ratio)
    if show_morse_graph:
        width_ratios.append(
            1.0
            if finite_relation_annotations is not None
            else (0.70 if len(components) <= 4 else 1.0)
        )
    figure = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    grid = figure.add_gridspec(1, len(width_ratios), width_ratios=width_ratios)
    column = 0
    chart_axes: list[Axes] = []
    zoom_axes: list[Axes] = []
    for panel in layout.panels:
        axis = figure.add_subplot(grid[0, column])
        column += 1
        chart_axes.append(axis)
        _draw_chart_projection(
            axis,
            components,
            chart_kind=panel.chart_kind,
            projection=panel.projection,
            limits=panel.limits,
            labels=panel.labels,
            show_grid=show_grid,
            show_panel_title=show_panel_titles,
            spines_below_cells=float(frame_margin) > 0.0,
            outlined=panel.small,
        )
        if panel.zooms:
            rows = grid[0, column].subgridspec(len(panel.zooms), 1, hspace=0.45)
            column += 1
            for row, zoom in enumerate(panel.zooms):
                zoom_axis = figure.add_subplot(rows[row, 0])
                _draw_zoom_panel(zoom_axis, zoom, components, outlined=panel.small)
                _mark_zoom_window(axis, zoom)
                zoom_axes.append(zoom_axis)
    projection_axes = tuple(chart_axes[: len(layout.projections)])
    handle_axes = tuple(chart_axes[len(layout.projections) :])

    graph_axis = figure.add_subplot(grid[0, column]) if show_morse_graph else None
    if graph_axis is not None:
        _draw_morse_graph(
            graph_axis,
            layout.order,
            components,
            conley_indices=(
                None
                if finite_relation_annotations is None
                else finite_relation_annotations.shift_classes
            ),
            show_component_sizes=show_component_sizes,
            show_title=show_panel_titles,
            graph_title="Morse graph",
            blocked_index_nodes=layout.blocked_nodes,
        )

    if title is not None:
        figure.suptitle(title)
    if show_legend:
        legend = [
            patches.Patch(
                facecolor=component.color,
                edgecolor="#202020",
                label=(
                    f"{component.label}: {len(component.base_boxes)} base, "
                    f"{len(component.handle_boxes)} handle"
                ),
            )
            for component in components
        ]
        if legend:
            figure.legend(
                handles=legend,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.025),
                ncol=min(3, len(legend)),
                frameon=False,
                fontsize=7.5,
            )
    if show_status_note and finite_relation_annotations is not None:
        figure.text(
            0.995,
            0.002,
            (
                r"finite sampled-relation Conley index over $\mathbb{F}_5$; "
                "continuous-system certification not established"
            ),
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
    elif show_status_note:
        figure.text(
            0.995,
            0.002,
            _status_text(data),
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#666666",
        )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.82 if title is not None else 0.96,
        bottom=0.24 if show_legend else (0.18 if show_status_note else 0.15),
        wspace=0.32,
    )

    plot = AtlasHybridMorsePlot(
        figure=figure,
        projection_axes=projection_axes,
        handle_axes=handle_axes,
        morse_graph_axis=graph_axis,
        morse_graph=layout.order,
        projections=layout.projections,
        handle_projections=layout.handle_projections,
        components=components,
        data=data,
        zoom_axes=tuple(zoom_axes),
        zooms=layout.zooms,
    )
    if fig_fname is not None:
        output = Path(fig_fname)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return plot


# Sizes, in inches, of the axes of a panel drawn as its own figure.  A chart
# panel keeps about its size in the combined figure (3.4 inches high, with a
# zoom column and a Morse graph next to it); the handle chart is a little
# narrower, as there.  A zoom panel is square and about 0.6 times as wide as
# the chart panel, so its tick labels can be 8 points instead of 5.5.
PANEL_CHART_SIZE = (2.6, 2.75)
PANEL_HANDLE_SIZE = (2.4, 2.75)
PANEL_ZOOM_SIZE = (1.6, 1.6)
# Space around the axes (left, bottom, right, top), in inches, for the tick
# labels, the axis labels, and the letter of a zoom.  The files are saved
# with a tight bounding box, so unused space is cut.
PANEL_CHART_MARGINS = (0.75, 0.6, 0.15, 0.15)
PANEL_ZOOM_MARGINS = (0.8, 0.55, 0.15, 0.3)
# Font size, in points, of the node labels of the Morse graph drawn as its own
# figure.  The figure is the size of the graph layout in inches, so the labels
# print at this size when the figure is shown at its natural size.
PANEL_GRAPH_FONT_SIZE = 7.0


@dataclass(frozen=True)
class AtlasMorsePanelFigures:
    """Each panel of an Atlas Morse plot drawn as its own figure.

    ``figures`` and ``axes`` map a panel name to its figure and axes, in the
    order of the combined figure: ``base`` (``base-1``, ``base-2``, ... when
    there are several base projections), ``zoom-A``, ``zoom-B``, ... for the
    zooms of that chart, ``handle`` (named like ``base``) and its zooms, and
    ``graph``.
    """

    figures: Mapping[str, Figure]
    axes: Mapping[str, Axes]
    morse_graph: nx.DiGraph
    components: tuple[AtlasHybridMorseComponent, ...]
    data: AtlasMorsePlotData
    zooms: tuple[AtlasDetailZoom, ...] = ()

    def close(self) -> None:
        """Close every figure."""

        for figure in self.figures.values():
            plt.close(figure)


def _figure_with_axes(
    size: tuple[float, float],
    margins: tuple[float, float, float, float],
    dpi: int,
) -> tuple[Figure, Axes]:
    """A figure with one axes of ``size`` inches and ``margins`` inches around it."""

    left, bottom, right, top = margins
    width = left + size[0] + right
    height = bottom + size[1] + top
    figure = plt.figure(figsize=(width, height), dpi=dpi)
    axis = figure.add_axes(
        (left / width, bottom / height, size[0] / width, size[1] / height)
    )
    return figure, axis


def plot_atlas_hybrid_morse_panels(
    source: AtlasMorsePlotData | object,
    *,
    atlas_charts: SuspensionAtlasCharts | None = None,
    finite_relation_annotations: AtlasFiniteRelationIndexAnnotations | None = None,
    blocked_index_nodes: Iterable[int] | Mapping[int, str] = (),
    morse_nodes: Iterable[int] | None = None,
    proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    handle_proj_dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
    clist: Sequence[str],
    axis_labels: Sequence[str] | None = None,
    handle_axis_labels: Sequence[str] | None = None,
    show_handles: bool = False,
    show_morse_graph: bool = True,
    show_grid: bool = False,
    show_component_sizes: bool = False,
    base_view: str = "support",
    handle_view: str | None = None,
    frame_margin: float = 0.0,
    detail_zooms: bool = False,
    graph_font_size: float = PANEL_GRAPH_FONT_SIZE,
    dpi: int = 300,
) -> AtlasMorsePanelFigures:
    """Draw each panel of :func:`plot_atlas_hybrid_morse_sets` as its own figure.

    The options are those of :func:`plot_atlas_hybrid_morse_sets`, and each
    panel has the same colors, cells, limits, and zoom windows as there: a
    chart panel outlines the window of each of its zooms, with its letter,
    and a zoom panel has its letter as title.  A chart panel has axes of
    :data:`PANEL_CHART_SIZE` inches (:data:`PANEL_HANDLE_SIZE` for the handle
    chart) and a zoom panel :data:`PANEL_ZOOM_SIZE`, with larger tick labels
    than in the combined figure and the axis labels of its chart panel, so it
    can be shown apart from that panel.  The Morse graph is drawn with labels of
    ``graph_font_size`` points in a figure the size of its layout in inches,
    so a larger graph gives a larger figure, and the labels print at that
    size when the figure is shown at its natural size.
    """

    from .hybrid_morse_plot import _draw_morse_graph

    layout = _atlas_plot_layout(
        source,
        atlas_charts=atlas_charts,
        finite_relation_annotations=finite_relation_annotations,
        blocked_index_nodes=blocked_index_nodes,
        morse_nodes=morse_nodes,
        proj_dims=proj_dims,
        handle_proj_dims=handle_proj_dims,
        clist=clist,
        axis_labels=axis_labels,
        handle_axis_labels=handle_axis_labels,
        show_handles=show_handles,
        base_view=base_view,
        handle_view=handle_view,
        frame_margin=frame_margin,
        detail_zooms=detail_zooms,
    )
    components = layout.components
    figures: dict[str, Figure] = {}
    axes: dict[str, Axes] = {}
    try:
        for chart_kind, chart_projections in (
            ("base", layout.projections),
            ("handle", layout.handle_projections),
        ):
            chart_panels = [panel for panel in layout.panels if panel.chart_kind == chart_kind]
            for number, panel in enumerate(chart_panels, start=1):
                name = chart_kind if len(chart_projections) == 1 else f"{chart_kind}-{number}"
                figure, axis = _figure_with_axes(
                    PANEL_CHART_SIZE if chart_kind == "base" else PANEL_HANDLE_SIZE,
                    PANEL_CHART_MARGINS,
                    dpi,
                )
                figures[name], axes[name] = figure, axis
                _draw_chart_projection(
                    axis,
                    components,
                    chart_kind=chart_kind,
                    projection=panel.projection,
                    limits=panel.limits,
                    labels=panel.labels,
                    show_grid=show_grid,
                    show_panel_title=False,
                    spines_below_cells=float(frame_margin) > 0.0,
                    outlined=panel.small,
                )
                for zoom in panel.zooms:
                    _mark_zoom_window(axis, zoom)
                    zoom_name = f"zoom-{zoom.label}"
                    zoom_figure, zoom_axis = _figure_with_axes(
                        PANEL_ZOOM_SIZE, PANEL_ZOOM_MARGINS, dpi
                    )
                    figures[zoom_name], axes[zoom_name] = zoom_figure, zoom_axis
                    _draw_zoom_panel(
                        zoom_axis,
                        zoom,
                        components,
                        outlined=panel.small,
                        style=ZOOM_FIGURE_STYLE,
                        labels=panel.labels,
                    )
        if show_morse_graph:
            # Drawn first on a whole-figure axis, then the figure is given the
            # size of the layout: one unit of the axis is then one inch.
            figure = plt.figure(figsize=(1.0, 1.0), dpi=dpi)
            axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
            figures["graph"], axes["graph"] = figure, axis
            _draw_morse_graph(
                axis,
                layout.order,
                components,
                conley_indices=(
                    None
                    if finite_relation_annotations is None
                    else finite_relation_annotations.shift_classes
                ),
                show_component_sizes=show_component_sizes,
                show_title=False,
                blocked_index_nodes=layout.blocked_nodes,
                font_size=graph_font_size,
            )
            x_lower, x_upper = axis.get_xlim()
            y_lower, y_upper = axis.get_ylim()
            figure.set_size_inches(x_upper - x_lower, y_upper - y_lower)
    except BaseException:
        for figure in figures.values():
            plt.close(figure)
        raise
    return AtlasMorsePanelFigures(
        figures=figures,
        axes=axes,
        morse_graph=layout.order,
        components=components,
        data=layout.data,
        zooms=layout.zooms,
    )


__all__ = [
    "ATLAS_MORSE_PLOT_SCHEMA",
    "PHYSICAL_CONLEY_FINITE_RELATION_AUDIT_SCHEMA",
    "AtlasFiniteRelationIndexAnnotations",
    "AtlasMorseBox",
    "AtlasMorseNode",
    "AtlasMorsePlotData",
    "AtlasHybridMorseComponent",
    "AtlasHybridMorsePlot",
    "AtlasDetailZoom",
    "AtlasMorsePanelFigures",
    "CELL_OUTLINE_WIDTH",
    "DETAIL_ZOOM_MAX_WINDOW",
    "DETAIL_ZOOM_MIN_AREA",
    "PANEL_CHART_SIZE",
    "PANEL_GRAPH_FONT_SIZE",
    "PANEL_HANDLE_SIZE",
    "PANEL_ZOOM_SIZE",
    "extract_atlas_morse_plot_data",
    "atlas_morse_plot_data_payload",
    "save_atlas_morse_plot_data",
    "load_atlas_morse_plot_data",
    "load_atlas_finite_relation_index_annotations",
    "atlas_morse_components",
    "atlas_morse_hasse",
    "plot_atlas_hybrid_morse_panels",
    "plot_atlas_hybrid_morse_sets",
]
