"""Physical Conley-index audits for actual CMGDB Atlas relations.

The canonical path starts with the stored ``MapGraph`` and its Morse sets,
extracts CMGDB's top-cell candidate ``X=S union F(S), A=F(S) minus S``, and
forms the nerve of the actual Atlas rectangles in the reset quotient.  It
checks the finite pair, every acyclic carrier value, carrier subordination,
pair preservation, and ``dF=Fd`` before asking CMGDB for the finite-relation
shift class.  Outside-``X`` exits and callback failures remain in the cache.

The finite-relation result is deliberately separate from a Conley index of
the continuous suspension map.  The latter stays uncertified until a
whole-cell outer enclosure and an external continuous index-pair theorem are
available.  No analytic orbit label is substituted for either result.

The older common-refinement mapping-cylinder audit remains below as an
independent falsification tool; it is not the persisted physical result.
"""

from __future__ import annotations

import bisect
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable

import networkx as nx
import numpy as np

from ..src.suspension_complex import (
    CMGDBRelativeHomologyPayload,
    CubicalCell,
    CubicalGridComplex,
    FiniteCellComplex,
    GuardPrismCell,
    PhaseSliceCell,
    RelativeCellPair,
    SuspensionBaseCell,
)
from ..src.atlas_conley import (
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    prepare_atlas_relation_conley_2d,
)


Cell = Hashable
MODULUS = 5


def _coordinate(value: float) -> float:
    """Canonical coordinate key without changing reported physical bounds."""

    return round(float(value), 12)


def _bounds_key(chart_id: int, bounds: Sequence[float]) -> tuple[object, ...]:
    return (int(chart_id), *(_coordinate(value) for value in bounds))


def _add_scaled_chain(
    accumulator: dict[Cell, int],
    chain: Mapping[Cell, int],
    scale: int,
) -> None:
    for cell, coefficient in chain.items():
        value = accumulator.get(cell, 0) + scale * coefficient
        if value:
            accumulator[cell] = value
        else:
            accumulator.pop(cell, None)


@dataclass(frozen=True, order=True)
class AtlasTopCellRecord:
    """One tagged top cell and its complete MapGraph image."""

    index: int
    chart_id: int
    bounds: tuple[float, ...]
    image: tuple[int, ...]
    morse_node: int | None
    relation_evaluated: bool = True
    callback_failed_samples: int = 0
    callback_failure_reasons: tuple[tuple[str, int], ...] = ()

    @property
    def empty_image(self) -> bool:
        return not self.image

    def to_json_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "chart_id": self.chart_id,
            "bounds": list(self.bounds),
            "image": list(self.image),
            "empty_image": self.empty_image,
            "relation_evaluated": self.relation_evaluated,
            "morse_node": self.morse_node,
            "callback_failed_samples": self.callback_failed_samples,
            "callback_failure_reasons": dict(self.callback_failure_reasons),
        }


@dataclass(frozen=True)
class AtlasRelationSnapshot:
    """Serializable provenance snapshot of one physical Atlas computation."""

    model: str
    depth: int
    t_star: float
    cells: tuple[AtlasTopCellRecord, ...]
    morse_nodes: tuple[int, ...]
    morse_edges: tuple[tuple[int, int], ...]
    base_chart_id: int
    handle_chart_id: int
    whole_cell_outer_enclosure_certified: bool
    global_attractor_lattice_interpretation: bool
    box_map_failed_samples: int
    box_map_empty_images: int
    box_map_unresolved_stage_edges: int
    relation_scope: str = "full_atlas"

    def __post_init__(self) -> None:
        if tuple(cell.index for cell in self.cells) != tuple(range(len(self.cells))):
            raise ValueError("Atlas relation cells must be indexed consecutively")
        if self.relation_scope not in {"full_atlas", "targeted_index_candidates"}:
            raise ValueError("unknown Atlas relation scope")

    @property
    def adjacency(self) -> tuple[tuple[int, ...], ...]:
        return tuple(cell.image for cell in self.cells)

    def morse_set(self, node: int) -> frozenset[int]:
        result = frozenset(
            cell.index for cell in self.cells if cell.morse_node == int(node)
        )
        if not result:
            raise ValueError(f"Morse node {node} has no cells in this snapshot")
        return result

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": "physical-conley-atlas-relation-v1",
            "metadata": {
                "model": self.model,
                "depth": self.depth,
                "t_star": self.t_star,
                "base_chart_id": self.base_chart_id,
                "handle_chart_id": self.handle_chart_id,
                "whole_cell_outer_enclosure_certified": (
                    self.whole_cell_outer_enclosure_certified
                ),
                "global_attractor_lattice_interpretation": (
                    self.global_attractor_lattice_interpretation
                ),
                "box_map_failed_samples_across_cmgdb_passes": (
                    self.box_map_failed_samples
                ),
                "box_map_empty_images_across_cmgdb_passes": (
                    self.box_map_empty_images
                ),
                "box_map_unresolved_stage_edges": (
                    self.box_map_unresolved_stage_edges
                ),
                "relation_scope": self.relation_scope,
                "evaluated_relation_sources": sum(
                    cell.relation_evaluated for cell in self.cells
                ),
                "provenance": (
                    "Direct extraction from CMGDB MapGraph adjacency, Atlas "
                    "chart boxes, callback diagnostics, and MorseGraph sets."
                ),
            },
            "morse_graph": {
                "nodes": list(self.morse_nodes),
                "edges": [list(edge) for edge in self.morse_edges],
            },
            "cells": [cell.to_json_dict() for cell in self.cells],
        }

    def write_gzip_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as stream:
            json.dump(self.to_json_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

    @classmethod
    def read_gzip_json(cls, path: str | Path) -> "AtlasRelationSnapshot":
        """Load a full or targeted relation cache without rerunning CMGDB."""

        with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema") != "physical-conley-atlas-relation-v1":
            raise ValueError("unsupported physical Conley relation-cache schema")
        metadata = payload["metadata"]
        morse_graph = payload["morse_graph"]
        cells = tuple(
            AtlasTopCellRecord(
                index=int(cell["index"]),
                chart_id=int(cell["chart_id"]),
                bounds=tuple(float(value) for value in cell["bounds"]),
                image=tuple(int(value) for value in cell["image"]),
                morse_node=(
                    None if cell["morse_node"] is None else int(cell["morse_node"])
                ),
                relation_evaluated=bool(cell.get("relation_evaluated", True)),
                callback_failed_samples=int(cell["callback_failed_samples"]),
                callback_failure_reasons=tuple(
                    sorted(
                        (str(reason), int(count))
                        for reason, count in cell["callback_failure_reasons"].items()
                    )
                ),
            )
            for cell in payload["cells"]
        )
        return cls(
            model=str(metadata["model"]),
            depth=int(metadata["depth"]),
            t_star=float(metadata["t_star"]),
            cells=cells,
            morse_nodes=tuple(int(node) for node in morse_graph["nodes"]),
            morse_edges=tuple(
                (int(source), int(target))
                for source, target in morse_graph["edges"]
            ),
            base_chart_id=int(metadata["base_chart_id"]),
            handle_chart_id=int(metadata["handle_chart_id"]),
            whole_cell_outer_enclosure_certified=bool(
                metadata["whole_cell_outer_enclosure_certified"]
            ),
            global_attractor_lattice_interpretation=bool(
                metadata["global_attractor_lattice_interpretation"]
            ),
            box_map_failed_samples=int(
                metadata["box_map_failed_samples_across_cmgdb_passes"]
            ),
            box_map_empty_images=int(
                metadata["box_map_empty_images_across_cmgdb_passes"]
            ),
            box_map_unresolved_stage_edges=int(
                metadata["box_map_unresolved_stage_edges"]
            ),
            relation_scope=str(metadata.get("relation_scope", "full_atlas")),
        )


def snapshot_from_acceptance(
    result: Any,
    *,
    model: str,
    base_chart_id: int = 0,
    handle_chart_id: int = 1,
) -> AtlasRelationSnapshot:
    """Extract full adjacency, boxes, Morse membership, and failure flags."""

    tokens = tuple(result.cell_tokens)
    if len(tokens) != int(result.map_graph.num_vertices()):
        raise ValueError("acceptance tokens and MapGraph vertex count disagree")

    node_by_index: dict[int, int] = {}
    for node in result.morse_graph.vertices():
        for raw_index in result.morse_graph.morse_set(node):
            index = int(raw_index)
            if index in node_by_index:
                raise ValueError("one Atlas cell belongs to two Morse sets")
            node_by_index[index] = int(node)

    failures_by_source: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    for diagnostic in result.box_map_diagnostics.retained_source_records:
        key = _bounds_key(diagnostic.source_chart_id, diagnostic.source_bounds)
        failures_by_source[key].update(failure.reason for failure in diagnostic.failures)

    cells = []
    for index, token in enumerate(tokens):
        if int(token.index) != index:
            raise ValueError("Atlas token indices are not MapGraph indices")
        bounds = tuple(float(value) for value in token.bounds)
        failures = failures_by_source.get(
            _bounds_key(int(token.chart_id), bounds),
            Counter(),
        )
        cells.append(
            AtlasTopCellRecord(
                index=index,
                chart_id=int(token.chart_id),
                bounds=bounds,
                image=tuple(
                    sorted(int(target) for target in result.map_graph.adjacencies(index))
                ),
                morse_node=node_by_index.get(index),
                callback_failed_samples=sum(failures.values()),
                callback_failure_reasons=tuple(sorted(failures.items())),
            )
        )

    return AtlasRelationSnapshot(
        model=model,
        depth=int(result.depth),
        t_star=float(result.t_star),
        cells=tuple(cells),
        morse_nodes=tuple(sorted(int(node) for node in result.morse_graph.vertices())),
        morse_edges=tuple(
            sorted(
                (int(source), int(target))
                for source, target in result.morse_graph.edges()
            )
        ),
        base_chart_id=int(base_chart_id),
        handle_chart_id=int(handle_chart_id),
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=int(result.box_map_diagnostics.failed_samples),
        box_map_empty_images=int(result.box_map_diagnostics.empty_images),
        box_map_unresolved_stage_edges=int(
            result.box_map_diagnostics.unresolved_stage_edges
        ),
    )


@dataclass(frozen=True)
class TopCellIndexPair:
    """CMGDB's top-cell candidate, before taking cellular closures."""

    morse_node: int
    s_cells: frozenset[int]
    x_cells: frozenset[int]
    a_cells: frozenset[int]
    f_s_cells: frozenset[int]
    unevaluated_s_sources: tuple[int, ...]
    unevaluated_x_sources: tuple[int, ...]
    empty_s_sources: tuple[int, ...]
    first_condition_violations: tuple[tuple[int, tuple[int, ...]], ...]
    second_condition_violations: tuple[tuple[int, tuple[int, ...]], ...]
    a_targets_outside_x: tuple[tuple[int, tuple[int, ...]], ...]
    recurrent_components_in_x: tuple[frozenset[int], ...]
    recurrent_components_not_s_or_a: tuple[frozenset[int], ...]
    s_is_strongly_connected: bool

    @property
    def pair_invariance_passed(self) -> bool:
        return bool(
            not self.unevaluated_s_sources
            and not self.unevaluated_x_sources
            and not self.empty_s_sources
            and not self.first_condition_violations
            and not self.second_condition_violations
            and self.s_is_strongly_connected
        )

    @property
    def combinatorial_isolation_passed(self) -> bool:
        return bool(
            self.pair_invariance_passed
            and not self.recurrent_components_not_s_or_a
        )

    def summary(self) -> dict[str, object]:
        return {
            "construction": "X=S_union_F(S), A=F(S)_minus_S",
            "closures_taken_at_this_stage": False,
            "morse_node": self.morse_node,
            "s_top_cells": len(self.s_cells),
            "f_s_top_cells": len(self.f_s_cells),
            "x_top_cells": len(self.x_cells),
            "a_top_cells": len(self.a_cells),
            "unevaluated_S_sources": list(self.unevaluated_s_sources),
            "unevaluated_X_sources": list(self.unevaluated_x_sources),
            "empty_s_sources": list(self.empty_s_sources),
            "first_condition": "F(X\\A) subset X",
            "first_condition_violations": [
                [source, list(targets)]
                for source, targets in self.first_condition_violations
            ],
            "second_condition": "F(A) intersection X subset A",
            "second_condition_violations": [
                [source, list(targets)]
                for source, targets in self.second_condition_violations
            ],
            "a_targets_outside_x": [
                [source, list(targets)]
                for source, targets in self.a_targets_outside_x
            ],
            "s_is_strongly_connected": self.s_is_strongly_connected,
            "recurrent_component_sizes_in_x": [
                len(component) for component in self.recurrent_components_in_x
            ],
            "recurrent_components_not_s_or_a": [
                sorted(component)
                for component in self.recurrent_components_not_s_or_a
            ],
            "pair_invariance_passed": self.pair_invariance_passed,
            "combinatorial_isolation_passed": self.combinatorial_isolation_passed,
        }


def extract_top_cell_index_pair(
    snapshot: AtlasRelationSnapshot,
    morse_node: int,
) -> TopCellIndexPair:
    """Extract and independently check the standard CMGDB candidate pair."""

    adjacency = snapshot.adjacency
    s_cells = snapshot.morse_set(morse_node)
    unevaluated_s = tuple(
        sorted(source for source in s_cells if not snapshot.cells[source].relation_evaluated)
    )
    if unevaluated_s:
        raise ValueError(
            "Morse-set relation was not evaluated on sources "
            f"{list(unevaluated_s)!r}"
        )
    f_s = frozenset(
        target for source in s_cells for target in adjacency[source]
    )
    x_cells = s_cells | f_s
    a_cells = x_cells - s_cells
    unevaluated_x = tuple(
        sorted(source for source in x_cells if not snapshot.cells[source].relation_evaluated)
    )

    empty_s = tuple(sorted(source for source in s_cells if not adjacency[source]))
    first = tuple(
        (source, tuple(sorted(set(adjacency[source]) - x_cells)))
        for source in sorted(s_cells)
        if set(adjacency[source]) - x_cells
    )
    second = tuple(
        (source, tuple(sorted((set(adjacency[source]) & x_cells) - a_cells)))
        for source in sorted(a_cells)
        if (set(adjacency[source]) & x_cells) - a_cells
    )
    outside = tuple(
        (source, tuple(sorted(set(adjacency[source]) - x_cells)))
        for source in sorted(a_cells)
        if set(adjacency[source]) - x_cells
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(x_cells)
    graph.add_edges_from(
        (source, target)
        for source in x_cells
        for target in adjacency[source]
        if target in x_cells
    )
    recurrent = []
    for component in nx.strongly_connected_components(graph):
        frozen = frozenset(int(value) for value in component)
        if len(frozen) > 1 or any(graph.has_edge(value, value) for value in frozen):
            recurrent.append(frozen)
    recurrent.sort(key=lambda component: min(component))
    unexpected = tuple(
        component
        for component in recurrent
        if component != s_cells and not component <= a_cells
    )
    s_graph = graph.subgraph(s_cells)
    strongly_connected = bool(s_cells) and nx.is_strongly_connected(s_graph)

    return TopCellIndexPair(
        morse_node=int(morse_node),
        s_cells=s_cells,
        x_cells=frozenset(x_cells),
        a_cells=frozenset(a_cells),
        f_s_cells=f_s,
        unevaluated_s_sources=unevaluated_s,
        unevaluated_x_sources=unevaluated_x,
        empty_s_sources=empty_s,
        first_condition_violations=first,
        second_condition_violations=second,
        a_targets_outside_x=outside,
        recurrent_components_in_x=tuple(recurrent),
        recurrent_components_not_s_or_a=unexpected,
        s_is_strongly_connected=strongly_connected,
    )


@dataclass(frozen=True)
class PhysicalMappingCylinder:
    """A cellular realization of the two tagged Atlas charts."""

    complex: FiniteCellComplex
    base_complex: CubicalGridComplex
    handle_complex: CubicalGridComplex
    base_first_coordinates: tuple[float, ...]
    base_second_coordinates: tuple[float, ...]
    handle_coordinates: tuple[float, ...]
    phase_coordinates: tuple[float, ...]
    top_to_atlas: Mapping[Cell, int]
    atlas_to_top: Mapping[int, tuple[Cell, ...]]
    bottom_vertex_targets: tuple[tuple[int, int], ...]
    reset_vertex_targets: tuple[tuple[int, int], ...]
    attachment_chain_maps_validated: bool

    @property
    def top_cells(self) -> tuple[Cell, ...]:
        return self.complex.cells_of_dimension(2)

    def summary(self) -> dict[str, object]:
        return {
            "complex_kind": "two-chart cellular mapping cylinder",
            "dimension": self.complex.max_dimension,
            "cells_by_dimension": [
                len(self.complex.cells_of_dimension(dimension))
                for dimension in range(self.complex.max_dimension + 1)
            ],
            "top_cells": len(self.top_cells),
            "base_refinement": [
                len(self.base_first_coordinates) - 1,
                len(self.base_second_coordinates) - 1,
            ],
            "handle_refinement": [
                len(self.handle_coordinates) - 1,
                len(self.phase_coordinates) - 1,
            ],
            "bottom_vertex_targets": [
                list(target) for target in self.bottom_vertex_targets
            ],
            "reset_vertex_targets": [
                list(target) for target in self.reset_vertex_targets
            ],
            "attachment_chain_maps_validated": (
                self.attachment_chain_maps_validated
            ),
            "boundary_squared_zero_validated": True,
            "atlas_edges_inferred": 0,
        }


def _chart_breakpoints(
    snapshot: AtlasRelationSnapshot,
    chart_id: int,
    axis: int,
) -> tuple[float, ...]:
    values = set()
    for cell in snapshot.cells:
        if cell.chart_id != chart_id:
            continue
        dimension = len(cell.bounds) // 2
        if dimension != 2:
            raise ValueError("physical Conley adapter currently requires 2D charts")
        values.add(_coordinate(cell.bounds[axis]))
        values.add(_coordinate(cell.bounds[dimension + axis]))
    if len(values) < 2:
        raise ValueError(f"Atlas chart {chart_id} has no usable axis {axis}")
    return tuple(sorted(values))


def _find_coordinate(values: Sequence[float], target: float) -> int:
    key = _coordinate(target)
    index = bisect.bisect_left(values, key)
    if index >= len(values) or _coordinate(values[index]) != key:
        raise ValueError(f"coordinate {target} is absent from the cellular refinement")
    return index


def _find_interval(values: Sequence[float], point: float) -> int:
    index = bisect.bisect_right(values, float(point)) - 1
    if index == len(values) - 1:
        index -= 1
    if index < 0 or not values[index] <= point <= values[index + 1]:
        raise ValueError(f"point {point} lies outside coordinate intervals")
    return index


def _base_vertical_chain(
    fixed_first: int,
    start_second: int,
    end_second: int,
) -> dict[CubicalCell, int]:
    if start_second == end_second:
        return {}
    sign = 1 if end_second > start_second else -1
    lower = min(start_second, end_second)
    upper = max(start_second, end_second)
    return {
        CubicalCell((fixed_first, second), (False, True)): sign
        for second in range(lower, upper)
    }


def _attachment_map(
    handle: CubicalGridComplex,
    base: CubicalGridComplex,
    vertex_targets: Sequence[tuple[int, int]],
) -> dict[CubicalCell, dict[CubicalCell, int]]:
    images: dict[CubicalCell, dict[CubicalCell, int]] = {}
    for vertex_index, target in enumerate(vertex_targets):
        source = CubicalCell((vertex_index,), (False,))
        images[source] = {CubicalCell(tuple(target), (False, False)): 1}
    for edge_index in range(handle.subdivisions[0]):
        source = CubicalCell((edge_index,), (True,))
        first = vertex_targets[edge_index]
        second = vertex_targets[edge_index + 1]
        if first[0] != second[0]:
            raise ValueError("physical attachments must lie on one base section")
        images[source] = _base_vertical_chain(first[0], first[1], second[1])

    for source in handle.cells:
        left: dict[Cell, int] = {}
        for target, coefficient in images[source].items():
            _add_scaled_chain(left, base.boundary(target), coefficient)
        right: dict[Cell, int] = {}
        for face, coefficient in handle.boundary(source).items():
            _add_scaled_chain(right, images[face], coefficient)
        if left != right:
            raise ValueError(
                f"attachment is not a cellular chain map on {source!r}: "
                f"dF={left!r}, Fd={right!r}"
            )
    return images


def _build_mapping_cylinder_chain_complex(
    base: CubicalGridComplex,
    handle: CubicalGridComplex,
    *,
    slabs: int,
    bottom_map: Mapping[CubicalCell, Mapping[CubicalCell, int]],
    reset_map: Mapping[CubicalCell, Mapping[CubicalCell, int]],
    handle_id: str,
) -> FiniteCellComplex:
    base_refs = {cell: SuspensionBaseCell(cell) for cell in base.cells}
    dimensions: dict[Cell, int] = {
        base_refs[cell]: base.dimension(cell) for cell in base.cells
    }
    boundaries: dict[Cell, dict[Cell, int]] = {
        base_refs[cell]: {
            base_refs[face]: coefficient
            for face, coefficient in base.boundary(cell).items()
        }
        for cell in base.cells
    }

    slices: dict[tuple[CubicalCell, int], PhaseSliceCell] = {}
    prisms: dict[tuple[CubicalCell, int], GuardPrismCell] = {}
    for source in handle.cells:
        for level in range(1, slabs):
            slices[(source, level)] = PhaseSliceCell(handle_id, source, level)
        for slab in range(slabs):
            prisms[(source, slab)] = GuardPrismCell(handle_id, source, slab)

    for source in handle.cells:
        source_dimension = handle.dimension(source)
        for level in range(1, slabs):
            cell = slices[(source, level)]
            dimensions[cell] = source_dimension
            boundaries[cell] = {
                slices[(face, level)]: coefficient
                for face, coefficient in handle.boundary(source).items()
            }

        for slab in range(slabs):
            cell = prisms[(source, slab)]
            dimensions[cell] = source_dimension + 1
            boundary: dict[Cell, int] = {
                prisms[(face, slab)]: coefficient
                for face, coefficient in handle.boundary(source).items()
            }
            endpoint_sign = -1 if source_dimension % 2 == 0 else 1
            if slab == 0:
                bottom = {
                    base_refs[target]: coefficient
                    for target, coefficient in bottom_map[source].items()
                }
            else:
                bottom = {slices[(source, slab)]: 1}
            _add_scaled_chain(boundary, bottom, endpoint_sign)

            if slab + 1 == slabs:
                top = {
                    base_refs[target]: coefficient
                    for target, coefficient in reset_map[source].items()
                }
            else:
                top = {slices[(source, slab + 1)]: 1}
            _add_scaled_chain(boundary, top, -endpoint_sign)
            boundaries[cell] = boundary

    return FiniteCellComplex(
        dimensions,
        boundaries,
        metadata={
            "kind": "physical-two-chart-mapping-cylinder",
            "handle_id": handle_id,
            "phase_slabs": slabs,
        },
    )


def _atlas_interval_lookup(
    snapshot: AtlasRelationSnapshot,
) -> dict[tuple[object, ...], int]:
    lookup = {}
    for cell in snapshot.cells:
        key = _bounds_key(cell.chart_id, cell.bounds)
        if key in lookup:
            raise ValueError(f"duplicate Atlas box geometry for key {key!r}")
        lookup[key] = cell.index
    return lookup


def build_physical_mapping_cylinder(
    snapshot: AtlasRelationSnapshot,
    *,
    bottom_first_coordinate: float,
    reset_first_coordinate: float,
    reset_second_coordinate: Any,
    handle_id: str,
) -> PhysicalMappingCylinder:
    """Build an exact cellular common refinement of both chart attachments."""

    original_first = _chart_breakpoints(
        snapshot, snapshot.base_chart_id, 0
    )
    original_second = _chart_breakpoints(
        snapshot, snapshot.base_chart_id, 1
    )
    handle_coordinates = _chart_breakpoints(
        snapshot, snapshot.handle_chart_id, 0
    )
    phase_coordinates = _chart_breakpoints(
        snapshot, snapshot.handle_chart_id, 1
    )
    reset_values = tuple(
        _coordinate(reset_second_coordinate(value))
        for value in handle_coordinates
    )
    first_coordinates = tuple(
        sorted(
            set(original_first)
            | {
                _coordinate(bottom_first_coordinate),
                _coordinate(reset_first_coordinate),
            }
        )
    )
    second_coordinates = tuple(
        sorted(set(original_second) | set(handle_coordinates) | set(reset_values))
    )

    base = CubicalGridComplex(
        (len(first_coordinates) - 1, len(second_coordinates) - 1)
    )
    handle = CubicalGridComplex((len(handle_coordinates) - 1,))
    bottom_first = _find_coordinate(first_coordinates, bottom_first_coordinate)
    reset_first = _find_coordinate(first_coordinates, reset_first_coordinate)
    bottom_targets = tuple(
        (bottom_first, _find_coordinate(second_coordinates, value))
        for value in handle_coordinates
    )
    reset_targets = tuple(
        (reset_first, _find_coordinate(second_coordinates, value))
        for value in reset_values
    )
    bottom_map = _attachment_map(handle, base, bottom_targets)
    reset_map = _attachment_map(handle, base, reset_targets)
    complex_ = _build_mapping_cylinder_chain_complex(
        base,
        handle,
        slabs=len(phase_coordinates) - 1,
        bottom_map=bottom_map,
        reset_map=reset_map,
        handle_id=handle_id,
    )

    lookup = _atlas_interval_lookup(snapshot)
    top_to_atlas: dict[Cell, int] = {}
    atlas_to_top: dict[int, list[Cell]] = {
        cell.index: [] for cell in snapshot.cells
    }

    for first_index in range(base.subdivisions[0]):
        first_midpoint = 0.5 * (
            first_coordinates[first_index] + first_coordinates[first_index + 1]
        )
        original_first_index = _find_interval(original_first, first_midpoint)
        for second_index in range(base.subdivisions[1]):
            second_midpoint = 0.5 * (
                second_coordinates[second_index]
                + second_coordinates[second_index + 1]
            )
            original_second_index = _find_interval(original_second, second_midpoint)
            bounds = (
                original_first[original_first_index],
                original_second[original_second_index],
                original_first[original_first_index + 1],
                original_second[original_second_index + 1],
            )
            atlas_index = lookup[
                _bounds_key(snapshot.base_chart_id, bounds)
            ]
            cell = SuspensionBaseCell(
                CubicalCell((first_index, second_index), (True, True))
            )
            top_to_atlas[cell] = atlas_index
            atlas_to_top[atlas_index].append(cell)

    for handle_index in range(handle.subdivisions[0]):
        source_edge = CubicalCell((handle_index,), (True,))
        for slab in range(len(phase_coordinates) - 1):
            bounds = (
                handle_coordinates[handle_index],
                phase_coordinates[slab],
                handle_coordinates[handle_index + 1],
                phase_coordinates[slab + 1],
            )
            atlas_index = lookup[
                _bounds_key(snapshot.handle_chart_id, bounds)
            ]
            cell = GuardPrismCell(handle_id, source_edge, slab)
            top_to_atlas[cell] = atlas_index
            atlas_to_top[atlas_index].append(cell)

    if set(top_to_atlas) != set(complex_.cells_of_dimension(2)):
        missing = set(complex_.cells_of_dimension(2)) - set(top_to_atlas)
        raise ValueError(
            "not every cellular top cell has Atlas provenance: "
            f"{sorted(missing, key=repr)[:5]!r}"
        )
    if any(not top_cells for top_cells in atlas_to_top.values()):
        missing = [index for index, top_cells in atlas_to_top.items() if not top_cells]
        raise ValueError(f"Atlas cells lost under common refinement: {missing[:10]!r}")

    return PhysicalMappingCylinder(
        complex=complex_,
        base_complex=base,
        handle_complex=handle,
        base_first_coordinates=first_coordinates,
        base_second_coordinates=second_coordinates,
        handle_coordinates=handle_coordinates,
        phase_coordinates=phase_coordinates,
        top_to_atlas=MappingProxyType(top_to_atlas),
        atlas_to_top=MappingProxyType(
            {index: tuple(cells) for index, cells in atlas_to_top.items()}
        ),
        bottom_vertex_targets=bottom_targets,
        reset_vertex_targets=reset_targets,
        attachment_chain_maps_validated=True,
    )


def _rank_sparse_entries(
    entries: Sequence[tuple[int, int, int]],
    column_count: int,
    *,
    modulus: int = MODULUS,
) -> int:
    columns: list[dict[int, int]] = [dict() for _ in range(column_count)]
    for row, column, coefficient in entries:
        value = (columns[column].get(row, 0) + coefficient) % modulus
        if value:
            columns[column][row] = value
        else:
            columns[column].pop(row, None)

    pivots: dict[int, dict[int, int]] = {}
    for original in columns:
        vector = dict(original)
        while vector:
            pivot = max(vector)
            if pivot not in pivots:
                inverse = pow(vector[pivot], -1, modulus)
                pivots[pivot] = {
                    row: coefficient * inverse % modulus
                    for row, coefficient in vector.items()
                    if coefficient % modulus
                }
                break
            scale = vector[pivot]
            for row, coefficient in pivots[pivot].items():
                value = (vector.get(row, 0) - scale * coefficient) % modulus
                if value:
                    vector[row] = value
                else:
                    vector.pop(row, None)
    return len(pivots)


def _relative_betti(
    complex_: FiniteCellComplex,
    p1_cells: Collection[Cell],
    p0_cells: Collection[Cell],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not p1_cells:
        zeros = (0,) * (complex_.max_dimension + 1)
        return zeros, zeros
    pair = RelativeCellPair(complex_, p1_cells, p0_cells)
    counts = pair.cell_counts
    boundaries = pair.boundary_entries(modulus=MODULUS)
    ranks = tuple(
        0
        if dimension == 0
        else _rank_sparse_entries(boundaries[dimension], counts[dimension])
        for dimension in range(len(counts))
    )
    betti = tuple(
        counts[dimension]
        - ranks[dimension]
        - (ranks[dimension + 1] if dimension + 1 < len(ranks) else 0)
        for dimension in range(len(counts))
    )
    return counts, betti


@dataclass(frozen=True)
class CarrierFiberWitness:
    """One exact non-acyclic or trivial CMGDB relative-fiber witness."""

    source_cell: str
    source_dimension: int
    incident_x_atlas_sources: tuple[int, ...]
    incident_a_atlas_sources: tuple[int, ...]
    x_image_atlas_targets: tuple[int, ...]
    a_image_atlas_targets: tuple[int, ...]
    relative_cell_counts: tuple[int, ...]
    relative_betti_numbers: tuple[int, ...]
    classification: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_cell": self.source_cell,
            "source_dimension": self.source_dimension,
            "incident_x_atlas_sources": list(self.incident_x_atlas_sources),
            "incident_a_atlas_sources": list(self.incident_a_atlas_sources),
            "x_image_atlas_targets": list(self.x_image_atlas_targets),
            "a_image_atlas_targets": list(self.a_image_atlas_targets),
            "relative_cell_counts": list(self.relative_cell_counts),
            "relative_betti_numbers": list(self.relative_betti_numbers),
            "classification": self.classification,
        }


@dataclass(frozen=True)
class _FiberData:
    x_image_cells: frozenset[Cell]
    a_image_cells: frozenset[Cell]
    fiber_cells: frozenset[Cell]
    x_atlas_sources: tuple[int, ...]
    a_atlas_sources: tuple[int, ...]
    x_atlas_targets: tuple[int, ...]
    a_atlas_targets: tuple[int, ...]
    cell_counts: tuple[int, ...]
    betti: tuple[int, ...]
    classification: str


@dataclass(frozen=True)
class CarriedRelativeChainMap:
    """An explicitly validated quotient chain map over ``GF(5)``."""

    pair: RelativeCellPair
    images: Mapping[Cell, Mapping[Cell, int]]
    payload: CMGDBRelativeHomologyPayload
    carrier_support_validated: bool
    chain_map_equation_validated: bool
    quotient_pair_preserved: bool

    def summary(self) -> dict[str, object]:
        nonzero_by_dimension = []
        for basis in self.pair.basis_by_dimension:
            nonzero_by_dimension.append(
                sum(len(self.images[source]) for source in basis)
            )
        return {
            "constructed": True,
            "coefficient_field": MODULUS,
            "relative_cell_counts": list(self.pair.cell_counts),
            "nonzero_chain_entries_by_dimension": nonzero_by_dimension,
            "carrier_support_validated": self.carrier_support_validated,
            "chain_map_equation_validated": self.chain_map_equation_validated,
            "quotient_pair_preserved": self.quotient_pair_preserved,
        }


@dataclass(frozen=True)
class CellularCarrierAudit:
    """Cellular closures, relative fibers, and chain-selection result."""

    refined_s_top_cells: int
    refined_x_top_cells: int
    refined_a_top_cells: int
    p1_cell_counts: tuple[int, ...]
    p0_cell_counts: tuple[int, ...]
    relative_cell_counts: tuple[int, ...]
    fiber_classification_counts: Mapping[str, int]
    nonacyclic_witnesses: tuple[CarrierFiberWitness, ...]
    trivial_witnesses: tuple[CarrierFiberWitness, ...]
    carried_chain_map: CarriedRelativeChainMap | None
    chain_construction_failure: Mapping[str, object] | None
    atlas_targets_outside_pair_x: int

    @property
    def all_relative_fibers_acyclic(self) -> bool:
        return bool(
            self.fiber_classification_counts.get("nonacyclic", 0) == 0
            and self.fiber_classification_counts.get("trivial", 0) == 0
        )

    @property
    def chain_map_constructed(self) -> bool:
        return self.carried_chain_map is not None

    def summary(self) -> dict[str, object]:
        return {
            "top_cell_refinement": {
                "S": self.refined_s_top_cells,
                "X": self.refined_x_top_cells,
                "A": self.refined_a_top_cells,
            },
            "cellular_closures": {
                "P1_cells_by_dimension": list(self.p1_cell_counts),
                "P0_cells_by_dimension": list(self.p0_cell_counts),
                "relative_cells_by_dimension": list(self.relative_cell_counts),
                "P0_subset_P1": True,
            },
            "fiber_semantics": (
                "closure(union F(X-neighbor top cells)) relative to "
                "closure(union F(A-neighbor top cells))"
            ),
            "fiber_classification_counts": dict(self.fiber_classification_counts),
            "all_relative_fibers_acyclic": self.all_relative_fibers_acyclic,
            "nonacyclic_witnesses": [
                witness.to_json_dict() for witness in self.nonacyclic_witnesses
            ],
            "trivial_witnesses": [
                witness.to_json_dict() for witness in self.trivial_witnesses
            ],
            "atlas_targets_outside_pair_x": self.atlas_targets_outside_pair_x,
            "chain_map": (
                self.carried_chain_map.summary()
                if self.carried_chain_map is not None
                else {
                    "constructed": False,
                    "failure": dict(self.chain_construction_failure or {}),
                }
            ),
        }


def _fiber_classification(betti: Sequence[int]) -> str:
    if all(value == 0 for value in betti):
        return "trivial"
    if betti and betti[0] == 1 and all(value == 0 for value in betti[1:]):
        return "acyclic"
    return "nonacyclic"


def _solve_boundary_in_cells(
    complex_: FiniteCellComplex,
    allowed_cells: Sequence[Cell],
    rhs: Mapping[Cell, int],
    quotient_cells: frozenset[Cell],
) -> dict[Cell, int] | None:
    """Solve ``boundary(chain)=rhs`` over GF(5) using sparse elimination."""

    if not rhs:
        return {}
    columns: list[dict[Cell, int]] = []
    for cell in allowed_cells:
        columns.append(
            {
                face: coefficient % MODULUS
                for face, coefficient in complex_.boundary(cell).items()
                if face in quotient_cells and coefficient % MODULUS
            }
        )

    pivots: dict[Cell, tuple[dict[Cell, int], dict[int, int]]] = {}
    for column_index, original in enumerate(columns):
        vector = dict(original)
        combination = {column_index: 1}
        while vector:
            pivot = max(vector, key=repr)
            if pivot not in pivots:
                inverse = pow(vector[pivot], -1, MODULUS)
                vector = {
                    row: coefficient * inverse % MODULUS
                    for row, coefficient in vector.items()
                    if coefficient % MODULUS
                }
                combination = {
                    index: coefficient * inverse % MODULUS
                    for index, coefficient in combination.items()
                    if coefficient % MODULUS
                }
                pivots[pivot] = (vector, combination)
                break
            pivot_vector, pivot_combination = pivots[pivot]
            scale = vector[pivot]
            for row, coefficient in pivot_vector.items():
                value = (vector.get(row, 0) - scale * coefficient) % MODULUS
                if value:
                    vector[row] = value
                else:
                    vector.pop(row, None)
            for index, coefficient in pivot_combination.items():
                value = (
                    combination.get(index, 0) - scale * coefficient
                ) % MODULUS
                if value:
                    combination[index] = value
                else:
                    combination.pop(index, None)

    remainder = {
        cell: coefficient % MODULUS
        for cell, coefficient in rhs.items()
        if coefficient % MODULUS
    }
    solution: dict[int, int] = {}
    while remainder:
        pivot = max(remainder, key=repr)
        if pivot not in pivots:
            return None
        pivot_vector, pivot_combination = pivots[pivot]
        scale = remainder[pivot]
        for row, coefficient in pivot_vector.items():
            value = (remainder.get(row, 0) - scale * coefficient) % MODULUS
            if value:
                remainder[row] = value
            else:
                remainder.pop(row, None)
        for index, coefficient in pivot_combination.items():
            value = (solution.get(index, 0) + scale * coefficient) % MODULUS
            if value:
                solution[index] = value
            else:
                solution.pop(index, None)
    return {allowed_cells[index]: coefficient for index, coefficient in solution.items()}


def _construct_carried_relative_chain_map(
    complex_: FiniteCellComplex,
    pair: RelativeCellPair,
    fiber_by_source: Mapping[Cell, _FiberData],
) -> tuple[CarriedRelativeChainMap | None, Mapping[str, object] | None]:
    quotient_cells = frozenset(pair.p1_cells - pair.p0_cells)
    images: dict[Cell, dict[Cell, int]] = {}
    order = {cell: index for index, cell in enumerate(complex_.cells)}

    for dimension, basis in enumerate(pair.basis_by_dimension):
        for source in basis:
            fiber = fiber_by_source[source]
            allowed = tuple(
                sorted(
                    (
                        target
                        for target in fiber.fiber_cells
                        if target in quotient_cells
                        and complex_.dimension(target) == dimension
                    ),
                    key=order.__getitem__,
                )
            )
            if dimension == 0:
                images[source] = {} if not allowed else {allowed[0]: 1}
                continue

            rhs: dict[Cell, int] = {}
            for face, incidence in complex_.boundary(source).items():
                if face not in quotient_cells:
                    continue
                for target, coefficient in images[face].items():
                    value = (
                        rhs.get(target, 0) + incidence * coefficient
                    ) % MODULUS
                    if value:
                        rhs[target] = value
                    else:
                        rhs.pop(target, None)
            solution = _solve_boundary_in_cells(
                complex_,
                allowed,
                rhs,
                quotient_cells,
            )
            if solution is None:
                return None, {
                    "reason": "no_preboundary_in_declared_relative_fiber",
                    "source_cell": repr(source),
                    "source_dimension": dimension,
                    "allowed_target_cells": len(allowed),
                    "rhs_terms": len(rhs),
                    "fiber_relative_betti": list(fiber.betti),
                }
            images[source] = solution

    for source in quotient_cells:
        if not set(images[source]) <= fiber_by_source[source].fiber_cells:
            return None, {
                "reason": "constructed_chain_leaves_carrier_fiber",
                "source_cell": repr(source),
            }
        left: dict[Cell, int] = {}
        for target, coefficient in images[source].items():
            for face, incidence in complex_.boundary(target).items():
                if face not in quotient_cells:
                    continue
                value = (left.get(face, 0) + coefficient * incidence) % MODULUS
                if value:
                    left[face] = value
                else:
                    left.pop(face, None)
        right: dict[Cell, int] = {}
        for face, incidence in complex_.boundary(source).items():
            if face not in quotient_cells:
                continue
            for target, coefficient in images[face].items():
                value = (
                    right.get(target, 0) + incidence * coefficient
                ) % MODULUS
                if value:
                    right[target] = value
                else:
                    right.pop(target, None)
        if left != right:
            return None, {
                "reason": "constructed_map_fails_chain_equation",
                "source_cell": repr(source),
                "boundary_after_map": repr(left),
                "map_after_boundary": repr(right),
            }

    chain_entries = []
    for basis in pair.basis_by_dimension:
        row_of = {cell: row for row, cell in enumerate(basis)}
        entries = []
        for column, source in enumerate(basis):
            for target, coefficient in images[source].items():
                entries.append((row_of[target], column, coefficient % MODULUS))
        chain_entries.append(tuple(entries))
    payload = CMGDBRelativeHomologyPayload(
        cell_counts=pair.cell_counts,
        boundary_entries=pair.boundary_entries(modulus=MODULUS),
        chain_map_entries=tuple(chain_entries),
        basis_by_dimension=pair.basis_by_dimension,
    )
    return (
        CarriedRelativeChainMap(
            pair=pair,
            images=MappingProxyType(
                {
                    source: MappingProxyType(dict(image))
                    for source, image in images.items()
                }
            ),
            payload=payload,
            carrier_support_validated=True,
            chain_map_equation_validated=True,
            quotient_pair_preserved=True,
        ),
        None,
    )


def audit_cellular_carrier(
    snapshot: AtlasRelationSnapshot,
    top_pair: TopCellIndexPair,
    physical: PhysicalMappingCylinder,
    *,
    witness_limit: int = 20,
    attempt_chain_map: bool = True,
) -> CellularCarrierAudit:
    """Audit CMGDB relative fibers and attempt an explicit chain selector."""

    complex_ = physical.complex

    def expand(atlas_cells: Collection[int]) -> frozenset[Cell]:
        return frozenset(
            top
            for atlas_index in atlas_cells
            for top in physical.atlas_to_top[atlas_index]
        )

    s_top = expand(top_pair.s_cells)
    x_top = expand(top_pair.x_cells)
    a_top = expand(top_pair.a_cells)
    p1_cells = complex_.closure(x_top)
    p0_cells = complex_.closure(a_top)
    if not p0_cells <= p1_cells:
        raise ValueError("cellular closure of A is not contained in closure of X")
    pair = RelativeCellPair(complex_, p1_cells, p0_cells)
    quotient_cells = frozenset(p1_cells - p0_cells)

    closure_by_top = {
        top: complex_.closure((top,)) for top in x_top
    }
    incident_x: dict[Cell, set[int]] = defaultdict(set)
    incident_a: dict[Cell, set[int]] = defaultdict(set)
    for top in x_top:
        atlas_source = physical.top_to_atlas[top]
        for cell in closure_by_top[top]:
            incident_x[cell].add(atlas_source)
            if atlas_source in top_pair.a_cells:
                incident_a[cell].add(atlas_source)

    fiber_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], _FiberData] = {}

    def fiber_for(source: Cell) -> _FiberData:
        x_sources = tuple(sorted(incident_x[source]))
        a_sources = tuple(sorted(incident_a[source]))
        key = (x_sources, a_sources)
        if key in fiber_cache:
            return fiber_cache[key]

        x_targets = tuple(
            sorted(
                {
                    target
                    for atlas_source in x_sources
                    for target in snapshot.cells[atlas_source].image
                    if target in top_pair.x_cells
                }
            )
        )
        a_targets = tuple(
            sorted(
                {
                    target
                    for atlas_source in a_sources
                    for target in snapshot.cells[atlas_source].image
                    if target in top_pair.x_cells
                }
            )
        )
        x_target_top = expand(x_targets)
        a_target_top = expand(a_targets)
        x_image = complex_.closure(x_target_top) if x_target_top else frozenset()
        a_image = complex_.closure(a_target_top) if a_target_top else frozenset()
        if not a_image <= x_image:
            raise ValueError(
                "relative fiber A-image is not contained in its X-image"
            )
        counts, betti = _relative_betti(complex_, x_image, a_image)
        value = _FiberData(
            x_image_cells=x_image,
            a_image_cells=a_image,
            fiber_cells=frozenset(x_image - a_image),
            x_atlas_sources=x_sources,
            a_atlas_sources=a_sources,
            x_atlas_targets=x_targets,
            a_atlas_targets=a_targets,
            cell_counts=counts,
            betti=betti,
            classification=_fiber_classification(betti),
        )
        fiber_cache[key] = value
        return value

    fiber_by_source: dict[Cell, _FiberData] = {}
    classifications: Counter[str] = Counter()
    nonacyclic = []
    trivial = []
    for source in (
        cell for cell in complex_.cells if cell in quotient_cells
    ):
        fiber = fiber_for(source)
        fiber_by_source[source] = fiber
        classifications[fiber.classification] += 1
        if fiber.classification not in {"nonacyclic", "trivial"}:
            continue
        witness = CarrierFiberWitness(
            source_cell=repr(source),
            source_dimension=complex_.dimension(source),
            incident_x_atlas_sources=fiber.x_atlas_sources,
            incident_a_atlas_sources=fiber.a_atlas_sources,
            x_image_atlas_targets=fiber.x_atlas_targets,
            a_image_atlas_targets=fiber.a_atlas_targets,
            relative_cell_counts=fiber.cell_counts,
            relative_betti_numbers=fiber.betti,
            classification=fiber.classification,
        )
        collection = nonacyclic if fiber.classification == "nonacyclic" else trivial
        if len(collection) < witness_limit:
            collection.append(witness)

    chain_map = None
    chain_failure: Mapping[str, object] | None = {
        "reason": "chain_construction_not_requested"
    }
    if attempt_chain_map:
        chain_map, chain_failure = _construct_carried_relative_chain_map(
            complex_,
            pair,
            fiber_by_source,
        )

    p1_counts = tuple(
        sum(cell in p1_cells for cell in complex_.cells_of_dimension(dimension))
        for dimension in range(complex_.max_dimension + 1)
    )
    p0_counts = tuple(
        sum(cell in p0_cells for cell in complex_.cells_of_dimension(dimension))
        for dimension in range(complex_.max_dimension + 1)
    )
    outside_count = sum(
        len(set(snapshot.cells[source].image) - top_pair.x_cells)
        for source in top_pair.x_cells
    )
    return CellularCarrierAudit(
        refined_s_top_cells=len(s_top),
        refined_x_top_cells=len(x_top),
        refined_a_top_cells=len(a_top),
        p1_cell_counts=p1_counts,
        p0_cell_counts=p0_counts,
        relative_cell_counts=pair.cell_counts,
        fiber_classification_counts=MappingProxyType(dict(classifications)),
        nonacyclic_witnesses=tuple(nonacyclic),
        trivial_witnesses=tuple(trivial),
        carried_chain_map=chain_map,
        chain_construction_failure=chain_failure,
        atlas_targets_outside_pair_x=outside_count,
    )


@dataclass(frozen=True)
class PhysicalConleyCandidateAudit:
    """All gates and optional CMGDB shift-class output for one Morse node."""

    candidate_name: str
    snapshot: AtlasRelationSnapshot
    top_pair: TopCellIndexPair
    physical_complex: PhysicalMappingCylinder
    cellular_audit: CellularCarrierAudit
    failed_sample_sources_in_s: tuple[int, ...]
    failed_sample_sources_in_x: tuple[int, ...]
    failed_samples_in_s: int
    failed_samples_in_x: int
    cmgdb_shift_result: Mapping[str, object] | None
    withheld_reasons: tuple[str, ...]

    @property
    def all_gates_passed(self) -> bool:
        return not self.withheld_reasons

    def summary(self) -> dict[str, object]:
        return {
            "candidate": self.candidate_name,
            "model": self.snapshot.model,
            "depth": self.snapshot.depth,
            "t_star": self.snapshot.t_star,
            "morse_node": self.top_pair.morse_node,
            "top_cell_pair": self.top_pair.summary(),
            "cellular_realization": self.physical_complex.summary(),
            "cellular_carrier": self.cellular_audit.summary(),
            "physical_relation_gates": {
                "whole_cell_outer_enclosure_certified": (
                    self.snapshot.whole_cell_outer_enclosure_certified
                ),
                "box_map_unresolved_stage_edges": (
                    self.snapshot.box_map_unresolved_stage_edges
                ),
                "failed_sample_sources_in_S": list(
                    self.failed_sample_sources_in_s
                ),
                "failed_sample_sources_in_X": list(
                    self.failed_sample_sources_in_x
                ),
                "failed_samples_in_S_across_cmgdb_passes": (
                    self.failed_samples_in_s
                ),
                "failed_samples_in_X_across_cmgdb_passes": (
                    self.failed_samples_in_x
                ),
            },
            "all_gates_passed": self.all_gates_passed,
            "shift_class_status": (
                "computed" if self.cmgdb_shift_result is not None else "withheld"
            ),
            "withheld_reasons": list(self.withheld_reasons),
            "cmgdb_shift_result": (
                None
                if self.cmgdb_shift_result is None
                else dict(self.cmgdb_shift_result)
            ),
            "analytic_conley_label_attached": False,
        }


def audit_physical_conley_candidate(
    snapshot: AtlasRelationSnapshot,
    physical: PhysicalMappingCylinder,
    *,
    morse_node: int,
    candidate_name: str,
    witness_limit: int = 20,
) -> PhysicalConleyCandidateAudit:
    """Run every gate and invoke CMGDB only if none fails."""

    top_pair = extract_top_cell_index_pair(snapshot, morse_node)
    cellular = audit_cellular_carrier(
        snapshot,
        top_pair,
        physical,
        witness_limit=witness_limit,
        attempt_chain_map=True,
    )
    failed_s_sources = tuple(
        sorted(
            source
            for source in top_pair.s_cells
            if snapshot.cells[source].callback_failed_samples
        )
    )
    failed_x_sources = tuple(
        sorted(
            source
            for source in top_pair.x_cells
            if snapshot.cells[source].callback_failed_samples
        )
    )
    failed_s = sum(
        snapshot.cells[source].callback_failed_samples
        for source in top_pair.s_cells
    )
    failed_x = sum(
        snapshot.cells[source].callback_failed_samples
        for source in top_pair.x_cells
    )

    reasons = []
    if not snapshot.whole_cell_outer_enclosure_certified:
        reasons.append("whole_cell_outer_enclosure_not_certified")
    if snapshot.box_map_unresolved_stage_edges:
        reasons.append("unresolved_sample_event_stage_edges")
    if not top_pair.pair_invariance_passed:
        reasons.append("top_cell_index_pair_invariance_failed")
    if not top_pair.combinatorial_isolation_passed:
        reasons.append("relative_combinatorial_isolation_failed")
    if failed_s_sources:
        reasons.append("callback_sample_failures_inside_morse_set")
    if not physical.attachment_chain_maps_validated:
        reasons.append("reset_attachments_not_cellular")
    if not cellular.all_relative_fibers_acyclic:
        reasons.append("relative_carrier_fibers_not_all_acyclic")
    if not cellular.chain_map_constructed:
        reasons.append("carried_relative_chain_map_not_constructed")

    cmgdb_result = None
    if not reasons:
        try:
            import CMGDB
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("local CMGDB fork is not installed") from error
        payload = cellular.carried_chain_map
        if payload is None:  # pragma: no cover - guarded above
            raise AssertionError("missing chain map after every gate passed")
        cmgdb_result = CMGDB.ComputeRelativeHomologyShiftClass(
            *payload.payload.as_compute_args()
        )

    return PhysicalConleyCandidateAudit(
        candidate_name=candidate_name,
        snapshot=snapshot,
        top_pair=top_pair,
        physical_complex=physical,
        cellular_audit=cellular,
        failed_sample_sources_in_s=failed_s_sources,
        failed_sample_sources_in_x=failed_x_sources,
        failed_samples_in_s=failed_s,
        failed_samples_in_x=failed_x,
        cmgdb_shift_result=cmgdb_result,
        withheld_reasons=tuple(reasons),
    )


def _chain_equation_failures(
    complex_: FiniteCellComplex,
    chain_map: Any,
    *,
    modulus: int = MODULUS,
    witness_limit: int = 20,
) -> tuple[dict[str, object], ...]:
    """Independently recompute ``dF=Fd`` for a selected cellular map."""

    failures = []
    for source in complex_.cells:
        left: dict[Cell, int] = {}
        for target, coefficient in chain_map.image(source).items():
            for face, incidence in complex_.boundary(target).items():
                value = (left.get(face, 0) + coefficient * incidence) % modulus
                if value:
                    left[face] = value
                else:
                    left.pop(face, None)
        right: dict[Cell, int] = {}
        for face, incidence in complex_.boundary(source).items():
            for target, coefficient in chain_map.image(face).items():
                value = (right.get(target, 0) + incidence * coefficient) % modulus
                if value:
                    right[target] = value
                else:
                    right.pop(target, None)
        if left != right and len(failures) < witness_limit:
            failures.append(
                {
                    "source": repr(source),
                    "dimension": complex_.dimension(source),
                    "dF": repr(left),
                    "Fd": repr(right),
                }
            )
    return tuple(failures)


@dataclass(frozen=True)
class AtlasNerveFiniteRelationAudit:
    """Finite-relation Conley result with separate continuous-system gates."""

    candidate_name: str
    snapshot: AtlasRelationSnapshot
    top_pair: TopCellIndexPair
    nerve: AtlasQuotientNerveComplex2D
    pair: AtlasRelativeIndexPair2D
    preparation: Any
    failed_sources_in_x: tuple[int, ...]
    chain_equation_failures: tuple[Mapping[str, object], ...]
    finite_relation_shift_class: Mapping[str, object] | None
    finite_relation_blockers: tuple[str, ...]
    continuous_system_blockers: tuple[str, ...]

    @property
    def finite_relation_conley_index_computed(self) -> bool:
        return self.finite_relation_shift_class is not None

    def summary(self) -> dict[str, object]:
        complex_ = self.pair.complex
        relative_pair = self.pair.relative_pair
        carrier = self.preparation.carrier
        chain_map = self.preparation.chain_map
        p0_counts = tuple(
            sum(
                cell in relative_pair.p0_cells
                for cell in complex_.cells_of_dimension(dimension)
            )
            for dimension in range(complex_.max_dimension + 1)
        )
        carrier_sizes = tuple(len(carrier.image(cell)) for cell in complex_.cells)
        nonzero_entries = tuple(
            sum(len(chain_map.image(cell)) for cell in complex_.cells_of_dimension(dimension))
            for dimension in range(complex_.max_dimension + 1)
        )
        evaluated_x = tuple(
            source
            for source in sorted(self.top_pair.x_cells)
            if self.snapshot.cells[source].relation_evaluated
        )
        failure_reason_counts: Counter[str] = Counter()
        failure_certificates = []
        for source in self.failed_sources_in_x:
            record = self.snapshot.cells[source]
            failure_reason_counts.update(dict(record.callback_failure_reasons))
            failure_certificates.append(
                {
                    "source": source,
                    "pair_membership": (
                        "S" if source in self.top_pair.s_cells else "A"
                    ),
                    "chart_id": record.chart_id,
                    "bounds": list(record.bounds),
                    "failed_samples": record.callback_failed_samples,
                    "reasons": dict(record.callback_failure_reasons),
                }
            )
        return {
            "candidate": self.candidate_name,
            "model": self.snapshot.model,
            "depth": self.snapshot.depth,
            "t_star": self.snapshot.t_star,
            "relation_provenance": {
                "scope": self.snapshot.relation_scope,
                "X_sources_evaluated": len(evaluated_x),
                "X_sources_total": len(self.top_pair.x_cells),
                "failed_sources_in_X": list(self.failed_sources_in_x),
                "failed_sources_in_S": [
                    source
                    for source in self.failed_sources_in_x
                    if source in self.top_pair.s_cells
                ],
                "failed_sources_in_A": [
                    source
                    for source in self.failed_sources_in_x
                    if source in self.top_pair.a_cells
                ],
                "failed_samples_in_X": sum(
                    self.snapshot.cells[source].callback_failed_samples
                    for source in self.failed_sources_in_x
                ),
                "callback_failure_reason_counts": dict(failure_reason_counts),
                "callback_failure_certificates": failure_certificates,
                "unresolved_event_stage_edges": (
                    self.snapshot.box_map_unresolved_stage_edges
                ),
                "original_exit_edges_retained": True,
            },
            "top_cell_pair": self.top_pair.summary(),
            "quotient_nerve": {
                "kind": "actual Atlas rectangles in reset quotient",
                "maximum_dimension": complex_.max_dimension,
                "cells_by_dimension": [
                    len(complex_.cells_of_dimension(dimension))
                    for dimension in range(complex_.max_dimension + 1)
                ],
                "finite_intersections_verified_contractible": bool(
                    self.nerve.metadata[
                        "finite_intersections_verified_contractible"
                    ]
                ),
                "boundary_squared_zero_validated": True,
                "vertices_are_actual_atlas_boxes": True,
                "analytic_orbit_skeleton_used": False,
                "interior_seam_subcomplexes": list(
                    self.nerve.metadata.get("interior_seam_subcomplexes", ())
                ),
                "interior_seam_subcomplex_audits": dict(
                    self.nerve.metadata.get(
                        "interior_seam_subcomplex_audits", {}
                    )
                ),
            },
            "cellular_pair": {
                "P1_cells_by_dimension": [
                    len(complex_.cells_of_dimension(dimension))
                    for dimension in range(complex_.max_dimension + 1)
                ],
                "P0_cells_by_dimension": list(p0_counts),
                "relative_basis_by_dimension": list(relative_pair.cell_counts),
                "P0_subset_P1": True,
                "carrier_preserves_pair": carrier.preserves_pair(relative_pair),
                "selected_chain_map_preserves_P1": chain_map.preserves(
                    relative_pair.p1_cells
                ),
                "selected_chain_map_preserves_P0": chain_map.preserves(
                    relative_pair.p0_cells
                ),
            },
            "carrier_certificate": {
                "construction": self.preparation.carrier_construction,
                "coefficient_field": "GF(5)",
                "carrier_values_checked": len(complex_.cells),
                "all_carrier_values_acyclic": carrier.acyclicity_validated,
                "carrier_value_cell_count_min": min(carrier_sizes),
                "carrier_value_cell_count_max": max(carrier_sizes),
                "face_nesting_validated": True,
                "selected_chain_map_subordinate": carrier.carries(chain_map),
                "chain_equation_dF_equals_Fd": not self.chain_equation_failures,
                "chain_equation_failure_witnesses": [
                    dict(witness) for witness in self.chain_equation_failures
                ],
                "nonzero_chain_entries_by_dimension": list(nonzero_entries),
            },
            "finite_relation_conley_index": (
                None
                if self.finite_relation_shift_class is None
                else dict(self.finite_relation_shift_class)
            ),
            "finite_relation_shift_class": (
                None
                if self.finite_relation_shift_class is None
                else list(self.finite_relation_shift_class.get("shift_class", ()))
            ),
            "finite_relation_blockers": list(self.finite_relation_blockers),
            "continuous_system_conley_index_certified": False,
            "continuous_system_blockers": list(self.continuous_system_blockers),
            "whole_cell_outer_enclosure_certified": (
                self.snapshot.whole_cell_outer_enclosure_certified
            ),
            "external_continuous_index_pair_theorem_available": False,
            "analytic_conley_label_attached": False,
        }


def audit_atlas_nerve_finite_relation(
    snapshot: AtlasRelationSnapshot,
    *,
    morse_node: int,
    candidate_name: str,
    gluing: AtlasResetGluing2D,
    interior_seam_subcomplexes: Collection[str] = (),
) -> AtlasNerveFiniteRelationAudit:
    """Compute an honest finite-relation index and retain continuous blockers."""

    top_pair = extract_top_cell_index_pair(snapshot, morse_node)
    selected_cells = tuple(
        AtlasRectangleCell2D(
            index=source,
            chart_id=snapshot.cells[source].chart_id,
            bounds=tuple(snapshot.cells[source].bounds),
        )
        for source in sorted(top_pair.x_cells)
    )
    nerve = AtlasQuotientNerveComplex2D(
        selected_cells,
        gluing,
        interior_seam_subcomplexes=interior_seam_subcomplexes,
    )
    pair = AtlasRelativeIndexPair2D(
        nerve,
        top_pair.x_cells,
        top_pair.a_cells,
        index_pair_certified=False,
    )
    relation = {
        source: tuple(
            target
            for target in snapshot.cells[source].image
            if target in top_pair.x_cells
        )
        for source in top_pair.x_cells
    }
    preparation = prepare_atlas_relation_conley_2d(
        pair,
        top_relation=relation,
        outer_enclosure_certified=False,
        use_exit_component_carrier=True,
    )
    chain_failures = _chain_equation_failures(
        pair.complex,
        preparation.chain_map,
    )
    failed_sources = tuple(
        source
        for source in sorted(top_pair.x_cells)
        if snapshot.cells[source].callback_failed_samples
    )
    finite_blockers = []
    if not top_pair.pair_invariance_passed:
        finite_blockers.append("finite_top_cell_pair_invariance_failed")
    if top_pair.unevaluated_x_sources:
        finite_blockers.append("mapgraph_relation_not_evaluated_on_all_X_sources")
    if not top_pair.combinatorial_isolation_passed:
        finite_blockers.append("finite_combinatorial_isolation_failed")
    if not preparation.carrier.acyclicity_validated:
        finite_blockers.append("carrier_acyclicity_not_validated")
    if not preparation.carrier.preserves_pair(pair.relative_pair):
        finite_blockers.append("carrier_does_not_preserve_quotient_pair")
    if not preparation.carrier.carries(preparation.chain_map):
        finite_blockers.append("selected_chain_map_leaves_carrier")
    if chain_failures:
        finite_blockers.append("selected_map_fails_chain_equation")

    finite_result = None
    if not finite_blockers:
        finite_result = preparation.compute_finite_relation_shift_class()

    continuous_blockers = []
    if not snapshot.whole_cell_outer_enclosure_certified:
        continuous_blockers.append("whole_cell_outer_enclosure_not_certified")
    continuous_blockers.append("no_external_continuous_index_pair_theorem")
    if failed_sources:
        continuous_blockers.append("callback_sample_failures_inside_X")
    if snapshot.box_map_unresolved_stage_edges:
        continuous_blockers.append("unresolved_sample_event_stage_edges")

    return AtlasNerveFiniteRelationAudit(
        candidate_name=candidate_name,
        snapshot=snapshot,
        top_pair=top_pair,
        nerve=nerve,
        pair=pair,
        preparation=preparation,
        failed_sources_in_x=failed_sources,
        chain_equation_failures=chain_failures,
        finite_relation_shift_class=finite_result,
        finite_relation_blockers=tuple(finite_blockers),
        continuous_system_blockers=tuple(continuous_blockers),
    )


def build_ball_mapping_cylinder(
    snapshot: AtlasRelationSnapshot,
    *,
    restitution: float = 0.8,
) -> PhysicalMappingCylinder:
    return build_physical_mapping_cylinder(
        snapshot,
        bottom_first_coordinate=0.0,
        reset_first_coordinate=0.0,
        reset_second_coordinate=lambda velocity: -float(restitution) * velocity,
        handle_id="bouncing-ball-impact",
    )


def build_wheel_mapping_cylinder(
    snapshot: AtlasRelationSnapshot,
    *,
    alpha: float = 0.4,
    gamma: float = 0.2,
) -> PhysicalMappingCylinder:
    impact_factor = float(np.cos(2.0 * alpha))
    return build_physical_mapping_cylinder(
        snapshot,
        bottom_first_coordinate=alpha + gamma,
        reset_first_coordinate=gamma - alpha,
        reset_second_coordinate=lambda velocity: impact_factor * velocity,
        handle_id="rimless-wheel-impact",
    )


__all__ = [
    "AtlasTopCellRecord",
    "AtlasRelationSnapshot",
    "TopCellIndexPair",
    "PhysicalMappingCylinder",
    "CarrierFiberWitness",
    "CarriedRelativeChainMap",
    "CellularCarrierAudit",
    "PhysicalConleyCandidateAudit",
    "AtlasNerveFiniteRelationAudit",
    "snapshot_from_acceptance",
    "extract_top_cell_index_pair",
    "build_physical_mapping_cylinder",
    "build_ball_mapping_cylinder",
    "build_wheel_mapping_cylinder",
    "audit_cellular_carrier",
    "audit_physical_conley_candidate",
    "audit_atlas_nerve_finite_relation",
]
