"""Conditional finite Conley pipeline for persisted Garcia Atlas relations.

This module consumes ``garcia-walker-open-exit-atlas-relation-v1`` artifacts.
It does not rerun the ODE or replace the fixed-time suspension map by a return
map.  The guard inclusion and reset are covered by sampled-and-bloated target
families under an explicit outer-cover assumption.  Every resulting carrier
must be face compatible and acyclic, and both attachment selectors must lift
to verified integral chain maps before a double mapping cylinder is built.

The ambient Atlas may be mixed depth.  The current sparse cellular adapter
requires the candidate ``X`` itself to have one common per-axis dyadic depth;
otherwise a conforming adaptive cubical complex is still required and the
construction stops explicitly.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

import networkx as nx
import numpy as np

from ..src.atlas_conley import (
    AtlasMappingCylinderConleyPreparation,
    AtlasMappingCylinderRegistry,
    AtlasMappingCylinderRelativePair,
    AtlasMappingCylinderTopCell,
    prepare_atlas_mapping_cylinder_conley,
)
from ..src.cmgdb_suspension_boxmap import SuspensionAtlasCharts
from ..src.suspension_complex import (
    CellularAttachmentMap,
    CrossComplexAcyclicCarrier,
    CubicalCell,
    DoubleMappingCylinderComplex,
    DoubleMappingCylinderHandle,
    SparseCubicalGridComplex,
)
from .garcia_passive_walker_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    garcia_passive_walker_atlas_charts,
)
from .garcia_passive_walker_suspension import reset_map


GARCIA_RELATION_SCHEMA = "garcia-walker-open-exit-atlas-relation-v1"
GARCIA_HANDLE_ID = "garcia-heelstrike"
SAMPLED_BLOAT_ASSUMPTION = (
    "The closed axis-aligned hull of the declared tensor samples, expanded by "
    "the declared padding and clipped to the analytically valid ambient base "
    "chart, is assumed to contain the full guard/reset image of each source "
    "cell."
)


@dataclass(frozen=True)
class PersistedGarciaAtlasCell:
    index: int
    chart_id: int
    axis_depth: int
    coordinates: tuple[int, int, int, int]
    bounds: tuple[float, ...]
    image: frozenset[int]
    relation_evaluated: bool
    open_exit: Mapping[str, Any]


@dataclass(frozen=True)
class PersistedGarciaRelation:
    metadata: Mapping[str, Any]
    chart_payload: Mapping[str, Any]
    candidate: Mapping[str, Any]
    morse_graph: Mapping[str, Any]
    reference_audit: Mapping[str, Any]
    cells: tuple[PersistedGarciaAtlasCell, ...]
    source_path: str | None = None

    @cached_property
    def cell_by_index(self) -> Mapping[int, PersistedGarciaAtlasCell]:
        return MappingProxyType({cell.index: cell for cell in self.cells})

    @property
    def x_cells(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.candidate.get("X", ()))

    @property
    def s_cells(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.candidate.get("S", ()))

    @property
    def a_cells(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.candidate.get("A", ()))

    @property
    def discovery_gates_passed(self) -> bool:
        return self.candidate["discovery_gates_passed"] is True


def _strict_bool(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be a JSON boolean")
    return value


def _nonnegative_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _read_payload(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream), str(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream), str(path)


def load_persisted_garcia_relation(
    source: str | Path | Mapping[str, Any],
) -> PersistedGarciaRelation:
    """Parse and structurally validate one persisted local-relation artifact."""

    payload, source_path = _read_payload(source)
    if payload.get("schema") != GARCIA_RELATION_SCHEMA:
        raise ValueError(
            f"expected schema {GARCIA_RELATION_SCHEMA!r}, got "
            f"{payload.get('schema')!r}"
        )
    metadata = payload.get("metadata")
    charts = payload.get("charts")
    candidate = payload.get("candidate")
    morse_graph = payload.get("morse_graph")
    reference_audit = payload.get("reference_audit")
    raw_cells = payload.get("cells")
    if not isinstance(metadata, Mapping) or not isinstance(charts, Mapping):
        raise ValueError("Garcia relation artifact is missing metadata or charts")
    if not isinstance(candidate, Mapping) or not isinstance(raw_cells, list):
        raise ValueError("Garcia relation artifact is missing candidate or cells")
    if not isinstance(morse_graph, Mapping) or not isinstance(reference_audit, Mapping):
        raise ValueError("Garcia relation artifact is missing Morse/reference audit data")
    _strict_bool(
        candidate.get("discovery_gates_passed"),
        "candidate.discovery_gates_passed",
    )

    cells: list[PersistedGarciaAtlasCell] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("Garcia cell records must be mappings")
        coordinates = tuple(int(value) for value in raw.get("coordinates", ()))
        bounds = tuple(float(value) for value in raw.get("bounds", ()))
        if len(coordinates) != 4 or len(bounds) != 8:
            raise ValueError("Garcia cells need four dyadic coordinates and eight bounds")
        relation_evaluated = _strict_bool(
            raw.get("relation_evaluated"),
            f"cell {raw.get('index')}.relation_evaluated",
        )
        raw_open_exit = raw.get("open_exit")
        if not isinstance(raw_open_exit, Mapping):
            raise ValueError(f"cell {raw.get('index')} is missing open-exit provenance")
        has_open_exit = _strict_bool(
            raw_open_exit.get("has_open_exit"),
            f"cell {raw.get('index')}.open_exit.has_open_exit",
        )
        unresolved_edges: list[Any] = []
        if "unresolved_stage_edges" in raw_open_exit:
            unresolved_edges = raw_open_exit["unresolved_stage_edges"]
            if not isinstance(unresolved_edges, list) or any(
                not isinstance(edge, list)
                or len(edge) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in edge
                )
                for edge in unresolved_edges
            ):
                raise ValueError(
                    f"cell {raw.get('index')}.open_exit.unresolved_stage_edges "
                    "must be a list of integer edge pairs"
                )
        exit_counts = {
            name: _nonnegative_int(
                raw_open_exit.get(name),
                f"cell {raw.get('index')}.open_exit.{name}",
            )
            for name in (
                "failed_samples_per_invocation",
                "missing_in_domain_target_cells",
                "ambient_boundary_pieces",
                "wholly_outside_active_family_pieces",
            )
        }
        exit_flags = {
            name: _strict_bool(
                raw_open_exit.get(name),
                f"cell {raw.get('index')}.open_exit.{name}",
            )
            for name in ("explicit_empty_callback", "mapgraph_empty_image")
        }
        recomputed_open_exit = bool(
            any(exit_counts.values())
            or unresolved_edges
            or any(exit_flags.values())
        )
        if has_open_exit != recomputed_open_exit:
            raise ValueError(
                f"cell {raw.get('index')}.open_exit.has_open_exit disagrees "
                "with its raw exit provenance"
            )
        cell = PersistedGarciaAtlasCell(
            index=int(raw["index"]),
            chart_id=int(raw["chart_id"]),
            axis_depth=int(raw["axis_depth"]),
            coordinates=coordinates,  # type: ignore[arg-type]
            bounds=bounds,
            image=frozenset(int(value) for value in raw.get("image", ())),
            relation_evaluated=relation_evaluated,
            open_exit=MappingProxyType(dict(raw_open_exit)),
        )
        if exit_flags["mapgraph_empty_image"] != (not cell.image):
            raise ValueError(
                f"cell {cell.index}.open_exit.mapgraph_empty_image disagrees "
                "with its stored relation image"
            )
        if cell.chart_id not in (BASE_CHART_ID, HANDLE_CHART_ID):
            raise ValueError(f"cell {cell.index} uses unknown chart {cell.chart_id}")
        if cell.axis_depth < 0:
            raise ValueError(f"cell {cell.index} has negative dyadic depth")
        subdivisions = 2**cell.axis_depth
        if any(value < 0 or value >= subdivisions for value in cell.coordinates):
            raise ValueError(f"cell {cell.index} has out-of-range dyadic coordinates")
        if not all(math.isfinite(value) for value in cell.bounds):
            raise ValueError(f"cell {cell.index} has nonfinite bounds")
        cells.append(cell)

    ordered = tuple(sorted(cells, key=lambda cell: cell.index))
    if tuple(cell.index for cell in ordered) != tuple(range(len(ordered))):
        raise ValueError("persisted Garcia Atlas indices must be contiguous from zero")
    descriptors = {
        (cell.chart_id, cell.axis_depth, cell.coordinates) for cell in ordered
    }
    if len(descriptors) != len(ordered):
        raise ValueError("persisted Garcia cells contain duplicate dyadic geometry")
    for cell in ordered:
        for ancestor_depth in range(cell.axis_depth):
            shift = cell.axis_depth - ancestor_depth
            ancestor = (
                cell.chart_id,
                ancestor_depth,
                tuple(value >> shift for value in cell.coordinates),
            )
            if ancestor in descriptors:
                raise ValueError("persisted Garcia dyadic family is not an antichain")
    known = frozenset(cell.index for cell in ordered)
    x_cells = frozenset(int(value) for value in candidate.get("X", ()))
    s_cells = frozenset(int(value) for value in candidate.get("S", ()))
    a_cells = frozenset(int(value) for value in candidate.get("A", ()))
    if not x_cells or not s_cells:
        raise ValueError("persisted Garcia candidate needs nonempty S and X")
    if x_cells != s_cells | a_cells or s_cells & a_cells:
        raise ValueError("candidate must satisfy X = S disjoint-union A")
    if not x_cells <= known:
        raise ValueError("candidate refers to unknown Atlas cells")
    for cell in ordered:
        unknown = cell.image.difference(known)
        if unknown:
            raise ValueError(
                f"relation from cell {cell.index} refers to unknown targets "
                f"{sorted(unknown)!r}"
            )

    cell_by_index = {cell.index: cell for cell in ordered}
    f_s = frozenset(
        target for source_index in s_cells for target in cell_by_index[source_index].image
    )
    expected_x = s_cells | f_s
    expected_a = f_s - s_cells
    if x_cells != expected_x or a_cells != expected_a:
        raise ValueError(
            "candidate equations disagree with the stored relation: expected "
            f"X=S union F(S)={sorted(expected_x)!r}, "
            f"A=F(S)\\S={sorted(expected_a)!r}"
        )

    graph = nx.DiGraph()
    graph.add_nodes_from(known)
    graph.add_edges_from(
        (cell.index, target) for cell in ordered for target in cell.image
    )
    recurrent_components = [
        frozenset(component)
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1
        or any(graph.has_edge(index, index) for index in component)
    ]
    if s_cells not in recurrent_components:
        raise ValueError("candidate S is not a recurrent SCC of the stored relation")

    morse_node = candidate.get("morse_node")
    if isinstance(morse_node, bool) or not isinstance(morse_node, int):
        raise ValueError("candidate.morse_node must be an integer")
    morse_sets = morse_graph.get("morse_sets")
    if not isinstance(morse_sets, Mapping) or str(morse_node) not in morse_sets:
        raise ValueError("candidate Morse node is absent from stored morse_sets")
    stored_morse_set = frozenset(int(value) for value in morse_sets[str(morse_node)])
    if stored_morse_set != s_cells:
        raise ValueError("candidate S disagrees with its stored Morse set")

    computed_pair_violations = tuple(
        source_index
        for source_index in sorted(a_cells)
        if (cell_by_index[source_index].image & x_cells) - a_cells
    )
    stored_pair_violations = tuple(
        int(value)
        for value in candidate.get("pair_second_condition_violations", ())
    )
    if stored_pair_violations != computed_pair_violations:
        raise ValueError(
            "stored pair-second-condition audit disagrees with the raw relation"
        )

    computed_open_exit_s = tuple(
        source_index
        for source_index in sorted(s_cells)
        if cell_by_index[source_index].open_exit["has_open_exit"] is True
    )
    if "open_exit_sources_in_S" in candidate and tuple(
        int(value) for value in candidate["open_exit_sources_in_S"]
    ) != computed_open_exit_s:
        raise ValueError("stored S open-exit audit disagrees with source provenance")
    computed_unresolved_s = tuple(
        source_index
        for source_index in sorted(s_cells)
        if cell_by_index[source_index].open_exit.get("unresolved_stage_edges", [])
    )
    if "unresolved_stage_sources_in_S" in candidate and tuple(
        int(value) for value in candidate["unresolved_stage_sources_in_S"]
    ) != computed_unresolved_s:
        raise ValueError(
            "stored S unresolved-stage audit disagrees with source provenance"
        )
    computed_missing_in_domain_s = tuple(
        source_index
        for source_index in sorted(s_cells)
        if cell_by_index[source_index].open_exit[
            "missing_in_domain_target_cells"
        ]
        > 0
    )
    if "missing_in_domain_sources_in_S" in candidate and tuple(
        int(value) for value in candidate["missing_in_domain_sources_in_S"]
    ) != computed_missing_in_domain_s:
        raise ValueError(
            "stored S missing-support audit disagrees with source provenance"
        )

    _nonnegative_int(reference_audit.get("misses"), "reference_audit.misses")
    _nonnegative_int(
        reference_audit.get("evaluation_failures"),
        "reference_audit.evaluation_failures",
    )
    for candidate_key, audit_key in (
        ("reference_endpoint_misses", "misses"),
        ("reference_evaluation_failures", "evaluation_failures"),
    ):
        if candidate_key in candidate and _nonnegative_int(
            candidate[candidate_key], f"candidate.{candidate_key}"
        ) != reference_audit[audit_key]:
            raise ValueError(
                f"candidate.{candidate_key} disagrees with reference_audit.{audit_key}"
            )

    return PersistedGarciaRelation(
        metadata=MappingProxyType(dict(metadata)),
        chart_payload=MappingProxyType(dict(charts)),
        candidate=MappingProxyType(dict(candidate)),
        morse_graph=MappingProxyType(dict(morse_graph)),
        reference_audit=MappingProxyType(dict(reference_audit)),
        cells=ordered,
        source_path=source_path,
    )


@dataclass(frozen=True)
class GarciaAttachmentCarrierRecord:
    map_name: str
    source_cell: CubicalCell
    sample_count: int
    sampled_bounds: tuple[float, ...]
    bloated_bounds: tuple[float, ...]
    target_top_coordinates: tuple[tuple[int, int, int, int], ...]
    ambient_clip_axes: tuple[int, ...]


@dataclass(frozen=True)
class GarciaMappingCylinderTopology:
    relation: PersistedGarciaRelation
    charts: SuspensionAtlasCharts
    axis_depth: int
    base_complex: SparseCubicalGridComplex
    guard_complex: SparseCubicalGridComplex
    guard_carrier: CrossComplexAcyclicCarrier
    reset_carrier: CrossComplexAcyclicCarrier
    guard_attachment: CellularAttachmentMap
    reset_attachment: CellularAttachmentMap
    cylinder: DoubleMappingCylinderComplex
    registry: AtlasMappingCylinderRegistry
    pair: AtlasMappingCylinderRelativePair
    attachment_records: tuple[GarciaAttachmentCarrierRecord, ...]
    candidate_base_coordinates: frozenset[tuple[int, int, int, int]]
    attachment_support_coordinates: frozenset[tuple[int, int, int, int]]
    auxiliary_base_coordinates: frozenset[tuple[int, int, int, int]]
    attachment_samples_per_axis: int
    attachment_padding_cells: float
    outer_cover_assumption: str = SAMPLED_BLOAT_ASSUMPTION

    @property
    def continuous_system_conley_index_certified(self) -> bool:
        return False


def _stored_chart_bounds(
    relation: PersistedGarciaRelation,
    label: str,
) -> tuple[tuple[float, float], ...]:
    raw = relation.chart_payload.get(label)
    if not isinstance(raw, Mapping):
        raise ValueError(f"artifact is missing {label!r} chart metadata")
    bounds = tuple(tuple(float(value) for value in interval) for interval in raw["bounds"])
    if any(len(interval) != 2 for interval in bounds):
        raise ValueError(f"artifact {label} chart bounds are malformed")
    return bounds  # type: ignore[return-value]


def _cell_parameter_points(
    cell: CubicalCell,
    chart_bounds: Sequence[Sequence[float]],
    subdivisions: int,
    samples_per_axis: int,
) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for axis, interval in enumerate(chart_bounds):
        lower = float(interval[0]) + (
            float(interval[1]) - float(interval[0])
        ) * cell.anchor[axis] / subdivisions
        if cell.spanning[axis]:
            upper = float(interval[0]) + (
                float(interval[1]) - float(interval[0])
            ) * (cell.anchor[axis] + 1) / subdivisions
            axes.append(np.linspace(lower, upper, samples_per_axis))
        else:
            axes.append(np.asarray((lower,), dtype=np.float64))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in itertools.product(*axes)
    )


def _closed_grid_cover(
    bounds: tuple[float, ...],
    chart_bounds: Sequence[Sequence[float]],
    subdivisions: int,
) -> tuple[tuple[int, int, int, int], ...]:
    dimension = len(chart_bounds)
    lower = bounds[:dimension]
    upper = bounds[dimension:]
    coordinate_ranges = []
    tolerance = 1.0e-12
    for axis, interval in enumerate(chart_bounds):
        span = float(interval[1]) - float(interval[0])
        normalized_lower = (lower[axis] - float(interval[0])) / span
        normalized_upper = (upper[axis] - float(interval[0])) / span
        first = max(0, int(math.ceil(subdivisions * normalized_lower - tolerance)) - 1)
        last = min(
            subdivisions - 1,
            int(math.floor(subdivisions * normalized_upper + tolerance)),
        )
        coordinate_ranges.append(range(first, last + 1))
    return tuple(
        tuple(int(value) for value in coordinates)
        for coordinates in itertools.product(*coordinate_ranges)
    )  # type: ignore[return-value]


def _attachment_coordinate_generators(
    *,
    map_name: str,
    guard_complex: SparseCubicalGridComplex,
    charts: SuspensionAtlasCharts,
    samples_per_axis: int,
    padding_cells: float,
) -> tuple[
    dict[CubicalCell, tuple[tuple[int, int, int, int], ...]],
    tuple[GarciaAttachmentCarrierRecord, ...],
]:
    subdivisions = guard_complex.subdivisions[0]
    base_bounds = tuple(tuple(float(value) for value in item) for item in charts.base_bounds)
    base_widths = tuple(
        (interval[1] - interval[0]) / subdivisions for interval in base_bounds
    )
    coordinate_generators: dict[
        CubicalCell, tuple[tuple[int, int, int, int], ...]
    ] = {}
    records: list[GarciaAttachmentCarrierRecord] = []

    for source in guard_complex.cells:
        parameters = _cell_parameter_points(
            source,
            charts.guard_bounds,
            subdivisions,
            samples_per_axis,
        )
        images: list[np.ndarray] = []
        for intrinsic in parameters:
            guard = np.asarray(charts.guard_embedding(intrinsic), dtype=np.float64)
            if guard.shape != (4,) or not np.all(np.isfinite(guard)):
                raise ValueError("Garcia guard embedding returned an invalid state")
            if abs(float(guard[2] - 2.0 * guard[0])) > 1.0e-10:
                raise ValueError("Garcia guard embedding violates phi = 2 theta")
            image = guard if map_name == "guard" else np.asarray(reset_map(guard))
            if image.shape != (4,) or not np.all(np.isfinite(image)):
                raise ValueError(f"Garcia {map_name} map returned an invalid state")
            images.append(image)

        stacked = np.vstack(images)
        sampled_lower = np.min(stacked, axis=0)
        sampled_upper = np.max(stacked, axis=0)
        raw_lower = sampled_lower - padding_cells * np.asarray(base_widths)
        raw_upper = sampled_upper + padding_cells * np.asarray(base_widths)
        ambient_lower = np.asarray([interval[0] for interval in base_bounds])
        ambient_upper = np.asarray([interval[1] for interval in base_bounds])
        if np.any(sampled_lower < ambient_lower - 1.0e-12) or np.any(
            sampled_upper > ambient_upper + 1.0e-12
        ):
            raise ValueError(
                f"sampled Garcia {map_name} image leaves the declared base chart"
            )
        clip_axes = tuple(
            int(axis)
            for axis in range(4)
            if raw_lower[axis] < ambient_lower[axis]
            or raw_upper[axis] > ambient_upper[axis]
        )
        bloated_lower = np.maximum(raw_lower, ambient_lower)
        bloated_upper = np.minimum(raw_upper, ambient_upper)
        if np.any(bloated_lower > bloated_upper):
            raise ValueError(f"Garcia {map_name} bloat misses the ambient base chart")
        bloated = tuple(float(value) for value in (*bloated_lower, *bloated_upper))
        target_coordinates = _closed_grid_cover(
            bloated,
            base_bounds,
            subdivisions,
        )
        if not target_coordinates:
            raise ValueError(f"sampled/bloated {map_name} carrier is empty")
        coordinate_generators[source] = target_coordinates
        records.append(
            GarciaAttachmentCarrierRecord(
                map_name=map_name,
                source_cell=source,
                sample_count=len(parameters),
                sampled_bounds=tuple(
                    float(value) for value in (*sampled_lower, *sampled_upper)
                ),
                bloated_bounds=bloated,
                target_top_coordinates=target_coordinates,
                ambient_clip_axes=clip_axes,
            )
        )
    return coordinate_generators, tuple(records)


def build_garcia_mapping_cylinder_topology(
    source: str | Path | Mapping[str, Any] | PersistedGarciaRelation,
    *,
    attachment_samples_per_axis: int = 3,
    attachment_padding_cells: float = 1.0,
) -> GarciaMappingCylinderTopology:
    """Build the conditional Garcia reset quotient from one persisted relation."""

    relation = (
        source
        if isinstance(source, PersistedGarciaRelation)
        else load_persisted_garcia_relation(source)
    )
    if attachment_samples_per_axis < 2:
        raise ValueError("attachment_samples_per_axis must be at least two")
    if not math.isfinite(attachment_padding_cells) or attachment_padding_cells <= 0.0:
        raise ValueError("attachment_padding_cells must be finite and positive")
    if not all(relation.cell_by_index[index].relation_evaluated for index in relation.x_cells):
        raise ValueError("MapGraph relation was not evaluated on every candidate X source")
    if int(relation.metadata.get("base_chart_id", BASE_CHART_ID)) != BASE_CHART_ID:
        raise ValueError("persisted Garcia base chart id disagrees with the adapter")
    if int(relation.metadata.get("handle_chart_id", HANDLE_CHART_ID)) != HANDLE_CHART_ID:
        raise ValueError("persisted Garcia handle chart id disagrees with the adapter")

    candidate_depths = {
        relation.cell_by_index[index].axis_depth for index in relation.x_cells
    }
    if len(candidate_depths) != 1:
        raise ValueError(
            "candidate X contains mixed dyadic depths; a conforming adaptive "
            "mapping-cylinder complex is required"
        )
    axis_depth = next(iter(candidate_depths))
    subdivisions = 2**axis_depth
    base_records = tuple(
        relation.cell_by_index[index]
        for index in sorted(relation.x_cells)
        if relation.cell_by_index[index].chart_id == BASE_CHART_ID
    )
    handle_records = tuple(
        relation.cell_by_index[index]
        for index in sorted(relation.x_cells)
        if relation.cell_by_index[index].chart_id == HANDLE_CHART_ID
    )
    if not base_records or not handle_records:
        raise ValueError("candidate X must contain both base and handle Atlas cells")

    stored_base_bounds = _stored_chart_bounds(relation, "base")
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=stored_base_bounds,
        guard_delta=float(relation.metadata.get("guard_delta", 0.05)),
        transversality_eta=float(
            relation.metadata.get("transversality_eta", 0.10)
        ),
    )
    stored_handle_bounds = _stored_chart_bounds(relation, "handle")
    if not np.allclose(
        np.asarray(stored_handle_bounds),
        np.asarray(charts.handle_bounds),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("persisted handle chart disagrees with Garcia parametrization")
    for record in relation.cells:
        chart_bounds = charts.bounds_for(record.chart_id)
        record_subdivisions = 2**record.axis_depth
        lower = tuple(
            interval[0]
            + (interval[1] - interval[0]) * coordinate / record_subdivisions
            for interval, coordinate in zip(chart_bounds, record.coordinates)
        )
        upper = tuple(
            interval[0]
            + (interval[1] - interval[0])
            * (coordinate + 1)
            / record_subdivisions
            for interval, coordinate in zip(chart_bounds, record.coordinates)
        )
        expected = np.asarray((*lower, *upper), dtype=np.float64)
        if not np.allclose(
            np.asarray(record.bounds), expected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"persisted bounds of Atlas cell {record.index} disagree with "
                "its dyadic chart address"
            )

    guard_coordinates = tuple(
        sorted({record.coordinates[:3] for record in handle_records})
    )
    guard_complex = SparseCubicalGridComplex(
        (subdivisions,) * 3,
        guard_coordinates,
    )
    guard_coordinate_generators, guard_records = _attachment_coordinate_generators(
        map_name="guard",
        guard_complex=guard_complex,
        charts=charts,
        samples_per_axis=attachment_samples_per_axis,
        padding_cells=float(attachment_padding_cells),
    )
    reset_coordinate_generators, reset_records = _attachment_coordinate_generators(
        map_name="reset",
        guard_complex=guard_complex,
        charts=charts,
        samples_per_axis=attachment_samples_per_axis,
        padding_cells=float(attachment_padding_cells),
    )
    candidate_base_coordinates = frozenset(
        record.coordinates for record in base_records
    )
    attachment_support_coordinates = frozenset(
        coordinates
        for coordinate_generators in (
            guard_coordinate_generators,
            reset_coordinate_generators,
        )
        for targets in coordinate_generators.values()
        for coordinates in targets
    )
    ambient_base_coordinates = (
        candidate_base_coordinates | attachment_support_coordinates
    )
    base_complex = SparseCubicalGridComplex(
        (subdivisions,) * 4,
        ambient_base_coordinates,
    )
    guard_generators = {
        source_cell: tuple(
            base_complex.top_cell_at(coordinates) for coordinates in targets
        )
        for source_cell, targets in guard_coordinate_generators.items()
    }
    reset_generators = {
        source_cell: tuple(
            base_complex.top_cell_at(coordinates) for coordinates in targets
        )
        for source_cell, targets in reset_coordinate_generators.items()
    }
    guard_carrier = CrossComplexAcyclicCarrier(
        guard_complex,
        base_complex,
        guard_generators,
        modulus=5,
        validate_acyclic=True,
    )
    reset_carrier = CrossComplexAcyclicCarrier(
        guard_complex,
        base_complex,
        reset_generators,
        modulus=5,
        validate_acyclic=True,
    )
    guard_attachment = guard_carrier.construct_integral_attachment_map()
    reset_attachment = reset_carrier.construct_integral_attachment_map()
    cylinder = DoubleMappingCylinderComplex(
        base_complex,
        (
            DoubleMappingCylinderHandle(
                GARCIA_HANDLE_ID,
                guard_complex,
                guard_attachment,
                reset_attachment,
                slabs=subdivisions,
            ),
        ),
    )

    registered: list[AtlasMappingCylinderTopCell] = []
    for record in (*base_records, *handle_records):
        if record.chart_id == BASE_CHART_ID:
            complex_cell = cylinder.base_cell(
                base_complex.top_cell_at(record.coordinates)
            )
        else:
            complex_cell = cylinder.prism_cell(
                GARCIA_HANDLE_ID,
                guard_complex.top_cell_at(record.coordinates[:3]),
                record.coordinates[3],
            )
        registered.append(
            AtlasMappingCylinderTopCell(record.index, record.chart_id, complex_cell)
        )
    registry = AtlasMappingCylinderRegistry(
        cylinder,
        registered,
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )
    pair = AtlasMappingCylinderRelativePair(
        registry,
        relation.x_cells,
        relation.a_cells,
        index_pair_certified=False,
    )
    return GarciaMappingCylinderTopology(
        relation=relation,
        charts=charts,
        axis_depth=axis_depth,
        base_complex=base_complex,
        guard_complex=guard_complex,
        guard_carrier=guard_carrier,
        reset_carrier=reset_carrier,
        guard_attachment=guard_attachment,
        reset_attachment=reset_attachment,
        cylinder=cylinder,
        registry=registry,
        pair=pair,
        attachment_records=guard_records + reset_records,
        candidate_base_coordinates=candidate_base_coordinates,
        attachment_support_coordinates=attachment_support_coordinates,
        auxiliary_base_coordinates=(
            attachment_support_coordinates - candidate_base_coordinates
        ),
        attachment_samples_per_axis=int(attachment_samples_per_axis),
        attachment_padding_cells=float(attachment_padding_cells),
    )


def _exit_components(
    topology: GarciaMappingCylinderTopology,
) -> Mapping[int, frozenset[int]]:
    pair = topology.pair
    closures = {
        index: pair.complex.closure((topology.registry.complex_cell(index),))
        for index in pair.p0_atlas_cells
    }
    adjacency = {index: set() for index in pair.p0_atlas_cells}
    for first, second in itertools.combinations(sorted(pair.p0_atlas_cells), 2):
        if closures[first].intersection(closures[second]):
            adjacency[first].add(second)
            adjacency[second].add(first)
    result: dict[int, frozenset[int]] = {}
    remaining = set(pair.p0_atlas_cells)
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            neighbors = adjacency[current].intersection(remaining)
            remaining.difference_update(neighbors)
            component.update(neighbors)
            frontier.extend(neighbors)
        frozen = frozenset(component)
        for index in component:
            result[index] = frozen
    return MappingProxyType(result)


def _coface_intersection_carrier(
    topology: GarciaMappingCylinderTopology,
) -> tuple[Mapping[Any, Collection[Any]], Mapping[int, frozenset[int]]]:
    relation = topology.relation
    pair = topology.pair
    exit_components = _exit_components(topology)
    raw_relation: dict[int, frozenset[int]] = {}
    top_values: dict[int, frozenset[Any]] = {}

    for source in sorted(pair.p1_atlas_cells):
        raw_targets = relation.cell_by_index[source].image
        raw_relation[source] = raw_targets
        if source in pair.p0_atlas_cells:
            # This is only an auxiliary pair-preserving carrier value.  The
            # raw relation, including targets outside X, remains recorded in
            # ``raw_relation`` and is never rewritten as component edges.
            targets = exit_components[source]
        else:
            targets = raw_targets
            outside = targets.difference(pair.p1_atlas_cells)
            if outside:
                raise ValueError(
                    f"non-exit source {source} leaves candidate X through "
                    f"{sorted(outside)!r}"
                )
            if not targets:
                raise ValueError(f"non-exit source {source} has empty MapGraph image")
        top_values[source] = pair.complex.closure(
            topology.registry.complex_cell(target) for target in targets
        )

    cofaces: dict[Any, list[int]] = {cell: [] for cell in pair.complex.cells}
    for index in sorted(pair.p1_atlas_cells):
        closure = pair.complex.closure((topology.registry.complex_cell(index),))
        for cell in closure:
            cofaces[cell].append(index)

    generators: dict[Any, Collection[Any]] = {}
    for cell in pair.complex.cells:
        containing = cofaces[cell]
        if not containing:
            raise AssertionError(f"relative-complex cell {cell!r} has no Atlas coface")
        value = set(top_values[containing[0]])
        for index in containing[1:]:
            value.intersection_update(top_values[index])
        if not value:
            raise ValueError(
                "coface-intersection fixed-time carrier is empty on cell "
                f"{cell!r}; face-level enclosure data or refinement is required"
            )
        generators[cell] = frozenset(value)
    return MappingProxyType(generators), MappingProxyType(raw_relation)


def prepare_persisted_garcia_finite_conley(
    topology: GarciaMappingCylinderTopology,
) -> AtlasMappingCylinderConleyPreparation:
    """Construct the conditional finite shift-class input from persisted data."""

    relation = topology.relation
    if not relation.discovery_gates_passed:
        raise ValueError(
            "persisted Garcia candidate failed its discovery/exit gates; no finite "
            "Conley result is promoted"
        )
    unevaluated = tuple(
        source
        for source in sorted(relation.s_cells)
        if not relation.cell_by_index[source].relation_evaluated
    )
    if unevaluated:
        raise ValueError(f"candidate S contains unevaluated sources {unevaluated!r}")
    missing_in_domain = tuple(
        source
        for source in sorted(relation.s_cells)
        if _nonnegative_int(
            relation.cell_by_index[source].open_exit[
                "missing_in_domain_target_cells"
            ],
            f"cell {source}.open_exit.missing_in_domain_target_cells",
        )
        > 0
    )
    if missing_in_domain:
        raise ValueError(
            "candidate S has in-domain targets omitted by the active family at "
            f"sources {missing_in_domain!r}"
        )
    open_exits = tuple(
        source
        for source in sorted(relation.s_cells)
        if relation.cell_by_index[source].open_exit["has_open_exit"] is True
    )
    if open_exits:
        raise ValueError(
            f"candidate S contains explicit open-exit sources {open_exits!r}"
        )
    unresolved = tuple(
        source
        for source in sorted(relation.s_cells)
        if relation.cell_by_index[source].open_exit.get(
            "unresolved_stage_edges", []
        )
    )
    if unresolved:
        raise ValueError(
            f"candidate S contains unresolved-stage sources {unresolved!r}"
        )
    pair_violations = tuple(
        source
        for source in sorted(relation.a_cells)
        if (relation.cell_by_index[source].image & relation.x_cells)
        - relation.a_cells
    )
    if pair_violations:
        raise ValueError(
            "persisted Garcia candidate violates the second pair condition at "
            f"sources {pair_violations!r}"
        )
    reference_misses = _nonnegative_int(
        relation.reference_audit["misses"], "reference_audit.misses"
    )
    reference_failures = _nonnegative_int(
        relation.reference_audit["evaluation_failures"],
        "reference_audit.evaluation_failures",
    )
    if reference_misses or reference_failures:
        raise ValueError(
            "persisted Garcia reference audit is incomplete: "
            f"misses={reference_misses}, evaluation_failures={reference_failures}"
        )
    missing_labels = tuple(relation.candidate.get("reference_missing_labels", ()))
    labels_total = _nonnegative_int(
        relation.candidate.get("reference_labels_total", 0),
        "candidate.reference_labels_total",
    )
    labels_recovered = _nonnegative_int(
        relation.candidate.get("reference_labels_recovered", 0),
        "candidate.reference_labels_recovered",
    )
    if not labels_total or labels_recovered != labels_total or missing_labels:
        raise ValueError(
            "persisted Garcia reference labels were not all recovered by candidate S"
        )

    boundary_sources: list[int] = []
    for source in sorted(relation.s_cells):
        record = relation.cell_by_index[source]
        chart_bounds = topology.charts.bounds_for(record.chart_id)
        lower = record.bounds[:4]
        upper = record.bounds[4:]
        touches = any(
            not (
                record.chart_id == HANDLE_CHART_ID and axis == 3
            )
            and (
                abs(lower[axis] - interval[0]) <= 1.0e-12
                or abs(upper[axis] - interval[1]) <= 1.0e-12
            )
            for axis, interval in enumerate(chart_bounds)
        )
        if touches:
            boundary_sources.append(source)
    if boundary_sources:
        raise ValueError(
            "candidate S touches a nonglued ambient-chart boundary at sources "
            f"{tuple(boundary_sources)!r}"
        )

    generators, raw_relation = _coface_intersection_carrier(topology)
    return prepare_atlas_mapping_cylinder_conley(
        topology.pair,
        top_relation=raw_relation,
        cell_carrier_generators=generators,
        outer_enclosure_certified=False,
        modulus=5,
    )


__all__ = [
    "GARCIA_HANDLE_ID",
    "GARCIA_RELATION_SCHEMA",
    "SAMPLED_BLOAT_ASSUMPTION",
    "GarciaAttachmentCarrierRecord",
    "GarciaMappingCylinderTopology",
    "PersistedGarciaAtlasCell",
    "PersistedGarciaRelation",
    "build_garcia_mapping_cylinder_topology",
    "load_persisted_garcia_relation",
    "prepare_persisted_garcia_finite_conley",
]
