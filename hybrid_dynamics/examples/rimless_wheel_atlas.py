"""Rimless-wheel computation through CMGDB's tagged suspension box map.

This module is the first physical integration of
``CMGDB.AtlasModel``.  CMGDB still owns the finite grid, graph, SCC, and
Morse-order computation; the only hybrid-specific replacement is its box-map
callback.  The two charts are

* the physical base chart ``(theta, omega)``; and
* the intrinsic reset-handle chart ``(omega_guard, s)``.

The handle is attached only to the outgoing part of the impact section
(``omega_guard >= 0``).  The wheel's event is configured for increasing
crossings, so adding a handle at negative angular velocity would introduce a
reset that the hybrid system does not execute.

The acceptance diagnostics are intentionally stronger than merely obtaining
a Morse graph.  They probe a dense analytic walking-cycle reference, test
every nonempty recorded image for connectedness in the reset-glued quotient,
test the gait candidate's base support, and identify the saddle-to-gait Morse
order.  These finite checks can falsify a sampled relation but cannot certify
the whole-cell outer-enclosure hypothesis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from ..src.atlas_conley import (
    AffineBoundaryEmbedding2D,
    AtlasResetGluing2D,
)
from ..src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    SuspensionAtlasCharts,
    SuspensionBoxMapDiagnostics,
    build_cmgdb_atlas_model,
)
from ..src.fixed_time_relation_audit import (
    CellSetConnectivity,
    EndpointCoverageAudit,
    EndpointCoverageWitness,
    EndpointProbe,
    RelationConnectivityAudit,
)
from .rimless_wheel import RimlessWheel
from .rimless_wheel_suspension import walking_fixed_point_speed


BASE_CHART_ID = 0
HANDLE_CHART_ID = 1
DEFAULT_BASE_BOUNDS = ((-0.2, 0.6), (-0.5, 1.0))
DEFAULT_GUARD_SPEED_BOUNDS = (0.0, 1.0)
DEFAULT_T_STAR = 2.0


@dataclass(frozen=True, order=True)
class AtlasWheelCell:
    """One CMGDB cell with its chart tag and closed rectangle."""

    index: int
    chart_id: int
    bounds: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.bounds) // 2

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.bounds[: self.dimension], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.bounds[self.dimension :], dtype=np.float64)


class RimlessWheelQuotientIncidence:
    """Closed-cell incidence after guard/reset seam identification."""

    def __init__(
        self,
        charts: SuspensionAtlasCharts,
        *,
        alpha: float,
        gamma: float,
        atol: float = 1e-12,
    ) -> None:
        self.charts = charts
        self.guard_angle = float(alpha + gamma)
        self.reset_angle = float(gamma - alpha)
        self.impact_factor = float(np.cos(2.0 * alpha))
        self.atol = float(atol)

    def _rectangles_intersect(
        self,
        first_lower: np.ndarray,
        first_upper: np.ndarray,
        second_lower: np.ndarray,
        second_upper: np.ndarray,
    ) -> bool:
        return bool(
            np.all(first_lower <= second_upper + self.atol)
            and np.all(second_lower <= first_upper + self.atol)
        )

    def intersects(self, first: AtlasWheelCell, second: AtlasWheelCell) -> bool:
        if first == second:
            return True
        if first.chart_id == second.chart_id:
            return self._rectangles_intersect(
                first.lower,
                first.upper,
                second.lower,
                second.upper,
            )
        if {first.chart_id, second.chart_id} != {
            self.charts.base_chart_id,
            self.charts.handle_chart_id,
        }:
            return False
        base = first if first.chart_id == self.charts.base_chart_id else second
        handle = first if first.chart_id == self.charts.handle_chart_id else second
        omega_lower = float(handle.lower[0])
        omega_upper = float(handle.upper[0])

        if handle.lower[-1] <= self.atol:
            guard_lower = np.asarray((self.guard_angle, omega_lower))
            guard_upper = np.asarray((self.guard_angle, omega_upper))
            if self._rectangles_intersect(
                base.lower, base.upper, guard_lower, guard_upper
            ):
                return True
        if handle.upper[-1] >= 1.0 - self.atol:
            reset_omega = sorted(
                (self.impact_factor * omega_lower, self.impact_factor * omega_upper)
            )
            reset_lower = np.asarray((self.reset_angle, reset_omega[0]))
            reset_upper = np.asarray((self.reset_angle, reset_omega[1]))
            if self._rectangles_intersect(
                base.lower, base.upper, reset_lower, reset_upper
            ):
                return True
        return False

    def connected_components(
        self,
        cells: Collection[AtlasWheelCell],
    ) -> tuple[frozenset[AtlasWheelCell], ...]:
        remaining = set(cells)
        components: list[frozenset[AtlasWheelCell]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                neighbors = {
                    candidate
                    for candidate in remaining
                    if self.intersects(current, candidate)
                }
                remaining.difference_update(neighbors)
                component.update(neighbors)
                stack.extend(neighbors)
            components.append(frozenset(component))
        components.sort(key=lambda component: min(component))
        return tuple(components)


@dataclass(frozen=True)
class RimlessWheelAtlasSetup:
    """Objects needed to run the native CMGDB computation."""

    wheel: RimlessWheel
    charts: SuspensionAtlasCharts
    box_map: CMGDBSuspensionBoxMap
    model: Any


@dataclass(frozen=True)
class RimlessWheelAtlasAcceptance:
    """Morse output and falsification diagnostics for one fixed depth."""

    depth: int
    subdivisions: int
    t_star: float
    morse_graph: Any
    map_graph: Any
    cell_tokens: tuple[AtlasWheelCell, ...]
    relation: Mapping[AtlasWheelCell, frozenset[AtlasWheelCell]]
    gait_node: int | None
    saddle_node: int | None
    gait_reference_node_hits: Mapping[int, int]
    reference_endpoint_audit: EndpointCoverageAudit
    image_connectivity_audit: RelationConnectivityAudit
    empty_image_sources: tuple[AtlasWheelCell, ...]
    gait_base_connectivity: CellSetConnectivity | None
    saddle_to_gait_path: tuple[int, ...]
    saddle_to_gait_direct: bool
    box_map_diagnostics: SuspensionBoxMapDiagnostics

    @property
    def morse_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                (int(source), int(target))
                for source, target in self.morse_graph.edges()
            )
        )

    @property
    def reference_covered(self) -> bool:
        return self.reference_endpoint_audit.passed

    @property
    def image_values_connected(self) -> bool:
        return self.image_connectivity_audit.passed

    @property
    def gait_base_connected(self) -> bool:
        return bool(
            self.gait_base_connectivity is not None
            and self.gait_base_connectivity.connected
        )

    def summary(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "subdivisions_per_axis": self.subdivisions,
            "t_star": self.t_star,
            "relation_scope": "nonempty local-window relation",
            "map_cells": int(self.map_graph.num_vertices()),
            "morse_nodes": int(self.morse_graph.num_vertices()),
            "morse_edges": [list(edge) for edge in self.morse_edges],
            "gait_node": self.gait_node,
            "saddle_node": self.saddle_node,
            "saddle_to_gait_path": list(self.saddle_to_gait_path),
            "saddle_to_gait_direct": self.saddle_to_gait_direct,
            "reference_probes": len(self.reference_endpoint_audit.witnesses),
            "reference_misses": len(self.reference_endpoint_audit.missed),
            "reference_evaluation_failures": len(
                self.reference_endpoint_audit.evaluation_failures
            ),
            "nonempty_images": len(self.image_connectivity_audit.images),
            "disconnected_images": len(self.image_connectivity_audit.disconnected),
            "empty_images": len(self.empty_image_sources),
            "empty_sources_excluded_from_connectivity_audit": True,
            "gait_base_connected": self.gait_base_connected,
            "gait_base_component_sizes": (
                []
                if self.gait_base_connectivity is None
                else [
                    len(component)
                    for component in self.gait_base_connectivity.components
                ]
            ),
            "box_map_failed_samples": self.box_map_diagnostics.failed_samples,
            "box_map_unresolved_stage_edges": (
                self.box_map_diagnostics.unresolved_stage_edges
            ),
            "whole_cell_outer_enclosure_certified": False,
            "global_attractor_lattice_interpretation": False,
            "conley_index_computed": False,
        }


def rimless_wheel_atlas_charts(
    *,
    base_bounds: Sequence[Sequence[float]] = DEFAULT_BASE_BOUNDS,
    guard_speed_bounds: Sequence[float] = DEFAULT_GUARD_SPEED_BOUNDS,
    guard_angle: float = 0.6,
) -> SuspensionAtlasCharts:
    """Return ``(theta,omega)`` and outgoing ``(omega_guard,s)`` charts."""

    bounds = tuple(
        (float(interval[0]), float(interval[1])) for interval in base_bounds
    )
    guard_interval = (
        float(guard_speed_bounds[0]),
        float(guard_speed_bounds[1]),
    )
    angle = float(guard_angle)
    return SuspensionAtlasCharts(
        base_bounds=bounds,
        guard_bounds=(guard_interval,),
        guard_coordinates=lambda guard: np.asarray([guard[1]], dtype=np.float64),
        guard_embedding=lambda intrinsic: np.asarray(
            [angle, intrinsic[0]], dtype=np.float64
        ),
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )


def rimless_wheel_atlas_reset_gluing(
    *, alpha: float = 0.4, gamma: float = 0.2
) -> AtlasResetGluing2D:
    """Return the exact affine guard/reset seam data for the Atlas nerve."""

    alpha = float(alpha)
    gamma = float(gamma)
    if not np.isfinite(alpha) or not np.isfinite(gamma):
        raise ValueError("alpha and gamma must be finite")
    impact_factor = float(np.cos(2.0 * alpha))
    if abs(impact_factor) <= 1.0e-15:
        raise ValueError("the initial cellular 2D adapter requires injective reset")
    return AtlasResetGluing2D(
        BASE_CHART_ID,
        HANDLE_CHART_ID,
        guard=AffineBoundaryEmbedding2D(0, alpha + gamma, 1.0),
        reset=AffineBoundaryEmbedding2D(0, gamma - alpha, impact_factor),
    )


def build_rimless_wheel_atlas_model(
    *,
    depth: int,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    alpha: float = 0.4,
    gamma: float = 0.2,
    max_step: float = 0.02,
    diagnostics_limit: int = 25_000,
) -> RimlessWheelAtlasSetup:
    """Build the wheel and install its tagged box map in ``AtlasModel``."""

    _validate_even_depth(depth)
    bounds = DEFAULT_BASE_BOUNDS
    wheel = RimlessWheel(
        domain_bounds=list(bounds),
        alpha=alpha,
        gamma=gamma,
        max_jumps=24,
    )
    charts = rimless_wheel_atlas_charts(
        base_bounds=bounds,
        guard_speed_bounds=(max(0.0, bounds[1][0]), bounds[1][1]),
        guard_angle=alpha + gamma,
    )
    box_map = CMGDBSuspensionBoxMap(
        wheel.system,
        charts,
        t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        max_jumps=24,
        max_step=max_step,
        require_domain_path=True,
        diagnostics_limit=diagnostics_limit,
    )
    model = build_cmgdb_atlas_model(box_map, depth=depth)
    return RimlessWheelAtlasSetup(
        wheel=wheel,
        charts=charts,
        box_map=box_map,
        model=model,
    )


def compute_rimless_wheel_atlas_acceptance(
    *,
    depth: int = 10,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    reference_base_samples: int = 161,
    reference_handle_samples: int = 81,
) -> RimlessWheelAtlasAcceptance:
    """Run CMGDB and audit the walking cycle, quotient, and Morse order."""

    setup = build_rimless_wheel_atlas_model(
        depth=depth,
        t_star=t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
    )
    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB fork is not installed") from error

    setup.box_map.reset_diagnostics()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(setup.model)
    subdivisions = _subdivisions_for_depth(depth)
    tokens = _atlas_cells(
        morse_graph,
        int(map_graph.num_vertices()),
    )
    relation = _atlas_relation(map_graph, tokens)
    incidence = RimlessWheelQuotientIncidence(
        setup.charts,
        alpha=setup.wheel.alpha,
        gamma=setup.wheel.gamma,
    )

    nonempty_relation = {
        source: targets for source, targets in relation.items() if targets
    }
    empty_sources = tuple(source for source, targets in relation.items() if not targets)
    image_audit = _audit_relation_connectivity(nonempty_relation, incidence)

    probes, reference_source_tokens = _walking_cycle_probes(
        setup,
        tokens,
        base_samples=reference_base_samples,
        handle_samples=reference_handle_samples,
    )
    reference_audit = _audit_atlas_endpoint_probes(
        relation,
        probes,
        tokens,
        setup.box_map,
    )
    node_by_cell = _morse_node_by_phase_cell(morse_graph)
    reference_hits = Counter(
        node_by_cell[index]
        for index, token in enumerate(tokens)
        if token in reference_source_tokens and index in node_by_cell
    )
    gait_node = _unique_largest_hit(reference_hits)

    saddle_tokens = _cells_containing_point(
        tokens,
        setup.charts.base_chart_id,
        (0.0, 0.0),
    )
    saddle_hits = Counter(
        node_by_cell[index]
        for index, token in enumerate(tokens)
        if token in saddle_tokens and index in node_by_cell
    )
    saddle_node = _unique_largest_hit(saddle_hits)

    gait_base_connectivity = None
    if gait_node is not None:
        gait_cells = {
            tokens[int(index)]
            for index in morse_graph.morse_set(gait_node)
            if tokens[int(index)].chart_id == setup.charts.base_chart_id
        }
        if gait_cells:
            components = incidence.connected_components(gait_cells)
            gait_base_connectivity = CellSetConnectivity(
                label="rimless-gait-base",
                cells=frozenset(gait_cells),
                components=components,
            )

    order = nx.DiGraph()
    order.add_nodes_from(int(node) for node in morse_graph.vertices())
    order.add_edges_from(
        (int(source), int(target)) for source, target in morse_graph.edges()
    )
    path: tuple[int, ...] = ()
    if (
        saddle_node is not None
        and gait_node is not None
        and nx.has_path(order, saddle_node, gait_node)
    ):
        path = tuple(
            int(node)
            for node in nx.shortest_path(order, saddle_node, gait_node)
        )

    return RimlessWheelAtlasAcceptance(
        depth=depth,
        subdivisions=subdivisions,
        t_star=float(t_star),
        morse_graph=morse_graph,
        map_graph=map_graph,
        cell_tokens=tokens,
        relation=relation,
        gait_node=gait_node,
        saddle_node=saddle_node,
        gait_reference_node_hits=dict(sorted(reference_hits.items())),
        reference_endpoint_audit=reference_audit,
        image_connectivity_audit=image_audit,
        empty_image_sources=empty_sources,
        gait_base_connectivity=gait_base_connectivity,
        saddle_to_gait_path=path,
        saddle_to_gait_direct=(
            saddle_node is not None
            and gait_node is not None
            and order.has_edge(saddle_node, gait_node)
        ),
        box_map_diagnostics=setup.box_map.diagnostics(),
    )


def _validate_even_depth(depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise ValueError("depth must be an integer")
    if depth < 2 or depth % 2:
        raise ValueError("wheel acceptance uses an even depth of at least two")


def _subdivisions_for_depth(depth: int) -> int:
    _validate_even_depth(depth)
    return 2 ** (depth // 2)


def _atlas_cells(
    morse_graph: Any,
    cell_count: int,
) -> tuple[AtlasWheelCell, ...]:
    cells: list[AtlasWheelCell] = []
    for index in range(cell_count):
        chart_id, raw_bounds = morse_graph.phase_space_chart_box(index)
        cells.append(
            AtlasWheelCell(
                index=index,
                chart_id=int(chart_id),
                bounds=tuple(float(value) for value in raw_bounds),
            )
        )
    return tuple(cells)


def _atlas_relation(
    map_graph: Any,
    tokens: Sequence[AtlasWheelCell],
) -> dict[AtlasWheelCell, frozenset[AtlasWheelCell]]:
    if int(map_graph.num_vertices()) != len(tokens):
        raise ValueError("CMGDB map graph and Atlas token counts disagree")
    return {
        source: frozenset(tokens[int(target)] for target in map_graph.adjacencies(index))
        for index, source in enumerate(tokens)
    }


def _walking_cycle_probes(
    setup: RimlessWheelAtlasSetup,
    cells: Sequence[AtlasWheelCell],
    *,
    base_samples: int,
    handle_samples: int,
) -> tuple[tuple[EndpointProbe, ...], frozenset[AtlasWheelCell]]:
    if base_samples < 2 or handle_samples < 1:
        raise ValueError("reference sampling counts are too small")
    alpha = setup.wheel.alpha
    gamma = setup.wheel.gamma
    speed = walking_fixed_point_speed(alpha=alpha, gamma=gamma)
    initial = np.asarray((gamma - alpha, speed), dtype=np.float64)
    trajectory = setup.wheel.system.simulate(
        initial,
        (0.0, 8.0),
        max_jumps=1,
        dense_output=True,
        max_step=0.005,
    )
    if not trajectory.jump_times or trajectory.segments[0].solution is None:
        raise RuntimeError("analytic walking reference did not reach the impact guard")
    impact_time = float(trajectory.jump_times[0])
    guard_state, reset_state = trajectory.jump_states[0]
    base_times = np.linspace(0.0, impact_time, base_samples, endpoint=False)
    base_states = np.asarray(trajectory.segments[0].solution(base_times), dtype=float).T

    probes: list[EndpointProbe] = []
    source_tokens: set[AtlasWheelCell] = set()
    for sample_index, state in enumerate(base_states):
        sources = _cells_containing_point(
            cells,
            setup.charts.base_chart_id,
            state,
        )
        endpoint = setup.box_map.evaluate_point(BASE_CHART_ID, state)
        source_tokens.update(sources)
        probes.extend(
            EndpointProbe(
                source=source,
                endpoint=endpoint,
                initial_state=tuple(float(value) for value in state),
                label=f"walking-base-{sample_index}",
            )
            for source in sources
        )

    phases = np.linspace(0.0, 1.0, handle_samples + 2)[1:-1]
    for sample_index, phase in enumerate(phases):
        coordinates = setup.charts.encode_handle(guard_state, float(phase))
        sources = _cells_containing_point(
            cells,
            setup.charts.handle_chart_id,
            coordinates,
        )
        endpoint = setup.box_map.evaluate_point(HANDLE_CHART_ID, coordinates)
        source_tokens.update(sources)
        probes.extend(
            EndpointProbe(
                source=source,
                endpoint=endpoint,
                initial_state=coordinates,
                label=f"walking-handle-{sample_index}",
            )
            for source in sources
        )
    return tuple(probes), frozenset(source_tokens)


def _cells_containing_point(
    cells: Sequence[AtlasWheelCell],
    chart_id: int,
    coordinates: Sequence[float],
    *,
    atol: float = 1e-10,
) -> frozenset[AtlasWheelCell]:
    point = np.asarray(coordinates, dtype=np.float64)
    result = frozenset(
        cell
        for cell in cells
        if cell.chart_id == chart_id
        and np.all(point >= cell.lower - atol)
        and np.all(point <= cell.upper + atol)
    )
    if not result:
        raise ValueError(f"point {tuple(point)} lies outside Atlas chart {chart_id}")
    return result


def _audit_atlas_endpoint_probes(
    relation: Mapping[AtlasWheelCell, Collection[AtlasWheelCell]],
    probes: Sequence[EndpointProbe],
    cells: Sequence[AtlasWheelCell],
    box_map: CMGDBSuspensionBoxMap,
) -> EndpointCoverageAudit:
    witnesses = []
    for probe in probes:
        chart_id, coordinates, _stratum = box_map.encode_endpoint(probe.endpoint)
        containing = _cells_containing_point(cells, chart_id, coordinates)
        recorded = frozenset(relation[probe.source])
        witnesses.append(
            EndpointCoverageWitness(
                probe=probe,
                containing_target_cells=containing,
                recorded_targets=recorded,
            )
        )
    return EndpointCoverageAudit(tuple(witnesses))


def _audit_relation_connectivity(
    relation: Mapping[AtlasWheelCell, Collection[AtlasWheelCell]],
    incidence: RimlessWheelQuotientIncidence,
) -> RelationConnectivityAudit:
    images = tuple(
        CellSetConnectivity(
            label=source,
            cells=frozenset(targets),
            components=incidence.connected_components(targets),
        )
        for source, targets in sorted(relation.items())
    )
    return RelationConnectivityAudit(images)


def _morse_node_by_phase_cell(morse_graph: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in morse_graph.vertices():
        for index in morse_graph.morse_set(node):
            if int(index) in result:
                raise ValueError("a phase-space cell belongs to two Morse sets")
            result[int(index)] = int(node)
    return result


def _unique_largest_hit(hits: Mapping[int, int]) -> int | None:
    if not hits:
        return None
    best = max(hits.values())
    winners = sorted(node for node, count in hits.items() if count == best)
    return winners[0] if len(winners) == 1 else None


__all__ = [
    "BASE_CHART_ID",
    "HANDLE_CHART_ID",
    "DEFAULT_BASE_BOUNDS",
    "DEFAULT_GUARD_SPEED_BOUNDS",
    "DEFAULT_T_STAR",
    "AtlasWheelCell",
    "RimlessWheelQuotientIncidence",
    "RimlessWheelAtlasSetup",
    "RimlessWheelAtlasAcceptance",
    "rimless_wheel_atlas_charts",
    "rimless_wheel_atlas_reset_gluing",
    "build_rimless_wheel_atlas_model",
    "compute_rimless_wheel_atlas_acceptance",
]
