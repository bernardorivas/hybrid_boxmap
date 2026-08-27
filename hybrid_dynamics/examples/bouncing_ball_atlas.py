"""Bouncing-ball computation through CMGDB's tagged suspension box map.

CMGDB owns the Atlas grid, directed graph, SCC decomposition, and Morse order.
The hybrid-specific callback returns a finite union of rectangles in two
tagged charts:

* the physical base chart ``(h, v)`` on ``[0,2] x [-5,5]``; and
* the outgoing reset-handle chart ``(v_guard, s)`` on ``[-5,0] x [0,1]``.

The acceptance probes use the closed-form ballistic flow, rather than the
pointwise event-driven evaluator used by the callback. They cover rebound
arcs, both quotient seams,
the rest fiber, and geometrically shrinking bounce velocities near the Zeno
limit.  These finite checks can falsify a sampled relation, but do not certify
the whole-cell outer-enclosure condition.  Empty images and failed callback
samples caused by exits from the rectangular window are reported explicitly.
No Conley index is assigned by this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
from ..src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
    SuspensionSample,
)
from .bouncing_ball import BouncingBall


BASE_CHART_ID = 0
HANDLE_CHART_ID = 1
DEFAULT_BASE_BOUNDS = ((0.0, 2.0), (-5.0, 5.0))
DEFAULT_GUARD_VELOCITY_BOUNDS = (-5.0, 0.0)
DEFAULT_T_STAR = 1.5
DEFAULT_GRAVITY = 9.81
DEFAULT_RESTITUTION = 0.8


_DOMAIN_EXIT_FAILURE_MARKERS = (
    "leaves the declared state-space bounds",
    "trajectory leaves the declared state space",
    "reset leaves the declared state space",
    "exits the declared state space",
    "exits the declared nonrectangular state space",
)


def _is_domain_exit_failure_reason(reason: str) -> bool:
    """Return whether a sampled callback failure records a domain exit."""

    return any(marker in reason for marker in _DOMAIN_EXIT_FAILURE_MARKERS)


def _state_copy(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, order=True)
class AtlasBallCell:
    """One CMGDB Atlas cell with its chart tag and physical coordinates."""

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


class BouncingBallQuotientIncidence:
    """Closed-cell incidence after the two reset-handle face gluings."""

    def __init__(
        self,
        charts: SuspensionAtlasCharts,
        *,
        restitution: float,
        atol: float = 1e-12,
    ) -> None:
        self.charts = charts
        self.restitution = float(restitution)
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

    def intersects(self, first: AtlasBallCell, second: AtlasBallCell) -> bool:
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
        velocity_lower = float(handle.lower[0])
        velocity_upper = float(handle.upper[0])

        if handle.lower[-1] <= self.atol:
            guard_lower = np.asarray((0.0, velocity_lower), dtype=np.float64)
            guard_upper = np.asarray((0.0, velocity_upper), dtype=np.float64)
            if self._rectangles_intersect(
                base.lower,
                base.upper,
                guard_lower,
                guard_upper,
            ):
                return True

        if handle.upper[-1] >= 1.0 - self.atol:
            reset_velocities = sorted(
                (-self.restitution * velocity_lower, -self.restitution * velocity_upper)
            )
            reset_lower = np.asarray((0.0, reset_velocities[0]), dtype=np.float64)
            reset_upper = np.asarray((0.0, reset_velocities[1]), dtype=np.float64)
            if self._rectangles_intersect(
                base.lower,
                base.upper,
                reset_lower,
                reset_upper,
            ):
                return True
        return False

    def connected_components(
        self,
        cells: Collection[AtlasBallCell],
    ) -> tuple[frozenset[AtlasBallCell], ...]:
        remaining = set(cells)
        components: list[frozenset[AtlasBallCell]] = []
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
class BouncingBallAtlasSetup:
    """Objects needed to run one native CMGDB computation."""

    ball: BouncingBall
    charts: SuspensionAtlasCharts
    box_map: CMGDBSuspensionBoxMap
    model: Any


@dataclass(frozen=True)
class BouncingBallAtlasAcceptance:
    """Morse output and falsification diagnostics for one fixed depth."""

    depth: int
    subdivisions: int
    t_star: float
    morse_graph: Any
    map_graph: Any
    cell_tokens: tuple[AtlasBallCell, ...]
    relation: Mapping[AtlasBallCell, frozenset[AtlasBallCell]]
    zeno_node: int | None
    zeno_reference_node_hits: Mapping[int, int]
    reference_endpoint_audit: EndpointCoverageAudit
    reference_guard_speeds: tuple[float, ...]
    image_connectivity_audit: RelationConnectivityAudit
    empty_image_sources: tuple[AtlasBallCell, ...]
    zeno_support_connectivity: CellSetConnectivity | None
    zeno_base_connectivity: CellSetConnectivity | None
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
    def zeno_support_connected(self) -> bool:
        return bool(
            self.zeno_support_connectivity is not None
            and self.zeno_support_connectivity.connected
        )

    @property
    def zeno_base_connected(self) -> bool:
        return bool(
            self.zeno_base_connectivity is not None
            and self.zeno_base_connectivity.connected
        )

    @property
    def one_morse_node(self) -> bool:
        return int(self.morse_graph.num_vertices()) == 1 and self.zeno_node is not None

    @property
    def local_window_gates_passed(self) -> bool:
        """Finite local-window gates; exits do not become a global claim."""

        return bool(
            self.reference_covered
            and self.image_values_connected
            and self.zeno_support_connected
            and self.zeno_base_connected
            and self.box_map_diagnostics.unresolved_stage_edges == 0
            and self.one_morse_node
        )

    def _failure_summary(self) -> dict[str, object]:
        records = self.box_map_diagnostics.retained_source_records
        failures = [failure for record in records for failure in record.failures]
        reasons = Counter(failure.reason for failure in failures)
        exit_failures = sum(
            count
            for reason, count in reasons.items()
            if _is_domain_exit_failure_reason(reason)
        )
        return {
            "retained_callback_records": len(records),
            "callback_records_with_failures": sum(bool(record.failures) for record in records),
            "callback_records_with_only_failures": sum(
                bool(record.failures) and record.successful_samples == 0
                for record in records
            ),
            "domain_exit_sample_failures": exit_failures,
            "other_sample_failures": len(failures) - exit_failures,
            "failure_reasons": dict(sorted(reasons.items())),
        }

    def _morse_set_sizes(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for node in self.morse_graph.vertices():
            indices = [int(index) for index in self.morse_graph.morse_set(node)]
            base = sum(
                self.cell_tokens[index].chart_id == BASE_CHART_ID
                for index in indices
            )
            handle = len(indices) - base
            result[str(int(node))] = {
                "base": int(base),
                "handle": int(handle),
                "total": len(indices),
            }
        return result

    def summary(self) -> dict[str, object]:
        positive_speeds = [speed for speed in self.reference_guard_speeds if speed > 0.0]
        result: dict[str, object] = {
            "depth": self.depth,
            "subdivisions_per_axis": self.subdivisions,
            "t_star": self.t_star,
            "relation_scope": "nonempty local-window relation",
            "map_cells": int(self.map_graph.num_vertices()),
            "morse_nodes": int(self.morse_graph.num_vertices()),
            "morse_edges": [list(edge) for edge in self.morse_edges],
            "morse_set_cell_counts": self._morse_set_sizes(),
            "expected_m0_node": self.zeno_node,
            "zeno_reference_node_hits": dict(self.zeno_reference_node_hits),
            "reference_guard_speed_count": len(self.reference_guard_speeds),
            "reference_min_positive_guard_speed": min(positive_speeds),
            "reference_max_guard_speed": max(self.reference_guard_speeds),
            "reference_includes_exact_rest_fiber": 0.0 in self.reference_guard_speeds,
            "reference_probes": len(self.reference_endpoint_audit.witnesses),
            "reference_misses": len(self.reference_endpoint_audit.missed),
            "reference_evaluation_failures": len(
                self.reference_endpoint_audit.evaluation_failures
            ),
            "nonempty_images": len(self.image_connectivity_audit.images),
            "disconnected_images": len(self.image_connectivity_audit.disconnected),
            "empty_images": len(self.empty_image_sources),
            "empty_sources_excluded_from_connectivity_audit": True,
            "zeno_support_connected_in_quotient": self.zeno_support_connected,
            "zeno_support_component_sizes": (
                []
                if self.zeno_support_connectivity is None
                else [
                    len(component)
                    for component in self.zeno_support_connectivity.components
                ]
            ),
            "zeno_base_connected": self.zeno_base_connected,
            "zeno_base_component_sizes": (
                []
                if self.zeno_base_connectivity is None
                else [len(component) for component in self.zeno_base_connectivity.components]
            ),
            "box_map_source_boxes_across_cmgdb_passes": (
                self.box_map_diagnostics.source_boxes
            ),
            "box_map_sampled_points_across_cmgdb_passes": (
                self.box_map_diagnostics.sampled_points
            ),
            "box_map_failed_samples": self.box_map_diagnostics.failed_samples,
            "box_map_empty_images_across_cmgdb_passes": (
                self.box_map_diagnostics.empty_images
            ),
            "box_map_unresolved_stage_edges": (
                self.box_map_diagnostics.unresolved_stage_edges
            ),
            "local_window_gates_passed": self.local_window_gates_passed,
            "whole_cell_outer_enclosure_certified": False,
            "global_attractor_lattice_interpretation": False,
            "conley_index_computed": False,
        }
        result.update(self._failure_summary())
        return result


def bouncing_ball_atlas_charts(
    *,
    base_bounds: Sequence[Sequence[float]] = DEFAULT_BASE_BOUNDS,
    guard_velocity_bounds: Sequence[float] = DEFAULT_GUARD_VELOCITY_BOUNDS,
) -> SuspensionAtlasCharts:
    """Return physical ``(h,v)`` and outgoing ``(v_guard,s)`` charts."""

    bounds = tuple(
        (float(interval[0]), float(interval[1])) for interval in base_bounds
    )
    guard_interval = (
        float(guard_velocity_bounds[0]),
        float(guard_velocity_bounds[1]),
    )
    if guard_interval[1] > 0.0:
        raise ValueError("the bouncing-ball handle is restricted to v_guard <= 0")
    return SuspensionAtlasCharts(
        base_bounds=bounds,
        guard_bounds=(guard_interval,),
        guard_coordinates=lambda guard: np.asarray([guard[1]], dtype=np.float64),
        guard_embedding=lambda intrinsic: np.asarray(
            [0.0, intrinsic[0]], dtype=np.float64
        ),
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )


def bouncing_ball_atlas_reset_gluing(
    *, restitution: float = DEFAULT_RESTITUTION
) -> AtlasResetGluing2D:
    """Return the exact affine seam data used by the physical Atlas nerve."""

    restitution = float(restitution)
    if not np.isfinite(restitution) or restitution <= 0.0 or restitution >= 1.0:
        raise ValueError("the cellular 2D reset requires 0 < restitution < 1")
    return AtlasResetGluing2D(
        BASE_CHART_ID,
        HANDLE_CHART_ID,
        guard=AffineBoundaryEmbedding2D(0, 0.0, 1.0),
        reset=AffineBoundaryEmbedding2D(0, 0.0, -restitution),
    )


def build_bouncing_ball_atlas_model(
    *,
    depth: int,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    gravity: float = DEFAULT_GRAVITY,
    restitution: float = DEFAULT_RESTITUTION,
    max_step: float = 0.02,
    diagnostics_limit: int = 25_000,
) -> BouncingBallAtlasSetup:
    """Build the ball and install its tagged box map in ``AtlasModel``."""

    _validate_even_depth(depth)
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(restitution) or not 0.0 <= restitution < 1.0:
        raise ValueError("restitution must satisfy 0 <= restitution < 1")
    bounds = DEFAULT_BASE_BOUNDS
    ball = BouncingBall(
        domain_bounds=list(bounds),
        g=float(gravity),
        c=float(restitution),
        max_jumps=24,
    )
    charts = bouncing_ball_atlas_charts(
        base_bounds=bounds,
        guard_velocity_bounds=(bounds[1][0], min(0.0, bounds[1][1])),
    )
    box_map = CMGDBSuspensionBoxMap(
        ball.system,
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
    return BouncingBallAtlasSetup(
        ball=ball,
        charts=charts,
        box_map=box_map,
        model=model,
    )


def analytic_bouncing_ball_suspension_endpoint(
    source_chart_id: int,
    coordinates: Sequence[float],
    total_time: float,
    *,
    gravity: float = DEFAULT_GRAVITY,
    restitution: float = DEFAULT_RESTITUTION,
    atol: float = 1e-12,
) -> SuspensionSample:
    """Evaluate the unit-handle suspension using the exact ballistic flow.

    This reference is independent of the sampled ODE/event evaluator used by
    :class:`CMGDBSuspensionBoxMap`.  Each impact consumes one full unit of
    suspension time, so only finitely many impacts can occur for finite
    ``total_time``, including at the Zeno rest state.
    """

    total_time = float(total_time)
    gravity = float(gravity)
    restitution = float(restitution)
    atol = float(atol)
    if not np.isfinite(total_time) or total_time < 0.0:
        raise ValueError("total_time must be finite and non-negative")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(restitution) or not 0.0 <= restitution < 1.0:
        raise ValueError("restitution must satisfy 0 <= restitution < 1")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")

    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("bouncing-ball chart coordinates must be finite and 2D")

    if int(source_chart_id) == BASE_CHART_ID:
        state = values
        if state[0] < -atol:
            raise ValueError("height must be non-negative")
        return _analytic_advance_from_base(
            np.asarray((max(0.0, state[0]), state[1]), dtype=np.float64),
            total_time,
            gravity=gravity,
            restitution=restitution,
            jump_offset=0,
            continuous_offset=0.0,
            atol=atol,
        )

    if int(source_chart_id) != HANDLE_CHART_ID:
        raise ValueError(f"unknown bouncing-ball chart id {source_chart_id}")
    guard_velocity, phase = (float(value) for value in values)
    if guard_velocity > atol:
        raise ValueError("the outgoing bouncing-ball guard has v <= 0")
    if phase < -atol or phase > 1.0 + atol:
        raise ValueError("handle phase must lie in [0,1]")
    phase = min(max(phase, 0.0), 1.0)
    guard = _state_copy((0.0, min(guard_velocity, 0.0)))
    reset = _state_copy((0.0, -restitution * guard[1]))
    distance_to_reset = 1.0 - phase

    if total_time < distance_to_reset - atol:
        return HandleSuspensionSample(
            guard_state=guard,
            reset_state=reset,
            phase=float(phase + total_time),
            total_time=total_time,
            continuous_time=0.0,
            jump_index=0,
        )
    if total_time <= distance_to_reset + atol:
        return BaseSuspensionSample(
            state=reset,
            total_time=total_time,
            continuous_time=0.0,
            jumps_completed=1,
        )
    return _analytic_advance_from_base(
        reset,
        total_time - distance_to_reset,
        gravity=gravity,
        restitution=restitution,
        jump_offset=1,
        continuous_offset=0.0,
        atol=atol,
        reported_total_time=total_time,
    )


def _analytic_advance_from_base(
    state: Sequence[float],
    duration: float,
    *,
    gravity: float,
    restitution: float,
    jump_offset: int,
    continuous_offset: float,
    atol: float,
    reported_total_time: float | None = None,
) -> SuspensionSample:
    point = np.asarray(state, dtype=np.float64).copy()
    remaining = max(0.0, float(duration))
    jumps = int(jump_offset)
    continuous = float(continuous_offset)
    total = float(duration if reported_total_time is None else reported_total_time)

    # Every loop consumes one unit handle unless it returns.  This explicit
    # bound also makes accidental non-progress impossible at the rest state.
    for _iteration in range(int(np.ceil(remaining + atol)) + 3):
        impact_time = _analytic_next_impact_time(
            point,
            gravity=gravity,
            atol=atol,
        )
        if remaining < impact_time - atol:
            endpoint = _ballistic_flow(point, remaining, gravity)
            return BaseSuspensionSample(
                state=_state_copy(endpoint),
                total_time=total,
                continuous_time=continuous + remaining,
                jumps_completed=jumps,
            )

        guard = _ballistic_flow(point, impact_time, gravity)
        guard[0] = 0.0
        guard = _state_copy(guard)
        if remaining <= impact_time + atol:
            return BaseSuspensionSample(
                state=guard,
                total_time=total,
                continuous_time=continuous + impact_time,
                jumps_completed=jumps,
            )

        remaining -= impact_time
        continuous += impact_time
        reset = _state_copy((0.0, -restitution * guard[1]))
        if remaining < 1.0 - atol:
            return HandleSuspensionSample(
                guard_state=guard,
                reset_state=reset,
                phase=float(remaining),
                total_time=total,
                continuous_time=continuous,
                jump_index=jumps,
            )
        if remaining <= 1.0 + atol:
            return BaseSuspensionSample(
                state=reset,
                total_time=total,
                continuous_time=continuous,
                jumps_completed=jumps + 1,
            )
        remaining -= 1.0
        jumps += 1
        point = np.asarray(reset, dtype=np.float64)
    raise RuntimeError("analytic suspension evaluator exhausted its event budget")


def _analytic_next_impact_time(
    state: Sequence[float],
    *,
    gravity: float,
    atol: float,
) -> float:
    height, velocity = (float(value) for value in state)
    if height < -atol:
        raise ValueError("height must be non-negative")
    height = max(0.0, height)
    discriminant = velocity * velocity + 2.0 * gravity * height
    return float((velocity + np.sqrt(max(0.0, discriminant))) / gravity)


def _ballistic_flow(
    state: Sequence[float],
    duration: float,
    gravity: float,
) -> np.ndarray:
    height, velocity = (float(value) for value in state)
    time = float(duration)
    return np.asarray(
        (
            height + velocity * time - 0.5 * gravity * time * time,
            velocity - gravity * time,
        ),
        dtype=np.float64,
    )


def compute_bouncing_ball_atlas_acceptance(
    *,
    depth: int = 8,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    reference_speed_samples: int = 25,
    reference_base_samples: int = 17,
    reference_handle_samples: int = 17,
    gravity: float = DEFAULT_GRAVITY,
    restitution: float = DEFAULT_RESTITUTION,
) -> BouncingBallAtlasAcceptance:
    """Run CMGDB and audit the analytic shrinking-bounce reference family."""

    setup = build_bouncing_ball_atlas_model(
        depth=depth,
        t_star=t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        gravity=gravity,
        restitution=restitution,
    )
    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB fork is not installed") from error

    setup.box_map.reset_diagnostics()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(setup.model)
    subdivisions = _subdivisions_for_depth(depth)
    tokens = _atlas_cells(morse_graph, int(map_graph.num_vertices()))
    relation = _atlas_relation(map_graph, tokens)
    incidence = BouncingBallQuotientIncidence(
        setup.charts,
        restitution=setup.ball.c,
    )

    nonempty_relation = {
        source: targets for source, targets in relation.items() if targets
    }
    empty_sources = tuple(source for source, targets in relation.items() if not targets)
    image_audit = _audit_relation_connectivity(nonempty_relation, incidence)

    probes, zeno_tokens, guard_speeds = _shrinking_bounce_reference_probes(
        setup,
        tokens,
        speed_samples=reference_speed_samples,
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
    zeno_hits = Counter(
        node_by_cell[index]
        for index, token in enumerate(tokens)
        if token in zeno_tokens and index in node_by_cell
    )
    zeno_node = _unique_largest_hit(zeno_hits)

    zeno_support_connectivity = None
    zeno_base_connectivity = None
    if zeno_node is not None:
        zeno_cells = {
            tokens[int(index)] for index in morse_graph.morse_set(zeno_node)
        }
        if zeno_cells:
            zeno_support_connectivity = CellSetConnectivity(
                label="bouncing-ball-zeno-support",
                cells=frozenset(zeno_cells),
                components=incidence.connected_components(zeno_cells),
            )
        zeno_base_cells = {
            cell for cell in zeno_cells if cell.chart_id == setup.charts.base_chart_id
        }
        if zeno_base_cells:
            zeno_base_connectivity = CellSetConnectivity(
                label="bouncing-ball-zeno-base",
                cells=frozenset(zeno_base_cells),
                components=incidence.connected_components(zeno_base_cells),
            )

    return BouncingBallAtlasAcceptance(
        depth=depth,
        subdivisions=subdivisions,
        t_star=float(t_star),
        morse_graph=morse_graph,
        map_graph=map_graph,
        cell_tokens=tokens,
        relation=relation,
        zeno_node=zeno_node,
        zeno_reference_node_hits=dict(sorted(zeno_hits.items())),
        reference_endpoint_audit=reference_audit,
        reference_guard_speeds=guard_speeds,
        image_connectivity_audit=image_audit,
        empty_image_sources=empty_sources,
        zeno_support_connectivity=zeno_support_connectivity,
        zeno_base_connectivity=zeno_base_connectivity,
        box_map_diagnostics=setup.box_map.diagnostics(),
    )


def _reference_speeds(count: int, maximum: float) -> tuple[float, ...]:
    if count < 5:
        raise ValueError("reference_speed_samples must be at least five")
    if maximum <= 0.0:
        raise ValueError("maximum reference speed must be positive")
    positive_count = count - 1
    logarithmic_count = max(2, positive_count // 2)
    linear_count = positive_count - logarithmic_count
    logarithmic = np.geomspace(1e-6, min(0.25, maximum), logarithmic_count)
    linear = np.linspace(min(0.25, maximum), maximum, linear_count + 1)[1:]
    speeds = np.unique(np.concatenate(([0.0], logarithmic, linear)))
    return tuple(float(value) for value in speeds)


def _shrinking_bounce_reference_probes(
    setup: BouncingBallAtlasSetup,
    cells: Sequence[AtlasBallCell],
    *,
    speed_samples: int,
    base_samples: int,
    handle_samples: int,
) -> tuple[tuple[EndpointProbe, ...], frozenset[AtlasBallCell], tuple[float, ...]]:
    if base_samples < 2 or handle_samples < 3:
        raise ValueError("reference sampling counts are too small")
    maximum = min(
        abs(setup.charts.guard_bounds[0][0]),
        setup.charts.base_bounds[1][1] / setup.ball.c,
    )
    speeds = _reference_speeds(speed_samples, maximum)
    probes: list[EndpointProbe] = []
    zeno_tokens: set[AtlasBallCell] = set()
    zeno_threshold = speeds[min(3, len(speeds) - 1)]

    def append_probe(
        chart_id: int,
        coordinates: Sequence[float],
        label: str,
        *,
        zeno: bool,
    ) -> None:
        sources = _cells_containing_point(cells, chart_id, coordinates)
        endpoint = analytic_bouncing_ball_suspension_endpoint(
            chart_id,
            coordinates,
            setup.box_map.t_star,
            gravity=setup.ball.g,
            restitution=setup.ball.c,
        )
        if zeno:
            zeno_tokens.update(sources)
        probes.extend(
            EndpointProbe(
                source=source,
                endpoint=endpoint,
                initial_state=tuple(float(value) for value in coordinates),
                label=label,
            )
            for source in sources
        )

    phases = np.linspace(0.0, 1.0, handle_samples)
    for speed_index, speed in enumerate(speeds):
        near_zeno = speed <= zeno_threshold
        guard_velocity = -speed
        guard = np.asarray((0.0, guard_velocity), dtype=np.float64)
        append_probe(
            BASE_CHART_ID,
            guard,
            f"guard-{speed_index}",
            zeno=near_zeno,
        )
        for phase_index, phase in enumerate(phases):
            handle_coordinates = (guard_velocity, float(phase))
            append_probe(
                HANDLE_CHART_ID,
                handle_coordinates,
                f"handle-{speed_index}-{phase_index}",
                zeno=near_zeno,
            )

        rebound_velocity = setup.ball.c * speed
        flight_time = 0.0 if speed == 0.0 else 2.0 * rebound_velocity / setup.ball.g
        base_times = (
            np.asarray((0.0,))
            if speed == 0.0
            else np.linspace(0.0, flight_time, base_samples, endpoint=False)
        )
        for time_index, time in enumerate(base_times):
            state = _ballistic_flow((0.0, rebound_velocity), float(time), setup.ball.g)
            state[0] = max(0.0, state[0])
            append_probe(
                BASE_CHART_ID,
                state,
                f"rebound-{speed_index}-{time_index}",
                zeno=near_zeno,
            )
    return tuple(probes), frozenset(zeno_tokens), speeds


def _validate_even_depth(depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise ValueError("depth must be an integer")
    if depth < 2 or depth % 2:
        raise ValueError("ball acceptance uses an even depth of at least two")


def _subdivisions_for_depth(depth: int) -> int:
    _validate_even_depth(depth)
    return 2 ** (depth // 2)


def _atlas_cells(
    morse_graph: Any,
    cell_count: int,
) -> tuple[AtlasBallCell, ...]:
    cells: list[AtlasBallCell] = []
    for index in range(cell_count):
        chart_id, raw_bounds = morse_graph.phase_space_chart_box(index)
        cells.append(
            AtlasBallCell(
                index=index,
                chart_id=int(chart_id),
                bounds=tuple(float(value) for value in raw_bounds),
            )
        )
    return tuple(cells)


def _atlas_relation(
    map_graph: Any,
    tokens: Sequence[AtlasBallCell],
) -> dict[AtlasBallCell, frozenset[AtlasBallCell]]:
    if int(map_graph.num_vertices()) != len(tokens):
        raise ValueError("CMGDB map graph and Atlas token counts disagree")
    return {
        source: frozenset(tokens[int(target)] for target in map_graph.adjacencies(index))
        for index, source in enumerate(tokens)
    }


def _cells_containing_point(
    cells: Sequence[AtlasBallCell],
    chart_id: int,
    coordinates: Sequence[float],
    *,
    atol: float = 1e-10,
) -> frozenset[AtlasBallCell]:
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
    relation: Mapping[AtlasBallCell, Collection[AtlasBallCell]],
    probes: Sequence[EndpointProbe],
    cells: Sequence[AtlasBallCell],
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
    relation: Mapping[AtlasBallCell, Collection[AtlasBallCell]],
    incidence: BouncingBallQuotientIncidence,
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
    "DEFAULT_GUARD_VELOCITY_BOUNDS",
    "DEFAULT_T_STAR",
    "DEFAULT_GRAVITY",
    "DEFAULT_RESTITUTION",
    "AtlasBallCell",
    "BouncingBallQuotientIncidence",
    "BouncingBallAtlasSetup",
    "BouncingBallAtlasAcceptance",
    "bouncing_ball_atlas_charts",
    "bouncing_ball_atlas_reset_gluing",
    "build_bouncing_ball_atlas_model",
    "analytic_bouncing_ball_suspension_endpoint",
    "compute_bouncing_ball_atlas_acceptance",
]
