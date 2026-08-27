"""Shared result types for the physical suspension examples.

This module deliberately materializes a *sampled point-image relation*.  The
trajectory endpoints use the exact unit-handle clock ``T = t + j``, but the
resulting finite relation is not certified to enclose the image of every point
in a grid cell.  Keeping that distinction in the result object prevents the
example runners from accidentally presenting an exploratory SCC computation
as a rigorous Conley-index computation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np

from ..src.grid import Grid
from ..src.hybrid_system import HybridSystem
from ..src.hybrid_trajectory import HybridTrajectory
from ..src.implicit_phase_scc import (
    FullSCCReconstruction,
    PhaseGadgetDescriptor,
    VirtualPhaseNode,
    reconstruct_from_base_descriptors,
)
from ..src.sampled_suspension import (
    BaseCell,
    BaseSuspensionSample,
    HandleSuspensionSample,
    PhaseCell,
    SuspensionSample,
    locate_augmented_cells,
    sample_suspension_trajectory,
)
from ..src.suspension_complex import (
    CMGDBRelativeHomologyPayload,
    CellularChainMap,
    CellularResetMap,
    FiniteCellComplex,
    FixedTimeCarrier,
    RelativeCellPair,
    ResetHandle,
    SuspensionCellComplex,
)


@dataclass(frozen=True)
class CemeteryCell:
    """An explicit absorbing compactification cell, not a physical state."""

    label: str


@dataclass(frozen=True)
class PhysicalEndpointWitness:
    """One numerically propagated endpoint used to check a physical model."""

    name: str
    initial_state: tuple[float, ...]
    endpoint: SuspensionSample


@dataclass(frozen=True)
class ResetHandleIngredients:
    """Base-grid data needed to build one reset mapping cylinder."""

    handle_id: Hashable
    guard_cell: BaseCell
    reset_cells: tuple[BaseCell, ...]
    phase_cells: tuple[PhaseCell, ...]
    guard_state: tuple[float, ...]
    reset_state: tuple[float, ...]


@dataclass(frozen=True)
class SuspensionComplexIngredients:
    """Topology-bearing inputs, before construction of a cellular complex."""

    base_bounds: tuple[tuple[float, float], ...]
    base_subdivisions: tuple[int, ...]
    handles: tuple[ResetHandleIngredients, ...]
    phase_slabs: int
    failure_cells: tuple[CemeteryCell, ...] = ()
    status: str = (
        "physical-grid guard/reset attachments are materialized; the separate "
        "validated orbit skeleton is not yet a physical-grid incidence complex/carrier"
    )


FiniteSuspensionCell = Hashable


@dataclass(frozen=True)
class FixedTimeSuspensionPipelineResult:
    """Common output contract for the three physical example runners."""

    model_name: str
    t_star: float
    sampled_orbit_period: float
    handle_slabs: int
    relation: Mapping[FiniteSuspensionCell, frozenset[FiniteSuspensionCell]]
    relation_graph: nx.DiGraph
    phase_cells: frozenset[PhaseCell]
    recurrent_sccs: tuple[frozenset[FiniteSuspensionCell], ...]
    phase_descriptors: tuple[PhaseGadgetDescriptor, ...]
    direct_base_edges: frozenset[
        tuple[FiniteSuspensionCell, FiniteSuspensionCell]
    ]
    scc_reconstruction: FullSCCReconstruction
    descriptor_expansion_matches_relation: bool
    suspension_complex_ingredients: SuspensionComplexIngredients
    suspension_complex: SuspensionCellComplex
    relative_pair: RelativeCellPair
    fixed_time_carrier: FixedTimeCarrier
    cellular_chain_map: CellularChainMap
    cmgdb_payload: CMGDBRelativeHomologyPayload
    cmgdb_result: Mapping[str, object] | None
    witnesses: tuple[PhysicalEndpointWitness, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    relation_hash: str = ""
    trajectory_status: str = (
        "genuine event-driven ODE/reset computation with unit suspension handles"
    )
    relation_status: str = (
        "sampling-derived point-image cover; not a verified whole-cell outer enclosure"
    )
    scc_status: str = "exact SCC computation for the materialized sampled relation"
    conley_index_status: str = (
        "not computed: the validated skeleton payload is not yet a proved "
        "sampled-grid index pair, and this runner does not invoke CMGDB"
    )
    skeleton_status: str = (
        "validated periodic-orbit cell skeleton with the identity representative of "
        "the time-map homotopy class; it is not a sampled-grid index pair"
    )

    def summary(self) -> dict[str, object]:
        """Return a compact, serialization-friendly runner summary."""

        result = {
            "model_name": self.model_name,
            "t_star": self.t_star,
            "sampled_orbit_period": self.sampled_orbit_period,
            "relation_nodes": self.relation_graph.number_of_nodes(),
            "relation_edges": self.relation_graph.number_of_edges(),
            "phase_cells": len(self.phase_cells),
            "recurrent_scc_sizes": [len(component) for component in self.recurrent_sccs],
            "reset_handles": len(self.suspension_complex_ingredients.handles),
            "descriptor_expansion_matches_relation": (
                self.descriptor_expansion_matches_relation
            ),
            "skeleton_betti_numbers_f5": self.suspension_complex.betti_numbers(
                modulus=5,
            ),
            "cmgdb_payload_cell_counts": self.cmgdb_payload.cell_counts,
            "relation_hash": self.relation_hash,
            "trajectory_status": self.trajectory_status,
            "relation_status": self.relation_status,
            "scc_status": self.scc_status,
            "conley_index_status": self.conley_index_status,
            "skeleton_status": self.skeleton_status,
        }
        if self.cmgdb_result is not None:
            result["cmgdb_shift_class"] = self.cmgdb_result.get("shift_class")
            result["cmgdb_homology_dimensions"] = self.cmgdb_result.get(
                "homology_dimensions",
            )
        return result


def build_periodic_orbit_skeleton(
    *,
    model_name: str,
    flow_arcs: int,
    handle_slabs: int,
) -> tuple[
    SuspensionCellComplex,
    RelativeCellPair,
    FixedTimeCarrier,
    CellularChainMap,
    CMGDBRelativeHomologyPayload,
]:
    """Build a validated cellular circle skeleton for a periodic hybrid orbit.

    ``flow_arcs=1`` models a one-step gait (or a subdivided Zeno rest fiber),
    while ``flow_arcs=2`` retains the two distinct stance arcs of a period-two
    gait.  This is topology-bearing local orbit data, not a replacement for the
    physical base grid or a proof that the sampled grid cells form an index
    pair.
    """

    if flow_arcs <= 0:
        raise ValueError("flow_arcs must be positive")
    dimensions: dict[Hashable, int] = {}
    boundaries: dict[Hashable, Mapping[Hashable, int]] = {}
    starts = []
    guards = []
    for arc in range(flow_arcs):
        start = (model_name, "orbit_start", arc)
        guard = (model_name, "orbit_guard", arc)
        edge = (model_name, "flow_arc", arc)
        starts.append(start)
        guards.append(guard)
        dimensions[start] = 0
        dimensions[guard] = 0
        dimensions[edge] = 1
        boundaries[start] = {}
        boundaries[guard] = {}
        boundaries[edge] = {start: -1, guard: 1}

    base = FiniteCellComplex(
        dimensions,
        boundaries,
        metadata={
            "kind": "periodic-hybrid-orbit-skeleton",
            "model_name": model_name,
            "flow_arcs": flow_arcs,
        },
    )
    reset = CellularResetMap.from_cell_map(
        base,
        guard_cells=guards,
        cell_map={
            guard: starts[(index + 1) % flow_arcs]
            for index, guard in enumerate(guards)
        },
    )
    suspension = SuspensionCellComplex.from_base(
        base,
        (
            ResetHandle(
                handle_id=(model_name, "orbit_reset"),
                reset=reset,
                slabs=handle_slabs,
            ),
        ),
    )
    pair = RelativeCellPair(suspension, suspension.cell_set, ())
    carrier = FixedTimeCarrier.identity(suspension, modulus=5)
    chain_map = CellularChainMap.identity(suspension, modulus=5)
    carrier.require_carries(chain_map)
    payload = chain_map.to_cmgdb_payload(pair)
    return suspension, pair, carrier, chain_map, payload


def _cell_key(cell: FiniteSuspensionCell) -> tuple[object, ...]:
    if isinstance(cell, BaseCell):
        return ("base", cell.index)
    if isinstance(cell, PhaseCell):
        return ("phase", cell.guard_cell, cell.slab)
    if isinstance(cell, CemeteryCell):
        return ("cemetery", cell.label)
    return (type(cell).__qualname__, repr(cell))


def _relation_digest(
    relation: Mapping[FiniteSuspensionCell, frozenset[FiniteSuspensionCell]],
) -> str:
    edges = sorted(
        ((_cell_key(source), _cell_key(target)))
        for source, targets in relation.items()
        for target in targets
    )
    payload = json.dumps(edges, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recurrent_components(
    graph: nx.DiGraph,
) -> tuple[frozenset[FiniteSuspensionCell], ...]:
    components = []
    for raw_component in nx.strongly_connected_components(graph):
        component = frozenset(raw_component)
        if len(component) > 1:
            components.append(component)
            continue
        node = next(iter(component))
        if graph.has_edge(node, node):
            components.append(component)
    components.sort(key=lambda value: tuple(sorted(map(repr, value))))
    return tuple(components)


def _sample_times(
    trajectory: HybridTrajectory,
    orbit_period: float,
    handle_slabs: int,
    sample_count: int,
) -> tuple[float, ...]:
    times = set(np.linspace(0.0, orbit_period, sample_count, endpoint=False))
    for jump_index, jump_time in enumerate(trajectory.jump_times):
        handle_start = float(jump_time) + jump_index
        if handle_start >= orbit_period - 1e-10:
            break
        times.add(handle_start)
        for slab in range(handle_slabs):
            times.add(handle_start + (slab + 0.5) / handle_slabs)
        handle_end = handle_start + 1.0
        if handle_end < orbit_period - 1e-10:
            times.add(handle_end)
    return tuple(sorted(time for time in times if 0.0 <= time < orbit_period))


def _handle_ingredients(
    model_name: str,
    trajectory: HybridTrajectory,
    grid: Grid,
    orbit_period: float,
    handle_slabs: int,
) -> tuple[ResetHandleIngredients, ...]:
    # One physical handle is keyed by its guard base cell.  Repeated visits to
    # the same guard cell share the same global PhaseCell values.
    reset_cells_by_guard: dict[int, set[int]] = {}
    representatives: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for jump_index, (jump_time, states) in enumerate(
        zip(trajectory.jump_times, trajectory.jump_states),
    ):
        if float(jump_time) + jump_index >= orbit_period - 1e-10:
            break
        guard_state, reset_state = states
        guard_cells = grid.find_boxes_containing_point(guard_state)
        reset_cells = grid.find_boxes_containing_point(reset_state)
        if not guard_cells or not reset_cells:
            raise ValueError("a guard or reset representative lies outside the base grid")
        for guard_cell in guard_cells:
            reset_cells_by_guard.setdefault(guard_cell, set()).update(reset_cells)
            representatives.setdefault(guard_cell, (guard_state, reset_state))

    handles = []
    for guard_cell in sorted(reset_cells_by_guard):
        guard_state, reset_state = representatives[guard_cell]
        reset_cells = tuple(
            BaseCell(index) for index in sorted(reset_cells_by_guard[guard_cell])
        )
        phase_cells = tuple(
            PhaseCell(guard_cell=guard_cell, slab=slab)
            for slab in range(handle_slabs)
        )
        handle_id = (model_name, "guard", guard_cell)
        handles.append(
            ResetHandleIngredients(
                handle_id=handle_id,
                guard_cell=BaseCell(guard_cell),
                reset_cells=reset_cells,
                phase_cells=phase_cells,
                guard_state=tuple(float(value) for value in guard_state),
                reset_state=tuple(float(value) for value in reset_state),
            ),
        )
    return tuple(handles)


def _fixed_time_phase_descriptors(
    model_name: str,
    graph: nx.DiGraph,
) -> tuple[
    tuple[PhaseGadgetDescriptor, ...],
    frozenset[tuple[FiniteSuspensionCell, FiniteSuspensionCell]],
    FullSCCReconstruction,
    bool,
]:
    """Encode exactly the phase part of a materialized fixed-time graph."""

    phase_nodes = tuple(node for node in graph.nodes if isinstance(node, PhaseCell))
    phase_set = set(phase_nodes)
    base_nodes = tuple(node for node in graph.nodes if node not in phase_set)
    descriptor_id = (model_name, "fixed_time_phase_relation")
    descriptor = PhaseGadgetDescriptor(
        descriptor_id=descriptor_id,
        phase_nodes=phase_nodes,
        phase_edges=tuple(
            (source, target)
            for source, target in graph.edges
            if source in phase_set and target in phase_set
        ),
        entries=tuple(
            (source, target)
            for source, target in graph.edges
            if source not in phase_set and target in phase_set
        ),
        exits=tuple(
            (source, target)
            for source, target in graph.edges
            if source in phase_set and target not in phase_set
        ),
    )
    direct_base_edges = frozenset(
        (source, target)
        for source, target in graph.edges
        if source not in phase_set and target not in phase_set
    )
    descriptors = (descriptor,)
    reconstruction = reconstruct_from_base_descriptors(
        base_nodes,
        descriptors,
        direct_base_edges=direct_base_edges,
    )

    def expanded(node: FiniteSuspensionCell) -> FiniteSuspensionCell:
        if isinstance(node, PhaseCell):
            return VirtualPhaseNode(descriptor_id, node)
        return node

    expected_nodes = {expanded(node) for node in graph.nodes}
    expected_edges = {
        (expanded(source), expanded(target)) for source, target in graph.edges
    }
    matches = (
        set(reconstruction.virtual_graph.nodes) == expected_nodes
        and set(reconstruction.virtual_graph.edges) == expected_edges
    )
    return descriptors, direct_base_edges, reconstruction, matches


def materialize_periodic_orbit_relation(
    *,
    model_name: str,
    system: HybridSystem,
    initial_state: Sequence[float],
    grid: Grid,
    t_star: float,
    orbit_period: float,
    handle_slabs: int,
    sample_count: int,
    max_jumps: int,
    max_step: float,
    skeleton_flow_arcs: int = 1,
    witnesses: Sequence[PhysicalEndpointWitness] = (),
    diagnostics: Mapping[str, object] | None = None,
    failure_cells: Sequence[CemeteryCell] = (),
    compute_cmgdb: bool = False,
) -> FixedTimeSuspensionPipelineResult:
    """Materialize the sampled time-``t_star`` relation along one orbit.

    A single event-driven trajectory is reused for all source/target samples.
    This makes the clock computation genuine and deterministic while retaining
    the explicit limitation that only finitely many points, not whole cells,
    were propagated.
    """

    if not np.isfinite(t_star) or t_star <= 0:
        raise ValueError("t_star must be finite and positive")
    if not np.isfinite(orbit_period) or orbit_period <= 0:
        raise ValueError("orbit_period must be finite and positive")
    if handle_slabs <= 0:
        raise ValueError("handle_slabs must be positive")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    initial = np.asarray(initial_state, dtype=np.float64)
    # Suspension time never advances more slowly than continuous time.  This
    # ordinary horizon therefore covers every requested source and target.
    ordinary_horizon = orbit_period + t_star + max_step
    trajectory = system.simulate(
        initial,
        (0.0, ordinary_horizon),
        max_jumps=max_jumps,
        dense_output=True,
        max_step=max_step,
        jump_time_penalty=False,
    )

    source_times = _sample_times(
        trajectory,
        orbit_period=orbit_period,
        handle_slabs=handle_slabs,
        sample_count=sample_count,
    )
    mutable_relation: dict[
        FiniteSuspensionCell, set[FiniteSuspensionCell]
    ] = {}
    sample_kind_counts = {"base": 0, "handle": 0}

    for source_time in source_times:
        source_sample = sample_suspension_trajectory(trajectory, source_time)
        target_sample = sample_suspension_trajectory(
            trajectory,
            source_time + t_star,
        )
        if isinstance(target_sample, BaseSuspensionSample):
            sample_kind_counts["base"] += 1
        elif isinstance(target_sample, HandleSuspensionSample):
            sample_kind_counts["handle"] += 1

        source_cells = locate_augmented_cells(source_sample, grid, handle_slabs)
        target_cells = locate_augmented_cells(target_sample, grid, handle_slabs)
        for source in source_cells:
            mutable_relation.setdefault(source, set()).update(target_cells)
        for target in target_cells:
            mutable_relation.setdefault(target, set())

    for failure_cell in failure_cells:
        mutable_relation.setdefault(failure_cell, set()).add(failure_cell)

    relation = {
        source: frozenset(targets)
        for source, targets in sorted(
            mutable_relation.items(), key=lambda item: _cell_key(item[0]),
        )
    }
    graph = nx.DiGraph()
    graph.graph.update(
        semantics="sampled fixed-time unit-handle suspension relation",
        model_name=model_name,
        t_star=float(t_star),
        whole_cell_outer_enclosure=False,
    )
    graph.add_nodes_from(relation)
    graph.add_edges_from(
        (source, target)
        for source, targets in relation.items()
        for target in targets
    )

    handles = _handle_ingredients(
        model_name,
        trajectory,
        grid,
        orbit_period,
        handle_slabs,
    )
    (
        descriptors,
        direct_base_edges,
        scc_reconstruction,
        descriptor_expansion_matches_relation,
    ) = _fixed_time_phase_descriptors(model_name, graph)
    if not descriptor_expansion_matches_relation:
        raise AssertionError(
            "fixed-time phase descriptor expansion does not match its source relation",
        )
    phase_cells = frozenset(
        cell
        for cell in graph.nodes
        if isinstance(cell, PhaseCell)
    ) | frozenset(
        phase_cell for handle in handles for phase_cell in handle.phase_cells
    )

    final_diagnostics = dict(diagnostics or {})
    final_diagnostics.update(
        source_samples=len(source_times),
        target_base_samples=sample_kind_counts["base"],
        target_handle_samples=sample_kind_counts["handle"],
        numerical_jump_times=tuple(float(time) for time in trajectory.jump_times),
    )
    ingredients = SuspensionComplexIngredients(
        base_bounds=tuple(
            (float(lower), float(upper)) for lower, upper in grid.bounds
        ),
        base_subdivisions=tuple(int(value) for value in grid.subdivisions),
        handles=handles,
        phase_slabs=handle_slabs,
        failure_cells=tuple(failure_cells),
    )
    (
        suspension_complex,
        relative_pair,
        fixed_time_carrier,
        cellular_chain_map,
        cmgdb_payload,
    ) = build_periodic_orbit_skeleton(
        model_name=model_name,
        flow_arcs=skeleton_flow_arcs,
        handle_slabs=handle_slabs,
    )
    cmgdb_result = None
    conley_index_status = (
        "not computed: the validated skeleton payload is not yet a proved "
        "sampled-grid index pair, and this runner did not invoke CMGDB"
    )
    if compute_cmgdb:
        try:
            import CMGDB
        except ImportError as error:
            raise RuntimeError(
                "compute_cmgdb=True requires a CMGDB build with "
                "ComputeRelativeHomologyShiftClass",
            ) from error
        if not hasattr(CMGDB, "ComputeRelativeHomologyShiftClass"):
            raise RuntimeError(
                "installed CMGDB predates ComputeRelativeHomologyShiftClass",
            )
        cmgdb_result = CMGDB.ComputeRelativeHomologyShiftClass(
            *cmgdb_payload.as_compute_args(),
        )
        conley_index_status = (
            "CMGDB shift class computed for the validated local orbit skeleton; "
            "not a Conley-index claim for the sampled physical grid relation"
        )
    return FixedTimeSuspensionPipelineResult(
        model_name=model_name,
        t_star=float(t_star),
        sampled_orbit_period=float(orbit_period),
        handle_slabs=handle_slabs,
        relation=relation,
        relation_graph=graph,
        phase_cells=phase_cells,
        recurrent_sccs=_recurrent_components(graph),
        phase_descriptors=descriptors,
        direct_base_edges=direct_base_edges,
        scc_reconstruction=scc_reconstruction,
        descriptor_expansion_matches_relation=(
            descriptor_expansion_matches_relation
        ),
        suspension_complex_ingredients=ingredients,
        suspension_complex=suspension_complex,
        relative_pair=relative_pair,
        fixed_time_carrier=fixed_time_carrier,
        cellular_chain_map=cellular_chain_map,
        cmgdb_payload=cmgdb_payload,
        cmgdb_result=cmgdb_result,
        witnesses=tuple(witnesses),
        diagnostics=final_diagnostics,
        relation_hash=_relation_digest(relation),
        conley_index_status=conley_index_status,
    )


__all__ = [
    "CemeteryCell",
    "PhysicalEndpointWitness",
    "ResetHandleIngredients",
    "SuspensionComplexIngredients",
    "FixedTimeSuspensionPipelineResult",
    "build_periodic_orbit_skeleton",
    "materialize_periodic_orbit_relation",
]
