"""CMGDB-style sampled box relation for a fixed-time suspension map.

Unlike the periodic-orbit smoke models, this module evaluates every declared
base-grid cell.  Reset handles are represented by globally identified
``PhaseCell`` tokens and evaluated with the unit suspension clock.  Images are
formed by the usual corner-sample, rectangular-hull, and cell-padding rule,
stratified so base and handle endpoints are never hulled together.

This is a reproducible finite sample-and-bloat image cover, in the usual
CMGDB exploratory style.  It does not by itself prove that the returned
multivalued relation contains the whole fixed-time image of every cell; that
outer-relation property is a separate hypothesis or check.  In particular,
base source cells are sampled at their corners and are not yet partitioned by
all interior impact-time strata.  ``guard_sampler`` declares the handle grid;
it does not repair an impact branch missed by every source corner.  Therefore
this module must not be used as an acyclic carrier for a Conley-index
calculation until a whole-cell event-stratified enclosure is supplied.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import numpy.typing as npt

from .grid import Grid
from .hybrid_system import HybridSystem
from .hybrid_trajectory import HybridTrajectory
from .implicit_phase_scc import (
    FullSCCReconstruction,
    PhaseGadgetDescriptor,
    VirtualPhaseNode,
)
from .sampled_suspension import (
    BaseCell,
    BaseSuspensionSample,
    HandleSuspensionSample,
    PhaseCell,
    SuspensionSample,
    sample_suspension_trajectory,
    simulate_suspension_endpoint,
    suspension_integration_jump_limit,
)


State = npt.NDArray[np.float64]
GuardSampler = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], Sequence[State]]
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class SuspensionCemeteryCell:
    """Absorbing compactification used for failed or exiting trajectories."""

    label: str = "exit"


@dataclass(frozen=True)
class GridResetHandle:
    """One guard-trace cell and its implicit unit-handle phase grid."""

    handle_id: Hashable
    guard_cell: BaseCell
    reset_cells: tuple[BaseCell, ...]
    phase_cells: tuple[PhaseCell, ...]
    guard_state: tuple[float, ...]
    reset_state: tuple[float, ...]


@dataclass(frozen=True)
class GridSuspensionIngredients:
    """Plotting and reconstruction metadata for the augmented grid."""

    base_bounds: tuple[tuple[float, float], ...]
    base_subdivisions: tuple[int, ...]
    handles: tuple[GridResetHandle, ...]
    phase_slabs: int
    failure_cells: tuple[SuspensionCemeteryCell, ...] = ()


FiniteSuspensionCell = BaseCell | PhaseCell | SuspensionCemeteryCell


@dataclass(frozen=True)
class _PhaseChartBoundary:
    """A final-time chart change pulled back to a source handle phase."""

    source_phase: float
    face: str
    guard_state: State
    jump_index: int


@dataclass(frozen=True)
class FixedTimeSuspensionGridResult:
    """Full finite relation produced by :func:`compute_fixed_time_suspension_grid`."""

    model_name: str
    t_star: float
    handle_slabs: int
    relation: Mapping[FiniteSuspensionCell, frozenset[FiniteSuspensionCell]]
    relation_graph: nx.DiGraph
    phase_cells: frozenset[PhaseCell]
    recurrent_sccs: tuple[frozenset[FiniteSuspensionCell], ...]
    phase_descriptors: tuple[PhaseGadgetDescriptor, ...]
    direct_base_edges: frozenset[tuple[FiniteSuspensionCell, FiniteSuspensionCell]]
    scc_reconstruction: FullSCCReconstruction
    descriptor_expansion_matches_relation: bool
    suspension_complex_ingredients: GridSuspensionIngredients
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    relation_status: str = (
        "cellwise corner-sampled fixed-time suspension image cover with padding; "
        "base-cell interior event strata are not completed and the whole-cell "
        "outer-relation property is not established"
    )
    conley_index_status: str = (
        "not computed for this physical grid relation"
    )

    def summary(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "t_star": self.t_star,
            "handle_slabs": self.handle_slabs,
            "base_cells": sum(
                isinstance(node, BaseCell) for node in self.relation_graph.nodes
            ),
            "phase_cells": len(self.phase_cells),
            "relation_nodes": self.relation_graph.number_of_nodes(),
            "relation_edges": self.relation_graph.number_of_edges(),
            "recurrent_scc_sizes": [len(value) for value in self.recurrent_sccs],
            "recurrent_base_cells": [
                sum(isinstance(node, BaseCell) for node in value)
                for value in self.recurrent_sccs
            ],
            "recurrent_phase_cells": [
                sum(isinstance(node, PhaseCell) for node in value)
                for value in self.recurrent_sccs
            ],
            "grid_subdivisions": self.diagnostics.get("grid_subdivisions"),
            "ambient_grid_cells": self.diagnostics.get("ambient_grid_cells"),
            "active_base_cells": self.diagnostics.get("active_base_cells"),
            "guard_trace_cells": self.diagnostics.get("guard_trace_cells"),
            "padding_cells": self.diagnostics.get("padding_cells"),
            "descriptor_expansion_matches_relation": (
                self.descriptor_expansion_matches_relation
            ),
            "relation_status": self.relation_status,
            "conley_index_status": self.conley_index_status,
        }


def _cell_key(cell: FiniteSuspensionCell) -> tuple[object, ...]:
    if isinstance(cell, BaseCell):
        return ("base", cell.index)
    if isinstance(cell, PhaseCell):
        return ("phase", cell.guard_cell, cell.slab)
    return ("cemetery", cell.label)


def _grid_cells_in_rectangle(
    grid: Grid,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    *,
    padding_cells: float,
) -> tuple[int, ...]:
    lower_array = np.asarray(lower, dtype=float) - padding_cells * grid.box_widths
    upper_array = np.asarray(upper, dtype=float) + padding_cells * grid.box_widths
    if np.any(upper_array < grid.bounds[:, 0]) or np.any(
        lower_array > grid.bounds[:, 1]
    ):
        return ()
    clipped_lower = np.maximum(lower_array, grid.bounds[:, 0])
    clipped_upper = np.minimum(upper_array, grid.bounds[:, 1])
    scaled_lower = (clipped_lower - grid.bounds[:, 0]) / grid.box_widths
    scaled_upper = (clipped_upper - grid.bounds[:, 0]) / grid.box_widths
    # The grid is a *closed* box cover.  If the lower endpoint lies on an
    # internal face, the box immediately to its left also intersects the
    # rectangle; the box to the right is already selected by the upper-index
    # formula.  A scale-aware tolerance absorbs the few ulps introduced by
    # reconstructing grid faces from bounds and widths.
    index_tolerance = (
        64.0
        * np.finfo(float).eps
        * np.maximum.reduce(
            (
                np.ones(grid.ndim, dtype=float),
                np.abs(scaled_lower),
                np.abs(scaled_upper),
                np.asarray(grid.subdivisions, dtype=float),
            )
        )
    )
    low_index = np.ceil(scaled_lower - index_tolerance).astype(int) - 1
    high_index = np.floor(scaled_upper + index_tolerance).astype(int)
    low_index = np.clip(low_index, 0, grid.subdivisions - 1)
    high_index = np.clip(high_index, low_index, grid.subdivisions - 1)
    ranges = [
        range(int(low_index[axis]), int(high_index[axis]) + 1)
        for axis in range(grid.ndim)
    ]
    return tuple(
        int(np.ravel_multi_index(coordinates, tuple(grid.subdivisions)))
        for coordinates in itertools.product(*ranges)
    )


def _base_cover(
    samples: Sequence[BaseSuspensionSample],
    grid: Grid,
    *,
    padding_cells: float,
) -> set[FiniteSuspensionCell]:
    if not samples:
        return set()
    # Different jump counts are distinct endpoint strata of the base chart.
    # Hulling them together would draw a fictitious rectangle across the
    # intervening reset handle.  This is the hybrid analogue of keeping
    # branches of a discontinuous map in separate image boxes.
    by_jump_count: dict[int, list[State]] = defaultdict(list)
    for sample in samples:
        by_jump_count[int(sample.jumps_completed)].append(sample.state)
    targets: set[FiniteSuspensionCell] = set()
    for branch_states in by_jump_count.values():
        states = np.asarray(branch_states, dtype=float)
        targets.update(
            BaseCell(index)
            for index in _grid_cells_in_rectangle(
                grid,
                np.min(states, axis=0),
                np.max(states, axis=0),
                padding_cells=padding_cells,
            )
        )
    return targets


def _handle_cover(
    samples: Sequence[HandleSuspensionSample],
    grid: Grid,
    *,
    handle_slabs: int,
    padding_cells: float,
    guard_cells: frozenset[int],
) -> set[FiniteSuspensionCell]:
    """Cover handle endpoints by branchwise guard/phase rectangles.

    ``jump_index`` is the branch identifier carried by a sampled suspension
    endpoint.  Images from different reset traversals are unioned, never
    hulled together.  Within one branch, however, the ordinary CMGDB rule is
    used: take the rectangular hull of all sampled guard states and of their
    phase coordinates, pad by the requested number of cells, and include every
    closed grid cell intersecting that rectangle.
    """

    targets: set[FiniteSuspensionCell] = set()
    by_branch: dict[int, list[HandleSuspensionSample]] = defaultdict(list)
    for sample in samples:
        by_branch[int(sample.jump_index)].append(sample)

    for branch_samples in by_branch.values():
        guard_states = np.asarray(
            [sample.guard_state for sample in branch_samples],
            dtype=float,
        )
        candidate_guard_cells = _grid_cells_in_rectangle(
            grid,
            np.min(guard_states, axis=0),
            np.max(guard_states, axis=0),
            padding_cells=padding_cells,
        )
        phases = np.asarray([sample.phase for sample in branch_samples], dtype=float)
        scaled_lower = float(np.min(phases) * handle_slabs - padding_cells)
        scaled_upper = float(np.max(phases) * handle_slabs + padding_cells)
        phase_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(scaled_lower), abs(scaled_upper), handle_slabs)
        )
        # Closed slabs use the same incidence convention as base boxes: an
        # endpoint exactly on a slab face belongs to both incident slabs.
        low_slab = int(np.ceil(scaled_lower - phase_tolerance)) - 1
        high_slab = int(np.floor(scaled_upper + phase_tolerance))
        low_slab = max(0, min(handle_slabs - 1, low_slab))
        high_slab = max(low_slab, min(handle_slabs - 1, high_slab))
        for guard_cell in candidate_guard_cells:
            if guard_cell not in guard_cells:
                continue
            targets.update(
                PhaseCell(guard_cell=guard_cell, slab=slab)
                for slab in range(low_slab, high_slab + 1)
            )
    return targets


def _terminal_handle_face_cover(
    guard_states: Sequence[State],
    grid: Grid,
    *,
    handle_slabs: int,
    padding_cells: float,
    guard_cells: frozenset[int],
) -> set[FiniteSuspensionCell]:
    """Cover the handle face identified with its reset endpoint in the quotient.

    A point at phase one is represented pointwise by its reset state in the
    base chart, but the *closed* augmented cell cover is also incident to the
    terminal handle slab.  This helper records that second chart incidence
    without constructing an invalid interior ``HandleSuspensionSample`` at
    phase one.
    """

    if not guard_states:
        return set()
    states = np.asarray(guard_states, dtype=float)
    candidate_guard_cells = _grid_cells_in_rectangle(
        grid,
        np.min(states, axis=0),
        np.max(states, axis=0),
        padding_cells=padding_cells,
    )
    # Expanding the terminal face by p phase cells includes the terminal slab
    # itself and every preceding closed slab reached by that padding.
    scaled_lower = float(handle_slabs - padding_cells)
    phase_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(scaled_lower), handle_slabs)
    )
    low_slab = int(np.ceil(scaled_lower - phase_tolerance)) - 1
    low_slab = max(0, min(handle_slabs - 1, low_slab))
    return {
        PhaseCell(guard_cell=guard_cell, slab=slab)
        for guard_cell in candidate_guard_cells
        if guard_cell in guard_cells
        for slab in range(low_slab, handle_slabs)
    }


def _initial_handle_face_cover(
    guard_states: Sequence[State],
    grid: Grid,
    *,
    handle_slabs: int,
    padding_cells: float,
    guard_cells: frozenset[int],
) -> set[FiniteSuspensionCell]:
    """Cover the handle face identified with its guard endpoint in the quotient."""

    if not guard_states:
        return set()
    states = np.asarray(guard_states, dtype=float)
    candidate_guard_cells = _grid_cells_in_rectangle(
        grid,
        np.min(states, axis=0),
        np.max(states, axis=0),
        padding_cells=padding_cells,
    )
    # This is the phase-zero counterpart of
    # ``_terminal_handle_face_cover``.  Closed phase padding reaches forward
    # from the initial face.
    scaled_upper = float(padding_cells)
    phase_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(scaled_upper), handle_slabs)
    )
    high_slab = int(np.floor(scaled_upper + phase_tolerance))
    high_slab = max(0, min(handle_slabs - 1, high_slab))
    return {
        PhaseCell(guard_cell=guard_cell, slab=slab)
        for guard_cell in candidate_guard_cells
        if guard_cell in guard_cells
        for slab in range(0, high_slab + 1)
    }


def _cover_samples(
    samples: Sequence[SuspensionSample | None],
    grid: Grid,
    *,
    handle_slabs: int,
    padding_cells: float,
    guard_cells: frozenset[int],
    cemetery: SuspensionCemeteryCell,
) -> set[FiniteSuspensionCell]:
    base = [sample for sample in samples if isinstance(sample, BaseSuspensionSample)]
    handle = [
        sample for sample in samples if isinstance(sample, HandleSuspensionSample)
    ]
    targets = _base_cover(base, grid, padding_cells=padding_cells)
    targets.update(
        _handle_cover(
            handle,
            grid,
            handle_slabs=handle_slabs,
            padding_cells=padding_cells,
            guard_cells=guard_cells,
        )
    )
    if any(sample is None for sample in samples) or (samples and not targets):
        targets.add(cemetery)
    return targets


def _restrict_targets_to_active_base(
    targets: set[FiniteSuspensionCell],
    *,
    active_base_cells: frozenset[int],
    cemetery: SuspensionCemeteryCell,
) -> set[FiniteSuspensionCell]:
    """Replace every target base cell outside the active family by cemetery."""

    restricted = {
        target
        for target in targets
        if not isinstance(target, BaseCell) or target.index in active_base_cells
    }
    if len(restricted) != len(targets):
        restricted.add(cemetery)
    return restricted


def _corner_indices(
    cell_index: int,
    subdivisions: npt.NDArray[np.int64],
) -> tuple[int, ...]:
    coordinates = np.asarray(
        np.unravel_index(cell_index, tuple(subdivisions)),
        dtype=int,
    )
    corner_shape = tuple(subdivisions + 1)
    return tuple(
        int(np.ravel_multi_index(coordinates + offset, corner_shape))
        for offset in itertools.product((0, 1), repeat=len(subdivisions))
    )


def _safe_endpoint(
    system: HybridSystem,
    state: npt.ArrayLike,
    total_time: float,
    *,
    max_jumps: int,
    max_step: float,
) -> SuspensionSample | None:
    try:
        sample = simulate_suspension_endpoint(
            system,
            state,
            total_time,
            max_jumps=max_jumps,
            max_step=max_step,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    if isinstance(sample, BaseSuspensionSample):
        if not np.all(np.isfinite(sample.state)) or not system.is_valid_state(sample.state):
            return None
    if isinstance(sample, HandleSuspensionSample):
        if any(
            not np.all(np.isfinite(state)) or not system.is_valid_state(state)
            for state in (sample.guard_state, sample.reset_state)
        ):
            return None
    return sample


def _phase_endpoint(
    system: HybridSystem,
    guard_state: State,
    reset_state: State,
    phase: float,
    t_star: float,
    *,
    max_jumps: int,
    max_step: float,
) -> SuspensionSample | None:
    target_phase = phase + t_star
    if target_phase < 1.0 - 1e-12:
        return HandleSuspensionSample(
            guard_state=guard_state,
            reset_state=reset_state,
            phase=float(target_phase),
            total_time=float(t_star),
            continuous_time=0.0,
            jump_index=0,
        )
    remaining = max(0.0, target_phase - 1.0)
    if remaining <= 1e-12:
        return BaseSuspensionSample(
            state=reset_state,
            total_time=float(t_star),
            continuous_time=0.0,
            jumps_completed=1,
        )
    sample = _safe_endpoint(
        system,
        reset_state,
        remaining,
        max_jumps=max_jumps,
        max_step=max_step,
    )
    if isinstance(sample, BaseSuspensionSample):
        return BaseSuspensionSample(
            state=sample.state,
            total_time=float(t_star),
            continuous_time=sample.continuous_time,
            jumps_completed=sample.jumps_completed + 1,
        )
    if isinstance(sample, HandleSuspensionSample):
        return HandleSuspensionSample(
            guard_state=sample.guard_state,
            reset_state=sample.reset_state,
            phase=sample.phase,
            total_time=float(t_star),
            continuous_time=sample.continuous_time,
            jump_index=sample.jump_index + 1,
        )
    return None


def _safe_reset_trajectory(
    system: HybridSystem,
    reset_state: State,
    total_time: float,
    *,
    max_jumps: int,
    max_step: float,
    atol: float = 1e-9,
) -> HybridTrajectory | None:
    """Record one reset-origin trajectory for repeated suspension sampling."""

    if total_time <= atol:
        return None
    try:
        integration_max_jumps = suspension_integration_jump_limit(
            total_time,
            max_jumps,
            atol=atol,
        )
        trajectory = system.simulate(
            reset_state,
            (0.0, total_time),
            max_jumps=integration_max_jumps,
            dense_output=True,
            max_step=max_step,
            jump_time_penalty=False,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    return trajectory if trajectory.segments else None


def _shift_reset_origin_sample(
    system: HybridSystem,
    sample: SuspensionSample,
    *,
    t_star: float,
) -> SuspensionSample | None:
    """Account for the already-completed source handle in sample metadata."""

    if isinstance(sample, BaseSuspensionSample):
        if not np.all(np.isfinite(sample.state)) or not system.is_valid_state(sample.state):
            return None
        return BaseSuspensionSample(
            state=sample.state,
            total_time=float(t_star),
            continuous_time=sample.continuous_time,
            jumps_completed=sample.jumps_completed + 1,
        )
    return HandleSuspensionSample(
        guard_state=sample.guard_state,
        reset_state=sample.reset_state,
        phase=sample.phase,
        total_time=float(t_star),
        continuous_time=sample.continuous_time,
        jump_index=sample.jump_index + 1,
    )


def _phase_endpoint_from_reset_trajectory(
    system: HybridSystem,
    reset_state: State,
    phase: float,
    t_star: float,
    trajectory: HybridTrajectory | None,
    *,
    atol: float = 1e-9,
) -> SuspensionSample | None:
    """Sample a phase source using a reset-origin trajectory cached by guard."""

    remaining = max(0.0, float(phase + t_star - 1.0))
    if remaining <= atol:
        return BaseSuspensionSample(
            state=reset_state,
            total_time=float(t_star),
            continuous_time=0.0,
            jumps_completed=1,
        )
    if trajectory is None:
        return None
    try:
        sample = sample_suspension_trajectory(
            trajectory,
            remaining,
            atol=atol,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    return _shift_reset_origin_sample(system, sample, t_star=t_star)


def _phase_chart_boundaries(
    trajectory: HybridTrajectory | None,
    *,
    t_star: float,
    phase_lower: float,
    phase_upper: float,
    atol: float = 1e-9,
) -> tuple[_PhaseChartBoundary, ...]:
    """Pull every recorded future handle entry and exit into a source slab.

    If the reset-origin trajectory has ordinary jump time ``tau_j``, its
    suspension handle begins at ``tau_j + j`` and ends one unit later.  Since a
    source phase ``s`` leaves its current handle with residual budget
    ``t_star + s - 1``, these two chart boundaries pull back by
    ``s = boundary - t_star + 1``.
    """

    if trajectory is None or not trajectory.segments:
        return ()
    origin = float(trajectory.segments[0].t_start)
    boundaries: list[_PhaseChartBoundary] = []
    for jump_index, (jump_time_raw, states) in enumerate(
        zip(trajectory.jump_times, trajectory.jump_states),
    ):
        jump_time = float(jump_time_raw)
        guard_state, _reset_state = states
        entry_time = jump_time - origin + jump_index
        for face, boundary_time in (("initial", entry_time), ("terminal", entry_time + 1.0)):
            source_phase = boundary_time - t_star + 1.0
            scale = max(
                1.0,
                abs(boundary_time),
                abs(t_star),
                abs(phase_lower),
                abs(phase_upper),
            )
            tolerance = max(atol, 64.0 * np.finfo(float).eps * scale)
            if not (
                phase_lower - tolerance
                <= source_phase
                <= phase_upper + tolerance
            ):
                continue
            boundaries.append(
                _PhaseChartBoundary(
                    source_phase=float(
                        min(max(source_phase, phase_lower), phase_upper)
                    ),
                    face=face,
                    guard_state=np.asarray(guard_state, dtype=float),
                    jump_index=jump_index + 1,
                )
            )
    return tuple(boundaries)


def _event_driven_phase_samples(
    system: HybridSystem,
    guard_state: State,
    reset_state: State,
    trajectory: HybridTrajectory | None,
    *,
    phase_lower: float,
    phase_upper: float,
    t_star: float,
    atol: float = 1e-9,
) -> tuple[list[SuspensionSample | None], tuple[_PhaseChartBoundary, ...]]:
    """Sample every final-time chart stratum over one source phase slab."""

    boundaries = list(
        _phase_chart_boundaries(
            trajectory,
            t_star=t_star,
            phase_lower=phase_lower,
            phase_upper=phase_upper,
            atol=atol,
        )
    )
    current_exit_phase = 1.0 - t_star
    tolerance = max(
        atol,
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(t_star), abs(phase_lower), abs(phase_upper)),
    )
    if (
        phase_lower - tolerance
        <= current_exit_phase
        <= phase_upper + tolerance
    ):
        boundaries.append(
            _PhaseChartBoundary(
                source_phase=float(
                    min(max(current_exit_phase, phase_lower), phase_upper)
                ),
                face="terminal",
                guard_state=np.asarray(guard_state, dtype=float),
                jump_index=0,
            )
        )

    cuts = sorted(
        {
            float(phase_lower),
            float(phase_upper),
            *(float(boundary.source_phase) for boundary in boundaries),
        }
    )
    # Boundary points themselves live in the base quotient chart.  A midpoint
    # in every nonempty open stratum ensures that a handle interval bracketed
    # by two base-chart boundary values is not lost.
    source_phases = list(cuts)
    source_phases.extend(
        (lower + upper) / 2.0
        for lower, upper in zip(cuts, cuts[1:])
        if upper - lower > 2.0 * tolerance
    )
    source_phases.sort()
    samples = [
        _phase_endpoint_from_reset_trajectory(
            system,
            reset_state,
            phase,
            t_star,
            trajectory,
            atol=atol,
        )
        for phase in source_phases
    ]
    return samples, tuple(boundaries)


def _recurrent_components(
    graph: nx.DiGraph,
) -> tuple[frozenset[FiniteSuspensionCell], ...]:
    recurrent = []
    for raw_component in nx.strongly_connected_components(graph):
        component = frozenset(raw_component)
        if len(component) > 1:
            recurrent.append(component)
            continue
        node = next(iter(component))
        if graph.has_edge(node, node):
            recurrent.append(component)
    recurrent.sort(key=lambda value: tuple(sorted((_cell_key(node) for node in value))))
    return tuple(recurrent)


def _phase_realized_macro_edges(
    graph: nx.DiGraph,
    phase_set: set[PhaseCell],
) -> frozenset[tuple[FiniteSuspensionCell, FiniteSuspensionCell]]:
    """Return the exact base relation realized through phase-only interiors.

    The phase-induced SCC condensation is computed once.  Exit target sets are
    propagated backwards through its DAG, replacing one descendant search per
    base entry.  The SCC/DAG work is ``O(V+E)``; materializing the returned
    relation is necessarily output-sensitive because that relation can itself
    contain quadratically many base pairs.
    """

    if not phase_set:
        return frozenset()

    phase_graph = graph.subgraph(phase_set)
    phase_components = tuple(
        frozenset(component)
        for component in nx.strongly_connected_components(phase_graph)
    )
    component_of = {
        node: component_index
        for component_index, component in enumerate(phase_components)
        for node in component
    }
    component_graph = nx.DiGraph()
    component_graph.add_nodes_from(range(len(phase_components)))
    exits: list[set[FiniteSuspensionCell]] = [
        set() for _ in phase_components
    ]
    entries: list[tuple[FiniteSuspensionCell, int]] = []

    for source, target in graph.edges:
        source_is_phase = source in phase_set
        target_is_phase = target in phase_set
        if source_is_phase and target_is_phase:
            source_component = component_of[source]
            target_component = component_of[target]
            if source_component != target_component:
                component_graph.add_edge(source_component, target_component)
        elif source_is_phase:
            exits[component_of[source]].add(target)
        elif target_is_phase:
            entries.append((source, component_of[target]))

    reachable_exits = [set(targets) for targets in exits]
    for component in reversed(tuple(nx.topological_sort(component_graph))):
        for successor in component_graph.successors(component):
            reachable_exits[component].update(reachable_exits[successor])

    return frozenset(
        (base_source, base_target)
        for base_source, phase_component in entries
        for base_target in reachable_exits[phase_component]
    )


def _linear_scc_reconstruction(
    virtual_graph: nx.DiGraph,
    *,
    macro_edges_realized_by_phase: frozenset[
        tuple[FiniteSuspensionCell, FiniteSuspensionCell]
    ],
    direct_base_edges: frozenset[
        tuple[FiniteSuspensionCell, FiniteSuspensionCell]
    ],
) -> FullSCCReconstruction:
    """Build the complete SCC partition and condensation in ``O(V+E)``."""

    components = tuple(
        frozenset(component)
        for component in nx.strongly_connected_components(virtual_graph)
    )
    component_of: dict[Hashable, int] = {}
    condensation = nx.DiGraph()
    condensation.graph["semantics"] = "full suspension SCC condensation"

    for component_index, component in enumerate(components):
        base_members = frozenset(
            node for node in component if not isinstance(node, VirtualPhaseNode)
        )
        phase_members = component.difference(base_members)
        condensation.add_node(
            component_index,
            members=component,
            base_members=base_members,
            phase_members=phase_members,
            kind="base-containing" if base_members else "phase-only",
        )
        for node in component:
            component_of[node] = component_index

    for source, target in virtual_graph.edges:
        source_component = component_of[source]
        target_component = component_of[target]
        if source_component != target_component:
            condensation.add_edge(source_component, target_component)

    if not nx.is_directed_acyclic_graph(condensation):
        raise AssertionError("SCC condensation must be acyclic")
    if len(component_of) != virtual_graph.number_of_nodes():
        raise AssertionError("SCC reconstruction omitted a virtual graph node")

    return FullSCCReconstruction(
        components=components,
        component_of=component_of,
        condensation=condensation,
        virtual_graph=virtual_graph,
        macro_edges_realized_by_phase=macro_edges_realized_by_phase,
        retained_direct_base_edges=direct_base_edges,
    )


def _descriptor_reconstruction(
    model_name: str,
    graph: nx.DiGraph,
) -> tuple[
    tuple[PhaseGadgetDescriptor, ...],
    frozenset[tuple[FiniteSuspensionCell, FiniteSuspensionCell]],
    FullSCCReconstruction,
    bool,
]:
    phase_nodes = tuple(node for node in graph.nodes if isinstance(node, PhaseCell))
    phase_set = set(phase_nodes)
    descriptor_id = (model_name, "full_grid_fixed_time_phase_relation")
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
    direct = frozenset(
        (source, target)
        for source, target in graph.edges
        if source not in phase_set and target not in phase_set
    )
    def expanded(node: FiniteSuspensionCell) -> Hashable:
        return VirtualPhaseNode(descriptor_id, node) if node in phase_set else node

    expanded_of = {node: expanded(node) for node in graph.nodes}
    virtual_graph = nx.DiGraph()
    virtual_graph.graph.update(graph.graph)
    virtual_graph.graph["semantics"] = (
        "exact virtual expansion of fixed-time suspension relation"
    )
    for node, attributes in graph.nodes(data=True):
        copied = dict(attributes)
        copied.setdefault("kind", "phase" if node in phase_set else "base")
        virtual_graph.add_node(expanded_of[node], **copied)
    for source, target, attributes in graph.edges(data=True):
        copied = dict(attributes)
        if source in phase_set and target in phase_set:
            copied.setdefault("kind", "phase")
        elif source in phase_set:
            copied.setdefault("kind", "phase_exit")
        elif target in phase_set:
            copied.setdefault("kind", "phase_entry")
        else:
            copied.setdefault("kind", "base")
        virtual_graph.add_edge(
            expanded_of[source], expanded_of[target], **copied
        )

    expected_nodes = set(expanded_of.values())
    expected_edges = {
        (expanded_of[source], expanded_of[target]) for source, target in graph.edges
    }
    node_bijection = len(expected_nodes) == graph.number_of_nodes()
    node_set_equal = set(virtual_graph.nodes) == expected_nodes
    edge_set_equal = set(virtual_graph.edges) == expected_edges
    matches = node_bijection and node_set_equal and edge_set_equal
    parity_certificate = {
        "exact": matches,
        "node_bijection": node_bijection,
        "node_set_equal": node_set_equal,
        "edge_set_equal": edge_set_equal,
        "source_node_count": graph.number_of_nodes(),
        "expanded_node_count": virtual_graph.number_of_nodes(),
        "source_edge_count": graph.number_of_edges(),
        "expanded_edge_count": virtual_graph.number_of_edges(),
    }
    virtual_graph.graph["graph_parity_certificate"] = parity_certificate

    realized = _phase_realized_macro_edges(graph, phase_set)
    reconstruction = _linear_scc_reconstruction(
        virtual_graph,
        macro_edges_realized_by_phase=realized,
        direct_base_edges=direct,
    )
    return (descriptor,), direct, reconstruction, matches


def compute_fixed_time_suspension_grid(
    *,
    model_name: str,
    system: HybridSystem,
    grid: Grid,
    guard_sampler: GuardSampler,
    t_star: float = 0.5,
    handle_slabs: int = 16,
    padding_cells: float = 1.0,
    max_jumps: int = 12,
    max_step: float = 0.02,
    cemetery_label: str = "exit",
    progress_callback: ProgressCallback | None = None,
    active_base_cells: Collection[int] | None = None,
) -> FixedTimeSuspensionGridResult:
    """Compute a cellwise sampled relation for ``Phi_H^t_star``.

    ``guard_sampler(lower, upper)`` must return representative points of the
    guard trace in the given closed base cell.  Returning an empty sequence
    declares that the cell misses the guard.  The reset map is taken from
    ``system``.  Both the base and phase directions use closed-cell endpoint
    samples and the same ``padding_cells`` cover rule.  If
    ``active_base_cells`` is supplied, only those ambient grid indices are
    evaluated and declared; image boxes outside that family are represented by
    the absorbing cemetery cell.  Every finite ``t_star > 0`` is supported.
    For ``t_star >= 1``, reset-origin trajectories are cached and every
    successive handle entry/exit that changes the endpoint chart over a source
    phase slab is sampled explicitly.

    """

    if not np.isfinite(t_star) or t_star <= 0.0:
        raise ValueError("t_star must be finite and strictly positive")
    if handle_slabs <= 0:
        raise ValueError("handle_slabs must be positive")
    if not np.isfinite(padding_cells) or padding_cells < 0:
        raise ValueError("padding_cells must be finite and non-negative")

    if active_base_cells is None:
        active_cell_indices = tuple(grid.box_indices)
    else:
        requested_indices = tuple(active_base_cells)
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in requested_indices
        ):
            raise ValueError("active_base_cells must contain integer grid indices")
        active_cell_indices = tuple(
            sorted({int(index) for index in requested_indices})
        )
        if any(
            index < 0 or index >= grid.total_boxes
            for index in active_cell_indices
        ):
            raise ValueError("active_base_cells contains an out-of-range grid index")
    active_cell_ids = frozenset(active_cell_indices)
    subdivisions = np.asarray(grid.subdivisions, dtype=np.int64)

    guard_samples: dict[int, tuple[tuple[State, State], ...]] = {}
    handles: list[GridResetHandle] = []
    for cell_index in active_cell_indices:
        lower, upper = grid.get_box_bounds(cell_index)
        representatives = tuple(
            np.asarray(state, dtype=float) for state in guard_sampler(lower, upper)
        )
        if not representatives:
            continue
        pairs = []
        reset_cells: set[BaseCell] = set()
        for guard_state in representatives:
            if guard_state.shape != (grid.ndim,):
                raise ValueError("guard_sampler returned a state of the wrong dimension")
            reset_state = np.asarray(system.apply_reset_map(guard_state), dtype=float)
            if reset_state.shape != guard_state.shape or not np.all(np.isfinite(reset_state)):
                raise ValueError("reset map returned an invalid state")
            pairs.append((guard_state, reset_state))
            reset_cells.update(
                BaseCell(index)
                for index in _grid_cells_in_rectangle(
                    grid,
                    reset_state,
                    reset_state,
                    padding_cells=0.0,
                )
                if index in active_cell_ids
            )
        guard_samples[cell_index] = tuple(pairs)
        phase_cells = tuple(
            PhaseCell(guard_cell=cell_index, slab=slab)
            for slab in range(handle_slabs)
        )
        representative_guard, representative_reset = pairs[len(pairs) // 2]
        handles.append(
            GridResetHandle(
                handle_id=(model_name, "guard", cell_index),
                guard_cell=BaseCell(cell_index),
                reset_cells=tuple(sorted(reset_cells, key=lambda cell: cell.index)),
                phase_cells=phase_cells,
                guard_state=tuple(float(value) for value in representative_guard),
                reset_state=tuple(float(value) for value in representative_reset),
            )
        )
    guard_cell_ids = frozenset(guard_samples)

    # For long horizons, every phase cell over one guard trace samples the
    # same reset-origin suspension trajectory at different residual times.
    # Record it once per representative, both to expose all successive chart
    # crossings and to avoid reintegrating it for every phase slab.
    reset_trajectories: dict[tuple[int, int], HybridTrajectory | None] = {}
    if t_star >= 1.0:
        for guard_cell, pairs in guard_samples.items():
            for representative_index, (_guard_state, reset_state) in enumerate(pairs):
                reset_trajectories[(guard_cell, representative_index)] = (
                    _safe_reset_trajectory(
                        system,
                        reset_state,
                        t_star,
                        max_jumps=max_jumps,
                        max_step=max_step,
                    )
                )

    corner_shape = tuple(int(value) + 1 for value in subdivisions)
    active_corner_indices = tuple(
        sorted(
            {
                corner_index
                for cell_index in active_cell_indices
                for corner_index in _corner_indices(cell_index, subdivisions)
            }
        )
    )
    corner_axes = tuple(
        np.linspace(
            grid.bounds[axis, 0],
            grid.bounds[axis, 1],
            int(grid.subdivisions[axis]) + 1,
        )
        for axis in range(grid.ndim)
    )
    endpoint_samples: dict[int, SuspensionSample | None] = {}
    for position, corner_index in enumerate(active_corner_indices, start=1):
        coordinates = np.unravel_index(corner_index, corner_shape)
        point = np.asarray(
            [corner_axes[axis][coordinates[axis]] for axis in range(grid.ndim)],
            dtype=float,
        )
        endpoint_samples[corner_index] = _safe_endpoint(
            system,
            point,
            t_star,
            max_jumps=max_jumps,
            max_step=max_step,
        )
        if progress_callback is not None:
            progress_callback("base vertices", position, len(active_corner_indices))

    cemetery = SuspensionCemeteryCell(cemetery_label)
    mutable_relation: dict[FiniteSuspensionCell, set[FiniteSuspensionCell]] = {}
    for position, cell_index in enumerate(active_cell_indices, start=1):
        samples = [
            endpoint_samples[index]
            for index in _corner_indices(cell_index, subdivisions)
        ]
        mutable_relation[BaseCell(cell_index)] = _restrict_targets_to_active_base(
            _cover_samples(
                samples,
                grid,
                handle_slabs=handle_slabs,
                padding_cells=padding_cells,
                guard_cells=guard_cell_ids,
                cemetery=cemetery,
            ),
            active_base_cells=active_cell_ids,
            cemetery=cemetery,
        )
        if progress_callback is not None:
            progress_callback(
                "base cells",
                position,
                len(active_cell_indices),
            )

    phase_cells = tuple(
        PhaseCell(guard_cell=guard_cell, slab=slab)
        for guard_cell in sorted(guard_samples)
        for slab in range(handle_slabs)
    )
    phase_total = len(phase_cells)
    crossing_phase = 1.0 - t_star
    phase_chart_boundary_count = 0
    phase_source_sample_count = 0
    maximum_phase_chart_boundaries = 0
    for position, phase_cell in enumerate(phase_cells, start=1):
        phase_lower = phase_cell.slab / handle_slabs
        phase_upper = (phase_cell.slab + 1) / handle_slabs
        target_samples: list[SuspensionSample | None] = []
        chart_boundaries: list[_PhaseChartBoundary] = []
        if t_star < 1.0:
            # Preserve the established subunit corner rule exactly.  In this
            # regime the old current-handle quotient boundary remains the only
            # extra source sample inserted by this implementation.
            phase_tolerance = (
                64.0
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(crossing_phase),
                    abs(phase_lower),
                    abs(phase_upper),
                )
            )
            source_phases = [phase_lower, phase_upper]
            crosses_quotient = (
                phase_lower - phase_tolerance
                <= crossing_phase
                <= phase_upper + phase_tolerance
            )
            if crosses_quotient and not any(
                abs(phase - crossing_phase) <= phase_tolerance
                for phase in source_phases
            ):
                # The map changes charts at this interior source phase.
                source_phases.append(crossing_phase)
            for guard_state, reset_state in guard_samples[phase_cell.guard_cell]:
                for phase in source_phases:
                    target_samples.append(
                        _phase_endpoint(
                            system,
                            guard_state,
                            reset_state,
                            phase,
                            t_star,
                            max_jumps=max_jumps,
                            max_step=max_step,
                        )
                    )
            phase_source_sample_count += len(target_samples)
            if crosses_quotient:
                chart_boundaries.extend(
                    _PhaseChartBoundary(
                        source_phase=float(crossing_phase),
                        face="terminal",
                        guard_state=np.asarray(guard_state, dtype=float),
                        jump_index=0,
                    )
                    for guard_state, _reset_state in guard_samples[
                        phase_cell.guard_cell
                    ]
                )
        else:
            # Pull every future handle entry/exit back through the residual
            # suspension clock and sample each open chart stratum separately.
            for representative_index, (guard_state, reset_state) in enumerate(
                guard_samples[phase_cell.guard_cell]
            ):
                samples, boundaries = _event_driven_phase_samples(
                    system,
                    guard_state,
                    reset_state,
                    reset_trajectories[(
                        phase_cell.guard_cell,
                        representative_index,
                    )],
                    phase_lower=phase_lower,
                    phase_upper=phase_upper,
                    t_star=t_star,
                )
                target_samples.extend(samples)
                chart_boundaries.extend(boundaries)
            phase_source_sample_count += len(target_samples)
        phase_targets = _cover_samples(
            target_samples,
            grid,
            handle_slabs=handle_slabs,
            padding_cells=padding_cells,
            guard_cells=guard_cell_ids,
            cemetery=cemetery,
        )
        phase_chart_boundary_count += len(chart_boundaries)
        maximum_phase_chart_boundaries = max(
            maximum_phase_chart_boundaries,
            len(chart_boundaries),
        )
        boundary_guard_states: dict[tuple[str, int], list[State]] = defaultdict(list)
        for boundary in chart_boundaries:
            boundary_guard_states[(boundary.face, boundary.jump_index)].append(
                boundary.guard_state
            )
        for (face, _jump_index), boundary_states in boundary_guard_states.items():
            if face == "initial":
                phase_targets.update(
                    _initial_handle_face_cover(
                        boundary_states,
                        grid,
                        handle_slabs=handle_slabs,
                        padding_cells=padding_cells,
                        guard_cells=guard_cell_ids,
                    )
                )
            else:
                # A quotient endpoint is pointwise in the base chart and also
                # incident to the corresponding closed handle face.
                phase_targets.update(
                    _terminal_handle_face_cover(
                        boundary_states,
                        grid,
                        handle_slabs=handle_slabs,
                        padding_cells=padding_cells,
                        guard_cells=guard_cell_ids,
                    )
                )
        mutable_relation[phase_cell] = _restrict_targets_to_active_base(
            phase_targets,
            active_base_cells=active_cell_ids,
            cemetery=cemetery,
        )
        if progress_callback is not None:
            progress_callback("phase cells", position, phase_total)

    mutable_relation[cemetery] = {cemetery}
    relation = {
        source: frozenset(targets)
        for source, targets in sorted(
            mutable_relation.items(),
            key=lambda item: _cell_key(item[0]),
        )
    }
    graph = nx.DiGraph()
    graph.graph.update(
        semantics="cellwise sampled fixed-time unit-handle suspension relation",
        model_name=model_name,
        t_star=float(t_star),
        padding_cells=float(padding_cells),
        whole_cell_outer_enclosure=False,
        base_source_event_strata_completed=False,
    )
    graph.add_nodes_from(relation)
    graph.add_edges_from(
        (source, target)
        for source, targets in relation.items()
        for target in targets
    )
    descriptors, direct, reconstruction, matches = _descriptor_reconstruction(
        model_name,
        graph,
    )
    if not matches:
        raise AssertionError("implicit phase expansion differs from materialized graph")
    ingredients = GridSuspensionIngredients(
        base_bounds=tuple(
            (float(lower), float(upper)) for lower, upper in grid.bounds
        ),
        base_subdivisions=tuple(int(value) for value in grid.subdivisions),
        handles=tuple(handles),
        phase_slabs=handle_slabs,
        failure_cells=(cemetery,),
    )
    return FixedTimeSuspensionGridResult(
        model_name=model_name,
        t_star=float(t_star),
        handle_slabs=handle_slabs,
        relation=relation,
        relation_graph=graph,
        phase_cells=frozenset(phase_cells),
        recurrent_sccs=_recurrent_components(graph),
        phase_descriptors=descriptors,
        direct_base_edges=direct,
        scc_reconstruction=reconstruction,
        descriptor_expansion_matches_relation=matches,
        suspension_complex_ingredients=ingredients,
        diagnostics={
            "grid_subdivisions": tuple(int(value) for value in grid.subdivisions),
            "grid_cells": grid.total_boxes,
            "ambient_grid_cells": grid.total_boxes,
            "active_base_cells": len(active_cell_indices),
            "unique_corner_samples": len(active_corner_indices),
            "ambient_unique_corner_samples": int(np.prod(corner_shape)),
            "guard_trace_cells": len(guard_samples),
            "padding_cells": float(padding_cells),
            "whole_cell_outer_enclosure": False,
            "base_source_event_strata_completed": False,
            "phase_sampling_mode": (
                "event-driven-all-crossings"
                if t_star >= 1.0
                else "legacy-subunit-single-crossing"
            ),
            "reset_trajectory_simulations": len(reset_trajectories),
            "phase_source_samples": phase_source_sample_count,
            "phase_chart_boundaries": phase_chart_boundary_count,
            "maximum_phase_chart_boundaries_per_cell": (
                maximum_phase_chart_boundaries
            ),
            "graph_parity_certificate": dict(
                reconstruction.virtual_graph.graph["graph_parity_certificate"]
            ),
        },
    )


__all__ = [
    "SuspensionCemeteryCell",
    "GridResetHandle",
    "GridSuspensionIngredients",
    "FixedTimeSuspensionGridResult",
    "compute_fixed_time_suspension_grid",
]
