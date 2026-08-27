"""Sample the unit-handle suspension without constructing a suspension grid.

The simulator records ordinary continuous time and a list of instantaneous
resets.  For the suspension used by the attractor construction, a reset is
instead represented by a handle of duration one.  Thus elapsed suspension
time is ``T = (t - t0) + j``, where ``j`` is the number of handles already
traversed.

This module reconstructs that clock from an existing :class:`HybridTrajectory`.
It intentionally does *not* use the legacy ``jump_time_penalty`` option: an
epsilon deducted from an integration horizon is not a unit suspension handle.

The faithful finite representation has both base cells and handle-phase cells;
the node types and locator below make that distinction explicit.  The graph
utilities eliminate paths whose internal vertices are all handle-phase
vertices.  This preserves the induced base-to-base path relation.  It does not
preserve the full lattice of forward-invariant vertex sets.  The Morse
diagnostic is therefore deliberately narrower and rejects phase-only recurrent
components.

This module is the pointwise clock/reference layer.  The paper's base-grid
algorithm is implemented at graph level in :mod:`implicit_phase_scc`, which
retains phase paths as descriptors and reconstructs every SCC, including
transient phase-only components.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, Tuple, Union

import networkx as nx
import numpy as np
import numpy.typing as npt

from .grid import Grid
from .hybrid_trajectory import HybridTrajectory

if TYPE_CHECKING:
    from .hybrid_system import HybridSystem


State = npt.NDArray[np.float64]


def _state_copy(state: npt.ArrayLike) -> State:
    """Return a read-only float copy suitable for a frozen sample record."""
    result = np.asarray(state, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BaseSuspensionSample:
    """A suspension endpoint lying in the base space."""

    state: State
    total_time: float
    continuous_time: float
    jumps_completed: int


@dataclass(frozen=True)
class HandleSuspensionSample:
    """A suspension endpoint in the interior of a reset handle.

    ``phase`` is the normalized handle coordinate and is strictly between zero
    and one.  At phase zero and one the quotient point is returned as a
    :class:`BaseSuspensionSample` instead.
    """

    guard_state: State
    reset_state: State
    phase: float
    total_time: float
    continuous_time: float
    jump_index: int


SuspensionSample = Union[BaseSuspensionSample, HandleSuspensionSample]


@dataclass(frozen=True)
class BaseCell:
    """A base-grid cell in the finite augmented suspension representation."""

    index: int


@dataclass(frozen=True)
class PhaseCell:
    """A handle slab over a guard-intersecting base cell."""

    guard_cell: int
    slab: int


AugmentedCell = Union[BaseCell, PhaseCell]


def _base_sample(
    state: npt.ArrayLike,
    *,
    total_time: float,
    continuous_time: float,
    jumps_completed: int,
) -> BaseSuspensionSample:
    return BaseSuspensionSample(
        state=_state_copy(state),
        total_time=float(total_time),
        continuous_time=float(continuous_time),
        jumps_completed=jumps_completed,
    )


def sample_suspension_trajectory(
    trajectory: HybridTrajectory,
    total_time: float,
    *,
    atol: float = 1e-9,
) -> SuspensionSample:
    """Sample a trajectory using the unit-handle suspension clock.

    Args:
        trajectory: A trajectory computed with ordinary continuous time and
            instantaneous reset records.
        total_time: Elapsed suspension time ``T >= 0`` from the first segment.
        atol: Absolute tolerance used only to identify quotient endpoints.

    Returns:
        A base sample, or a handle sample carrying the guard state, reset state,
        and handle phase.

    Raises:
        ValueError: If the input is malformed, ``total_time`` is negative, or
            the recorded trajectory does not cover the requested suspension
            time.  In particular, an early integration failure is never
            silently replaced by the last available state.
    """
    total_time = float(total_time)
    atol = float(atol)
    if not np.isfinite(total_time) or total_time < 0:
        raise ValueError("total_time must be a finite non-negative number")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    if not trajectory.segments:
        raise ValueError("cannot sample an empty trajectory")
    if len(trajectory.jump_times) != len(trajectory.jump_states):
        raise ValueError("trajectory jump times and jump states are inconsistent")
    if len(trajectory.segments) not in {
        len(trajectory.jump_times),
        len(trajectory.jump_times) + 1,
    }:
        raise ValueError("trajectory segments and jumps are inconsistent")

    origin = float(trajectory.segments[0].t_start)
    previous_jump_time = origin

    # Handles are checked first so their quotient endpoints are unambiguous even
    # though adjacent continuous segments share the same ordinary event time.
    for jump_index, (jump_time_raw, states) in enumerate(
        zip(trajectory.jump_times, trajectory.jump_states),
    ):
        jump_time = float(jump_time_raw)
        if jump_time + atol < previous_jump_time:
            raise ValueError("trajectory jump times must be nondecreasing")
        previous_jump_time = jump_time

        guard_state, reset_state = states
        handle_start = jump_time - origin + jump_index
        handle_end = handle_start + 1.0

        if handle_start - atol <= total_time <= handle_end + atol:
            if total_time <= handle_start + atol:
                return _base_sample(
                    guard_state,
                    total_time=total_time,
                    continuous_time=jump_time,
                    jumps_completed=jump_index,
                )
            if total_time >= handle_end - atol:
                return _base_sample(
                    reset_state,
                    total_time=total_time,
                    continuous_time=jump_time,
                    jumps_completed=jump_index + 1,
                )

            return HandleSuspensionSample(
                guard_state=_state_copy(guard_state),
                reset_state=_state_copy(reset_state),
                phase=float(total_time - handle_start),
                total_time=total_time,
                continuous_time=jump_time,
                jump_index=jump_index,
            )

    # Segment position in the list is the number of recorded handles preceding
    # that segment.  This remains true when absolute HybridTime jump indices have
    # been offset by simulate_from_hybrid_time.
    for jumps_completed, segment in enumerate(trajectory.segments):
        segment_start = float(segment.t_start) - origin + jumps_completed
        segment_end = float(segment.t_end) - origin + jumps_completed
        if segment_start - atol <= total_time <= segment_end + atol:
            continuous_time = origin + total_time - jumps_completed
            continuous_time = min(
                max(continuous_time, float(segment.t_start)),
                float(segment.t_end),
            )
            if segment.solution is None:
                raise ValueError("trajectory segment has no interpolation function")
            return _base_sample(
                segment.solution(continuous_time),
                total_time=total_time,
                continuous_time=continuous_time,
                jumps_completed=jumps_completed,
            )

    raise ValueError(
        "recorded trajectory does not cover suspension time " f"T={total_time}",
    )


def suspension_integration_jump_limit(
    total_time: float,
    configured_max_jumps: int,
    *,
    atol: float = 1e-9,
) -> int:
    """Return the tight simulator jump cap needed for suspension time ``T``.

    The instantaneous-reset simulator records one event beyond its
    ``max_jumps`` parameter before it stops.  Since each completed suspension
    handle consumes one unit of ``T``, ``ceil(T - atol) - 1`` is therefore a
    sufficient cap, including zero-flow reset chains.  A smaller configured
    limit is retained and can make a later sampling request fail explicitly.

    This helper depends on the documented current event-recording convention
    of :meth:`HybridTrajectory.compute_trajectory`.
    """
    total_time = float(total_time)
    atol = float(atol)
    if not np.isfinite(total_time) or total_time < 0:
        raise ValueError("total_time must be a finite non-negative number")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    if isinstance(configured_max_jumps, (bool, np.bool_)) or not isinstance(
        configured_max_jumps,
        (int, np.integer),
    ):
        raise ValueError("configured_max_jumps must be a non-negative integer")
    if configured_max_jumps < 0:
        raise ValueError("configured_max_jumps must be a non-negative integer")

    unit_handle_budget = max(
        0,
        math.ceil(max(0.0, total_time - atol)) - 1,
    )
    return min(int(configured_max_jumps), unit_handle_budget)


def simulate_suspension_endpoint(
    system: "HybridSystem",
    initial_state: npt.ArrayLike,
    total_time: float,
    *,
    max_jumps: int | None = None,
    max_step: float | None = None,
    atol: float = 1e-9,
    require_domain_path: bool = False,
) -> SuspensionSample:
    """Simulate and sample the exact unit-handle suspension endpoint.

    Integrating ordinary continuous time up to ``total_time`` is sufficient:
    the continuous time needed to reach suspension time ``T`` is at most ``T``.
    The resulting reset records are then re-timed by
    :func:`sample_suspension_trajectory`.
    """
    total_time = float(total_time)
    initial = np.asarray(initial_state, dtype=np.float64)
    if not np.isfinite(total_time) or total_time < 0:
        raise ValueError("total_time must be a finite non-negative number")
    if total_time == 0:
        return _base_sample(
            initial,
            total_time=0.0,
            continuous_time=0.0,
            jumps_completed=0,
        )

    # Every completed reset handle consumes one full unit of suspension time.
    # ``HybridTrajectory.compute_trajectory`` records the event that makes its
    # internal jump count exceed ``max_jumps`` before stopping, so a suspension
    # horizon T needs at most ceil(T)-1 as that parameter.  The tolerance agrees
    # with the quotient-endpoint convention in ``sample_suspension_trajectory``:
    # a request within ``atol`` of an integer handle endpoint needs no following
    # continuous segment.  Respect a smaller caller/system jump limit, since in
    # that case failure to cover T is intentional and reported by the sampler.
    requested_max_jumps = system.max_jumps if max_jumps is None else max_jumps
    integration_max_jumps = suspension_integration_jump_limit(
        total_time,
        requested_max_jumps,
        atol=atol,
    )
    trajectory = system.simulate(
        initial,
        (0.0, total_time),
        max_jumps=integration_max_jumps,
        dense_output=True,
        max_step=max_step,
        jump_time_penalty=False,
    )
    if require_domain_path:
        bounds = (
            None
            if system.domain_bounds is None
            else np.asarray(system.domain_bounds, dtype=np.float64)
        )
        tolerance = (
            None
            if bounds is None
            else atol * np.maximum(1.0, np.max(np.abs(bounds), axis=1))
        )

        def inside(states: npt.ArrayLike) -> bool:
            values = np.atleast_2d(np.asarray(states, dtype=np.float64))
            if not np.all(np.isfinite(values)):
                return False
            if bounds is not None and tolerance is not None:
                if not bool(
                    np.all(values >= bounds[:, 0] - tolerance)
                    and np.all(values <= bounds[:, 1] + tolerance)
                ):
                    return False
            predicate = getattr(system, "domain_predicate", None)
            return predicate is None or all(bool(predicate(state)) for state in values)

        if any(not inside(segment.state_values) for segment in trajectory.segments):
            raise ValueError("trajectory leaves the declared state space")
        if any(
            not inside(state)
            for guard_reset in trajectory.jump_states
            for state in guard_reset
        ):
            raise ValueError("reset leaves the declared state space")
    return sample_suspension_trajectory(trajectory, total_time, atol=atol)


def locate_augmented_cells(
    sample: SuspensionSample,
    grid: Grid,
    handle_slabs: int,
    *,
    atol: float = 1e-10,
) -> frozenset[AugmentedCell]:
    """Locate an endpoint in the closed base-plus-handle cell cover.

    Base points on a grid boundary belong to every incident base cell.  Handle
    points use the incident cells of their guard endpoint and the uniform closed
    slab cover of ``[0, 1]``.  Consequently, a point on an interior slab boundary
    belongs to both adjacent slabs.

    This function constructs cell identities only.  It does not validate the
    sampled outgoing-image relation for those cells.
    """
    if handle_slabs <= 0:
        raise ValueError("handle_slabs must be positive")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")

    if isinstance(sample, BaseSuspensionSample):
        indices = grid.find_boxes_containing_point(sample.state, tolerance=atol)
        if not indices:
            raise ValueError("base sample lies outside the grid")
        return frozenset(BaseCell(index) for index in indices)

    if not 0.0 < sample.phase < 1.0:
        raise ValueError("a handle sample must have phase strictly between zero and one")
    guard_indices = grid.find_boxes_containing_point(
        sample.guard_state,
        tolerance=atol,
    )
    if not guard_indices:
        raise ValueError("handle guard state lies outside the base grid")

    scaled_phase = sample.phase * handle_slabs
    lower_slab = min(int(np.floor(scaled_phase)), handle_slabs - 1)
    slabs = {lower_slab}
    nearest_boundary = int(round(scaled_phase))
    if (
        0 < nearest_boundary < handle_slabs
        and abs(scaled_phase - nearest_boundary) <= atol * handle_slabs
    ):
        slabs = {nearest_boundary - 1, nearest_boundary}

    return frozenset(
        PhaseCell(guard_cell=guard_cell, slab=slab)
        for guard_cell in guard_indices
        for slab in slabs
    )


def build_augmented_outer_graph(
    cells: Collection[AugmentedCell],
    enclose_image: Callable[[AugmentedCell], Collection[AugmentedCell]],
) -> nx.DiGraph:
    """Build a finite outer-model graph from a whole-cell enclosure callback.

    Every declared cell is added, including cells with no outgoing edge.  The
    callback returns the caller's proposed cells for the fixed-time image of
    its input cell.  This function checks finite index bookkeeping but does not
    validate that callback relation.

    Raises:
        ValueError: If the cell family is empty or the callback returns a cell
            outside that family.
    """
    declared = tuple(dict.fromkeys(cells))
    if not declared:
        raise ValueError("cells must contain at least one augmented cell")
    declared_set = set(declared)

    graph = nx.DiGraph()
    graph.graph["semantics"] = "exact-time unit-handle suspension outer model"
    graph.add_nodes_from(declared)

    for source in declared:
        destinations = set(enclose_image(source))
        unknown = destinations.difference(declared_set)
        if unknown:
            raise ValueError(
                f"enclosure for {source!r} returned undeclared cells: {unknown!r}",
            )
        graph.add_edges_from((source, destination) for destination in destinations)

    return graph


def crossing_completed_state(sample: SuspensionSample) -> State:
    """Advance a handle-interior sample to its next base visit.

    Base samples are unchanged.  A point strictly inside a handle is sent to
    that handle's reset endpoint, which occurs at a *later* suspension time.
    Hence this operation is not the exact fixed-time suspension map on the base
    and is not asserted to be semiconjugate to it.  Using it as a base-only
    invariant-set computation requires a separate theorem; the faithful sampled
    representation is :func:`locate_augmented_cells` applied to the original
    base-or-handle sample.
    """
    if isinstance(sample, BaseSuspensionSample):
        return _state_copy(sample.state)
    return _state_copy(sample.reset_state)


def simulate_crossing_completed_state(
    system: "HybridSystem",
    initial_state: npt.ArrayLike,
    total_time: float,
    *,
    max_jumps: int | None = None,
    max_step: float | None = None,
    atol: float = 1e-9,
) -> State:
    """Simulate a suspension endpoint and return its crossing-completed base state."""
    return crossing_completed_state(
        simulate_suspension_endpoint(
            system,
            initial_state,
            total_time,
            max_jumps=max_jumps,
            max_step=max_step,
            atol=atol,
        ),
    )


def _validate_phase_nodes(
    graph: nx.DiGraph,
    phase_nodes: Collection[Hashable],
) -> set[Hashable]:
    if not graph.is_directed():
        raise TypeError("suspension path collapse requires a directed graph")
    phase = set(phase_nodes)
    missing = phase.difference(graph.nodes)
    if missing:
        raise ValueError(f"phase_nodes contains nodes absent from graph: {missing!r}")
    return phase


def collapse_phase_paths(
    graph: nx.DiGraph,
    phase_nodes: Collection[Hashable],
) -> nx.DiGraph:
    """Collapse paths whose internal vertices are all phase-only vertices.

    The output contains precisely the non-phase (base) vertices.  It has an edge
    ``u -> v`` exactly when the input has a path from ``u`` to ``v`` whose
    internal vertices, if any, are all in ``phase_nodes``.  Reaching another base
    vertex stops the search, so the function does not take a transitive closure
    through base space.

    Node attributes and graph attributes are copied.  Edge attributes are not,
    because one macro edge can represent several different input paths.
    """
    phase = _validate_phase_nodes(graph, phase_nodes)
    collapsed = nx.DiGraph()
    collapsed.graph.update(graph.graph)

    base_nodes = [node for node in graph.nodes if node not in phase]
    collapsed.add_nodes_from((node, dict(graph.nodes[node])) for node in base_nodes)

    for source in base_nodes:
        pending: list[Hashable] = []
        visited_phase: set[Hashable] = set()

        for successor in graph.successors(source):
            if successor in phase:
                pending.append(successor)
            else:
                collapsed.add_edge(source, successor)

        while pending:
            current = pending.pop()
            if current in visited_phase:
                continue
            visited_phase.add(current)

            for successor in graph.successors(current):
                if successor in phase:
                    if successor not in visited_phase:
                        pending.append(successor)
                else:
                    collapsed.add_edge(source, successor)

    return collapsed


def _recurrent_components(graph: nx.DiGraph) -> tuple[frozenset[Hashable], ...]:
    recurrent = []
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            recurrent.append(frozenset(component))
            continue
        node = next(iter(component))
        if graph.has_edge(node, node):
            recurrent.append(frozenset(component))
    return tuple(recurrent)


MorseOrder = FrozenSet[Tuple[FrozenSet[Hashable], FrozenSet[Hashable]]]


def _strict_component_reachability(
    graph: nx.DiGraph,
    components: tuple[frozenset[Hashable], ...],
    labels: tuple[frozenset[Hashable], ...],
) -> MorseOrder:
    """Return the strict reachability order between labeled SCCs."""
    if len(components) != len(labels):
        raise ValueError("components and labels must have the same length")

    relation = set()
    for source_index, source in enumerate(components):
        source_node = next(iter(source))
        for target_index, target in enumerate(components):
            if source_index == target_index:
                continue
            target_node = next(iter(target))
            if nx.has_path(graph, source_node, target_node):
                relation.add((labels[source_index], labels[target_index]))
    return frozenset(relation)


@dataclass(frozen=True)
class RecurrentMorseCollapseDiagnostic:
    """Comparison of recurrent SCCs and their strict reachability order."""

    equivalent: bool
    component_sets_equivalent: bool
    reachability_order_equivalent: bool
    phase_only_recurrent_components: tuple[frozenset[Hashable], ...]
    full_recurrent_base_traces: tuple[frozenset[Hashable], ...]
    collapsed_recurrent_components: tuple[frozenset[Hashable], ...]
    full_recurrent_order: MorseOrder
    collapsed_recurrent_order: MorseOrder


class PhaseOnlyRecurrentComponentError(ValueError):
    """Raised when handle-phase vertices support recurrence without a base vertex."""


def diagnose_recurrent_morse_collapse(
    graph: nx.DiGraph,
    phase_nodes: Collection[Hashable],
) -> RecurrentMorseCollapseDiagnostic:
    """Compare recurrent SCCs with those of the phase-path-collapsed graph.

    A recurrent SCC containing base vertices is compared through its intersection
    with the base.  A recurrent SCC containing only phase vertices has no such
    image and makes ``equivalent`` false.
    """
    phase = _validate_phase_nodes(graph, phase_nodes)
    full_recurrent = _recurrent_components(graph)
    phase_only = tuple(component for component in full_recurrent if component <= phase)
    base_traces = tuple(
        frozenset(node for node in component if node not in phase)
        for component in full_recurrent
        if not component <= phase
    )
    represented_full = tuple(
        component for component in full_recurrent if not component <= phase
    )
    collapsed_graph = collapse_phase_paths(graph, phase)
    collapsed_recurrent = _recurrent_components(collapsed_graph)
    full_order = _strict_component_reachability(
        graph,
        represented_full,
        base_traces,
    )
    collapsed_order = _strict_component_reachability(
        collapsed_graph,
        collapsed_recurrent,
        collapsed_recurrent,
    )

    component_sets_equivalent = set(base_traces) == set(collapsed_recurrent)
    reachability_order_equivalent = full_order == collapsed_order
    equivalent = (
        not phase_only
        and component_sets_equivalent
        and reachability_order_equivalent
    )
    return RecurrentMorseCollapseDiagnostic(
        equivalent=equivalent,
        component_sets_equivalent=component_sets_equivalent,
        reachability_order_equivalent=reachability_order_equivalent,
        phase_only_recurrent_components=phase_only,
        full_recurrent_base_traces=base_traces,
        collapsed_recurrent_components=collapsed_recurrent,
        full_recurrent_order=full_order,
        collapsed_recurrent_order=collapsed_order,
    )


def assert_recurrent_morse_equivalence(
    graph: nx.DiGraph,
    phase_nodes: Collection[Hashable],
) -> RecurrentMorseCollapseDiagnostic:
    """Return a successful diagnostic or fail when Morse collapse is unsafe."""
    diagnostic = diagnose_recurrent_morse_collapse(graph, phase_nodes)
    if diagnostic.phase_only_recurrent_components:
        raise PhaseOnlyRecurrentComponentError(
            "phase-only recurrent SCCs cannot be represented on the base: "
            f"{diagnostic.phase_only_recurrent_components!r}",
        )
    if not diagnostic.equivalent:
        raise AssertionError(
            "recurrent SCCs or their reachability order do not agree after "
            "phase-path collapse: "
            f"full base traces={diagnostic.full_recurrent_base_traces!r}, "
            f"collapsed={diagnostic.collapsed_recurrent_components!r}, "
            f"full order={diagnostic.full_recurrent_order!r}, "
            f"collapsed order={diagnostic.collapsed_recurrent_order!r}",
        )
    return diagnostic


__all__ = [
    "BaseSuspensionSample",
    "HandleSuspensionSample",
    "SuspensionSample",
    "BaseCell",
    "PhaseCell",
    "AugmentedCell",
    "sample_suspension_trajectory",
    "suspension_integration_jump_limit",
    "simulate_suspension_endpoint",
    "locate_augmented_cells",
    "build_augmented_outer_graph",
    "crossing_completed_state",
    "simulate_crossing_completed_state",
    "collapse_phase_paths",
    "RecurrentMorseCollapseDiagnostic",
    "PhaseOnlyRecurrentComponentError",
    "diagnose_recurrent_morse_collapse",
    "assert_recurrent_morse_equivalence",
]
