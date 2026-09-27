"""Sampled multivalued map on the paper suspension grid and its Morse graph.

This module implements the computation of the manuscript's Examples section
on the grid ``Xi_n`` of :mod:`suspension_grid`:

* every element (atom) of ``Xi_n`` is sampled by a ``3 x 3`` tensor array on
  each of its elementary pieces (corners, edge midpoints, and center of each
  base cell, and likewise in the ``(u, s)`` coordinates of each handle piece);
* each sample is followed for ``tau`` units of suspension time with the
  unit-handle clock ``T = flow time + completed handles``;
* each endpoint is located in the closed pieces containing it, including both
  sides of an identification ``pi(g,0) = iota(g)`` or ``pi(g,1) = iota(r(g))``;
* endpoints outside the window ``D = Sigma(R)`` are discarded; and
* the target atoms are enlarged by one cell: every atom whose closure meets
  the closure of a target atom in ``Sigma X`` is added.

The last rule is the implemented ``epsilon_n`` of
``def:suspension-multivalued-map``: ``F_n(xi)`` consists of the atoms meeting
the union of the closed atoms that contain sampled endpoints of ``xi``.

The Morse graph is computed from the strongly connected components of this
explicit relation with :mod:`scipy.sparse.csgraph`; a component is a Morse set
if it carries an edge.  Morse sets are ordered by reachability.

The relation is sampled, not a certified outer approximation.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse import csgraph

from .hybrid_system import HybridSystem
from .suspension_grid import (
    DyadicBaseWindow,
    GuardResetSpec,
    SuspensionGrid,
)


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

ENDPOINT_FAILED = -1
ENDPOINT_BASE = 0
ENDPOINT_HANDLE = 1


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------


@dataclass
class SuspensionGridProblem:
    """A hybrid system with its window, guard parametrization, and ``tau``."""

    system: HybridSystem
    window: DyadicBaseWindow
    guard: GuardResetSpec
    tau: float
    max_step: float | None = None
    name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unit-handle suspension semiflow
# ---------------------------------------------------------------------------


@dataclass
class _FlowSegment:
    start: float
    end: float
    solution: Any
    exit_time: float
    initial_state: FloatArray


@dataclass
class _HandleSegment:
    start: float
    end: float
    guard_state: FloatArray
    reset_state: FloatArray
    exit_time: float


@dataclass
class SuspensionPath:
    """A trajectory of the suspension semiflow on ``[0, total_time]``.

    Flow segments carry the dense ODE solution; handle segments carry the
    guard and reset states.  ``exit_time`` is the first recorded suspension
    time at which the path lies outside the window ``D`` (``inf`` if never).
    """

    segments: list[_FlowSegment | _HandleSegment]
    total_time: float
    atol: float = 1.0e-9

    @property
    def exit_time(self) -> float:
        return min((segment.exit_time for segment in self.segments), default=np.inf)

    def evaluate(self, times: npt.ArrayLike) -> tuple[IntArray, FloatArray, FloatArray]:
        """Endpoints at suspension times ``times``.

        Returns ``(kind, state, phase)``.  For a handle endpoint ``state`` is
        the guard point ``g`` and ``phase`` lies strictly in ``(0, 1)``; at
        phase zero or one the base point ``g`` or ``r(g)`` is returned, as in
        :func:`sampled_suspension.sample_suspension_trajectory`.
        """

        values = np.atleast_1d(np.asarray(times, dtype=np.float64))
        kind = np.full(values.shape[0], ENDPOINT_FAILED, dtype=np.int64)
        state = np.full((values.shape[0], 2), np.nan)
        phase = np.full(values.shape[0], np.nan)
        pending = np.ones(values.shape[0], dtype=bool)
        for segment in self.segments:
            if not np.any(pending):
                break
            if isinstance(segment, _HandleSegment):
                at_start = pending & (np.abs(values - segment.start) <= self.atol)
                kind[at_start] = ENDPOINT_BASE
                state[at_start] = segment.guard_state
                pending &= ~at_start
                at_end = pending & (np.abs(values - segment.end) <= self.atol)
                kind[at_end] = ENDPOINT_BASE
                state[at_end] = segment.reset_state
                pending &= ~at_end
                inside = pending & (values > segment.start) & (values < segment.end)
                kind[inside] = ENDPOINT_HANDLE
                state[inside] = segment.guard_state
                phase[inside] = values[inside] - segment.start
                pending &= ~inside
            else:
                inside = (
                    pending
                    & (values >= segment.start - self.atol)
                    & (values <= segment.end + self.atol)
                )
                if np.any(inside):
                    local = np.clip(values[inside], segment.start, segment.end) - segment.start
                    local_end = segment.end - segment.start
                    local = np.clip(local, 0.0, local_end)
                    if local_end == 0.0:
                        points = np.repeat(
                            segment.initial_state.reshape(1, -1), local.size, axis=0
                        )
                    else:
                        points = np.asarray(segment.solution(local)).T.reshape(-1, 2)
                    kind[inside] = ENDPOINT_BASE
                    state[inside] = points
                    pending &= ~inside
        return kind, state, phase


class SuspensionFlow:
    """Event-driven unit-handle suspension semiflow of a :class:`HybridSystem`.

    Flow segments are integrated with :func:`scipy.integrate.solve_ivp` using
    the system's vector field, terminal guard event (including its crossing
    direction), tolerances, and reset map, exactly as
    :meth:`HybridSystem.simulate` does.  Each reset is represented by a handle
    of duration one, so the clock is ``T = flow time + completed handles``
    (the semantics of :mod:`sampled_suspension`).

    Whether a path starts on a handle is decided by the caller from the
    explicit guard ``G cap R`` of the grid (``start_on_handle``): a base point
    on the guard enters its handle immediately, while a point merely lying on
    the event hypersurface outside ``G`` flows.  This agrees with
    :meth:`HybridSystem.jumps_at_start` on the event surface.
    """

    def __init__(
        self,
        system: HybridSystem,
        *,
        max_step: float | None = None,
        in_window: Callable[[FloatArray], npt.NDArray[np.bool_]] | None = None,
        guard_in_window: Callable[[FloatArray], npt.NDArray[np.bool_]] | None = None,
        atol: float = 1.0e-9,
    ) -> None:
        self.system = system
        self.max_step = max_step
        self.in_window = in_window
        self.guard_in_window = guard_in_window
        self.atol = float(atol)

    def _flow(self, state: FloatArray, duration: float) -> tuple[Any, float | None, FloatArray | None]:
        kwargs: dict[str, Any] = {
            "fun": self.system.ode,
            "t_span": (0.0, float(duration)),
            "y0": np.asarray(state, dtype=np.float64),
            "events": self.system.event_function,
            "dense_output": True,
            "rtol": self.system.rtol,
            "atol": self.system.atol,
        }
        if self.max_step is not None:
            kwargs["max_step"] = self.max_step
        solution = solve_ivp(**kwargs)
        if solution.status == -1:
            raise RuntimeError(f"integration failed: {solution.message}")
        if solution.t_events and solution.t_events[0].size:
            event_time = float(solution.t_events[0][0])
            event_state = np.asarray(solution.y_events[0][0], dtype=np.float64)
            return solution, event_time, event_state
        return solution, None, None

    def _flow_exit(self, solution: Any, start: float, stop_local: float) -> float:
        if self.in_window is None:
            return np.inf
        times = solution.t[solution.t <= stop_local + self.atol]
        points = solution.y[:, : times.size].T
        if points.size == 0:
            return np.inf
        inside = self.in_window(points)
        outside = np.flatnonzero(~inside)
        return float(start + times[outside[0]]) if outside.size else np.inf

    def _guard_exit(self, guard_state: FloatArray, start: float) -> float:
        if self.guard_in_window is None:
            return np.inf
        inside = bool(self.guard_in_window(guard_state.reshape(1, -1))[0])
        return np.inf if inside else float(start)

    def path(
        self,
        initial_state: npt.ArrayLike,
        total_time: float,
        *,
        start_on_handle: bool = False,
        initial_phase: float = 0.0,
        max_segments: int = 100_000,
    ) -> SuspensionPath:
        """Integrate the suspension semiflow for ``total_time``.

        With ``start_on_handle=True``, ``initial_state`` is a guard point
        ``g`` and the path starts at ``pi(g, initial_phase)``.
        """

        total_time = float(total_time)
        segments: list[_FlowSegment | _HandleSegment] = []
        clock = 0.0
        state = np.asarray(initial_state, dtype=np.float64)
        pending_handle = bool(start_on_handle)
        phase0 = float(initial_phase) if start_on_handle else 0.0
        for _ in range(max_segments):
            if pending_handle:
                guard_state = state
                reset_state = np.asarray(
                    self.system.reset_map(guard_state), dtype=np.float64
                )
                start = clock - phase0
                end = start + 1.0
                segments.append(
                    _HandleSegment(
                        start=start,
                        end=end,
                        guard_state=guard_state,
                        reset_state=reset_state,
                        exit_time=self._guard_exit(guard_state, max(start, 0.0)),
                    )
                )
                clock = end
                phase0 = 0.0
                state = reset_state
                pending_handle = False
                if clock >= total_time - self.atol:
                    break
                continue
            remaining = total_time - clock
            if remaining <= self.atol:
                break
            solution, event_time, event_state = self._flow(state, remaining)
            if event_time is None:
                segments.append(
                    _FlowSegment(
                        start=clock,
                        end=total_time,
                        solution=solution.sol,
                        exit_time=self._flow_exit(solution, clock, remaining),
                        initial_state=state,
                    )
                )
                clock = total_time
                break
            segments.append(
                _FlowSegment(
                    start=clock,
                    end=clock + event_time,
                    solution=solution.sol,
                    exit_time=self._flow_exit(solution, clock, event_time),
                    initial_state=state,
                )
            )
            clock += event_time
            state = event_state
            pending_handle = True
        else:
            raise RuntimeError("suspension path exceeded max_segments")
        return SuspensionPath(segments=segments, total_time=total_time, atol=self.atol)


# ---------------------------------------------------------------------------
# Endpoint evaluation
# ---------------------------------------------------------------------------


@dataclass
class EndpointBatch:
    """Endpoints of a batch of sample points at suspension time ``tau``."""

    kind: IntArray
    state: FloatArray
    phase: FloatArray
    left_window: npt.NDArray[np.bool_]
    failures: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, size: int) -> "EndpointBatch":
        return cls(
            kind=np.full(size, ENDPOINT_FAILED, dtype=np.int64),
            state=np.full((size, 2), np.nan),
            phase=np.full(size, np.nan),
            left_window=np.zeros(size, dtype=bool),
        )

    @classmethod
    def concatenate(cls, batches: Sequence["EndpointBatch"]) -> "EndpointBatch":
        return cls(
            kind=np.concatenate([batch.kind for batch in batches]),
            state=np.concatenate([batch.state for batch in batches]),
            phase=np.concatenate([batch.phase for batch in batches]),
            left_window=np.concatenate([batch.left_window for batch in batches]),
            failures=[message for batch in batches for message in batch.failures],
        )


def _make_flow(problem: SuspensionGridProblem, level: int) -> SuspensionFlow:
    window = problem.window
    guard = problem.guard

    def in_window(points: FloatArray) -> npt.NDArray[np.bool_]:
        return window.contains(points, level)

    def guard_in_window(points: FloatArray) -> npt.NDArray[np.bool_]:
        u = guard.coordinate(points)
        span = guard.u_bounds[1] - guard.u_bounds[0]
        tolerance = 1.0e-9 * span
        return (u >= guard.u_bounds[0] - tolerance) & (u <= guard.u_bounds[1] + tolerance)

    return SuspensionFlow(
        problem.system,
        max_step=problem.max_step,
        in_window=in_window,
        guard_in_window=guard_in_window,
    )


def evaluate_base_endpoints(
    flow: SuspensionFlow,
    points: FloatArray,
    guard_u: FloatArray,
    tau: float,
) -> EndpointBatch:
    """Endpoints of base points; ``guard_u`` is finite exactly on ``G cap R``."""

    batch = EndpointBatch.empty(points.shape[0])
    for index in range(points.shape[0]):
        try:
            if np.isfinite(guard_u[index]):
                path = flow.path(points[index], tau, start_on_handle=True, initial_phase=0.0)
            else:
                path = flow.path(points[index], tau)
            kind, state, phase = path.evaluate(tau)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            batch.failures.append(f"base sample {points[index].tolist()}: {error}")
            continue
        batch.kind[index] = kind[0]
        batch.state[index] = state[0]
        batch.phase[index] = phase[0]
        batch.left_window[index] = path.exit_time < tau - flow.atol
    return batch


def evaluate_handle_endpoints(
    flow: SuspensionFlow,
    guard: GuardResetSpec,
    u_values: FloatArray,
    phases: FloatArray,
    tau: float,
    *,
    path_cache: dict[float, SuspensionPath] | None = None,
) -> EndpointBatch:
    """Endpoints of ``pi(gamma(u), s)`` for the tensor product ``u x s``.

    For each ``u`` one path is integrated from ``pi(gamma(u), 0)`` for
    ``tau + 1`` units; the endpoint of ``pi(gamma(u), s)`` at time ``tau``
    is that path at time ``tau + s``.  Result order is ``u``-major.  Paths
    are stored in ``path_cache`` (keyed by ``u``) when it is given.
    """

    batches: list[EndpointBatch] = []
    guard_points = guard.gamma(u_values)
    reset_points = guard.reset(u_values)
    for index in range(u_values.shape[0]):
        batch = EndpointBatch.empty(phases.shape[0])
        try:
            system_reset = np.asarray(
                flow.system.reset_map(guard_points[index]), dtype=np.float64
            )
            if not np.allclose(system_reset, reset_points[index], rtol=1.0e-10, atol=1.0e-12):
                raise ValueError(
                    "the guard specification's reset_point disagrees with the "
                    f"system reset at u={u_values[index]!r}"
                )
            key = float(u_values[index])
            path = None if path_cache is None else path_cache.get(key)
            if path is None:
                path = flow.path(
                    guard_points[index], tau + 1.0, start_on_handle=True, initial_phase=0.0
                )
                if path_cache is not None:
                    path_cache[key] = path
            times = tau + phases
            kind, state, phase = path.evaluate(times)
            exit_time = path.exit_time
            batch.kind[:] = kind
            batch.state[:] = state
            batch.phase[:] = phase
            # The source point pi(g, s) sits at path time s.
            batch.left_window[:] = (exit_time > phases + flow.atol) & (
                exit_time < times - flow.atol
            )
            if exit_time <= flow.atol:
                # The source itself lies outside the window: not a sample of D.
                raise ValueError("handle sample lies outside the window")
        except (RuntimeError, ValueError, FloatingPointError) as error:
            batch = EndpointBatch.empty(phases.shape[0])
            batch.failures.append(f"handle sample u={u_values[index]!r}: {error}")
        batches.append(batch)
    return EndpointBatch.concatenate(batches)


# Worker state for process-parallel base evaluation.
_WORKER: dict[str, Any] = {}


def _initialize_worker(factory: Callable[[], SuspensionGridProblem], level: int) -> None:
    problem = factory()
    _WORKER["flow"] = _make_flow(problem, level)
    _WORKER["tau"] = problem.tau


def _worker_base_chunk(arguments: tuple[FloatArray, FloatArray]) -> EndpointBatch:
    points, guard_u = arguments
    return evaluate_base_endpoints(_WORKER["flow"], points, guard_u, _WORKER["tau"])


# ---------------------------------------------------------------------------
# The relation
# ---------------------------------------------------------------------------


@dataclass
class SuspensionGridRelation:
    """The sampled multivalued map ``F_n`` on the atoms of ``Xi_n``.

    ``matrix`` is the Boolean adjacency (``matrix[a, b] = 1`` iff
    ``b in F_n(a)``) after the one-cell padding; ``sampled`` is the relation
    before padding (atoms containing sampled endpoints).
    """

    grid: SuspensionGrid
    tau: float
    matrix: sparse.csr_matrix
    sampled: sparse.csr_matrix
    statistics: dict[str, Any]

    @property
    def n_atoms(self) -> int:
        return self.grid.n_atoms

    @property
    def n_edges(self) -> int:
        return int(self.matrix.nnz)

    def image(self, atom: int) -> IntArray:
        start, stop = self.matrix.indptr[atom], self.matrix.indptr[atom + 1]
        return self.matrix.indices[start:stop].astype(np.int64)

    def image_of(self, atoms: npt.ArrayLike) -> IntArray:
        selected = np.zeros(self.n_atoms, dtype=bool)
        selected[np.asarray(atoms, dtype=np.int64)] = True
        reached = np.asarray(self.matrix[selected].sum(axis=0)).ravel() > 0
        return np.flatnonzero(reached)

    def forward_closure(self, atoms: npt.ArrayLike) -> IntArray:
        """Smallest forward-invariant set ``U`` of atoms containing ``atoms``."""

        return _forward_closure(self.matrix, atoms)


def _forward_closure(matrix: sparse.csr_matrix, seeds: npt.ArrayLike) -> IntArray:
    """Vertices reachable from ``seeds`` (including them) by breadth-first search."""

    start = np.unique(np.asarray(seeds, dtype=np.int64))
    n = matrix.shape[0]
    if start.size == 0:
        return start
    # A virtual source joined to every seed.
    extra = sparse.csr_matrix(
        (np.ones(start.size), (np.zeros(start.size, dtype=np.int64), start)),
        shape=(1, n + 1),
    )
    augmented = sparse.vstack(
        (
            sparse.hstack((matrix, sparse.csr_matrix((n, 1)))).tocsr(),
            extra,
        )
    ).tocsr()
    order = csgraph.breadth_first_order(
        augmented, n, directed=True, return_predecessors=False
    )
    return np.sort(order[order < n]).astype(np.int64)


def _unique_rows(values: IntArray) -> tuple[IntArray, IntArray]:
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)


def _locate_endpoints(
    grid: SuspensionGrid, batch: EndpointBatch
) -> tuple[IntArray, IntArray]:
    """``(endpoint index, piece)`` incidences of closed pieces containing endpoints."""

    base_kind = np.flatnonzero(batch.kind == ENDPOINT_BASE)
    handle_kind = np.flatnonzero(batch.kind == ENDPOINT_HANDLE)
    located_base = grid.locate_base_points(batch.state[base_kind])
    u_endpoint = (
        grid.guard.coordinate(batch.state[handle_kind]) if handle_kind.size else np.zeros(0)
    )
    located_handle = grid.locate_handle_points(u_endpoint, batch.phase[handle_kind])
    rows = np.concatenate(
        (base_kind[located_base.point_index], handle_kind[located_handle.point_index])
    )
    pieces = np.concatenate((located_base.piece, located_handle.piece))
    return rows.astype(np.int64), pieces.astype(np.int64)


class _BaseEvaluator:
    """Serial or process-parallel evaluation of base sample points."""

    def __init__(
        self,
        grid: SuspensionGrid,
        flow: SuspensionFlow,
        tau: float,
        *,
        workers: int,
        problem_factory: Callable[[], SuspensionGridProblem] | None,
        chunk_size: int,
    ) -> None:
        self.grid = grid
        self.flow = flow
        self.tau = tau
        self.chunk_size = int(chunk_size)
        self.executor = None
        if workers > 1:
            if problem_factory is None:
                raise ValueError("parallel evaluation requires a picklable problem_factory")
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(problem_factory, grid.level),
            )

    def __call__(self, points: FloatArray) -> EndpointBatch:
        guard_u = self.grid.guard_membership(points) if points.size else np.zeros(0)
        if self.executor is None or points.shape[0] <= self.chunk_size:
            return evaluate_base_endpoints(self.flow, points, guard_u, self.tau)
        chunks = [
            (points[start : start + self.chunk_size], guard_u[start : start + self.chunk_size])
            for start in range(0, points.shape[0], self.chunk_size)
        ]
        return EndpointBatch.concatenate(list(self.executor.map(_worker_base_chunk, chunks)))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None


def _lattice_edges(sample_of_piece: IntArray, q: int) -> tuple[IntArray, IntArray]:
    """Edges of each piece's ``(q+1) x (q+1)`` sample lattice with owners."""

    size = q + 1
    index = np.arange(size * size).reshape(size, size)
    local = np.concatenate(
        (
            np.stack((index[:-1, :].ravel(), index[1:, :].ravel()), axis=1),
            np.stack((index[:, :-1].ravel(), index[:, 1:].ravel()), axis=1),
        )
    )
    pairs = sample_of_piece[:, local]  # (pieces, edges, 2)
    owners = np.repeat(np.arange(sample_of_piece.shape[0]), local.shape[0])
    pairs = np.sort(pairs.reshape(-1, 2), axis=1)
    return pairs, owners


def compute_suspension_grid_relation(
    grid: SuspensionGrid,
    problem: SuspensionGridProblem,
    *,
    samples_per_axis: int = 3,
    padding: bool = True,
    exit_policy: str = "endpoint",
    gap_refinement_depth: int = 0,
    workers: int = 1,
    problem_factory: Callable[[], SuspensionGridProblem] | None = None,
    chunk_size: int = 256,
    progress: Callable[[str], None] | None = None,
) -> SuspensionGridRelation:
    """Sample ``F_n`` on ``Xi_n`` as described in the Examples section.

    The default (``samples_per_axis=3``, ``padding=True``,
    ``gap_refinement_depth=0``) is the rule of the manuscript: a ``3 x 3``
    tensor array on every elementary piece, the closed atoms containing the
    endpoints, and one-atom padding.

    ``exit_policy='endpoint'`` discards only endpoints outside ``D`` (targets
    outside the window); ``'path'`` additionally discards endpoints whose
    trajectory left ``D`` before time ``tau`` and returned.  Both counts are
    reported in ``statistics``.

    ``gap_refinement_depth > 0`` enables an additional rule that is *not*
    part of the manuscript's description: an edge of a piece's sample lattice
    whose two endpoints lie in ``D`` but in atoms whose closures do not meet
    is bisected, up to the given depth, and the endpoints of the inserted
    samples are added to the image of every piece containing that edge.  The
    image of a connected piece under ``f_tau`` is connected, so this closes
    sampling gaps (it is the sampled analogue of following the image of each
    lattice edge).  Unresolved gaps at the final depth are counted.

    With ``workers > 1`` the base samples are evaluated in worker processes,
    each rebuilding the problem from the picklable ``problem_factory``.
    """

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least two (a tensor rule with corners)")
    if exit_policy not in {"endpoint", "path"}:
        raise ValueError("exit_policy must be 'endpoint' or 'path'")
    if gap_refinement_depth < 0:
        raise ValueError("gap_refinement_depth must be nonnegative")
    started = time.perf_counter()
    q = int(samples_per_axis) - 1
    tau = float(problem.tau)
    flow = _make_flow(problem, grid.level)

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    evaluate_base = _BaseEvaluator(
        grid,
        flow,
        tau,
        workers=workers,
        problem_factory=problem_factory,
        chunk_size=chunk_size,
    )
    try:
        # Base samples on the lattice of step width / q.
        offsets = np.arange(q + 1)
        ax, ay = np.meshgrid(offsets, offsets, indexing="ij")
        tensor = np.stack((ax.ravel(), ay.ravel()), axis=1)
        base_lattice = (q * grid.base_addresses[:, None, :] + tensor[None, :, :]).reshape(-1, 2)
        base_keys, base_inverse = _unique_rows(base_lattice)
        lower = np.array([grid.window.ambient_bounds[0][0], grid.window.ambient_bounds[1][0]])
        base_points = lower + base_keys * (grid.cell_widths / q)
        base_sample_of_piece = base_inverse.reshape(grid.n_base, (q + 1) ** 2)
        report(f"base samples: {base_points.shape[0]} unique points")

        t0 = time.perf_counter()
        base_batch = evaluate_base(base_points)
        base_seconds = time.perf_counter() - t0
        report(f"base endpoints evaluated in {base_seconds:.1f} s")

        # Handle samples on the (u, s) lattice.
        u_lattice = np.empty(q * grid.n_guard + 1)
        for step in range(q):
            u_lattice[step:-1:q] = grid.u_edges[:-1] + step / q * np.diff(grid.u_edges)
        u_lattice[-1] = grid.u_edges[-1]
        s_lattice = np.arange(q * grid.n_phase + 1) / (q * grid.n_phase)
        t0 = time.perf_counter()
        path_cache: dict[float, SuspensionPath] = {}
        handle_batch = evaluate_handle_endpoints(
            flow,
            grid.guard,
            u_lattice,
            s_lattice,
            tau,
            path_cache=path_cache if gap_refinement_depth > 0 else None,
        )
        handle_seconds = time.perf_counter() - t0
        report(f"handle endpoints evaluated in {handle_seconds:.1f} s")
        guard_index, phase_index = grid.handle_indices(np.arange(grid.n_base, grid.n_pieces))
        handle_sample_of_piece = (
            (q * guard_index[:, None, None] + offsets[None, :, None]) * s_lattice.size
            + (q * phase_index[:, None, None] + offsets[None, None, :])
        ).reshape(grid.n_handle, (q + 1) ** 2)

        # All endpoints in one index space: base samples first.
        endpoints = EndpointBatch.concatenate((base_batch, handle_batch))
        n_base_samples = base_points.shape[0]
        sample_of_piece = np.concatenate(
            (base_sample_of_piece, handle_sample_of_piece + n_base_samples)
        )
        handle_u, handle_s = np.meshgrid(u_lattice, s_lattice, indexing="ij")
        source_coordinates = np.concatenate(
            (base_points, np.stack((handle_u.ravel(), handle_s.ravel()), axis=1))
        )
        source_is_handle = np.r_[
            np.zeros(n_base_samples, dtype=bool), np.ones(handle_u.size, dtype=bool)
        ]

        t0 = time.perf_counter()
        rows, pieces = _locate_endpoints(grid, endpoints)
        locate_seconds = time.perf_counter() - t0

        extra_sources: list[IntArray] = []  # (piece, endpoint) pairs from refinement
        refinement_stats: dict[str, Any] = {"gap_refinement_depth": int(gap_refinement_depth)}
        if gap_refinement_depth > 0:
            t0 = time.perf_counter()
            closure = grid.atom_adjacency + sparse.identity(grid.n_atoms, format="csr")
            edge_pairs, edge_owner_pieces = _lattice_edges(sample_of_piece, q)
            unique_edges, edge_inverse = _unique_rows(edge_pairs)
            # Owners of each unique edge (one or two pieces).
            owner_matrix = sparse.csr_matrix(
                (np.ones(edge_inverse.size), (edge_inverse, edge_owner_pieces)),
                shape=(unique_edges.shape[0], grid.n_pieces),
            )
            lattice_keys = set(float(value) for value in u_lattice)
            left = unique_edges[:, 0].copy()
            right = unique_edges[:, 1].copy()
            segment_edge = np.arange(unique_edges.shape[0])
            inserted = 0
            unresolved = 0
            for depth in range(gap_refinement_depth + 1):
                n_endpoints = endpoints.kind.shape[0]
                located = np.zeros(n_endpoints, dtype=bool)
                located[rows] = True
                usable = located & ((~endpoints.left_window) if exit_policy == "path" else True)
                atoms_of = sparse.csr_matrix(
                    (np.ones(rows.size), (rows, grid.atom_of_piece[pieces])),
                    shape=(n_endpoints, grid.n_atoms),
                )
                both = usable[left] & usable[right]
                touching = np.zeros(left.size, dtype=bool)
                if np.any(both):
                    candidates = np.flatnonzero(both)
                    near = (atoms_of[left[candidates]] @ closure).multiply(
                        atoms_of[right[candidates]]
                    )
                    touching[candidates] = np.asarray(near.sum(axis=1)).ravel() > 0
                gap = both & ~touching
                if depth == gap_refinement_depth or not np.any(gap):
                    unresolved = int(np.count_nonzero(gap))
                    break
                gap_left = left[gap]
                gap_right = right[gap]
                gap_edge = segment_edge[gap]
                middle = 0.5 * (source_coordinates[gap_left] + source_coordinates[gap_right])
                middle_handle = source_is_handle[gap_left]
                new_batch = EndpointBatch.empty(middle.shape[0])
                base_rows = np.flatnonzero(~middle_handle)
                if base_rows.size:
                    evaluated = evaluate_base(middle[base_rows])
                    new_batch.kind[base_rows] = evaluated.kind
                    new_batch.state[base_rows] = evaluated.state
                    new_batch.phase[base_rows] = evaluated.phase
                    new_batch.left_window[base_rows] = evaluated.left_window
                    new_batch.failures.extend(evaluated.failures)
                handle_rows = np.flatnonzero(middle_handle)
                if handle_rows.size:
                    u_values = middle[handle_rows, 0]
                    for u_value in np.unique(u_values):
                        members = handle_rows[u_values == u_value]
                        evaluated = evaluate_handle_endpoints(
                            flow,
                            grid.guard,
                            np.array([u_value]),
                            middle[members, 1],
                            tau,
                            path_cache=path_cache,
                        )
                        new_batch.kind[members] = evaluated.kind
                        new_batch.state[members] = evaluated.state
                        new_batch.phase[members] = evaluated.phase
                        new_batch.left_window[members] = evaluated.left_window
                        new_batch.failures.extend(evaluated.failures)
                    # Paths at new guard coordinates serve one round only.
                    for key in list(path_cache):
                        if key not in lattice_keys:
                            del path_cache[key]
                offset = endpoints.kind.shape[0]
                new_ids = offset + np.arange(middle.shape[0])
                endpoints = EndpointBatch.concatenate((endpoints, new_batch))
                source_coordinates = np.concatenate((source_coordinates, middle))
                source_is_handle = np.concatenate((source_is_handle, middle_handle))
                new_rows, new_pieces = _locate_endpoints(grid, new_batch)
                rows = np.concatenate((rows, offset + new_rows))
                pieces = np.concatenate((pieces, new_pieces))
                owners = owner_matrix[gap_edge].tocoo()
                extra_sources.append(
                    np.stack((owners.col, new_ids[owners.row]), axis=1).astype(np.int64)
                )
                inserted += middle.shape[0]
                left = np.concatenate((gap_left, new_ids))
                right = np.concatenate((new_ids, gap_right))
                segment_edge = np.concatenate((gap_edge, gap_edge))
                report(f"gap refinement depth {depth + 1}: {middle.shape[0]} samples")
            refinement_stats.update(
                {
                    "gap_refinement_samples": int(inserted),
                    "unresolved_gap_segments": int(unresolved),
                    "seconds_gap_refinement": time.perf_counter() - t0,
                }
            )
    finally:
        evaluate_base.close()

    n_endpoints = endpoints.kind.shape[0]
    located_any = np.zeros(n_endpoints, dtype=bool)
    located_any[rows] = True
    failed = endpoints.kind == ENDPOINT_FAILED
    exits = ~failed & ~located_any
    returned = located_any & endpoints.left_window
    usable = located_any.copy()
    if exit_policy == "path":
        usable &= ~endpoints.left_window
    keep = usable[rows]
    endpoint_atoms = sparse.csr_matrix(
        (
            np.ones(int(np.count_nonzero(keep))),
            (rows[keep], grid.atom_of_piece[pieces[keep]]),
        ),
        shape=(n_endpoints, grid.n_atoms),
    )

    # Source atom -> sample endpoints.
    source_pieces = np.repeat(np.arange(grid.n_pieces), sample_of_piece.shape[1])
    source_samples = sample_of_piece.reshape(-1)
    if extra_sources:
        extra = np.concatenate(extra_sources)
        source_pieces = np.concatenate((source_pieces, extra[:, 0]))
        source_samples = np.concatenate((source_samples, extra[:, 1]))
    sources = sparse.csr_matrix(
        (np.ones(source_samples.size), (grid.atom_of_piece[source_pieces], source_samples)),
        shape=(grid.n_atoms, n_endpoints),
    )
    sources.data[:] = 1.0
    sampled = (sources @ endpoint_atoms).tocsr()
    sampled.data[:] = 1.0
    sampled.eliminate_zeros()
    if padding:
        closure = grid.atom_adjacency + sparse.identity(grid.n_atoms, format="csr")
        matrix = (sampled @ closure).tocsr()
        matrix.data[:] = 1.0
    else:
        matrix = sampled.copy()
    matrix.sort_indices()

    exit_samples = sources @ exits.astype(np.float64)
    returned_samples = sources @ returned.astype(np.float64)
    failed_samples = sources @ failed.astype(np.float64)
    image_sizes = np.diff(matrix.indptr)
    statistics = {
        "samples_per_axis": int(samples_per_axis),
        "exit_policy": exit_policy,
        "padding": "one atom (closures meet in Sigma X)" if padding else "none",
        "image_rule": (
            "paper: 3x3 samples per piece, containing atoms, one-atom padding"
            if gap_refinement_depth == 0 and samples_per_axis == 3 and padding
            else "modified (see gap_refinement_depth, samples_per_axis, padding)"
        ),
        "unique_base_samples": int(n_base_samples),
        "unique_handle_samples": int(handle_batch.kind.size),
        "handle_paths": int(u_lattice.size),
        "evaluated_endpoints": int(n_endpoints),
        "failed_endpoints": int(np.count_nonzero(failed)),
        "failure_messages": endpoints.failures[:20],
        "discarded_exit_endpoints": int(np.count_nonzero(exits)),
        "left_and_returned_endpoints": int(np.count_nonzero(returned)),
        "source_atoms_with_exit": int(np.count_nonzero(exit_samples > 0)),
        "source_atoms_with_left_and_returned": int(np.count_nonzero(returned_samples > 0)),
        "source_atoms_with_failed_samples": int(np.count_nonzero(failed_samples > 0)),
        "atoms_with_empty_image": int(np.count_nonzero(image_sizes == 0)),
        "sampled_edges": int(sampled.nnz),
        "edges": int(matrix.nnz),
        "seconds_base_endpoints": base_seconds,
        "seconds_handle_endpoints": handle_seconds,
        "seconds_locate": locate_seconds,
        **refinement_stats,
        "seconds_total": time.perf_counter() - started,
    }
    return SuspensionGridRelation(
        grid=grid,
        tau=tau,
        matrix=matrix,
        sampled=sampled,
        statistics=statistics,
    )


# ---------------------------------------------------------------------------
# Morse graph
# ---------------------------------------------------------------------------


@dataclass
class SuspensionMorseGraph:
    """Morse sets (recurrent SCCs) of ``F_n`` ordered by reachability.

    ``edges`` is the Hasse diagram: ``(p, q)`` means ``M(q)`` is reachable
    from ``M(p)`` with no Morse set strictly between.  Node ``0`` is minimal
    (an attractor), following the CMGDB convention of the paper figures.
    """

    morse_sets: tuple[IntArray, ...]
    edges: tuple[tuple[int, int], ...]
    reachable: tuple[frozenset[int], ...]
    component_of_atom: IntArray
    n_components: int

    def base_readouts(self, grid: SuspensionGrid) -> tuple[IntArray, ...]:
        """``d_n^{-1}(M)`` for every Morse set ``M``."""

        return tuple(grid.base_readout(morse_set) for morse_set in self.morse_sets)


def compute_suspension_morse_graph(relation: SuspensionGridRelation) -> SuspensionMorseGraph:
    """Strongly connected components, Morse sets, and their order."""

    matrix = relation.matrix
    n_components, labels = csgraph.connected_components(
        matrix, directed=True, connection="strong"
    )
    sizes = np.bincount(labels, minlength=n_components)
    self_loop = np.zeros(n_components, dtype=bool)
    diagonal = matrix.diagonal() > 0
    self_loop[labels[diagonal]] = True
    recurrent = np.flatnonzero((sizes > 1) | self_loop)
    raw_sets = [np.flatnonzero(labels == component) for component in recurrent]

    # Reachability among Morse sets through the condensation of the relation.
    coo = matrix.tocoo()
    condensation = sparse.csr_matrix(
        (np.ones(coo.nnz), (labels[coo.row], labels[coo.col])),
        shape=(n_components, n_components),
    )
    recurrent_set = set(recurrent.tolist())
    reach: list[set[int]] = []
    for component in recurrent:
        order = csgraph.breadth_first_order(
            condensation, int(component), directed=True, return_predecessors=False
        )
        reach.append(set(order.tolist()).intersection(recurrent_set))
    component_to_raw = {int(component): index for index, component in enumerate(recurrent)}
    successors = [
        {component_to_raw[component] for component in hit} - {index}
        for index, hit in enumerate(reach)
    ]

    # Number the Morse sets so that sinks come first (height = longest path
    # to a minimal element), ties broken by the smallest atom index.
    heights: dict[int, int] = {}

    def height(index: int) -> int:
        if index not in heights:
            heights[index] = 0 if not successors[index] else 1 + max(
                height(other) for other in successors[index]
            )
        return heights[index]

    order = sorted(range(len(raw_sets)), key=lambda index: (height(index), int(raw_sets[index][0])))
    new_index = {old: new for new, old in enumerate(order)}
    morse_sets = tuple(raw_sets[old] for old in order)
    reachable = tuple(
        frozenset(new_index[other] for other in successors[old]) for old in order
    )
    hasse: list[tuple[int, int]] = []
    for source, targets in enumerate(reachable):
        for target in targets:
            if not any(target in reachable[middle] for middle in targets if middle != target):
                hasse.append((source, target))
    return SuspensionMorseGraph(
        morse_sets=morse_sets,
        edges=tuple(sorted(hasse)),
        reachable=reachable,
        component_of_atom=labels.astype(np.int64),
        n_components=int(n_components),
    )


# ---------------------------------------------------------------------------
# Connectivity diagnostics
# ---------------------------------------------------------------------------


def atom_set_components(grid: SuspensionGrid, atoms: npt.ArrayLike) -> int:
    """Connected components of ``|U|`` in ``Sigma X`` (closures meeting)."""

    values = np.unique(np.asarray(atoms, dtype=np.int64))
    if values.size == 0:
        return 0
    sub = grid.atom_adjacency[values][:, values]
    return int(csgraph.connected_components(sub, directed=False)[0])


def relation_image_connectivity(relation: SuspensionGridRelation) -> dict[str, int]:
    """Count atoms whose recorded image is disconnected in ``Sigma X``.

    The image of a connected atom under the continuous map ``f_tau`` is
    connected, so a disconnected recorded image (restricted to the window)
    indicates either an exit through the window boundary or a sampling gap.
    """

    grid = relation.grid
    disconnected_padded = 0
    disconnected_sampled = 0
    for atom in range(grid.n_atoms):
        image = relation.image(atom)
        if image.size and atom_set_components(grid, image) > 1:
            disconnected_padded += 1
        start, stop = relation.sampled.indptr[atom], relation.sampled.indptr[atom + 1]
        sampled = relation.sampled.indices[start:stop]
        if sampled.size and atom_set_components(grid, sampled) > 1:
            disconnected_sampled += 1
    return {
        "atoms": grid.n_atoms,
        "disconnected_padded_images": disconnected_padded,
        "disconnected_sampled_images": disconnected_sampled,
    }


# ---------------------------------------------------------------------------
# Independent endpoint probes
# ---------------------------------------------------------------------------


def audit_suspension_grid_endpoints(
    relation: SuspensionGridRelation,
    problem: SuspensionGridProblem,
    *,
    base_cells: npt.ArrayLike | None = None,
    subdivision: int = 4,
    guard_intervals: npt.ArrayLike | None = None,
    phase_samples: int = 7,
    seed: int = 0,
):
    """Probe endpoints of dense source points against the recorded relation.

    Base cells are probed on a ``(subdivision + 1)^2`` tensor array (a
    different lattice from the ``3 x 3`` sampling rule) and handle pieces on
    random interior ``(u, s)`` points.  Each probe is located with
    :meth:`SuspensionGrid.locate_base_points` or
    :meth:`SuspensionGrid.locate_handle_points` and recorded as an
    :class:`EndpointCoverageWitness` of :mod:`fixed_time_relation_audit`.  A
    probe is covered when one of the atoms containing its endpoint lies in the
    image of the source atom.  Probes whose endpoint leaves ``D`` are skipped
    (targets outside the window are discarded by construction).
    """

    from .fixed_time_relation_audit import (
        EndpointCoverageAudit,
        EndpointCoverageWitness,
        EndpointEvaluationFailure,
        EndpointProbe,
    )
    from .sampled_suspension import BaseSuspensionSample, HandleSuspensionSample

    grid = relation.grid
    flow = _make_flow(problem, grid.level)
    tau = relation.tau
    rng = np.random.default_rng(seed)
    witnesses: list[EndpointCoverageWitness] = []
    failures: list[EndpointEvaluationFailure] = []

    def image_atoms(atom: int) -> frozenset[int]:
        return frozenset(int(value) for value in relation.image(atom))

    cells = (
        np.arange(grid.n_base)
        if base_cells is None
        else np.asarray(base_cells, dtype=np.int64)
    )
    fractions = np.linspace(0.0, 1.0, subdivision + 1)
    fx, fy = np.meshgrid(fractions, fractions, indexing="ij")
    tensor = np.stack((fx.ravel(), fy.ravel()), axis=1)
    bounds = grid.base_bounds(cells)
    for cell, rectangle in zip(cells, bounds):
        points = rectangle[:2] + tensor * (rectangle[2:] - rectangle[:2])
        guard_u = grid.guard_membership(points)
        batch = evaluate_base_endpoints(flow, points, guard_u, tau)
        source_atom = int(grid.d_map[cell])
        for index in range(points.shape[0]):
            if batch.kind[index] == ENDPOINT_FAILED:
                failures.append(
                    EndpointEvaluationFailure(
                        source=source_atom,  # type: ignore[arg-type]
                        initial_state=tuple(points[index].tolist()),
                        message="evaluation failed",
                    )
                )
                continue
            if batch.kind[index] == ENDPOINT_BASE:
                located = grid.locate_base_points(batch.state[index : index + 1])
                sample = BaseSuspensionSample(
                    state=batch.state[index].copy(),
                    total_time=tau,
                    continuous_time=np.nan,
                    jumps_completed=-1,
                )
            else:
                u = grid.guard.coordinate(batch.state[index : index + 1])
                located = grid.locate_handle_points(u, batch.phase[index : index + 1])
                sample = HandleSuspensionSample(
                    guard_state=batch.state[index].copy(),
                    reset_state=np.full(2, np.nan),
                    phase=float(batch.phase[index]),
                    total_time=tau,
                    continuous_time=np.nan,
                    jump_index=-1,
                )
            if located.piece.size == 0:
                continue
            witnesses.append(
                EndpointCoverageWitness(
                    probe=EndpointProbe(
                        source=source_atom,  # type: ignore[arg-type]
                        endpoint=sample,
                        initial_state=tuple(points[index].tolist()),
                        label=f"base-cell-{int(cell)}",
                    ),
                    containing_target_cells=frozenset(
                        int(value) for value in grid.atom_of_piece[located.piece]
                    ),
                    recorded_targets=image_atoms(source_atom),
                )
            )

    intervals = (
        np.arange(grid.n_guard)
        if guard_intervals is None
        else np.asarray(guard_intervals, dtype=np.int64)
    )
    for interval in intervals:
        u = rng.uniform(grid.u_edges[interval], grid.u_edges[interval + 1], size=1)
        phases = np.sort(rng.uniform(0.0, 1.0, size=phase_samples))
        batch = evaluate_handle_endpoints(flow, grid.guard, u, phases, tau)
        for index in range(phases.size):
            phase_cell = min(int(phases[index] * grid.n_phase), grid.n_phase - 1)
            source_atom = int(grid.atom_of_piece[grid.handle_piece(interval, phase_cell)])
            if batch.kind[index] == ENDPOINT_FAILED:
                continue
            if batch.kind[index] == ENDPOINT_BASE:
                located = grid.locate_base_points(batch.state[index : index + 1])
            else:
                located = grid.locate_handle_points(
                    grid.guard.coordinate(batch.state[index : index + 1]),
                    batch.phase[index : index + 1],
                )
            if located.piece.size == 0:
                continue
            sample = BaseSuspensionSample(
                state=batch.state[index].copy(),
                total_time=tau,
                continuous_time=np.nan,
                jumps_completed=-1,
            )
            witnesses.append(
                EndpointCoverageWitness(
                    probe=EndpointProbe(
                        source=source_atom,  # type: ignore[arg-type]
                        endpoint=sample,
                        initial_state=(float(u[0]), float(phases[index])),
                        label=f"handle-J{int(interval)}",
                    ),
                    containing_target_cells=frozenset(
                        int(value) for value in grid.atom_of_piece[located.piece]
                    ),
                    recorded_targets=image_atoms(source_atom),
                )
            )
    return EndpointCoverageAudit(tuple(witnesses), tuple(failures))


__all__ = [
    "ENDPOINT_BASE",
    "ENDPOINT_FAILED",
    "ENDPOINT_HANDLE",
    "EndpointBatch",
    "SuspensionFlow",
    "SuspensionGridProblem",
    "SuspensionGridRelation",
    "SuspensionMorseGraph",
    "SuspensionPath",
    "atom_set_components",
    "audit_suspension_grid_endpoints",
    "compute_suspension_grid_relation",
    "compute_suspension_morse_graph",
    "evaluate_base_endpoints",
    "evaluate_handle_endpoints",
    "relation_image_connectivity",
]
