"""Optional parallel endpoint-cache construction for the Garcia Atlas map.

The scientific box-map callback remains the sole owner of sampling, event
stratification, hulling, padding, and graph construction.  This module only
evaluates the callback's deterministic tensor nodes ahead of time.  Results
are returned in the exact cache format consumed by
``_AuditedGarciaSuspensionBoxMap``.

Each worker constructs its own walker and suspension box map.  In particular,
no SciPy integrator, hybrid trajectory, or chart closure is shared across
processes.  ``ProcessPoolExecutor.map`` preserves the first-seen source order,
and a bounded submission buffer avoids creating one future per lattice node.
If a process pool cannot be started or breaks, the whole precomputation is
repeated serially from a fresh deterministic task stream.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import time
import warnings
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from ..src.cmgdb_suspension_boxmap import CMGDBSuspensionBoxMap
from .garcia_passive_walker_atlas import (
    DEFAULT_BASE_BOUNDS,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
)


EndpointKey: TypeAlias = tuple[object, ...]
EndpointCacheValue: TypeAlias = tuple[bool, object]
EndpointCacheEntry: TypeAlias = tuple[EndpointKey, EndpointCacheValue]
SourceBox: TypeAlias = tuple[int, tuple[float, ...]]

PARITY_PROVENANCE = {
    "scope": "fresh evaluation of one fixed Atlas family",
    "source_box_order": "native Atlas cell-index order",
    "tensor_node_order": "numpy.linspace axes with lexicographic tensor product",
    "shared_node_rule": "first actual point for each cache key",
    "cache_key": "chart id plus four coordinates rounded to 14 decimals",
    "endpoint_evaluator": "CMGDBSuspensionBoxMap._evaluate_point",
    "scientific_postprocessing": (
        "unchanged serial callback event stratification, hulling, padding, and graph logic"
    ),
    "acceptance_test": (
        "exact depth-8 scientific payload parity for serial, 2, 4, and 8 workers"
    ),
    "cross_family_cache_reuse": (
        "excluded; separate family orders can select different last-bit representatives"
    ),
}


def endpoint_cache_key(
    chart_id: int,
    point: Sequence[float],
) -> EndpointKey:
    """Match the Garcia callback's fourteen-decimal lattice-cache key."""

    return (
        int(chart_id),
        *(round(float(value), 14) for value in point),
    )


@dataclass(frozen=True)
class GarciaEndpointPrecomputeConfig:
    """Everything needed to reconstruct the scientific evaluator in a worker."""

    t_star: float = 0.5
    samples_per_axis: int | tuple[int, ...] = 3
    padding_cells: float = 1.0
    gamma: float = DEFAULT_GAMMA
    guard_delta: float = DEFAULT_GUARD_DELTA
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA
    domain_bounds: tuple[tuple[float, float], ...] = DEFAULT_BASE_BOUNDS
    max_step: float | None = 0.02
    max_jumps: int = 20
    atol: float = 1.0e-9
    require_domain_path: bool = True
    coordinate_system: str = "physical"

    def __post_init__(self) -> None:
        if not np.isfinite(self.t_star) or self.t_star <= 0.0:
            raise ValueError("t_star must be finite and positive")
        if len(self.domain_bounds) != 4:
            raise ValueError("the Garcia domain must have four coordinate intervals")
        if isinstance(self.samples_per_axis, int):
            if self.samples_per_axis < 3:
                raise ValueError("samples_per_axis must be at least three")
        elif len(self.samples_per_axis) != 4 or any(
            count < 3 for count in self.samples_per_axis
        ):
            raise ValueError(
                "samples_per_axis must have four entries, each at least three"
            )
        if self.max_jumps < 0:
            raise ValueError("max_jumps must be non-negative")
        if self.coordinate_system not in {"physical", "guard_aligned"}:
            raise ValueError(
                "coordinate_system must be 'physical' or 'guard_aligned'"
            )


@dataclass(frozen=True)
class GarciaEndpointPrecomputeResult:
    """A deterministic cache payload plus execution provenance."""

    entries: tuple[EndpointCacheEntry, ...]
    logical_tensor_points: int
    duplicate_tensor_points: int
    requested_workers: int
    used_workers: int
    mode: str
    elapsed_seconds: float
    fallback_reason: str | None = None

    @property
    def unique_points(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class _EndpointTask:
    chart_id: int
    point: tuple[float, ...]

    @property
    def key(self) -> EndpointKey:
        return endpoint_cache_key(self.chart_id, self.point)


_WORKER_BOX_MAP: CMGDBSuspensionBoxMap | None = None


def _build_worker_box_map(
    config: GarciaEndpointPrecomputeConfig,
) -> CMGDBSuspensionBoxMap:
    if config.coordinate_system == "guard_aligned":
        walker = GuardAlignedGarciaPassiveWalker(
            gamma=config.gamma,
            guard_delta=config.guard_delta,
            transversality_eta=config.transversality_eta,
            domain_bounds=[tuple(interval) for interval in config.domain_bounds],
            max_jumps=config.max_jumps,
        )
        charts = garcia_guard_aligned_atlas_charts(
            base_bounds=walker.domain_bounds,
            guard_delta=config.guard_delta,
            transversality_eta=config.transversality_eta,
        )
    else:
        walker = GarciaPassiveWalker(
            gamma=config.gamma,
            guard_delta=config.guard_delta,
            transversality_eta=config.transversality_eta,
            domain_bounds=[tuple(interval) for interval in config.domain_bounds],
            max_jumps=config.max_jumps,
        )
        charts = garcia_passive_walker_atlas_charts(
            base_bounds=walker.domain_bounds,
            guard_delta=config.guard_delta,
            transversality_eta=config.transversality_eta,
        )
    return CMGDBSuspensionBoxMap(
        walker.system,
        charts,
        config.t_star,
        samples_per_axis=config.samples_per_axis,
        padding_cells=config.padding_cells,
        max_jumps=config.max_jumps,
        max_step=config.max_step,
        atol=config.atol,
        require_domain_path=config.require_domain_path,
        diagnostics_limit=0,
    )


def _initialize_worker(config: GarciaEndpointPrecomputeConfig) -> None:
    global _WORKER_BOX_MAP
    _WORKER_BOX_MAP = _build_worker_box_map(config)


def _evaluate_with(
    box_map: CMGDBSuspensionBoxMap,
    task: _EndpointTask,
) -> EndpointCacheEntry:
    try:
        # The ordinary runner suppresses these trajectory warnings around the
        # serial CMGDB call.  Spawned workers do not inherit that filter, so
        # reproduce it locally; failures remain explicit cache entries below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = box_map._evaluate_point(  # noqa: SLF001 - exact callback primitive
                task.chart_id,
                np.asarray(task.point, dtype=np.float64),
            )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        cached: EndpointCacheValue = (False, (type(error), error.args))
    else:
        cached = (True, value)
    return task.key, cached


def _evaluate_worker_task(task: _EndpointTask) -> EndpointCacheEntry:
    if _WORKER_BOX_MAP is None:  # pragma: no cover - executor invariant
        raise RuntimeError("Garcia endpoint worker was not initialized")
    return _evaluate_with(_WORKER_BOX_MAP, task)


def _evaluate_worker_batch(
    tasks: tuple[_EndpointTask, ...],
) -> tuple[EndpointCacheEntry, ...]:
    """Amortize process IPC while retaining task order within every batch."""

    return tuple(_evaluate_worker_task(task) for task in tasks)


def _tensor_shape(
    samples_per_axis: int | tuple[int, ...],
    dimension: int,
) -> tuple[int, ...]:
    if isinstance(samples_per_axis, int):
        return (samples_per_axis,) * dimension
    if len(samples_per_axis) != dimension:
        raise ValueError("samples_per_axis does not match source-box dimension")
    return samples_per_axis


def _normalized_source_boxes(source_boxes: Iterable[SourceBox]) -> tuple[SourceBox, ...]:
    normalized: list[SourceBox] = []
    for chart_id, bounds in source_boxes:
        flat = tuple(float(value) for value in bounds)
        if not flat or len(flat) % 2:
            raise ValueError("each source box needs flattened lower/upper bounds")
        dimension = len(flat) // 2
        values = np.asarray(flat, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("source bounds must be finite")
        if np.any(values[:dimension] > values[dimension:]):
            raise ValueError("source lower bound exceeds upper bound")
        normalized.append((int(chart_id), flat))
    return tuple(normalized)


def _unique_tensor_tasks(
    source_boxes: Sequence[SourceBox],
    samples_per_axis: int | tuple[int, ...],
) -> Iterator[_EndpointTask]:
    """Yield the first actual representative of every rounded lattice key."""

    seen: set[EndpointKey] = set()
    for chart_id, bounds in source_boxes:
        dimension = len(bounds) // 2
        lower = bounds[:dimension]
        upper = bounds[dimension:]
        shape = _tensor_shape(samples_per_axis, dimension)
        axes = tuple(
            np.linspace(lower[axis], upper[axis], shape[axis])
            for axis in range(dimension)
        )
        for grid_index in itertools.product(*(range(count) for count in shape)):
            point = tuple(
                float(axes[axis][grid_index[axis]]) for axis in range(dimension)
            )
            task = _EndpointTask(chart_id=chart_id, point=point)
            if task.key in seen:
                continue
            seen.add(task.key)
            yield task


def _logical_tensor_point_count(
    source_boxes: Sequence[SourceBox],
    samples_per_axis: int | tuple[int, ...],
) -> int:
    return sum(
        int(np.prod(_tensor_shape(samples_per_axis, len(bounds) // 2)))
        for _chart_id, bounds in source_boxes
    )


def _serial_entries(
    source_boxes: Sequence[SourceBox],
    config: GarciaEndpointPrecomputeConfig,
) -> tuple[EndpointCacheEntry, ...]:
    box_map = _build_worker_box_map(config)
    return tuple(
        _evaluate_with(box_map, task)
        for task in _unique_tensor_tasks(source_boxes, config.samples_per_axis)
    )


def _task_batches(
    tasks: Iterator[_EndpointTask],
    chunksize: int,
) -> Iterator[tuple[_EndpointTask, ...]]:
    while True:
        batch = tuple(itertools.islice(tasks, chunksize))
        if not batch:
            return
        yield batch


def _bounded_process_entries(
    executor: ProcessPoolExecutor,
    tasks: Iterator[_EndpointTask],
    *,
    chunksize: int,
    submission_buffer: int,
) -> tuple[EndpointCacheEntry, ...]:
    """Submit a bounded number of ordered batches, compatible with Python 3.10+."""

    batches = _task_batches(tasks, chunksize)
    pending = deque()
    for _ in range(submission_buffer):
        try:
            batch = next(batches)
        except StopIteration:
            break
        pending.append(executor.submit(_evaluate_worker_batch, batch))

    entries: list[EndpointCacheEntry] = []
    while pending:
        # Futures are consumed in submission order.  A later completed batch
        # never overtakes an earlier source node in the cache payload.
        entries.extend(pending.popleft().result())
        try:
            batch = next(batches)
        except StopIteration:
            continue
        pending.append(executor.submit(_evaluate_worker_batch, batch))
    return tuple(entries)


def precompute_garcia_endpoint_cache(
    source_boxes: Iterable[SourceBox],
    config: GarciaEndpointPrecomputeConfig,
    *,
    workers: int,
    chunksize: int = 32,
    submission_buffer: int | None = None,
    start_method: str = "spawn",
    serial_fallback: bool = True,
) -> GarciaEndpointPrecomputeResult:
    """Evaluate unique Garcia tensor nodes serially or in local processes.

    ``source_boxes`` order is semantically significant: it determines which
    unrounded floating-point representative is retained when adjacent dyadic
    boxes share the same fourteen-decimal cache key.  The local relation passes
    its Atlas insertion order here, matching the ordinary serial callback.

    ``workers=1`` is the explicit serial path.  Values greater than one use a
    process pool.  Pool setup/execution failures cause a complete serial retry
    when ``serial_fallback`` is true; scientific endpoint failures remain cache
    entries and never trigger fallback.
    """

    if workers < 1:
        raise ValueError("workers must be positive when precomputation is enabled")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    boxes = _normalized_source_boxes(source_boxes)
    logical_points = _logical_tensor_point_count(boxes, config.samples_per_axis)
    started = time.perf_counter()

    if workers == 1:
        entries = _serial_entries(boxes, config)
        return GarciaEndpointPrecomputeResult(
            entries=entries,
            logical_tensor_points=logical_points,
            duplicate_tensor_points=logical_points - len(entries),
            requested_workers=1,
            used_workers=1,
            mode="serial",
            elapsed_seconds=time.perf_counter() - started,
        )

    buffer_size = (
        max(workers * 4, 1) if submission_buffer is None else int(submission_buffer)
    )
    if buffer_size < 1:
        raise ValueError("submission_buffer must be positive")
    try:
        context = mp.get_context(start_method)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            entries = _bounded_process_entries(
                executor,
                _unique_tensor_tasks(boxes, config.samples_per_axis),
                chunksize=chunksize,
                submission_buffer=buffer_size,
            )
    except (BrokenExecutor, OSError, RuntimeError, ValueError) as error:
        if not serial_fallback:
            raise
        entries = _serial_entries(boxes, config)
        return GarciaEndpointPrecomputeResult(
            entries=entries,
            logical_tensor_points=logical_points,
            duplicate_tensor_points=logical_points - len(entries),
            requested_workers=workers,
            used_workers=1,
            mode="serial_fallback",
            elapsed_seconds=time.perf_counter() - started,
            fallback_reason=f"{type(error).__name__}: {error}",
        )

    return GarciaEndpointPrecomputeResult(
        entries=entries,
        logical_tensor_points=logical_points,
        duplicate_tensor_points=logical_points - len(entries),
        requested_workers=workers,
        used_workers=workers,
        mode="process_pool",
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "EndpointCacheEntry",
    "EndpointCacheValue",
    "EndpointKey",
    "GarciaEndpointPrecomputeConfig",
    "GarciaEndpointPrecomputeResult",
    "PARITY_PROVENANCE",
    "SourceBox",
    "endpoint_cache_key",
    "precompute_garcia_endpoint_cache",
]
