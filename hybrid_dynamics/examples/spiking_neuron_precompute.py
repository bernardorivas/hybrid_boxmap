"""Deterministic parallel endpoint precompute for the neuron Atlas BoxMap.

Only pointwise suspension endpoints are evaluated in workers.  Source-box
stratification, finite-union hulling, L-domain splitting, padding, CMGDB graph
construction, and every audit remain in the parent process.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spiking_neuron import SpikingNeuron
from .spiking_neuron_atlas import (
    DEFAULT_MAX_STEP,
    DEFAULT_T_STAR,
    NeuronSuspensionBoxMap,
    spiking_neuron_atlas_charts,
)


EndpointKey = tuple[object, ...]
EndpointValue = tuple[bool, object]
EndpointEntry = tuple[EndpointKey, EndpointValue]
SourceBox = tuple[int, tuple[float, ...]]


@dataclass(frozen=True)
class NeuronEndpointPrecomputeConfig:
    t_star: float = DEFAULT_T_STAR
    samples_per_axis: int = 3
    padding_cells: float = 1.0
    max_step: float = DEFAULT_MAX_STEP
    max_jumps: int = 4
    atol: float = 1.0e-9

    def __post_init__(self) -> None:
        if not np.isfinite(self.t_star) or self.t_star <= 0.0:
            raise ValueError("t_star must be finite and positive")
        if self.samples_per_axis < 3:
            raise ValueError("samples_per_axis must be at least three")
        if not np.isfinite(self.max_step) or self.max_step <= 0.0:
            raise ValueError("max_step must be finite and positive")


@dataclass(frozen=True)
class NeuronEndpointPrecomputeResult:
    entries: tuple[EndpointEntry, ...]
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

    def metadata(self) -> dict[str, object]:
        return {
            "logical_tensor_points": self.logical_tensor_points,
            "unique_points": self.unique_points,
            "duplicate_tensor_points": self.duplicate_tensor_points,
            "requested_workers": self.requested_workers,
            "used_workers": self.used_workers,
            "mode": self.mode,
            "elapsed_seconds": self.elapsed_seconds,
            "fallback_reason": self.fallback_reason,
            "scientific_postprocessing": (
                "serial parent callback owns stratification, hulling, padding, "
                "L-domain split, and graph construction"
            ),
        }


@dataclass(frozen=True)
class _Task:
    chart_id: int
    point: tuple[float, ...]

    @property
    def key(self) -> EndpointKey:
        return (
            self.chart_id,
            *(round(float(value), 14) for value in self.point),
        )


_WORKER_MAP: NeuronSuspensionBoxMap | None = None


def _worker_map(config: NeuronEndpointPrecomputeConfig) -> NeuronSuspensionBoxMap:
    neuron = SpikingNeuron(max_jumps=config.max_jumps)
    return NeuronSuspensionBoxMap(
        neuron.system,
        spiking_neuron_atlas_charts(),
        config.t_star,
        samples_per_axis=config.samples_per_axis,
        padding_cells=config.padding_cells,
        max_jumps=config.max_jumps,
        max_step=config.max_step,
        atol=config.atol,
        require_domain_path=True,
        diagnostics_limit=0,
    )


def _initialize_worker(config: NeuronEndpointPrecomputeConfig) -> None:
    global _WORKER_MAP
    _WORKER_MAP = _worker_map(config)


def _evaluate(box_map: NeuronSuspensionBoxMap, task: _Task) -> EndpointEntry:
    try:
        value = super(NeuronSuspensionBoxMap, box_map)._evaluate_point(
            task.chart_id, np.asarray(task.point, dtype=np.float64)
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        cached: EndpointValue = (False, (type(error), error.args))
    else:
        cached = (True, value)
    return task.key, cached


def _evaluate_worker(task: _Task) -> EndpointEntry:
    if _WORKER_MAP is None:  # pragma: no cover - process invariant
        raise RuntimeError("neuron endpoint worker is uninitialized")
    return _evaluate(_WORKER_MAP, task)


def _evaluate_batch(tasks: tuple[_Task, ...]) -> tuple[EndpointEntry, ...]:
    return tuple(_evaluate_worker(task) for task in tasks)


def _normalized_boxes(boxes: Iterable[SourceBox]) -> tuple[SourceBox, ...]:
    result = []
    for chart_id, raw in boxes:
        bounds = tuple(float(value) for value in raw)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("neuron source boxes must have four finite bounds")
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            raise ValueError("source lower bound exceeds upper bound")
        result.append((int(chart_id), bounds))
    return tuple(result)


def _unique_tasks(boxes: Sequence[SourceBox], samples: int) -> Iterator[_Task]:
    seen: set[EndpointKey] = set()
    for chart_id, bounds in boxes:
        axes = (
            np.linspace(bounds[0], bounds[2], samples),
            np.linspace(bounds[1], bounds[3], samples),
        )
        for i, j in itertools.product(range(samples), repeat=2):
            task = _Task(chart_id, (float(axes[0][i]), float(axes[1][j])))
            if task.key not in seen:
                seen.add(task.key)
                yield task


def _batches(tasks: Iterator[_Task], chunksize: int) -> Iterator[tuple[_Task, ...]]:
    while True:
        batch = tuple(itertools.islice(tasks, chunksize))
        if not batch:
            return
        yield batch


def _parallel_entries(
    boxes: Sequence[SourceBox],
    config: NeuronEndpointPrecomputeConfig,
    *,
    workers: int,
    chunksize: int,
) -> tuple[EndpointEntry, ...]:
    context = mp.get_context("spawn")
    tasks = _unique_tasks(boxes, config.samples_per_axis)
    batches = _batches(tasks, chunksize)
    entries: list[EndpointEntry] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        pending = deque()
        for _ in range(max(2, 2 * workers)):
            try:
                pending.append(executor.submit(_evaluate_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            entries.extend(pending.popleft().result())
            try:
                pending.append(executor.submit(_evaluate_batch, next(batches)))
            except StopIteration:
                pass
    return tuple(entries)


def precompute_spiking_neuron_endpoints(
    source_boxes: Iterable[SourceBox],
    config: NeuronEndpointPrecomputeConfig,
    *,
    workers: int,
    chunksize: int = 64,
    serial_fallback: bool = True,
) -> NeuronEndpointPrecomputeResult:
    """Pre-evaluate every unique tensor node in deterministic source order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    boxes = _normalized_boxes(source_boxes)
    logical = len(boxes) * config.samples_per_axis**2
    started = time.perf_counter()
    if workers == 1:
        box_map = _worker_map(config)
        entries = tuple(
            _evaluate(box_map, task)
            for task in _unique_tasks(boxes, config.samples_per_axis)
        )
        return NeuronEndpointPrecomputeResult(
            entries,
            logical,
            logical - len(entries),
            1,
            1,
            "serial",
            time.perf_counter() - started,
        )
    try:
        entries = _parallel_entries(
            boxes, config, workers=workers, chunksize=chunksize
        )
    except (BrokenExecutor, OSError, RuntimeError) as error:
        if not serial_fallback:
            raise
        box_map = _worker_map(config)
        entries = tuple(
            _evaluate(box_map, task)
            for task in _unique_tasks(boxes, config.samples_per_axis)
        )
        return NeuronEndpointPrecomputeResult(
            entries,
            logical,
            logical - len(entries),
            workers,
            1,
            "serial_fallback",
            time.perf_counter() - started,
            f"{type(error).__name__}: {error}",
        )
    return NeuronEndpointPrecomputeResult(
        entries,
        logical,
        logical - len(entries),
        workers,
        workers,
        "process_pool",
        time.perf_counter() - started,
    )


def atlas_source_boxes(setup: Any) -> tuple[SourceBox, ...]:
    atlas = setup.model.phaseSpace()
    return tuple(
        (
            int(atlas.cell(index).chart_id),
            tuple(float(value) for value in atlas.cell(index).bounds),
        )
        for index in range(int(atlas.size()))
    )


__all__ = [
    "EndpointEntry",
    "NeuronEndpointPrecomputeConfig",
    "NeuronEndpointPrecomputeResult",
    "SourceBox",
    "atlas_source_boxes",
    "precompute_spiking_neuron_endpoints",
]
