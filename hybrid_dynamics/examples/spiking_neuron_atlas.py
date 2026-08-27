"""CMGDB Atlas benchmark for the compact spiking-neuron suspension.

The computation uses a full active cubical cover of the compact L-shaped
state space ``N`` and the outgoing reset handle ``(u_guard, s)``.  The ambient
dyadic address charts are deliberately larger than the active domain so every
important physical line is an exact cell face at the frozen per-axis depths
6, 7, and 8.  In particular, the guard ``v=35``, the interior reset seam
``v=-50``, the L-shaped notch, and the translation ``u -> u+100`` all align
with the cubical structure.

This is a sampled finite relation.  Its endpoint and connectivity audits are
falsifiers, not a claim of interval-rigorous whole-cell enclosure.  No
analytic Conley label is attached by this module.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..src.atlas_conley import (
    AffineBoundaryEmbedding2D,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasResetGluing2D,
)
from ..src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    SourceBoxMapDiagnostic,
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
from .spiking_neuron import (
    U_MAX,
    U_MIN,
    U_NOTCH,
    U_RESET_SHIFT,
    V_MIN,
    V_NOTCH,
    V_PEAK,
    V_RESET,
    SpikingNeuron,
    SpikingNeuronAnalyticAudit,
    spiking_neuron_analytic_audit,
)


BASE_CHART_ID = 0
HANDLE_CHART_ID = 1
BASE_AMBIENT_BOUNDS = ((-120.0, 200.0), (-400.0, 880.0))
HANDLE_U_AMBIENT_BOUNDS = (-400.0, 880.0)
ACTIVE_HANDLE_U_BOUNDS = (U_MIN, U_NOTCH)
DEFAULT_T_STAR = 0.5
DEFAULT_MAX_STEP = 0.02
FROZEN_TOTAL_DEPTHS = (12, 14, 16)
FROZEN_SAMPLING_SENSITIVITY = (12, 5)
BOXMAP_REVISION = "spiking-neuron-l-domain-split-v1"
PROTOCOL_REVISION = "spiking-neuron-atlas-protocol-v1"
ADAPTIVE_PREFLIGHT_REVISION = "spiking-neuron-error-driven-axis8-preflight-v1"


def _source_key(chart_id: int, bounds: Sequence[float]) -> tuple[object, ...]:
    return (int(chart_id), *(round(float(value), 14) for value in bounds))


def _point_key(chart_id: int, point: Sequence[float]) -> tuple[object, ...]:
    return (int(chart_id), *(round(float(value), 14) for value in point))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, order=True)
class NeuronDyadicCell:
    chart_id: int
    axis_depth: int
    coordinates: tuple[int, int]


@dataclass(frozen=True)
class SpikingNeuronActiveFamily:
    """Complete active cubical cover of ``N`` and ``(G cap N) x [0,1]``."""

    axis_depth: int
    cells: tuple[NeuronDyadicCell, ...]
    fingerprint: str

    @property
    def total_depth(self) -> int:
        return 2 * self.axis_depth

    @property
    def subdivisions_per_axis(self) -> int:
        return 2**self.axis_depth

    def chart_counts(self) -> dict[int, int]:
        return dict(Counter(cell.chart_id for cell in self.cells))

    def depth_counts(self) -> dict[int, int]:
        return {self.axis_depth: len(self.cells)}

    def tagged_cells(self) -> tuple[tuple[int, int, tuple[int, int]], ...]:
        return tuple(
            (cell.chart_id, cell.axis_depth, cell.coordinates) for cell in self.cells
        )


@dataclass(frozen=True)
class SpikingNeuronAdaptiveFamily:
    """A complete nonoverlapping mixed-depth cover derived from one grid."""

    coarse_axis_depth: int
    fine_axis_depth: int
    cells: tuple[NeuronDyadicCell, ...]
    refined_parents: tuple[NeuronDyadicCell, ...]
    fingerprint: str

    def chart_counts(self) -> dict[int, int]:
        return dict(Counter(cell.chart_id for cell in self.cells))

    def depth_counts(self) -> dict[int, int]:
        return dict(Counter(cell.axis_depth for cell in self.cells))

    def tagged_cells(self) -> tuple[tuple[int, int, tuple[int, int]], ...]:
        return tuple(
            (cell.chart_id, cell.axis_depth, cell.coordinates) for cell in self.cells
        )


def spiking_neuron_dyadic_bounds(cell: NeuronDyadicCell) -> tuple[float, ...]:
    """Decode one tagged dyadic address in the fixed neuron ambient charts."""

    charts = spiking_neuron_atlas_charts()
    intervals = charts.bounds_for(cell.chart_id)
    subdivisions = 2**cell.axis_depth
    lower = tuple(
        float(interval[0] + coordinate * (interval[1] - interval[0]) / subdivisions)
        for coordinate, interval in zip(cell.coordinates, intervals, strict=True)
    )
    upper = tuple(
        float(
            interval[0]
            + (coordinate + 1) * (interval[1] - interval[0]) / subdivisions
        )
        for coordinate, interval in zip(cell.coordinates, intervals, strict=True)
    )
    return (*lower, *upper)


def build_spiking_neuron_adaptive_family(
    *,
    coarse_axis_depth: int,
    refined_parents: Collection[NeuronDyadicCell],
    fine_axis_depth: int | None = None,
) -> SpikingNeuronAdaptiveFamily:
    """Replace selected coarse leaves by all four children, without holes."""

    fine = coarse_axis_depth + 1 if fine_axis_depth is None else int(fine_axis_depth)
    if fine != coarse_axis_depth + 1:
        raise ValueError("adaptive neuron preflight supports one dyadic level")
    coarse = build_spiking_neuron_active_family(coarse_axis_depth)
    selected = frozenset(refined_parents)
    unknown = selected.difference(coarse.cells)
    if unknown:
        raise ValueError("refined parents must belong to the complete coarse family")
    children = tuple(
        NeuronDyadicCell(
            parent.chart_id,
            fine,
            (2 * parent.coordinates[0] + first, 2 * parent.coordinates[1] + second),
        )
        for parent in sorted(selected)
        for first, second in itertools.product((0, 1), repeat=2)
    )
    leaves = tuple(sorted(set(coarse.cells).difference(selected)) + sorted(children))
    payload = [[cell.chart_id, cell.axis_depth, *cell.coordinates] for cell in leaves]
    return SpikingNeuronAdaptiveFamily(
        coarse_axis_depth=int(coarse_axis_depth),
        fine_axis_depth=fine,
        cells=leaves,
        refined_parents=tuple(sorted(selected)),
        fingerprint=_sha256(payload),
    )


def _aligned_index(value: float, interval: tuple[float, float], subdivisions: int) -> int:
    coordinate = subdivisions * (float(value) - interval[0]) / (
        interval[1] - interval[0]
    )
    rounded = int(round(coordinate))
    if abs(coordinate - rounded) > 1.0e-10:
        raise ValueError(f"{value} is not aligned with the declared dyadic grid")
    return rounded


def build_spiking_neuron_active_family(axis_depth: int) -> SpikingNeuronActiveFamily:
    """Construct the exact full-cell L-shaped base and guard-handle family."""

    if isinstance(axis_depth, bool) or not isinstance(axis_depth, (int, np.integer)):
        raise ValueError("axis_depth must be an integer")
    if axis_depth < 6:
        raise ValueError("the frozen neuron address chart starts at axis depth 6")
    subdivisions = 2**int(axis_depth)
    v_interval, u_interval = BASE_AMBIENT_BOUNDS
    v0 = _aligned_index(V_MIN, v_interval, subdivisions)
    v1 = _aligned_index(V_NOTCH, v_interval, subdivisions)
    vp = _aligned_index(V_PEAK, v_interval, subdivisions)
    u0 = _aligned_index(U_MIN, u_interval, subdivisions)
    u1 = _aligned_index(U_NOTCH, u_interval, subdivisions)
    u2 = _aligned_index(U_MAX, u_interval, subdivisions)

    base_indices = {
        (v_index, u_index)
        for v_index in range(v0, v1)
        for u_index in range(u0, u2)
    }
    base_indices.update(
        (v_index, u_index)
        for v_index in range(v1, vp)
        for u_index in range(u0, u1)
    )
    handle_indices = {
        (u_index, phase_index)
        for u_index in range(u0, u1)
        for phase_index in range(subdivisions)
    }
    cells = tuple(
        sorted(
            NeuronDyadicCell(BASE_CHART_ID, int(axis_depth), coordinates)
            for coordinates in base_indices
        )
        + sorted(
            NeuronDyadicCell(HANDLE_CHART_ID, int(axis_depth), coordinates)
            for coordinates in handle_indices
        )
    )
    payload = [
        [cell.chart_id, cell.axis_depth, *cell.coordinates] for cell in cells
    ]
    return SpikingNeuronActiveFamily(
        axis_depth=int(axis_depth),
        cells=cells,
        fingerprint=_sha256(payload),
    )


def spiking_neuron_atlas_charts() -> SuspensionAtlasCharts:
    """Return seam-aligned ambient ``(v,u)`` and ``(u_guard,s)`` charts."""

    return SuspensionAtlasCharts(
        base_bounds=BASE_AMBIENT_BOUNDS,
        guard_bounds=(HANDLE_U_AMBIENT_BOUNDS,),
        guard_coordinates=lambda guard: np.asarray((guard[1],), dtype=np.float64),
        guard_embedding=lambda intrinsic: np.asarray(
            (V_PEAK, intrinsic[0]), dtype=np.float64
        ),
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )


def spiking_neuron_atlas_reset_gluing() -> AtlasResetGluing2D:
    """Return the exact guard and interior-reset affine seam attachments."""

    return AtlasResetGluing2D(
        BASE_CHART_ID,
        HANDLE_CHART_ID,
        guard=AffineBoundaryEmbedding2D(0, V_PEAK, 1.0),
        reset=AffineBoundaryEmbedding2D(
            0, V_RESET, 1.0, offset=U_RESET_SHIFT
        ),
    )


def build_spiking_neuron_quotient_nerve(
    cells: Collection[AtlasNeuronCell],
    *,
    maximum_simplex_size: int = 16,
) -> AtlasQuotientNerveComplex2D:
    """Build the verified actual-cell nerve with an interior reset seam.

    The explicit opt-in does not assert validity.  The nerve constructor
    verifies that ``v=-50`` is a complete two-sided cubical subcomplex and
    rejects straddling, incomplete, or one-sided selected families.
    """

    return AtlasQuotientNerveComplex2D(
        (
            AtlasRectangleCell2D(cell.index, cell.chart_id, cell.bounds)
            for cell in cells
        ),
        spiking_neuron_atlas_reset_gluing(),
        maximum_simplex_size=maximum_simplex_size,
        interior_seam_subcomplexes=("reset",),
    )


@dataclass(frozen=True, order=True)
class AtlasNeuronCell:
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


class SpikingNeuronQuotientIncidence:
    """Closed-cell incidence in the reset-glued suspension quotient."""

    def __init__(self, charts: SuspensionAtlasCharts, *, atol: float = 1.0e-10):
        self.charts = charts
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

    def intersects(self, first: AtlasNeuronCell, second: AtlasNeuronCell) -> bool:
        if first == second:
            return True
        if first.chart_id == second.chart_id:
            return self._rectangles_intersect(
                first.lower, first.upper, second.lower, second.upper
            )
        if {first.chart_id, second.chart_id} != {BASE_CHART_ID, HANDLE_CHART_ID}:
            return False
        base = first if first.chart_id == BASE_CHART_ID else second
        handle = first if first.chart_id == HANDLE_CHART_ID else second
        u_lower, phase_lower = handle.lower
        u_upper, phase_upper = handle.upper
        if phase_lower <= self.atol:
            if self._rectangles_intersect(
                base.lower,
                base.upper,
                np.asarray((V_PEAK, u_lower)),
                np.asarray((V_PEAK, u_upper)),
            ):
                return True
        if phase_upper >= 1.0 - self.atol:
            if self._rectangles_intersect(
                base.lower,
                base.upper,
                np.asarray((V_RESET, u_lower + U_RESET_SHIFT)),
                np.asarray((V_RESET, u_upper + U_RESET_SHIFT)),
            ):
                return True
        return False

    def connected_components(
        self, cells: Collection[AtlasNeuronCell]
    ) -> tuple[frozenset[AtlasNeuronCell], ...]:
        remaining = set(cells)
        components: list[frozenset[AtlasNeuronCell]] = []
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


class SpikingNeuronSparseQuotientIncidence:
    """Exact constant-degree incidence for one uniform active neuron family.

    Closed same-chart cells meet precisely at Chebyshev-neighbor dyadic
    addresses.  Cross-chart neighbors are generated only on the guard and
    reset faces, including endpoint/corner incidence.  This avoids the
    quadratic all-pairs traversal when a Morse support contains most of ``N``.
    """

    def __init__(
        self,
        cells: Sequence[AtlasNeuronCell],
        charts: SuspensionAtlasCharts,
        *,
        axis_depth: int,
    ) -> None:
        self.cells = tuple(cells)
        self.charts = charts
        self.axis_depth = int(axis_depth)
        self.subdivisions = 2**self.axis_depth
        self._coordinates = {
            cell: self._dyadic_coordinates(cell) for cell in self.cells
        }
        self._cell_by_address = {
            (cell.chart_id, *self._coordinates[cell]): cell for cell in self.cells
        }
        if len(self._cell_by_address) != len(self.cells):
            raise ValueError("neuron Atlas cells do not have unique dyadic addresses")
        v_interval, u_interval = self.charts.base_bounds
        self._guard_v_index = _aligned_index(
            V_PEAK, v_interval, self.subdivisions
        )
        self._reset_v_index = _aligned_index(
            V_RESET, v_interval, self.subdivisions
        )
        self._reset_u_shift = _aligned_index(
            u_interval[0] + U_RESET_SHIFT,
            u_interval,
            self.subdivisions,
        )

    def _dyadic_coordinates(self, cell: AtlasNeuronCell) -> tuple[int, int]:
        bounds = self.charts.bounds_for(cell.chart_id)
        result = []
        for axis, interval in enumerate(bounds):
            coordinate = self.subdivisions * (
                float(cell.lower[axis]) - interval[0]
            ) / (interval[1] - interval[0])
            rounded = int(round(coordinate))
            expected_upper = interval[0] + (
                rounded + 1
            ) * (interval[1] - interval[0]) / self.subdivisions
            if (
                abs(coordinate - rounded) > 1.0e-9
                or abs(float(cell.upper[axis]) - expected_upper) > 1.0e-8
            ):
                raise ValueError("neuron Atlas cell is not a uniform dyadic top cell")
            result.append(rounded)
        return int(result[0]), int(result[1])

    def neighbors(self, cell: AtlasNeuronCell) -> frozenset[AtlasNeuronCell]:
        first, second = self._coordinates[cell]
        result = {
            candidate
            for delta_first, delta_second in itertools.product((-1, 0, 1), repeat=2)
            if (
                candidate := self._cell_by_address.get(
                    (
                        cell.chart_id,
                        first + delta_first,
                        second + delta_second,
                    )
                )
            )
            is not None
        }
        if cell.chart_id == HANDLE_CHART_ID:
            u_index, phase_index = first, second
            if phase_index == 0:
                for base_u in range(u_index - 1, u_index + 2):
                    candidate = self._cell_by_address.get(
                        (BASE_CHART_ID, self._guard_v_index - 1, base_u)
                    )
                    if candidate is not None:
                        result.add(candidate)
            if phase_index == self.subdivisions - 1:
                reset_u = u_index + self._reset_u_shift
                for base_v in (self._reset_v_index - 1, self._reset_v_index):
                    for base_u in range(reset_u - 1, reset_u + 2):
                        candidate = self._cell_by_address.get(
                            (BASE_CHART_ID, base_v, base_u)
                        )
                        if candidate is not None:
                            result.add(candidate)
        else:
            v_index, u_index = first, second
            if v_index == self._guard_v_index - 1:
                for handle_u in range(u_index - 1, u_index + 2):
                    candidate = self._cell_by_address.get(
                        (HANDLE_CHART_ID, handle_u, 0)
                    )
                    if candidate is not None:
                        result.add(candidate)
            if v_index in (self._reset_v_index - 1, self._reset_v_index):
                handle_u_center = u_index - self._reset_u_shift
                for handle_u in range(handle_u_center - 1, handle_u_center + 2):
                    candidate = self._cell_by_address.get(
                        (
                            HANDLE_CHART_ID,
                            handle_u,
                            self.subdivisions - 1,
                        )
                    )
                    if candidate is not None:
                        result.add(candidate)
        return frozenset(result)

    def connected_components(
        self, cells: Collection[AtlasNeuronCell]
    ) -> tuple[frozenset[AtlasNeuronCell], ...]:
        selected = set(cells)
        unknown = selected.difference(self._coordinates)
        if unknown:
            raise ValueError("connectivity request contains cells outside this Atlas")
        remaining = set(selected)
        components: list[frozenset[AtlasNeuronCell]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                neighbors = set(self.neighbors(current)).intersection(remaining)
                remaining.difference_update(neighbors)
                component.update(neighbors)
                stack.extend(neighbors)
            components.append(frozenset(component))
        components.sort(key=lambda component: min(component))
        return tuple(components)


class SpikingNeuronMixedQuotientIncidence:
    """Exact sparse incidence for a one-level mixed dyadic neuron family."""

    def __init__(
        self,
        cells: Sequence[AtlasNeuronCell],
        charts: SuspensionAtlasCharts,
        *,
        fine_axis_depth: int,
    ) -> None:
        self.cells = tuple(cells)
        self.charts = charts
        self.fine_axis_depth = int(fine_axis_depth)
        self.subdivisions = 2**self.fine_axis_depth
        self._ranges = {cell: self._fine_ranges(cell) for cell in self.cells}
        self._owner: dict[tuple[int, int, int], AtlasNeuronCell] = {}
        for cell, (first0, first1, second0, second1) in self._ranges.items():
            for first in range(first0, first1):
                for second in range(second0, second1):
                    address = (cell.chart_id, first, second)
                    previous = self._owner.setdefault(address, cell)
                    if previous != cell:
                        raise ValueError("mixed neuron leaves overlap at fine depth")
        v_interval, u_interval = self.charts.base_bounds
        self._guard_v_index = _aligned_index(
            V_PEAK, v_interval, self.subdivisions
        )
        self._reset_v_index = _aligned_index(
            V_RESET, v_interval, self.subdivisions
        )
        self._reset_u_shift = _aligned_index(
            u_interval[0] + U_RESET_SHIFT,
            u_interval,
            self.subdivisions,
        )

    def _fine_ranges(self, cell: AtlasNeuronCell) -> tuple[int, int, int, int]:
        intervals = self.charts.bounds_for(cell.chart_id)
        result: list[int] = []
        for axis, interval in enumerate(intervals):
            scale = self.subdivisions / (interval[1] - interval[0])
            lower = int(round((float(cell.lower[axis]) - interval[0]) * scale))
            upper = int(round((float(cell.upper[axis]) - interval[0]) * scale))
            if (
                abs(interval[0] + lower / scale - float(cell.lower[axis])) > 1.0e-8
                or abs(interval[0] + upper / scale - float(cell.upper[axis]))
                > 1.0e-8
                or upper <= lower
                or upper - lower not in (1, 2)
            ):
                raise ValueError(
                    "mixed neuron cell is not an axis-depth-7/8 dyadic leaf"
                )
            result.extend((lower, upper))
        return result[0], result[1], result[2], result[3]

    def neighbors(self, cell: AtlasNeuronCell) -> frozenset[AtlasNeuronCell]:
        if cell not in self._ranges:
            raise ValueError("connectivity request contains a cell outside this Atlas")
        first0, first1, second0, second1 = self._ranges[cell]
        result = {
            candidate
            for first in range(first0 - 1, first1 + 1)
            for second in range(second0 - 1, second1 + 1)
            if (
                candidate := self._owner.get((cell.chart_id, first, second))
            )
            is not None
        }
        if cell.chart_id == HANDLE_CHART_ID:
            u0, u1, phase0, phase1 = first0, first1, second0, second1
            if phase0 == 0:
                result.update(
                    candidate
                    for u_index in range(u0 - 1, u1 + 1)
                    if (
                        candidate := self._owner.get(
                            (BASE_CHART_ID, self._guard_v_index - 1, u_index)
                        )
                    )
                    is not None
                )
            if phase1 == self.subdivisions:
                result.update(
                    candidate
                    for v_index in (
                        self._reset_v_index - 1,
                        self._reset_v_index,
                    )
                    for u_index in range(
                        u0 + self._reset_u_shift - 1,
                        u1 + self._reset_u_shift + 1,
                    )
                    if (
                        candidate := self._owner.get(
                            (BASE_CHART_ID, v_index, u_index)
                        )
                    )
                    is not None
                )
        else:
            v0, v1, u0, u1 = first0, first1, second0, second1
            if v1 == self._guard_v_index:
                result.update(
                    candidate
                    for u_index in range(u0 - 1, u1 + 1)
                    if (
                        candidate := self._owner.get(
                            (HANDLE_CHART_ID, u_index, 0)
                        )
                    )
                    is not None
                )
            if v0 == self._reset_v_index or v1 == self._reset_v_index:
                result.update(
                    candidate
                    for u_index in range(
                        u0 - self._reset_u_shift - 1,
                        u1 - self._reset_u_shift + 1,
                    )
                    if (
                        candidate := self._owner.get(
                            (HANDLE_CHART_ID, u_index, self.subdivisions - 1)
                        )
                    )
                    is not None
                )
        return frozenset(result)

    def connected_components(
        self, cells: Collection[AtlasNeuronCell]
    ) -> tuple[frozenset[AtlasNeuronCell], ...]:
        selected = set(cells)
        unknown = selected.difference(self._ranges)
        if unknown:
            raise ValueError("connectivity request contains cells outside this Atlas")
        remaining = set(selected)
        components: list[frozenset[AtlasNeuronCell]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                neighbors = set(self.neighbors(current)).intersection(remaining)
                remaining.difference_update(neighbors)
                component.update(neighbors)
                stack.extend(neighbors)
            components.append(frozenset(component))
        components.sort(key=lambda component: min(component))
        return tuple(components)


@dataclass(frozen=True)
class SparseIncidenceParityAudit:
    axis_depth: int
    cells: int
    compared_cell_rows: int
    sparse_undirected_edges: int
    mismatched_rows: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatched_rows and self.compared_cell_rows == self.cells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "spiking-neuron-sparse-incidence-parity-v1",
            "axis_depth": self.axis_depth,
            "cells": self.cells,
            "compared_cell_rows": self.compared_cell_rows,
            "sparse_undirected_edges": self.sparse_undirected_edges,
            "mismatched_rows": list(self.mismatched_rows),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class MixedIncidenceParityAudit:
    fine_axis_depth: int
    cells: int
    leaf_depth_counts: tuple[tuple[int, int], ...]
    compared_cell_rows: int
    sparse_undirected_edges: int
    mismatched_rows: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatched_rows and self.compared_cell_rows == self.cells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "spiking-neuron-mixed-incidence-parity-v1",
            "fine_axis_depth": self.fine_axis_depth,
            "cells": self.cells,
            "leaf_depth_counts": {
                str(depth): count for depth, count in self.leaf_depth_counts
            },
            "compared_cell_rows": self.compared_cell_rows,
            "sparse_undirected_edges": self.sparse_undirected_edges,
            "mismatched_rows": list(self.mismatched_rows),
            "passed": self.passed,
        }


def audit_spiking_neuron_sparse_incidence_parity(
    cells: Sequence[AtlasNeuronCell],
    charts: SuspensionAtlasCharts,
    *,
    axis_depth: int,
) -> SparseIncidenceParityAudit:
    """Compare every sparse row with vectorized closed-rectangle incidence."""

    sparse = SpikingNeuronSparseQuotientIncidence(
        cells, charts, axis_depth=axis_depth
    )
    geometric = SpikingNeuronQuotientIncidence(charts)
    by_chart = {
        chart_id: tuple(cell for cell in cells if cell.chart_id == chart_id)
        for chart_id in (BASE_CHART_ID, HANDLE_CHART_ID)
    }
    lowers = {
        chart_id: np.asarray([cell.lower for cell in chart_cells], dtype=np.float64)
        for chart_id, chart_cells in by_chart.items()
    }
    uppers = {
        chart_id: np.asarray([cell.upper for cell in chart_cells], dtype=np.float64)
        for chart_id, chart_cells in by_chart.items()
    }
    mismatches = []
    edge_twice = 0
    for cell in cells:
        chart_cells = by_chart[cell.chart_id]
        same_mask = np.all(lowers[cell.chart_id] <= cell.upper + geometric.atol, axis=1) & np.all(
            uppers[cell.chart_id] >= cell.lower - geometric.atol, axis=1
        )
        expected = {
            chart_cells[index] for index in np.flatnonzero(same_mask)
        }
        other_chart = (
            HANDLE_CHART_ID if cell.chart_id == BASE_CHART_ID else BASE_CHART_ID
        )
        other_cells = by_chart[other_chart]
        other_lower = lowers[other_chart]
        other_upper = uppers[other_chart]
        cross_mask = np.zeros(len(other_cells), dtype=bool)
        if cell.chart_id == BASE_CHART_ID:
            guard_lower = np.column_stack(
                (np.full(len(other_cells), V_PEAK), other_lower[:, 0])
            )
            guard_upper = np.column_stack(
                (np.full(len(other_cells), V_PEAK), other_upper[:, 0])
            )
            reset_lower = np.column_stack(
                (
                    np.full(len(other_cells), V_RESET),
                    other_lower[:, 0] + U_RESET_SHIFT,
                )
            )
            reset_upper = np.column_stack(
                (
                    np.full(len(other_cells), V_RESET),
                    other_upper[:, 0] + U_RESET_SHIFT,
                )
            )
            cross_mask |= (
                other_lower[:, 1] <= geometric.atol
            ) & np.all(guard_lower <= cell.upper + geometric.atol, axis=1) & np.all(
                guard_upper >= cell.lower - geometric.atol, axis=1
            )
            cross_mask |= (
                other_upper[:, 1] >= 1.0 - geometric.atol
            ) & np.all(reset_lower <= cell.upper + geometric.atol, axis=1) & np.all(
                reset_upper >= cell.lower - geometric.atol, axis=1
            )
        else:
            if cell.lower[1] <= geometric.atol:
                guard_lower = np.asarray((V_PEAK, cell.lower[0]))
                guard_upper = np.asarray((V_PEAK, cell.upper[0]))
                cross_mask |= np.all(
                    other_lower <= guard_upper + geometric.atol, axis=1
                ) & np.all(other_upper >= guard_lower - geometric.atol, axis=1)
            if cell.upper[1] >= 1.0 - geometric.atol:
                reset_lower = np.asarray(
                    (V_RESET, cell.lower[0] + U_RESET_SHIFT)
                )
                reset_upper = np.asarray(
                    (V_RESET, cell.upper[0] + U_RESET_SHIFT)
                )
                cross_mask |= np.all(
                    other_lower <= reset_upper + geometric.atol, axis=1
                ) & np.all(other_upper >= reset_lower - geometric.atol, axis=1)
        expected.update(other_cells[index] for index in np.flatnonzero(cross_mask))
        observed = set(sparse.neighbors(cell))
        edge_twice += len(observed) - 1
        if observed != expected:
            mismatches.append(cell.index)
    return SparseIncidenceParityAudit(
        axis_depth=int(axis_depth),
        cells=len(cells),
        compared_cell_rows=len(cells),
        sparse_undirected_edges=edge_twice // 2,
        mismatched_rows=tuple(mismatches),
    )


def _vectorized_geometric_neighbors(
    cell: AtlasNeuronCell,
    *,
    by_chart: Mapping[int, tuple[AtlasNeuronCell, ...]],
    lowers: Mapping[int, np.ndarray],
    uppers: Mapping[int, np.ndarray],
    atol: float,
) -> set[AtlasNeuronCell]:
    chart_cells = by_chart[cell.chart_id]
    same_mask = np.all(
        lowers[cell.chart_id] <= cell.upper + atol, axis=1
    ) & np.all(uppers[cell.chart_id] >= cell.lower - atol, axis=1)
    expected = {chart_cells[index] for index in np.flatnonzero(same_mask)}
    other_chart = (
        HANDLE_CHART_ID if cell.chart_id == BASE_CHART_ID else BASE_CHART_ID
    )
    other_cells = by_chart[other_chart]
    other_lower = lowers[other_chart]
    other_upper = uppers[other_chart]
    cross_mask = np.zeros(len(other_cells), dtype=bool)
    if cell.chart_id == BASE_CHART_ID:
        guard_lower = np.column_stack(
            (np.full(len(other_cells), V_PEAK), other_lower[:, 0])
        )
        guard_upper = np.column_stack(
            (np.full(len(other_cells), V_PEAK), other_upper[:, 0])
        )
        reset_lower = np.column_stack(
            (
                np.full(len(other_cells), V_RESET),
                other_lower[:, 0] + U_RESET_SHIFT,
            )
        )
        reset_upper = np.column_stack(
            (
                np.full(len(other_cells), V_RESET),
                other_upper[:, 0] + U_RESET_SHIFT,
            )
        )
        cross_mask |= (other_lower[:, 1] <= atol) & np.all(
            guard_lower <= cell.upper + atol, axis=1
        ) & np.all(guard_upper >= cell.lower - atol, axis=1)
        cross_mask |= (other_upper[:, 1] >= 1.0 - atol) & np.all(
            reset_lower <= cell.upper + atol, axis=1
        ) & np.all(reset_upper >= cell.lower - atol, axis=1)
    else:
        if cell.lower[1] <= atol:
            guard_lower = np.asarray((V_PEAK, cell.lower[0]))
            guard_upper = np.asarray((V_PEAK, cell.upper[0]))
            cross_mask |= np.all(
                other_lower <= guard_upper + atol, axis=1
            ) & np.all(other_upper >= guard_lower - atol, axis=1)
        if cell.upper[1] >= 1.0 - atol:
            reset_lower = np.asarray(
                (V_RESET, cell.lower[0] + U_RESET_SHIFT)
            )
            reset_upper = np.asarray(
                (V_RESET, cell.upper[0] + U_RESET_SHIFT)
            )
            cross_mask |= np.all(
                other_lower <= reset_upper + atol, axis=1
            ) & np.all(other_upper >= reset_lower - atol, axis=1)
    expected.update(other_cells[index] for index in np.flatnonzero(cross_mask))
    return expected


def audit_spiking_neuron_mixed_incidence_parity(
    cells: Sequence[AtlasNeuronCell],
    charts: SuspensionAtlasCharts,
    *,
    fine_axis_depth: int,
) -> MixedIncidenceParityAudit:
    """Certify every mixed sparse incidence row against physical geometry."""

    sparse = SpikingNeuronMixedQuotientIncidence(
        cells, charts, fine_axis_depth=fine_axis_depth
    )
    geometric = SpikingNeuronQuotientIncidence(charts)
    by_chart = {
        chart_id: tuple(cell for cell in cells if cell.chart_id == chart_id)
        for chart_id in (BASE_CHART_ID, HANDLE_CHART_ID)
    }
    lowers = {
        chart_id: np.asarray([cell.lower for cell in chart_cells], dtype=np.float64)
        for chart_id, chart_cells in by_chart.items()
    }
    uppers = {
        chart_id: np.asarray([cell.upper for cell in chart_cells], dtype=np.float64)
        for chart_id, chart_cells in by_chart.items()
    }
    mismatches = []
    edge_twice = 0
    depth_counts: Counter[int] = Counter()
    for cell in cells:
        first0, first1, _second0, _second1 = sparse._ranges[cell]
        width = first1 - first0
        depth_counts[fine_axis_depth - int(round(np.log2(width)))] += 1
        expected = _vectorized_geometric_neighbors(
            cell,
            by_chart=by_chart,
            lowers=lowers,
            uppers=uppers,
            atol=geometric.atol,
        )
        observed = set(sparse.neighbors(cell))
        edge_twice += len(observed) - 1
        if observed != expected:
            mismatches.append(cell.index)
    return MixedIncidenceParityAudit(
        fine_axis_depth=int(fine_axis_depth),
        cells=len(cells),
        leaf_depth_counts=tuple(sorted(depth_counts.items())),
        compared_cell_rows=len(cells),
        sparse_undirected_edges=edge_twice // 2,
        mismatched_rows=tuple(mismatches),
    )


def _tagged_address(cell: NeuronDyadicCell) -> list[object]:
    return [cell.chart_id, cell.axis_depth, list(cell.coordinates)]


def _exact_uniform_seam_cofaces(
    cell: NeuronDyadicCell,
    *,
    cell_by_address: Mapping[tuple[int, int, int], NeuronDyadicCell],
) -> frozenset[NeuronDyadicCell]:
    """Return positive-length face cofaces, excluding corner-only contacts."""

    subdivisions = 2**cell.axis_depth
    v_interval, u_interval = BASE_AMBIENT_BOUNDS
    guard_v = _aligned_index(V_PEAK, v_interval, subdivisions)
    reset_v = _aligned_index(V_RESET, v_interval, subdivisions)
    reset_u_shift = _aligned_index(
        u_interval[0] + U_RESET_SHIFT, u_interval, subdivisions
    )
    first, second = cell.coordinates
    candidates: list[tuple[int, int, int]] = []
    if cell.chart_id == HANDLE_CHART_ID:
        if second == 0:
            candidates.append((BASE_CHART_ID, guard_v - 1, first))
        if second == subdivisions - 1:
            candidates.extend(
                (
                    (BASE_CHART_ID, reset_v - 1, first + reset_u_shift),
                    (BASE_CHART_ID, reset_v, first + reset_u_shift),
                )
            )
    elif cell.chart_id == BASE_CHART_ID:
        if first == guard_v - 1:
            candidates.append((HANDLE_CHART_ID, second, 0))
        if first in (reset_v - 1, reset_v):
            candidates.append(
                (
                    HANDLE_CHART_ID,
                    second - reset_u_shift,
                    subdivisions - 1,
                )
            )
    return frozenset(
        candidate
        for address in candidates
        if (candidate := cell_by_address.get(address)) is not None
    )


@dataclass(frozen=True)
class SpikingNeuronAdaptivePreflight:
    """Authenticated no-dynamics cost and witness audit for one clock."""

    t_star: float
    source_total_depth: int
    source_axis_depth: int
    fine_axis_depth: int
    samples_per_axis: int
    max_step: float
    max_jumps: int
    source_relation_csr_fingerprint: str
    source_provenance_trailer_fingerprint: str
    source_family_fingerprint: str
    unresolved_source_indices: tuple[int, ...]
    disconnected_image_source_indices: tuple[int, ...]
    disconnected_component_counts: tuple[tuple[int, int], ...]
    witness_source_indices: tuple[int, ...]
    unresolved_chart_counts: tuple[tuple[int, int], ...]
    disconnected_chart_counts: tuple[tuple[int, int], ...]
    witness_chart_counts: tuple[tuple[int, int], ...]
    unresolved_stage_edge_counts: tuple[tuple[str, int], ...]
    witness_bounds_by_chart: tuple[tuple[int, tuple[float, ...]], ...]
    same_chart_ring_parents: tuple[NeuronDyadicCell, ...]
    exact_seam_coface_parents: tuple[NeuronDyadicCell, ...]
    adaptive_family: SpikingNeuronAdaptiveFamily
    logical_tensor_points: int
    unique_tensor_points: int
    unique_tensor_points_by_chart: tuple[tuple[int, int], ...]
    tensor_point_fingerprint: str
    observed_uniform_unique_points: int
    observed_uniform_precompute_seconds: float
    no_dynamics_elapsed_seconds: float

    @property
    def projected_precompute_seconds(self) -> float:
        if self.observed_uniform_unique_points <= 0:
            return float("nan")
        return (
            self.observed_uniform_precompute_seconds
            * self.unique_tensor_points
            / self.observed_uniform_unique_points
        )

    def to_dict(self) -> dict[str, object]:
        unresolved = set(self.unresolved_source_indices)
        disconnected = set(self.disconnected_image_source_indices)
        refined = self.adaptive_family.refined_parents
        cost_gate = bool(
            len(self.adaptive_family.cells) <= 20_000
            and self.unique_tensor_points <= 80_000
            and self.unique_tensor_points * self.t_star / self.max_step
            <= 200_000_000
        )
        return {
            "schema": "spiking-neuron-adaptive-preflight-v1",
            "preflight_revision": ADAPTIVE_PREFLIGHT_REVISION,
            "dynamics_evaluated": False,
            "selection_uses_scc_or_morse_count": False,
            "selection_rule": (
                "depth-14 sources with unresolved stage or quotient-disconnected "
                "image, one closed same-chart ring, then positive-length exact "
                "guard/reset face cofaces; replace selected leaves by all axis-8 "
                "children"
            ),
            "t_star": self.t_star,
            "source_total_depth": self.source_total_depth,
            "source_axis_depth": self.source_axis_depth,
            "fine_axis_depth": self.fine_axis_depth,
            "samples_per_axis": self.samples_per_axis,
            "padding_cells": 1.0,
            "max_step": self.max_step,
            "max_jumps": self.max_jumps,
            "source_artifacts": {
                "relation_csr_fingerprint": self.source_relation_csr_fingerprint,
                "provenance_trailer_fingerprint": (
                    self.source_provenance_trailer_fingerprint
                ),
                "family_fingerprint": self.source_family_fingerprint,
            },
            "witnesses": {
                "unresolved_source_indices": list(self.unresolved_source_indices),
                "disconnected_image_source_indices": list(
                    self.disconnected_image_source_indices
                ),
                "overlap_source_indices": sorted(unresolved & disconnected),
                "union_source_indices": list(self.witness_source_indices),
                "counts": {
                    "unresolved": len(unresolved),
                    "disconnected_image": len(disconnected),
                    "overlap": len(unresolved & disconnected),
                    "union": len(self.witness_source_indices),
                },
                "disconnected_component_count_histogram": {
                    str(components): sources
                    for components, sources in self.disconnected_component_counts
                },
                "unresolved_chart_counts": {
                    str(chart): count for chart, count in self.unresolved_chart_counts
                },
                "disconnected_image_chart_counts": {
                    str(chart): count for chart, count in self.disconnected_chart_counts
                },
                "union_chart_counts": {
                    str(chart): count for chart, count in self.witness_chart_counts
                },
                "unresolved_stage_edge_histogram": dict(
                    self.unresolved_stage_edge_counts
                ),
                "union_bounds_envelope_by_chart": {
                    str(chart): list(bounds)
                    for chart, bounds in self.witness_bounds_by_chart
                },
            },
            "refinement": {
                "same_chart_closed_ring_parent_count": len(
                    self.same_chart_ring_parents
                ),
                "exact_seam_coface_parent_count": len(
                    self.exact_seam_coface_parents
                ),
                "new_exact_seam_coface_parent_count": len(
                    set(self.exact_seam_coface_parents).difference(
                        self.same_chart_ring_parents
                    )
                ),
                "seam_cofaces_already_in_ring": len(
                    set(self.exact_seam_coface_parents).intersection(
                        self.same_chart_ring_parents
                    )
                ),
                "refined_parent_count": len(refined),
                "refined_parent_chart_counts": {
                    str(chart): count
                    for chart, count in sorted(Counter(c.chart_id for c in refined).items())
                },
                "refined_parent_addresses": [
                    _tagged_address(cell) for cell in refined
                ],
                "coarse_leaves_retained": (
                    len(build_spiking_neuron_active_family(self.source_axis_depth).cells)
                    - len(refined)
                ),
                "fine_children_inserted": 4 * len(refined),
                "adaptive_leaf_cells": len(self.adaptive_family.cells),
                "adaptive_leaf_chart_counts": {
                    str(chart): count
                    for chart, count in sorted(
                        self.adaptive_family.chart_counts().items()
                    )
                },
                "adaptive_leaf_depth_counts": {
                    str(depth): count
                    for depth, count in sorted(
                        self.adaptive_family.depth_counts().items()
                    )
                },
                "adaptive_family_fingerprint": self.adaptive_family.fingerprint,
            },
            "exact_sample_cost": {
                "logical_tensor_points": self.logical_tensor_points,
                "unique_tensor_points": self.unique_tensor_points,
                "duplicate_tensor_points": (
                    self.logical_tensor_points - self.unique_tensor_points
                ),
                "unique_tensor_points_by_chart": {
                    str(chart): count
                    for chart, count in self.unique_tensor_points_by_chart
                },
                "tensor_point_fingerprint": self.tensor_point_fingerprint,
                "integration_horizon_over_max_step_units": (
                    self.unique_tensor_points * self.t_star / self.max_step
                ),
            },
            "fixed_caps": {
                "max_relation_edges": 50_000_000,
                "max_native_cache_bytes": 1 << 30,
                "max_csr_payload_bytes": 1 << 30,
                "preflight_active_cell_limit": 20_000,
                "preflight_unique_endpoint_limit": 80_000,
                "preflight_horizon_over_max_step_limit": 200_000_000,
            },
            "cost_gate_passed": cost_gate,
            "timing_calibration": {
                "observed_uniform_unique_points": self.observed_uniform_unique_points,
                "observed_uniform_precompute_seconds": (
                    self.observed_uniform_precompute_seconds
                ),
                "linear_projected_precompute_seconds_not_a_guarantee": (
                    self.projected_precompute_seconds
                ),
                "no_dynamics_preflight_seconds": self.no_dynamics_elapsed_seconds,
            },
            "launch_authorized_by_preflight": False,
        }


def _adaptive_tensor_point_cost(
    family: SpikingNeuronAdaptiveFamily,
    *,
    samples_per_axis: int,
) -> tuple[int, int, tuple[tuple[int, int], ...], str]:
    if samples_per_axis < 3:
        raise ValueError("adaptive tensor sampling requires at least three points")
    keys: set[tuple[object, ...]] = set()
    by_chart: Counter[int] = Counter()
    for cell in family.cells:
        bounds = spiking_neuron_dyadic_bounds(cell)
        axes = (
            np.linspace(bounds[0], bounds[2], samples_per_axis),
            np.linspace(bounds[1], bounds[3], samples_per_axis),
        )
        for first, second in itertools.product(*axes):
            key = _point_key(cell.chart_id, (first, second))
            if key not in keys:
                keys.add(key)
                by_chart[cell.chart_id] += 1
    encoded_keys = [list(key) for key in sorted(keys)]
    return (
        len(family.cells) * samples_per_axis**2,
        len(keys),
        tuple(sorted(by_chart.items())),
        _sha256(encoded_keys),
    )


def spiking_neuron_tensor_sample_cost(
    family: SpikingNeuronAdaptiveFamily,
    *,
    samples_per_axis: int,
) -> dict[str, object]:
    """Return the exact no-dynamics tensor-node cost for a mixed family."""

    logical, unique, by_chart, fingerprint = _adaptive_tensor_point_cost(
        family,
        samples_per_axis=samples_per_axis,
    )
    return {
        "logical_tensor_points": logical,
        "unique_tensor_points": unique,
        "duplicate_tensor_points": logical - unique,
        "unique_tensor_points_by_chart": {
            str(chart): count for chart, count in by_chart
        },
        "tensor_point_fingerprint": fingerprint,
    }


def build_spiking_neuron_adaptive_preflight(
    summary_path: str | Path,
    provenance_path: str | Path,
    *,
    fine_axis_depth: int = 8,
) -> SpikingNeuronAdaptivePreflight:
    """Derive the fixed error-driven mixed family without evaluating dynamics."""

    started = time.perf_counter()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    relation_csr = summary.get("relation_csr")
    if not isinstance(relation_csr, Mapping) or not isinstance(
        relation_csr.get("fingerprint"), str
    ):
        raise ValueError("adaptive preflight requires a fingerprinted CSR relation")
    provenance = validate_spiking_neuron_provenance(
        provenance_path,
        expected_relation_csr_fingerprint=str(relation_csr["fingerprint"]),
    )
    axis_depth = int(summary["axis_depth"])
    if axis_depth != 7 or fine_axis_depth != 8:
        raise ValueError("the frozen adaptive rule is depth-14 to axis-depth 8")
    if int(summary["samples_per_axis"]) != 3:
        raise ValueError("the adaptive source relation must use frozen 3x3 sampling")

    source_payloads = []
    with gzip.open(provenance_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("record") == "source":
                source_payloads.append(payload)
    if [int(payload["index"]) for payload in source_payloads] != list(
        range(len(source_payloads))
    ):
        raise ValueError("neuron provenance source indices are not contiguous")
    cells = tuple(
        AtlasNeuronCell(
            int(payload["index"]),
            int(payload["chart_id"]),
            tuple(float(value) for value in payload["bounds"]),
        )
        for payload in source_payloads
    )
    if len(cells) != int(summary["active_cells"]):
        raise ValueError("neuron provenance is missing active sources")
    incidence = SpikingNeuronSparseQuotientIncidence(
        cells, spiking_neuron_atlas_charts(), axis_depth=axis_depth
    )
    unresolved = tuple(
        cell.index
        for cell, payload in zip(cells, source_payloads, strict=True)
        if payload["unresolved_stage_edges"]
    )
    disconnected: list[int] = []
    component_histogram: Counter[int] = Counter()
    for cell, payload in zip(cells, source_payloads, strict=True):
        targets = tuple(cells[int(index)] for index in payload["targets"])
        components = incidence.connected_components(targets)
        if len(components) > 1:
            disconnected.append(cell.index)
            component_histogram[len(components)] += 1
    witness_indices = tuple(sorted(set(unresolved).union(disconnected)))
    unresolved_charts = Counter(cells[index].chart_id for index in unresolved)
    disconnected_charts = Counter(cells[index].chart_id for index in disconnected)
    witness_charts = Counter(cells[index].chart_id for index in witness_indices)
    stage_edges: Counter[str] = Counter(
        f"{int(edge[0])}->{int(edge[1])}"
        for index in unresolved
        for edge in source_payloads[index]["unresolved_stage_edges"]
    )
    witness_bounds = []
    for chart_id in sorted(witness_charts):
        chart_cells = tuple(
            cells[index]
            for index in witness_indices
            if cells[index].chart_id == chart_id
        )
        lowers = np.asarray([cell.lower for cell in chart_cells])
        uppers = np.asarray([cell.upper for cell in chart_cells])
        witness_bounds.append(
            (
                chart_id,
                tuple(
                    float(value)
                    for value in np.concatenate(
                        (np.min(lowers, axis=0), np.max(uppers, axis=0))
                    )
                ),
            )
        )

    coarse = build_spiking_neuron_active_family(axis_depth)
    dyadic_by_key = {
        _source_key(cell.chart_id, spiking_neuron_dyadic_bounds(cell)): cell
        for cell in coarse.cells
    }
    dyadic_by_index = tuple(
        dyadic_by_key[_source_key(cell.chart_id, cell.bounds)] for cell in cells
    )
    address_map = {
        (cell.chart_id, *cell.coordinates): cell for cell in coarse.cells
    }
    ring: set[NeuronDyadicCell] = set()
    for index in witness_indices:
        cell = dyadic_by_index[index]
        first, second = cell.coordinates
        ring.update(
            candidate
            for delta_first, delta_second in itertools.product((-1, 0, 1), repeat=2)
            if (
                candidate := address_map.get(
                    (
                        cell.chart_id,
                        first + delta_first,
                        second + delta_second,
                    )
                )
            )
            is not None
        )
    seam_cofaces = set().union(
        *(
            _exact_uniform_seam_cofaces(cell, cell_by_address=address_map)
            for cell in ring
        )
    ) if ring else set()
    refined = ring.union(seam_cofaces)
    family = build_spiking_neuron_adaptive_family(
        coarse_axis_depth=axis_depth,
        fine_axis_depth=fine_axis_depth,
        refined_parents=refined,
    )
    logical, unique, unique_by_chart, point_fingerprint = _adaptive_tensor_point_cost(
        family, samples_per_axis=int(summary["samples_per_axis"])
    )
    precompute = summary.get("endpoint_precompute")
    if not isinstance(precompute, Mapping):
        raise ValueError("source summary lacks endpoint precompute calibration")
    return SpikingNeuronAdaptivePreflight(
        t_star=float(summary["t_star"]),
        source_total_depth=int(summary["total_depth"]),
        source_axis_depth=axis_depth,
        fine_axis_depth=int(fine_axis_depth),
        samples_per_axis=int(summary["samples_per_axis"]),
        max_step=float(summary["max_step"]),
        max_jumps=int(summary["max_jumps"]),
        source_relation_csr_fingerprint=str(relation_csr["fingerprint"]),
        source_provenance_trailer_fingerprint=str(
            provenance["trailer_fingerprint"]
        ),
        source_family_fingerprint=str(summary["family_fingerprint"]),
        unresolved_source_indices=tuple(unresolved),
        disconnected_image_source_indices=tuple(disconnected),
        disconnected_component_counts=tuple(sorted(component_histogram.items())),
        witness_source_indices=witness_indices,
        unresolved_chart_counts=tuple(sorted(unresolved_charts.items())),
        disconnected_chart_counts=tuple(sorted(disconnected_charts.items())),
        witness_chart_counts=tuple(sorted(witness_charts.items())),
        unresolved_stage_edge_counts=tuple(sorted(stage_edges.items())),
        witness_bounds_by_chart=tuple(witness_bounds),
        same_chart_ring_parents=tuple(sorted(ring)),
        exact_seam_coface_parents=tuple(sorted(seam_cofaces)),
        adaptive_family=family,
        logical_tensor_points=logical,
        unique_tensor_points=unique,
        unique_tensor_points_by_chart=unique_by_chart,
        tensor_point_fingerprint=point_fingerprint,
        observed_uniform_unique_points=int(precompute["unique_points"]),
        observed_uniform_precompute_seconds=float(precompute["elapsed_seconds"]),
        no_dynamics_elapsed_seconds=time.perf_counter() - started,
    )


def load_spiking_neuron_adaptive_preflight(
    path: str | Path,
) -> tuple[dict[str, object], SpikingNeuronAdaptiveFamily]:
    """Strictly load a frozen preflight and reconstruct its complete family."""

    preflight_path = Path(path)
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "spiking-neuron-adaptive-preflight-v1":
        raise ValueError("unknown neuron adaptive preflight schema")
    if payload.get("preflight_revision") != ADAPTIVE_PREFLIGHT_REVISION:
        raise ValueError("unknown neuron adaptive preflight revision")
    if payload.get("dynamics_evaluated") is not False:
        raise ValueError("adaptive preflight must be a no-dynamics artifact")
    if payload.get("selection_uses_scc_or_morse_count") is not False:
        raise ValueError("adaptive family may not be selected by an SCC count")
    claimed_fingerprint = payload.get("fingerprint")
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("fingerprint", None)
    if claimed_fingerprint != _sha256(fingerprint_payload):
        raise ValueError("adaptive preflight fingerprint mismatch")
    source_paths = payload.get("source_paths")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_paths, Mapping) or not isinstance(
        source_artifacts, Mapping
    ):
        raise ValueError("adaptive preflight lacks bound source artifacts")
    source_summary_path = (preflight_path.parent / str(source_paths["summary"])).resolve()
    source_provenance_path = (
        preflight_path.parent / str(source_paths["provenance"])
    ).resolve()
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_relation = source_summary.get("relation_csr")
    if (
        not isinstance(source_relation, Mapping)
        or source_relation.get("fingerprint")
        != source_artifacts.get("relation_csr_fingerprint")
        or source_summary.get("family_fingerprint")
        != source_artifacts.get("family_fingerprint")
    ):
        raise ValueError("adaptive preflight source summary binding mismatch")
    source_provenance = validate_spiking_neuron_provenance(
        source_provenance_path,
        expected_relation_csr_fingerprint=str(source_relation["fingerprint"]),
    )
    if source_provenance.get("trailer_fingerprint") != source_artifacts.get(
        "provenance_trailer_fingerprint"
    ):
        raise ValueError("adaptive preflight source provenance binding mismatch")
    refinement = payload.get("refinement")
    sample_cost = payload.get("exact_sample_cost")
    if not isinstance(refinement, Mapping) or not isinstance(sample_cost, Mapping):
        raise ValueError("adaptive preflight lacks family or sample metadata")
    parents = tuple(
        NeuronDyadicCell(
            int(address[0]),
            int(address[1]),
            (int(address[2][0]), int(address[2][1])),
        )
        for address in refinement["refined_parent_addresses"]
    )
    family = build_spiking_neuron_adaptive_family(
        coarse_axis_depth=int(payload["source_axis_depth"]),
        fine_axis_depth=int(payload["fine_axis_depth"]),
        refined_parents=parents,
    )
    if family.fingerprint != refinement.get("adaptive_family_fingerprint"):
        raise ValueError("adaptive preflight family fingerprint mismatch")
    if len(family.cells) != int(refinement["adaptive_leaf_cells"]):
        raise ValueError("adaptive preflight family size mismatch")
    logical, unique, _by_chart, point_fingerprint = _adaptive_tensor_point_cost(
        family, samples_per_axis=int(payload["samples_per_axis"])
    )
    if (
        logical != int(sample_cost["logical_tensor_points"])
        or unique != int(sample_cost["unique_tensor_points"])
        or point_fingerprint != sample_cost.get("tensor_point_fingerprint")
    ):
        raise ValueError("adaptive preflight tensor sample fingerprint mismatch")
    return payload, family


@dataclass(frozen=True)
class NeuronClippingRecord:
    source_chart_id: int
    source_bounds: tuple[float, ...]
    raw_pieces: int
    clipped_pieces: int
    pieces_without_active_cover: int
    nonglued_open_boundary_pieces: int


class NeuronSuspensionBoxMap(CMGDBSuspensionBoxMap):
    """Cached BoxMap whose finite target union is intersected with active ``N``.

    A base rectangle is intersected independently with the left arm and lower
    right arm.  It is never replaced by a convex hull across the missing
    upper-right notch.  Padding outside the compact restriction is discarded,
    matching the requested CMGDB restricted-domain convention.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._point_lock = threading.Lock()
        self._point_cache: dict[tuple[object, ...], tuple[bool, object]] = {}
        self._point_cache_hits = 0
        self._point_cache_misses = 0
        self._precomputed_unclaimed: set[tuple[object, ...]] = set()
        self._source_lock = threading.Lock()
        self._source_cache: dict[
            tuple[object, ...], tuple[tuple[int, tuple[float, ...]], ...]
        ] = {}
        self._source_records: dict[tuple[object, ...], SourceBoxMapDiagnostic] = {}
        self._clipping_records: dict[tuple[object, ...], NeuronClippingRecord] = {}
        self._atlas: Any | None = None

    def bind_atlas(self, atlas: Any) -> None:
        self._atlas = atlas

    def seed_point_cache(
        self,
        entries: Sequence[tuple[tuple[object, ...], tuple[bool, object]]],
    ) -> None:
        seeded = {tuple(key): value for key, value in entries}
        if len(seeded) != len(entries):
            raise ValueError("endpoint precompute contains duplicate keys")
        with self._point_lock:
            if self._point_cache:
                raise RuntimeError("endpoint cache must be empty before seeding")
            self._point_cache = seeded
            self._point_cache_misses = len(seeded)
            self._precomputed_unclaimed = set(seeded)

    def point_cache_counts(self) -> tuple[int, int]:
        with self._point_lock:
            return self._point_cache_misses, self._point_cache_hits

    def clipping_records(self) -> tuple[NeuronClippingRecord, ...]:
        with self._source_lock:
            return tuple(self._clipping_records[key] for key in sorted(self._clipping_records))

    def source_records(self) -> Mapping[tuple[object, ...], SourceBoxMapDiagnostic]:
        with self._source_lock:
            return dict(self._source_records)

    def _evaluate_point(self, source_chart_id: int, point: np.ndarray):
        key = _point_key(source_chart_id, point)
        with self._point_lock:
            cached = self._point_cache.get(key)
            if cached is not None:
                if key in self._precomputed_unclaimed:
                    self._precomputed_unclaimed.remove(key)
                else:
                    self._point_cache_hits += 1
                succeeded, value = cached
                if succeeded:
                    return value
                error_type, error_args = value
                raise error_type(*error_args)
            self._point_cache_misses += 1
        try:
            value = super()._evaluate_point(source_chart_id, point)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            with self._point_lock:
                self._point_cache[key] = (False, (type(error), error.args))
            raise
        with self._point_lock:
            self._point_cache[key] = (True, value)
        return value

    def _record(self, record: SourceBoxMapDiagnostic) -> None:
        with self._source_lock:
            self._source_records[_source_key(record.source_chart_id, record.source_bounds)] = record
        super()._record(record)

    @staticmethod
    def _intersection_piece(
        chart_id: int,
        lower: np.ndarray,
        upper: np.ndarray,
        domain_lower: Sequence[float],
        domain_upper: Sequence[float],
    ) -> tuple[int, list[float]] | None:
        clipped_lower = np.maximum(lower, np.asarray(domain_lower, dtype=np.float64))
        clipped_upper = np.minimum(upper, np.asarray(domain_upper, dtype=np.float64))
        if np.any(clipped_lower > clipped_upper + 1.0e-12):
            return None
        return (
            int(chart_id),
            [float(value) for value in np.concatenate((clipped_lower, clipped_upper))],
        )

    def _clip_piece(
        self, chart_id: int, bounds: Sequence[float]
    ) -> list[tuple[int, list[float]]]:
        values = np.asarray(bounds, dtype=np.float64)
        lower, upper = values[:2], values[2:]
        domains = (
            (
                ((V_MIN, U_MIN), (V_NOTCH, U_MAX)),
                ((V_NOTCH, U_MIN), (V_PEAK, U_NOTCH)),
            )
            if chart_id == BASE_CHART_ID
            else (((U_MIN, 0.0), (U_NOTCH, 1.0)),)
        )
        result = []
        for domain_lower, domain_upper in domains:
            piece = self._intersection_piece(
                chart_id, lower, upper, domain_lower, domain_upper
            )
            if piece is not None:
                result.append(piece)
        unique = {
            (tag, *(round(value, 14) for value in piece_bounds)): (tag, piece_bounds)
            for tag, piece_bounds in result
        }
        return list(unique.values())

    @staticmethod
    def _has_nonglued_exit(chart_id: int, bounds: Sequence[float]) -> bool:
        values = np.asarray(bounds, dtype=np.float64)
        lower, upper = values[:2], values[2:]
        atol = 1.0e-12
        if chart_id == HANDLE_CHART_ID:
            return bool(lower[0] < U_MIN - atol or upper[0] > U_NOTCH + atol)
        if lower[0] < V_MIN - atol or lower[1] < U_MIN - atol:
            return True
        if upper[1] > U_MAX + atol:
            return True
        # Above u=160 only the left arm is physical.  This is an actual open
        # boundary of N, not a missing active cell.  The guard side v>35 is
        # glued and therefore deliberately not counted here.
        if upper[1] > U_NOTCH + atol and upper[0] > V_NOTCH + atol:
            return True
        return False

    def __call__(
        self, source_chart_id: int, source_bounds: Sequence[float]
    ) -> list[tuple[int, list[float]]]:
        key = _source_key(source_chart_id, source_bounds)
        with self._source_lock:
            cached = self._source_cache.get(key)
        if cached is not None:
            return [(tag, list(bounds)) for tag, bounds in cached]

        raw = super().__call__(source_chart_id, source_bounds)
        clipped = [
            piece
            for chart_id, bounds in raw
            for piece in self._clip_piece(chart_id, bounds)
        ]
        unique = {
            (tag, *(round(value, 14) for value in bounds)): (tag, bounds)
            for tag, bounds in clipped
        }
        clipped = list(unique.values())
        uncovered = 0
        if self._atlas is not None:
            uncovered = sum(
                not bool(self._atlas.cover(tag, bounds)) for tag, bounds in clipped
            )
        record = NeuronClippingRecord(
            source_chart_id=int(source_chart_id),
            source_bounds=tuple(float(value) for value in source_bounds),
            raw_pieces=len(raw),
            clipped_pieces=len(clipped),
            pieces_without_active_cover=int(uncovered),
            nonglued_open_boundary_pieces=sum(
                self._has_nonglued_exit(tag, bounds) for tag, bounds in raw
            ),
        )
        encoded = tuple(
            (int(tag), tuple(float(value) for value in bounds))
            for tag, bounds in clipped
        )
        with self._source_lock:
            self._source_cache[key] = encoded
            self._clipping_records[key] = record
        return [(tag, list(bounds)) for tag, bounds in encoded]


@dataclass(frozen=True)
class SpikingNeuronAtlasSetup:
    neuron: SpikingNeuron
    charts: SuspensionAtlasCharts
    family: SpikingNeuronActiveFamily | SpikingNeuronAdaptiveFamily
    box_map: NeuronSuspensionBoxMap
    model: Any


def _validate_total_depth(total_depth: int) -> int:
    if isinstance(total_depth, bool) or not isinstance(total_depth, (int, np.integer)):
        raise ValueError("total_depth must be an integer")
    if total_depth < 12 or total_depth % 2:
        raise ValueError("neuron total depth must be an even integer at least 12")
    return int(total_depth) // 2


def build_spiking_neuron_atlas_model(
    *,
    total_depth: int = 12,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    max_step: float = DEFAULT_MAX_STEP,
    max_jumps: int = 4,
    diagnostics_limit: int = 100_000,
    active_family: SpikingNeuronAdaptiveFamily | None = None,
    single_handle_bridge: bool = False,
    single_handle_bridge_max_bisections: int = 12,
) -> SpikingNeuronAtlasSetup:
    """Install the complete active ``N`` relation in a native CMGDB Atlas."""

    axis_depth = _validate_total_depth(total_depth)
    neuron = SpikingNeuron(max_jumps=max_jumps)
    charts = spiking_neuron_atlas_charts()
    family: SpikingNeuronActiveFamily | SpikingNeuronAdaptiveFamily
    if active_family is None:
        family = build_spiking_neuron_active_family(axis_depth)
    else:
        if active_family.fine_axis_depth != axis_depth:
            raise ValueError(
                "adaptive family fine depth must equal half the declared total depth"
            )
        family = active_family
    box_map = NeuronSuspensionBoxMap(
        neuron.system,
        charts,
        t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        max_jumps=max_jumps,
        max_step=max_step,
        require_domain_path=True,
        diagnostics_limit=diagnostics_limit,
        single_handle_bridge=single_handle_bridge,
        single_handle_bridge_max_bisections=(
            single_handle_bridge_max_bisections
        ),
    )
    model = build_cmgdb_atlas_model(
        box_map,
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    )
    box_map.bind_atlas(model.phaseSpace())
    return SpikingNeuronAtlasSetup(neuron, charts, family, box_map, model)


@dataclass(frozen=True)
class NeuronReferenceCycle:
    post_reset_u: float
    pre_reset_u: float
    flight_time: float
    suspension_period: float
    v_range: tuple[float, float]
    u_range: tuple[float, float]
    solution: Any

    def summary(self) -> dict[str, object]:
        return {
            "post_reset_u": self.post_reset_u,
            "pre_reset_u": self.pre_reset_u,
            "flight_time": self.flight_time,
            "suspension_period": self.suspension_period,
            "v_range": list(self.v_range),
            "u_range": list(self.u_range),
        }


def compute_neuron_reference_cycle() -> NeuronReferenceCycle:
    """Compute a high-accuracy Poincare fixed point independently of BoxMap."""

    neuron = SpikingNeuron(max_jumps=2, rtol=2.0e-12, atol=1.0e-13)

    def flight(post_reset_u: float, *, dense: bool = False):
        result = solve_ivp(
            neuron.system.ode,
            (0.0, 400.0),
            np.asarray((V_RESET, post_reset_u), dtype=np.float64),
            events=neuron.system.event_function,
            dense_output=dense,
            rtol=2.0e-12,
            atol=1.0e-13,
            max_step=0.02,
        )
        if not result.t_events or len(result.t_events[0]) != 1:
            raise RuntimeError("reference neuron flight did not reach the guard once")
        return result

    def return_residual(post_reset_u: float) -> float:
        result = flight(post_reset_u)
        pre_reset_u = float(result.y_events[0][0][1])
        return pre_reset_u + U_RESET_SHIFT - post_reset_u

    post_reset_u = float(brentq(return_residual, -100.0, 180.0, xtol=1.0e-12))
    result = flight(post_reset_u, dense=True)
    pre_reset_u = float(result.y_events[0][0][1])
    flight_time = float(result.t_events[0][0])
    dense_times = np.linspace(0.0, flight_time, 4001)
    dense_states = np.asarray(result.sol(dense_times), dtype=np.float64)
    return NeuronReferenceCycle(
        post_reset_u=post_reset_u,
        pre_reset_u=pre_reset_u,
        flight_time=flight_time,
        suspension_period=flight_time + 1.0,
        v_range=(float(np.min(dense_states[0])), float(np.max(dense_states[0]))),
        u_range=(float(np.min(dense_states[1])), float(np.max(dense_states[1]))),
        solution=result.sol,
    )


def _reference_endpoint_on_cycle(
    cycle: NeuronReferenceCycle,
    source_clock: float,
    t_star: float,
) -> SuspensionSample:
    """Advance one independent cycle point in the unit-handle clock."""

    absolute_target = float(source_clock) + float(t_star)
    completed_periods = int(np.floor(absolute_target / cycle.suspension_period))
    target_clock = absolute_target - completed_periods * cycle.suspension_period
    guard = np.asarray((V_PEAK, cycle.pre_reset_u), dtype=np.float64)
    reset = np.asarray((V_RESET, cycle.post_reset_u), dtype=np.float64)
    if target_clock < cycle.flight_time - 1.0e-11:
        state = np.asarray(cycle.solution(target_clock), dtype=np.float64)
        return BaseSuspensionSample(
            state, t_star, target_clock, completed_periods
        )
    phase = target_clock - cycle.flight_time
    if phase <= 1.0e-11:
        return BaseSuspensionSample(
            guard, t_star, cycle.flight_time, completed_periods
        )
    if phase >= 1.0 - 1.0e-11:
        return BaseSuspensionSample(
            reset, t_star, cycle.flight_time, completed_periods + 1
        )
    return HandleSuspensionSample(
        guard_state=guard,
        reset_state=reset,
        phase=float(phase),
        total_time=t_star,
        continuous_time=cycle.flight_time,
        jump_index=completed_periods,
    )


def _reference_endpoint_from_base(
    cycle: NeuronReferenceCycle,
    base_time: float,
    t_star: float,
) -> SuspensionSample:
    return _reference_endpoint_on_cycle(cycle, float(base_time), t_star)


def _reference_endpoint_from_handle(
    cycle: NeuronReferenceCycle,
    phase: float,
    t_star: float,
) -> SuspensionSample:
    return _reference_endpoint_on_cycle(
        cycle,
        cycle.flight_time + float(phase),
        t_star,
    )


def _atlas_cells(morse_graph: Any, count: int) -> tuple[AtlasNeuronCell, ...]:
    return tuple(
        AtlasNeuronCell(
            index=index,
            chart_id=int(morse_graph.phase_space_chart_box(index)[0]),
            bounds=tuple(
                float(value) for value in morse_graph.phase_space_chart_box(index)[1]
            ),
        )
        for index in range(count)
    )


def _atlas_relation(
    map_graph: Any, cells: Sequence[AtlasNeuronCell]
) -> dict[AtlasNeuronCell, frozenset[AtlasNeuronCell]]:
    return {
        source: frozenset(cells[int(target)] for target in map_graph.adjacencies(index))
        for index, source in enumerate(cells)
    }


def _cells_containing_point(
    cells: Sequence[AtlasNeuronCell],
    chart_id: int,
    coordinates: Sequence[float],
    *,
    atol: float = 1.0e-9,
) -> frozenset[AtlasNeuronCell]:
    point = np.asarray(coordinates, dtype=np.float64)
    located = frozenset(
        cell
        for cell in cells
        if cell.chart_id == chart_id
        and np.all(point >= cell.lower - atol)
        and np.all(point <= cell.upper + atol)
    )
    if not located:
        raise ValueError(f"reference point lies outside active chart {chart_id}")
    return located


def _reference_probes(
    setup: SpikingNeuronAtlasSetup,
    cells: Sequence[AtlasNeuronCell],
    cycle: NeuronReferenceCycle,
    *,
    base_samples: int,
    handle_samples: int,
) -> tuple[tuple[EndpointProbe, ...], Mapping[str, frozenset[AtlasNeuronCell]]]:
    if base_samples < 3 or handle_samples < 3:
        raise ValueError("reference sample counts must be at least three")
    probes: list[EndpointProbe] = []
    sources_by_label: dict[str, frozenset[AtlasNeuronCell]] = {}
    base_times = np.linspace(0.0, cycle.flight_time, base_samples, endpoint=False)
    for index, base_time in enumerate(base_times):
        state = np.asarray(cycle.solution(float(base_time)), dtype=np.float64)
        sources = _cells_containing_point(cells, BASE_CHART_ID, state)
        label = f"reference-base-{index}"
        sources_by_label[label] = sources
        endpoint = _reference_endpoint_from_base(cycle, float(base_time), setup.box_map.t_star)
        probes.extend(
            EndpointProbe(source, endpoint, tuple(float(x) for x in state), label)
            for source in sources
        )
    phases = np.linspace(0.0, 1.0, handle_samples + 2)[1:-1]
    guard = np.asarray((V_PEAK, cycle.pre_reset_u), dtype=np.float64)
    for index, phase in enumerate(phases):
        coordinates = setup.charts.encode_handle(guard, float(phase))
        sources = _cells_containing_point(cells, HANDLE_CHART_ID, coordinates)
        label = f"reference-handle-{index}"
        sources_by_label[label] = sources
        endpoint = _reference_endpoint_from_handle(cycle, float(phase), setup.box_map.t_star)
        probes.extend(
            EndpointProbe(source, endpoint, coordinates, label) for source in sources
        )
    return tuple(probes), sources_by_label


def _audit_reference_endpoints(
    relation: Mapping[AtlasNeuronCell, Collection[AtlasNeuronCell]],
    probes: Sequence[EndpointProbe],
    cells: Sequence[AtlasNeuronCell],
    box_map: NeuronSuspensionBoxMap,
) -> EndpointCoverageAudit:
    witnesses = []
    for probe in probes:
        chart_id, coordinates, _stratum = box_map.encode_endpoint(probe.endpoint)
        containing = _cells_containing_point(cells, chart_id, coordinates)
        witnesses.append(
            EndpointCoverageWitness(
                probe=probe,
                containing_target_cells=containing,
                recorded_targets=frozenset(relation[probe.source]),
            )
        )
    return EndpointCoverageAudit(tuple(witnesses))


def _audit_relation_connectivity(
    relation: Mapping[AtlasNeuronCell, Collection[AtlasNeuronCell]],
    incidence: SpikingNeuronQuotientIncidence,
) -> RelationConnectivityAudit:
    return RelationConnectivityAudit(
        tuple(
            CellSetConnectivity(
                label=source,
                cells=frozenset(targets),
                components=incidence.connected_components(targets),
            )
            for source, targets in sorted(relation.items())
        )
    )


def _morse_node_by_cell(morse_graph: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in morse_graph.vertices():
        for index in morse_graph.morse_set(node):
            if int(index) in result:
                raise ValueError("a phase-space cell belongs to two Morse sets")
            result[int(index)] = int(node)
    return result


def _reference_nodes_covering_every_probe(
    sources_by_label: Mapping[str, Collection[AtlasNeuronCell]],
    node_by_cell: Mapping[int, int],
) -> tuple[int, ...]:
    candidate_sets = []
    for sources in sources_by_label.values():
        candidate_sets.append(
            {node_by_cell[source.index] for source in sources if source.index in node_by_cell}
        )
    if not candidate_sets:
        return ()
    return tuple(sorted(set.intersection(*candidate_sets)))


def _touches_nonglued_boundary(cell: AtlasNeuronCell, *, atol: float = 1.0e-10) -> bool:
    lower, upper = cell.lower, cell.upper
    if cell.chart_id == HANDLE_CHART_ID:
        return bool(lower[0] <= U_MIN + atol or upper[0] >= U_NOTCH - atol)
    v0, u0 = lower
    v1, u1 = upper
    if v0 <= V_MIN + atol or u0 <= U_MIN + atol:
        return True
    if v1 <= V_NOTCH + atol and u1 >= U_MAX - atol:
        return True
    if v0 >= V_NOTCH - atol and u1 >= U_NOTCH - atol:
        return True
    if v0 <= V_NOTCH + atol <= v1 and u0 >= U_NOTCH - atol:
        return True
    return False


@dataclass(frozen=True)
class SpikingNeuronAtlasAcceptance:
    protocol_revision: str
    total_depth: int
    axis_depth: int
    samples_per_axis: int
    max_jumps: int
    t_star: float
    single_handle_bridge_enabled: bool
    single_handle_bridge_max_bisections: int
    elapsed_seconds: float
    analytic_audit: SpikingNeuronAnalyticAudit
    reference_cycle: NeuronReferenceCycle
    family: SpikingNeuronActiveFamily | SpikingNeuronAdaptiveFamily
    morse_graph: Any
    map_graph: Any
    cells: tuple[AtlasNeuronCell, ...]
    relation: Mapping[AtlasNeuronCell, frozenset[AtlasNeuronCell]]
    reference_endpoint_audit: EndpointCoverageAudit
    reference_nodes: tuple[int, ...]
    reference_node_hits: Mapping[int, int]
    image_connectivity_audit: RelationConnectivityAudit
    morse_support_connectivity: Mapping[int, CellSetConnectivity]
    sparse_incidence_parity: SparseIncidenceParityAudit | MixedIncidenceParityAudit
    empty_sources: tuple[AtlasNeuronCell, ...]
    failed_source_indices: tuple[int, ...]
    jump_limit_failure_source_indices: tuple[int, ...]
    raw_unresolved_source_indices: tuple[int, ...]
    unresolved_source_indices: tuple[int, ...]
    single_handle_bridge_source_indices: tuple[int, ...]
    failed_single_handle_bridge_source_indices: tuple[int, ...]
    missing_source_indices: tuple[int, ...]
    open_source_indices: tuple[int, ...]
    reference_open_boundary_indices: tuple[int, ...]
    box_map_diagnostics: SuspensionBoxMapDiagnostics
    clipping_records: tuple[NeuronClippingRecord, ...]
    endpoint_unique_evaluations: int
    endpoint_cache_reuses: int
    connectivity_algorithm: str = "exact_sparse_dyadic_plus_affine_seams"
    grid_kind: str = "uniform"
    source_total_depth: int | None = None
    endpoint_precompute: Mapping[str, object] | None = None
    relation_csr: Mapping[str, object] | None = None
    adaptive_preflight: Mapping[str, object] | None = None

    @property
    def morse_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted((int(a), int(b)) for a, b in self.morse_graph.edges())
        )

    @property
    def recurrent_source_indices(self) -> frozenset[int]:
        return frozenset(
            int(index)
            for node in self.morse_graph.vertices()
            for index in self.morse_graph.morse_set(node)
        )

    @property
    def reference_recurrent_source_indices(self) -> frozenset[int]:
        return frozenset(
            int(index)
            for node in self.reference_nodes
            for index in self.morse_graph.morse_set(node)
        )

    @property
    def recurrent_problem_indices(self) -> tuple[int, ...]:
        problems = (
            set(self.failed_source_indices)
            | set(self.unresolved_source_indices)
            | set(self.missing_source_indices)
            | set(self.open_source_indices)
            | {cell.index for cell in self.empty_sources}
        )
        return tuple(sorted(problems & set(self.recurrent_source_indices)))

    @property
    def reference_problem_indices(self) -> tuple[int, ...]:
        problems = (
            set(self.failed_source_indices)
            | set(self.unresolved_source_indices)
            | set(self.missing_source_indices)
            | set(self.open_source_indices)
            | {cell.index for cell in self.empty_sources}
        )
        return tuple(
            sorted(problems & set(self.reference_recurrent_source_indices))
        )

    @property
    def stage_gates_passed(self) -> bool:
        return bool(
            self.analytic_audit.passed
            and self.reference_endpoint_audit.passed
            and self.reference_nodes
            and self.image_connectivity_audit.passed
            and self.sparse_incidence_parity.passed
            and all(audit.connected for audit in self.morse_support_connectivity.values())
            and not self.jump_limit_failure_source_indices
            and not self.unresolved_source_indices
            and not self.failed_single_handle_bridge_source_indices
            and not self.reference_problem_indices
            and not self.reference_open_boundary_indices
        )

    def summary(self) -> dict[str, object]:
        morse_sizes = {
            str(int(node)): {
                "base": sum(
                    self.cells[int(index)].chart_id == BASE_CHART_ID
                    for index in self.morse_graph.morse_set(node)
                ),
                "handle": sum(
                    self.cells[int(index)].chart_id == HANDLE_CHART_ID
                    for index in self.morse_graph.morse_set(node)
                ),
                "total": len(tuple(self.morse_graph.morse_set(node))),
                "quotient_connected": self.morse_support_connectivity[int(node)].connected,
            }
            for node in self.morse_graph.vertices()
        }
        return {
            "schema": "spiking-neuron-atlas-acceptance-v1",
            "relation_stage_completed": True,
            "acceptance_audit_completed": True,
            "protocol_revision": self.protocol_revision,
            "boxmap_revision": BOXMAP_REVISION,
            "model": "compact_quadratic_integrate_and_fire",
            "formal_domain": "compact_L_shaped_N",
            "guard": "v=35, -300<=u<=160, increasing crossings",
            "reset": "(35,u)->(-50,u+100)",
            "total_depth": self.total_depth,
            "axis_depth": self.axis_depth,
            "subdivisions_per_axis": 2**self.axis_depth,
            "grid_kind": self.grid_kind,
            "source_total_depth": self.source_total_depth,
            "samples_per_axis": self.samples_per_axis,
            "padding_cells": 1.0,
            "t_star": self.t_star,
            "max_step": DEFAULT_MAX_STEP,
            "max_jumps": self.max_jumps,
            "elapsed_seconds": self.elapsed_seconds,
            "active_cells": len(self.cells),
            "active_chart_counts": self.family.chart_counts(),
            "active_leaf_depth_counts": self.family.depth_counts(),
            "family_fingerprint": self.family.fingerprint,
            "relation_edges": sum(len(targets) for targets in self.relation.values()),
            "morse_nodes": int(self.morse_graph.num_vertices()),
            "morse_edges": [list(edge) for edge in self.morse_edges],
            "morse_set_cell_counts": morse_sizes,
            "reference_cycle": self.reference_cycle.summary(),
            "reference_nodes_containing_complete_cycle": list(self.reference_nodes),
            "reference_node_hits": dict(self.reference_node_hits),
            "reference_endpoint_probes": len(self.reference_endpoint_audit.witnesses),
            "reference_endpoint_misses": len(self.reference_endpoint_audit.missed),
            "reference_endpoint_audit_passed": self.reference_endpoint_audit.passed,
            "nonempty_images": len(self.image_connectivity_audit.images),
            "connectivity_algorithm": self.connectivity_algorithm,
            "disconnected_images": len(self.image_connectivity_audit.disconnected),
            "all_morse_supports_quotient_connected": all(
                audit.connected for audit in self.morse_support_connectivity.values()
            ),
            "sparse_incidence_parity": self.sparse_incidence_parity.to_dict(),
            "empty_sources": len(self.empty_sources),
            "failed_sources": len(self.failed_source_indices),
            "jump_limit_failure_sources": len(
                self.jump_limit_failure_source_indices
            ),
            "raw_unresolved_sources": len(self.raw_unresolved_source_indices),
            "unresolved_sources": len(self.unresolved_source_indices),
            "single_handle_bridge_enabled": self.single_handle_bridge_enabled,
            "single_handle_bridge_max_bisections": (
                self.single_handle_bridge_max_bisections
            ),
            "single_handle_bridge_sources": len(
                self.single_handle_bridge_source_indices
            ),
            "failed_single_handle_bridge_sources": len(
                self.failed_single_handle_bridge_source_indices
            ),
            "single_handle_bridge_diagnostics": {
                "algorithm_revision": SINGLE_HANDLE_BRIDGE_ALGORITHM,
                "raw_unresolved_stage_edges": (
                    self.box_map_diagnostics.raw_unresolved_stage_edges
                ),
                "residual_unresolved_stage_edges": (
                    self.box_map_diagnostics.unresolved_stage_edges
                ),
                "attempts": self.box_map_diagnostics.single_handle_bridge_attempts,
                "synthesized": (
                    self.box_map_diagnostics.synthesized_single_handle_bridges
                ),
                "probe_points": (
                    self.box_map_diagnostics.single_handle_bridge_probe_points
                ),
                "failed_probes": (
                    self.box_map_diagnostics.single_handle_bridge_failed_probes
                ),
                "terminal_image_carrier_only": True,
                "intermediate_time_graph_edges_added": False,
            },
            "missing_sources": len(self.missing_source_indices),
            "open_sources": len(self.open_source_indices),
            "recurrent_problem_sources": len(self.recurrent_problem_indices),
            "reference_recurrent_problem_sources": len(
                self.reference_problem_indices
            ),
            "reference_node_open_boundary_cells": len(
                self.reference_open_boundary_indices
            ),
            "endpoint_unique_evaluations": self.endpoint_unique_evaluations,
            "endpoint_cache_reuses": self.endpoint_cache_reuses,
            "endpoint_precompute": (
                None if self.endpoint_precompute is None else dict(self.endpoint_precompute)
            ),
            "relation_csr": None if self.relation_csr is None else dict(self.relation_csr),
            "adaptive_preflight": (
                None
                if self.adaptive_preflight is None
                else dict(self.adaptive_preflight)
            ),
            "analytic_compact_domain_audit": self.analytic_audit.to_dict(),
            "dwell_handle_jump_bound": {
                "formula": "ceil(t_star/(1+dwell_time_lower_bound))",
                "required_upper_bound": int(
                    np.ceil(
                        self.t_star
                        / (1.0 + self.analytic_audit.dwell_time_lower_bound)
                    )
                ),
                "configured_max_jumps": self.max_jumps,
                "configured_limit_is_safe": self.max_jumps
                >= int(
                    np.ceil(
                        self.t_star
                        / (1.0 + self.analytic_audit.dwell_time_lower_bound)
                    )
                ),
            },
            "stage_gates_passed": self.stage_gates_passed,
            "scc_count_used_as_acceptance_gate": False,
            "analytic_conley_label_attached": False,
            "whole_cell_outer_enclosure_certified": False,
            "numerically_rigorous_outer_approximation_claimed": False,
            "finite_relation_conley_index_computed": False,
        }


def compute_spiking_neuron_atlas_acceptance(
    *,
    total_depth: int = 12,
    samples_per_axis: int = 3,
    t_star: float = DEFAULT_T_STAR,
    padding_cells: float = 1.0,
    max_step: float = DEFAULT_MAX_STEP,
    max_jumps: int = 4,
    reference_base_samples: int = 129,
    reference_handle_samples: int = 65,
    precomputed_entries: Sequence[
        tuple[tuple[object, ...], tuple[bool, object]]
    ] = (),
    endpoint_precompute: Mapping[str, object] | None = None,
    protocol_revision: str = PROTOCOL_REVISION,
    max_relation_edges: int = 50_000_000,
    max_native_cache_bytes: int = 1 << 30,
    active_family: SpikingNeuronAdaptiveFamily | None = None,
    source_total_depth: int | None = None,
    adaptive_preflight: Mapping[str, object] | None = None,
    single_handle_bridge: bool = False,
    single_handle_bridge_max_bisections: int = 12,
) -> SpikingNeuronAtlasAcceptance:
    """Compute one complete active-domain relation and all per-stage audits."""

    started = time.perf_counter()
    setup = build_spiking_neuron_atlas_model(
        total_depth=total_depth,
        t_star=t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        max_step=max_step,
        max_jumps=max_jumps,
        active_family=active_family,
        single_handle_bridge=single_handle_bridge,
        single_handle_bridge_max_bisections=(
            single_handle_bridge_max_bisections
        ),
    )
    if precomputed_entries:
        setup.box_map.seed_point_cache(precomputed_entries)
    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("the local CMGDB fork is not installed") from error

    cap_values = {
        "CMGDB_MAPGRAPH_HARD_MAX_VERTICES": len(setup.family.cells),
        "CMGDB_MAPGRAPH_HARD_MAX_EDGES": int(max_relation_edges),
        "CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES": int(max_native_cache_bytes),
    }
    previous_caps = {key: os.environ.get(key) for key in cap_values}
    try:
        for key, value in cap_values.items():
            os.environ[key] = str(value)
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(setup.model)
    finally:
        for key, previous in previous_caps.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    cells = _atlas_cells(morse_graph, int(map_graph.num_vertices()))
    relation = _atlas_relation(map_graph, cells)
    if isinstance(setup.family, SpikingNeuronAdaptiveFamily):
        incidence: SpikingNeuronSparseQuotientIncidence | SpikingNeuronMixedQuotientIncidence
        incidence = SpikingNeuronMixedQuotientIncidence(
            cells,
            setup.charts,
            fine_axis_depth=setup.family.fine_axis_depth,
        )
        sparse_parity: SparseIncidenceParityAudit | MixedIncidenceParityAudit
        sparse_parity = audit_spiking_neuron_mixed_incidence_parity(
            cells,
            setup.charts,
            fine_axis_depth=setup.family.fine_axis_depth,
        )
        connectivity_algorithm = (
            "exact_sparse_mixed_dyadic_plus_affine_seams_with_full_geometric_parity"
        )
    else:
        incidence = SpikingNeuronSparseQuotientIncidence(
            cells, setup.charts, axis_depth=setup.family.axis_depth
        )
        sparse_parity = audit_spiking_neuron_sparse_incidence_parity(
            cells, setup.charts, axis_depth=setup.family.axis_depth
        )
        connectivity_algorithm = "exact_sparse_dyadic_plus_affine_seams"
    if not sparse_parity.passed:
        raise RuntimeError("sparse neuron quotient incidence failed exact parity")
    nonempty = {source: targets for source, targets in relation.items() if targets}
    empty = tuple(source for source, targets in relation.items() if not targets)
    connectivity = _audit_relation_connectivity(nonempty, incidence)

    cycle = compute_neuron_reference_cycle()
    probes, sources_by_label = _reference_probes(
        setup,
        cells,
        cycle,
        base_samples=reference_base_samples,
        handle_samples=reference_handle_samples,
    )
    reference_audit = _audit_reference_endpoints(
        relation, probes, cells, setup.box_map
    )
    node_by_cell = _morse_node_by_cell(morse_graph)
    reference_nodes = _reference_nodes_covering_every_probe(
        sources_by_label, node_by_cell
    )
    reference_hits = Counter(
        node_by_cell[source.index]
        for sources in sources_by_label.values()
        for source in sources
        if source.index in node_by_cell
    )
    morse_connectivity = {
        int(node): CellSetConnectivity(
            label=f"morse-{int(node)}",
            cells=frozenset(cells[int(index)] for index in morse_graph.morse_set(node)),
            components=incidence.connected_components(
                {cells[int(index)] for index in morse_graph.morse_set(node)}
            ),
        )
        for node in morse_graph.vertices()
    }

    records = setup.box_map.source_records()
    index_by_key = {_source_key(cell.chart_id, cell.bounds): cell.index for cell in cells}
    failed = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if record.failures and key in index_by_key
        )
    )
    jump_limit_failures = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if any(
                "does not cover suspension time" in failure.reason
                or "maximum jump" in failure.reason.lower()
                for failure in record.failures
            )
            and key in index_by_key
        )
    )
    unresolved = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if record.unresolved_stage_edges and key in index_by_key
        )
    )
    raw_unresolved = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if record.raw_unresolved_stage_edges and key in index_by_key
        )
    )
    bridge_sources = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if record.single_handle_bridge_attempts and key in index_by_key
        )
    )
    failed_bridge_sources = tuple(
        sorted(
            index_by_key[key]
            for key, record in records.items()
            if any(
                not bridge.synthesized
                for bridge in record.single_handle_bridge_attempts
            )
            and key in index_by_key
        )
    )
    clipping = setup.box_map.clipping_records()
    missing = tuple(
        sorted(
            index_by_key[_source_key(record.source_chart_id, record.source_bounds)]
            for record in clipping
            if record.pieces_without_active_cover
        )
    )
    open_sources = tuple(
        sorted(
            index_by_key[_source_key(record.source_chart_id, record.source_bounds)]
            for record in clipping
            if record.nonglued_open_boundary_pieces
        )
    )
    reference_open = tuple(
        sorted(
            cell.index
            for node in reference_nodes
            for index in morse_graph.morse_set(node)
            for cell in (cells[int(index)],)
            if _touches_nonglued_boundary(cell)
        )
    )
    unique, reused = setup.box_map.point_cache_counts()
    return SpikingNeuronAtlasAcceptance(
        protocol_revision=str(protocol_revision),
        total_depth=int(total_depth),
        axis_depth=_validate_total_depth(total_depth),
        samples_per_axis=int(samples_per_axis),
        max_jumps=int(max_jumps),
        t_star=float(t_star),
        single_handle_bridge_enabled=bool(single_handle_bridge),
        single_handle_bridge_max_bisections=int(
            single_handle_bridge_max_bisections
        ),
        elapsed_seconds=time.perf_counter() - started,
        analytic_audit=spiking_neuron_analytic_audit(),
        reference_cycle=cycle,
        family=setup.family,
        morse_graph=morse_graph,
        map_graph=map_graph,
        cells=cells,
        relation=relation,
        reference_endpoint_audit=reference_audit,
        reference_nodes=reference_nodes,
        reference_node_hits=dict(sorted(reference_hits.items())),
        image_connectivity_audit=connectivity,
        morse_support_connectivity=morse_connectivity,
        sparse_incidence_parity=sparse_parity,
        empty_sources=empty,
        failed_source_indices=failed,
        jump_limit_failure_source_indices=jump_limit_failures,
        raw_unresolved_source_indices=raw_unresolved,
        unresolved_source_indices=unresolved,
        single_handle_bridge_source_indices=bridge_sources,
        failed_single_handle_bridge_source_indices=failed_bridge_sources,
        missing_source_indices=missing,
        open_source_indices=open_sources,
        reference_open_boundary_indices=reference_open,
        box_map_diagnostics=setup.box_map.diagnostics(),
        clipping_records=clipping,
        endpoint_unique_evaluations=unique,
        endpoint_cache_reuses=reused,
        connectivity_algorithm=connectivity_algorithm,
        grid_kind=(
            "mixed_axis_depth_7_8"
            if isinstance(setup.family, SpikingNeuronAdaptiveFamily)
            else "uniform"
        ),
        source_total_depth=(
            int(source_total_depth)
            if isinstance(setup.family, SpikingNeuronAdaptiveFamily)
            and source_total_depth is not None
            else None
        ),
        endpoint_precompute=endpoint_precompute,
        adaptive_preflight=adaptive_preflight,
    )


def plot_spiking_neuron_acceptance(
    result: SpikingNeuronAtlasAcceptance, path: str | Path
) -> Path:
    """Write a chart-aware base/handle view of every recurrent Morse support."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, (base_axis, handle_axis) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap("tab10")
    for order, node in enumerate(sorted(int(n) for n in result.morse_graph.vertices())):
        color = colors(order % 10)
        patches_by_chart: dict[int, list[Rectangle]] = defaultdict(list)
        for raw_index in result.morse_graph.morse_set(node):
            cell = result.cells[int(raw_index)]
            lower, upper = cell.lower, cell.upper
            patches_by_chart[cell.chart_id].append(
                Rectangle(
                    lower,
                    *(upper - lower),
                )
            )
        for chart_id, patches in patches_by_chart.items():
            axis = base_axis if chart_id == BASE_CHART_ID else handle_axis
            axis.add_collection(
                PatchCollection(
                    patches,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.28,
                    linewidth=0.25,
                )
            )
    cycle = result.reference_cycle
    times = np.linspace(0.0, cycle.flight_time, 1001)
    states = np.asarray(cycle.solution(times), dtype=np.float64)
    base_axis.plot(states[0], states[1], color="black", linewidth=1.3, label="reference")
    base_axis.plot(
        [V_PEAK, V_RESET],
        [cycle.pre_reset_u, cycle.post_reset_u],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    base_axis.set_xlim(V_MIN - 3.0, V_PEAK + 3.0)
    base_axis.set_ylim(U_MIN - 15.0, U_MAX + 15.0)
    base_axis.set_xlabel("v")
    base_axis.set_ylabel("u")
    base_axis.set_title("base chart (v, u)")
    handle_axis.plot(
        np.full(101, cycle.pre_reset_u),
        np.linspace(0.0, 1.0, 101),
        color="black",
        linewidth=1.3,
    )
    handle_axis.set_xlim(U_MIN - 15.0, U_NOTCH + 15.0)
    handle_axis.set_ylim(-0.02, 1.02)
    handle_axis.set_xlabel("u_guard")
    handle_axis.set_ylabel("s")
    handle_axis.set_title("reset-handle chart (u_guard, s)")
    figure.suptitle(
        f"Neuron Atlas: depth {result.total_depth}, {result.samples_per_axis}x"
        f"{result.samples_per_axis}, {result.morse_graph.num_vertices()} Morse nodes"
    )
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def write_spiking_neuron_provenance(
    result: SpikingNeuronAtlasAcceptance,
    path: str | Path,
) -> Path:
    """Atomically persist every source's callback and clipping provenance."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    records = {
        _source_key(record.source_chart_id, record.source_bounds): record
        for record in result.box_map_diagnostics.retained_source_records
    }
    clipping = {
        _source_key(record.source_chart_id, record.source_bounds): record
        for record in result.clipping_records
    }
    digest = hashlib.sha256()
    source_count = 0

    def encoded_line(payload: object) -> bytes:
        return _canonical_json(payload) + b"\n"

    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            header = result.summary()
            header_line = encoded_line({"record": "header", **header})
            digest.update(header_line)
            stream.write(header_line.decode("utf-8"))
            for cell in result.cells:
                key = _source_key(cell.chart_id, cell.bounds)
                source = records.get(key)
                clip = clipping.get(key)
                payload = {
                    "record": "source",
                    "index": cell.index,
                    "chart_id": cell.chart_id,
                    "bounds": list(cell.bounds),
                    "targets": sorted(target.index for target in result.relation[cell]),
                    "morse_node": next(
                        (
                            int(node)
                            for node in result.morse_graph.vertices()
                            if cell.index in set(int(i) for i in result.morse_graph.morse_set(node))
                        ),
                        None,
                    ),
                    "failures": [] if source is None else [f.reason for f in source.failures],
                    "unresolved_stage_edges": (
                        [] if source is None else [list(edge) for edge in source.unresolved_stage_edges]
                    ),
                    "raw_unresolved_stage_edges": (
                        []
                        if source is None
                        else [
                            list(edge)
                            for edge in source.raw_unresolved_stage_edges
                        ]
                    ),
                    "single_handle_bridge_attempts": (
                        []
                        if source is None
                        else [
                            bridge.to_dict()
                            for bridge in source.single_handle_bridge_attempts
                        ]
                    ),
                    "raw_pieces": None if clip is None else clip.raw_pieces,
                    "clipped_pieces": None if clip is None else clip.clipped_pieces,
                    "pieces_without_active_cover": (
                        None if clip is None else clip.pieces_without_active_cover
                    ),
                    "nonglued_open_boundary_pieces": (
                        None if clip is None else clip.nonglued_open_boundary_pieces
                    ),
                }
                line = encoded_line(payload)
                digest.update(line)
                stream.write(line.decode("utf-8"))
                source_count += 1
            trailer_fields = {
                "record": "trailer",
                "schema": "spiking-neuron-source-provenance-trailer-v1",
                "checkpoint_complete": True,
                "diagnostic_only": True,
                "source_records": source_count,
                "content_sha256": digest.hexdigest(),
                "relation_csr_fingerprint": (
                    None
                    if result.relation_csr is None
                    else result.relation_csr["fingerprint"]
                ),
            }
            trailer_fields["fingerprint"] = _sha256(trailer_fields)
            stream.write(encoded_line(trailer_fields).decode("utf-8"))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_spiking_neuron_provenance(
    path: str | Path,
    *,
    expected_relation_csr_fingerprint: str | None = None,
) -> dict[str, object]:
    """Strictly validate and fingerprint a diagnostic source checkpoint."""

    source = Path(path)
    digest = hashlib.sha256()
    source_records = 0
    previous_line: bytes | None = None
    previous_payload: Mapping[str, object] | None = None
    first_payload: Mapping[str, object] | None = None
    with gzip.open(source, "rb") as stream:
        for raw_line in stream:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError("malformed neuron provenance JSON line") from error
            if not isinstance(payload, Mapping):
                raise ValueError("neuron provenance records must be JSON objects")
            canonical = encoded_line_for_validation(payload)
            if canonical != raw_line:
                raise ValueError("neuron provenance JSON is not canonical")
            if first_payload is None:
                first_payload = payload
            if previous_line is not None:
                digest.update(previous_line)
                if previous_payload is not None and previous_payload.get("record") == "source":
                    source_records += 1
            previous_line = raw_line
            previous_payload = payload
    if first_payload is None or first_payload.get("record") != "header":
        raise ValueError("neuron provenance lacks its header")
    trailer = previous_payload
    if trailer is None or trailer.get("record") != "trailer":
        raise ValueError("neuron provenance lacks its authenticated trailer")
    if trailer.get("schema") != "spiking-neuron-source-provenance-trailer-v1":
        raise ValueError("unknown neuron provenance trailer schema")
    if not trailer.get("checkpoint_complete") or not trailer.get("diagnostic_only"):
        raise ValueError("neuron provenance trailer is incomplete or misclassified")
    if trailer.get("source_records") != source_records:
        raise ValueError("neuron provenance source count is inconsistent")
    if trailer.get("content_sha256") != digest.hexdigest():
        raise ValueError("neuron provenance content fingerprint mismatch")
    claimed_fingerprint = trailer.get("fingerprint")
    trailer_without_fingerprint = dict(trailer)
    trailer_without_fingerprint.pop("fingerprint", None)
    if claimed_fingerprint != _sha256(trailer_without_fingerprint):
        raise ValueError("neuron provenance trailer fingerprint mismatch")
    relation_fingerprint = trailer.get("relation_csr_fingerprint")
    if (
        expected_relation_csr_fingerprint is not None
        and relation_fingerprint != expected_relation_csr_fingerprint
    ):
        raise ValueError("neuron provenance is bound to a different CSR relation")
    return {
        "schema": "spiking-neuron-source-provenance-reference-v1",
        "path": source.name,
        "diagnostic_only": True,
        "source_records": source_records,
        "content_sha256": trailer["content_sha256"],
        "trailer_fingerprint": claimed_fingerprint,
        "file_sha256": _sha256_file(source),
        "relation_csr_fingerprint": relation_fingerprint,
    }


def authenticate_spiking_neuron_provenance(
    path: str | Path,
    *,
    relation_csr_fingerprint: str,
) -> dict[str, object]:
    """Upgrade a complete legacy diagnostic stream to the strict trailer.

    This is an atomic canonical rewrite of the same header/source records; it
    neither changes the relation nor invents missing source provenance.
    """

    target = Path(path)
    payloads: list[Mapping[str, object]] = []
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("neuron provenance records must be JSON objects")
            payloads.append(payload)
    if not payloads or payloads[0].get("record") != "header":
        raise ValueError("legacy neuron provenance lacks its header")
    if payloads[-1].get("record") == "trailer":
        return validate_spiking_neuron_provenance(
            target,
            expected_relation_csr_fingerprint=relation_csr_fingerprint,
        )
    source_payloads = [
        payload for payload in payloads[1:] if payload.get("record") == "source"
    ]
    if len(source_payloads) != len(payloads) - 1:
        raise ValueError("legacy neuron provenance contains an unknown record")
    declared_cells = payloads[0].get("active_cells")
    if declared_cells is not None and declared_cells != len(source_payloads):
        raise ValueError("legacy neuron provenance source count is incomplete")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with gzip.open(temporary, "wb") as stream:
            for payload in payloads:
                line = encoded_line_for_validation(payload)
                digest.update(line)
                stream.write(line)
            trailer = {
                "record": "trailer",
                "schema": "spiking-neuron-source-provenance-trailer-v1",
                "checkpoint_complete": True,
                "diagnostic_only": True,
                "source_records": len(source_payloads),
                "content_sha256": digest.hexdigest(),
                "relation_csr_fingerprint": str(relation_csr_fingerprint),
            }
            trailer["fingerprint"] = _sha256(trailer)
            stream.write(encoded_line_for_validation(trailer))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validate_spiking_neuron_provenance(
        target,
        expected_relation_csr_fingerprint=relation_csr_fingerprint,
    )


def encoded_line_for_validation(payload: Mapping[str, object]) -> bytes:
    """Canonical line helper kept public only for strict artifact tests."""

    return _canonical_json(payload) + b"\n"


def write_spiking_neuron_csr_checkpoint(
    result: SpikingNeuronAtlasAcceptance,
    path: str | Path,
    *,
    max_edges: int = 50_000_000,
    max_payload_bytes: int = 1 << 30,
) -> dict[str, object]:
    """Persist the authoritative native MapGraph in fingerprinted mmap CSR."""

    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("the local CMGDB fork is not installed") from error
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite CSR checkpoint {target}")
    configuration = {
        "schema": "spiking-neuron-mapgraph-csr-configuration-v1",
        "protocol_revision": result.protocol_revision,
        "boxmap_revision": BOXMAP_REVISION,
        "model": "compact_quadratic_integrate_and_fire",
        "formal_domain": "compact_L_shaped_N",
        "t_star": result.t_star,
        "total_depth": result.total_depth,
        "axis_depth": result.axis_depth,
        "grid_kind": result.grid_kind,
        "source_total_depth": result.source_total_depth,
        "samples_per_axis": result.samples_per_axis,
        "padding_cells": 1.0,
        "max_step": DEFAULT_MAX_STEP,
        "max_jumps": result.max_jumps,
        "single_handle_bridge_enabled": result.single_handle_bridge_enabled,
        "single_handle_bridge_max_bisections": (
            result.single_handle_bridge_max_bisections
        ),
        "single_handle_bridge_algorithm_revision": (
            SINGLE_HANDLE_BRIDGE_ALGORITHM
            if result.single_handle_bridge_enabled
            else None
        ),
        "family_fingerprint": result.family.fingerprint,
        "active_cells": len(result.cells),
        "active_leaf_depth_counts": result.family.depth_counts(),
        "adaptive_preflight_fingerprint": (
            None
            if result.adaptive_preflight is None
            else result.adaptive_preflight.get("fingerprint")
        ),
        "whole_cell_outer_enclosure_certified": False,
        "analytic_conley_label_attached": False,
    }
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=len(result.cells),
        max_edges=int(max_edges),
        max_payload_bytes=int(max_payload_bytes),
    )
    written = CMGDB.write_map_graph_csr_checkpoint(
        result.map_graph,
        target,
        configuration=configuration,
        caps=caps,
        target_dtype="auto",
    )
    metadata = CMGDB.read_map_graph_csr_metadata(written)
    fingerprint = metadata["fingerprint"]["sha256"]
    # Strictly load once now so a corrupt or configuration-mismatched artifact
    # cannot be published alongside a passing JSON summary.
    CMGDB.load_map_graph_csr_checkpoint(
        written,
        expected_configuration=configuration,
        caps=caps,
        expected_fingerprint=fingerprint,
    )
    return {
        "schema": "spiking-neuron-mapgraph-csr-reference-v1",
        "path": target.name,
        "fingerprint": fingerprint,
        "configuration_sha256": metadata["configuration_sha256"],
        "vertices": metadata["vertices"],
        "edges": metadata["edges"],
        "payload_bytes": metadata["payload_bytes"],
    }


__all__ = [
    "ACTIVE_HANDLE_U_BOUNDS",
    "ADAPTIVE_PREFLIGHT_REVISION",
    "AtlasNeuronCell",
    "BASE_AMBIENT_BOUNDS",
    "BASE_CHART_ID",
    "BOXMAP_REVISION",
    "DEFAULT_T_STAR",
    "FROZEN_SAMPLING_SENSITIVITY",
    "FROZEN_TOTAL_DEPTHS",
    "HANDLE_CHART_ID",
    "MixedIncidenceParityAudit",
    "NeuronClippingRecord",
    "NeuronDyadicCell",
    "NeuronReferenceCycle",
    "NeuronSuspensionBoxMap",
    "PROTOCOL_REVISION",
    "SpikingNeuronActiveFamily",
    "SpikingNeuronAdaptiveFamily",
    "SpikingNeuronAdaptivePreflight",
    "SpikingNeuronAtlasAcceptance",
    "SpikingNeuronAtlasSetup",
    "SpikingNeuronMixedQuotientIncidence",
    "SpikingNeuronQuotientIncidence",
    "SpikingNeuronSparseQuotientIncidence",
    "SparseIncidenceParityAudit",
    "audit_spiking_neuron_mixed_incidence_parity",
    "audit_spiking_neuron_sparse_incidence_parity",
    "authenticate_spiking_neuron_provenance",
    "build_spiking_neuron_active_family",
    "build_spiking_neuron_adaptive_family",
    "build_spiking_neuron_adaptive_preflight",
    "build_spiking_neuron_atlas_model",
    "build_spiking_neuron_quotient_nerve",
    "compute_neuron_reference_cycle",
    "compute_spiking_neuron_atlas_acceptance",
    "load_spiking_neuron_adaptive_preflight",
    "plot_spiking_neuron_acceptance",
    "spiking_neuron_atlas_charts",
    "spiking_neuron_dyadic_bounds",
    "spiking_neuron_tensor_sample_cost",
    "spiking_neuron_atlas_reset_gluing",
    "validate_spiking_neuron_provenance",
    "write_spiking_neuron_csr_checkpoint",
    "write_spiking_neuron_provenance",
]
