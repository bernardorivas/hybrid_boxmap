"""Honest finite-complex front end for physical Atlas Conley computations.

The vertices of :class:`AtlasQuotientNerveComplex2D` are the actual closed
base- and reset-handle rectangles used by ``CMGDB.AtlasModel``.  Intersections
are computed in the reset quotient, using explicit affine bottom and top seam
maps.  Every nonempty finite intersection admitted to the nerve is required
to be contractible.  Interior seams remain forbidden by default; the explicit
subcomplex opt-in additionally verifies no straddling, complete endpoint-face
coverage, and both incident base cofaces along every used seam segment.  Thus
this module never replaces the physical Atlas by an abstract periodic-orbit
circle and never forces a non-grid-aligned reset into a fictitious cubical
attachment.

For a candidate relative pair, the box map is evaluated independently on
every nerve simplex, including all faces.  Its tagged rectangles are covered
by the same Atlas.  The carrier of a simplex is the cellular closure of the
union of the raw image subcomplexes of all its faces.  Face nesting is then
automatic; acyclicity, pair preservation, carrier subordination, ``d^2 = 0``,
and ``dF = Fd`` are checked explicitly.  The subordinate chain selector is
constructed by the acyclic-carrier induction in :mod:`suspension_complex`.

This code does not certify an ODE enclosure or the index-pair axioms.  Its
finite-relation endpoint reports only the shift class of the validated nerve,
pair, carrier, and selected chain map.  The stronger continuous-system
endpoint remains locked unless the caller separately certifies both the
whole-cell outer enclosure and the index-pair obligations.

For dense four-dimensional Atlases, a closed-box nerve is combinatorially
impractical.  The scalable path uses the independent-guard double mapping
cylinder from :mod:`suspension_complex`.  An explicit registry preserves the
actual Atlas ids, and its preparation endpoint requires face-level carrier
data rather than inferring lower-dimensional images from pairwise adjacency.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .suspension_complex import (
    CMGDBRelativeHomologyPayload,
    CellularChainMap,
    DoubleMappingCylinderComplex,
    FiniteCellComplex,
    FixedTimeCarrier,
    GuardPrismCell,
    RelativeCellPair,
    SuspensionBaseCell,
)


Bounds2D = tuple[float, float, float, float]


class AtlasGoodCoverError(ValueError):
    """Raised when the actual quotient Atlas fails the verified good-cover gate."""


class PhysicalConleyCertificationError(RuntimeError):
    """Raised when a requested Conley endpoint lacks a required certificate."""


def _finite(value: float, description: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _canonical(value: float, digits: int) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result


def _normalize_bounds(bounds: Sequence[float], *, digits: int) -> Bounds2D:
    values = tuple(_canonical(_finite(value, "rectangle bound"), digits) for value in bounds)
    if len(values) != 4:
        raise ValueError("a two-dimensional Atlas rectangle needs four flattened bounds")
    x0, y0, x1, y1 = values
    if x0 > x1 or y0 > y1:
        raise ValueError("rectangle lower bounds must not exceed upper bounds")
    return values


def _intersection(
    rectangles: Sequence[Bounds2D],
    *,
    atol: float,
    digits: int,
) -> Bounds2D | None:
    if not rectangles:
        return None
    x0 = max(bounds[0] for bounds in rectangles)
    y0 = max(bounds[1] for bounds in rectangles)
    x1 = min(bounds[2] for bounds in rectangles)
    y1 = min(bounds[3] for bounds in rectangles)
    if x0 > x1 + atol or y0 > y1 + atol:
        return None
    if abs(x1 - x0) <= atol:
        x0 = x1 = 0.5 * (x0 + x1)
    if abs(y1 - y0) <= atol:
        y0 = y1 = 0.5 * (y0 + y1)
    return tuple(_canonical(value, digits) for value in (x0, y0, x1, y1))  # type: ignore[return-value]


def _intersects(first: Bounds2D, second: Bounds2D, atol: float) -> bool:
    return not (
        first[2] < second[0] - atol
        or second[2] < first[0] - atol
        or first[3] < second[1] - atol
        or second[3] < first[1] - atol
    )


@dataclass(frozen=True, order=True)
class AtlasRectangleCell2D:
    """One actual top-dimensional Atlas rectangle and its global grid index."""

    index: int
    chart_id: int
    bounds: Bounds2D

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Atlas cell index must be a nonnegative integer")
        if (
            isinstance(self.chart_id, bool)
            or not isinstance(self.chart_id, int)
            or self.chart_id < 0
        ):
            raise ValueError("Atlas chart id must be a nonnegative integer")
        bounds = tuple(float(value) for value in self.bounds)
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("Atlas cell bounds must contain four finite values")
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Atlas top cells must have positive area")
        object.__setattr__(self, "bounds", bounds)


@dataclass(frozen=True, order=True)
class AtlasNerveSimplex:
    """An oriented simplex, ordered by actual Atlas cell index."""

    vertices: tuple[int, ...]

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        if not vertices or tuple(sorted(set(vertices))) != vertices:
            raise ValueError("nerve simplex vertices must be nonempty, unique, and sorted")
        object.__setattr__(self, "vertices", vertices)

    @property
    def dimension(self) -> int:
        return len(self.vertices) - 1


@dataclass(frozen=True, order=True)
class QuotientRectangleRepresentative2D:
    """One chart representative of a subset of the reset quotient."""

    chart_id: int
    bounds: Bounds2D


@dataclass(frozen=True)
class AffineBoundaryEmbedding2D:
    """Affine embedding ``u -> base`` into an axis-aligned boundary line."""

    fixed_axis: int
    fixed_value: float
    scale: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.fixed_axis not in (0, 1):
            raise ValueError("fixed_axis must be 0 or 1")
        for name in ("fixed_value", "scale", "offset"):
            value = _finite(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.scale == 0.0:
            raise ValueError("an affine seam embedding must be injective")

    @property
    def varying_axis(self) -> int:
        return 1 - self.fixed_axis

    def point(self, intrinsic: float) -> tuple[float, float]:
        coordinates = [0.0, 0.0]
        coordinates[self.fixed_axis] = self.fixed_value
        coordinates[self.varying_axis] = self.scale * float(intrinsic) + self.offset
        return float(coordinates[0]), float(coordinates[1])

    def segment(self, lower: float, upper: float, *, digits: int) -> Bounds2D:
        first = self.point(lower)
        second = self.point(upper)
        return tuple(
            _canonical(value, digits)
            for value in (
                min(first[0], second[0]),
                min(first[1], second[1]),
                max(first[0], second[0]),
                max(first[1], second[1]),
            )
        )  # type: ignore[return-value]

    def inverse_interval(self, bounds: Bounds2D) -> tuple[float, float]:
        lower = bounds[self.varying_axis]
        upper = bounds[2 + self.varying_axis]
        values = ((lower - self.offset) / self.scale, (upper - self.offset) / self.scale)
        return min(values), max(values)


@dataclass(frozen=True)
class AtlasResetGluing2D:
    """Bottom/guard and top/reset identifications for a 2D Atlas handle."""

    base_chart_id: int
    handle_chart_id: int
    guard: AffineBoundaryEmbedding2D
    reset: AffineBoundaryEmbedding2D
    phase_axis: int = 1
    phase_bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.base_chart_id == self.handle_chart_id:
            raise ValueError("base and handle chart ids must be different")
        if self.phase_axis != 1:
            raise ValueError("the initial 2D implementation requires handle coordinates (u,s)")
        lower, upper = (float(value) for value in self.phase_bounds)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("phase_bounds must be a finite nondegenerate interval")
        object.__setattr__(self, "phase_bounds", (lower, upper))


@dataclass(frozen=True)
class QuotientIntersectionAudit2D:
    simplex: AtlasNerveSimplex
    nonempty: bool
    contractible: bool
    representatives: tuple[QuotientRectangleRepresentative2D, ...]
    reason: str


@dataclass(frozen=True)
class AtlasSeamSubcomplexAudit2D:
    """Constructor-verified certificate for one interior quotient seam.

    The certificate is never caller asserted.  It is emitted only after the
    selected base rectangles have been proved not to straddle the seam, every
    selected endpoint face of the handle has been covered completely by
    base-cell faces lying on that seam, and both incident base top-cell
    cofaces have been retained along every used segment of an interior seam.
    """

    name: str
    fixed_axis: int
    fixed_value: float
    base_face_cell_indices: tuple[int, ...]
    negative_side_base_cell_indices: tuple[int, ...]
    positive_side_base_cell_indices: tuple[int, ...]
    handle_face_cell_indices: tuple[int, ...]
    no_base_top_cell_straddles: bool
    complete_handle_face_attachment_cover: bool
    all_incident_top_cofaces_present: bool


@dataclass(frozen=True)
class _SeamOption:
    kind: str
    phase: float
    embedding: AffineBoundaryEmbedding2D
    base_bounds: Bounds2D
    handle_bounds: Bounds2D


class AtlasQuotientNerveComplex2D(FiniteCellComplex):
    """Verified nerve of actual Atlas rectangles in a two-chart reset quotient.

    By default both quotient seams must lie on the boundary of the selected
    base window, preserving the original conservative safeguard.  A caller
    may explicitly name an interior seam in ``interior_seam_subcomplexes``.
    That is only a request for constructor verification: it succeeds precisely
    when the selected base top cells do not straddle the seam, their seam
    faces completely cover every selected endpoint face of the handle, and
    both incident base cofaces are retained along every used interior-seam
    segment.  All finite quotient intersections are still enumerated and
    required to be contractible after this geometric gate passes.
    """

    def __init__(
        self,
        cells: Iterable[AtlasRectangleCell2D],
        gluing: AtlasResetGluing2D,
        *,
        atol: float = 1.0e-10,
        coordinate_digits: int = 12,
        maximum_simplex_size: int = 16,
        interior_seam_subcomplexes: Collection[str] = (),
    ) -> None:
        cell_data = tuple(sorted(cells, key=lambda cell: cell.index))
        if not cell_data:
            raise ValueError("an Atlas quotient nerve requires cells")
        if len({cell.index for cell in cell_data}) != len(cell_data):
            raise ValueError("Atlas cell indices must be unique")
        chart_ids = {cell.chart_id for cell in cell_data}
        allowed_chart_ids = {gluing.base_chart_id, gluing.handle_chart_id}
        if not chart_ids <= allowed_chart_ids:
            raise ValueError(
                "the 2D quotient nerve contains a chart outside its declared gluing"
            )
        if gluing.base_chart_id not in chart_ids:
            raise ValueError(
                "the initial 2D quotient nerve requires at least one base-chart cell"
            )
        if not math.isfinite(atol) or atol < 0.0:
            raise ValueError("atol must be finite and nonnegative")
        if maximum_simplex_size < 1:
            raise ValueError("maximum_simplex_size must be positive")
        if isinstance(interior_seam_subcomplexes, (str, bytes)):
            raise TypeError(
                "interior_seam_subcomplexes must be a collection of seam names"
            )
        requested_interior = tuple(interior_seam_subcomplexes)
        if any(not isinstance(name, str) for name in requested_interior):
            raise TypeError("interior seam names must be strings")
        if len(set(requested_interior)) != len(requested_interior):
            raise ValueError("interior seam names must be duplicate-free")
        unknown_interior = set(requested_interior).difference({"guard", "reset"})
        if unknown_interior:
            raise ValueError(
                "unknown interior seam names: "
                f"{sorted(unknown_interior)!r}"
            )
        if requested_interior and gluing.handle_chart_id not in chart_ids:
            raise ValueError(
                "interior seam verification requires selected handle-chart cells"
            )

        normalized = tuple(
            AtlasRectangleCell2D(
                cell.index,
                cell.chart_id,
                _normalize_bounds(cell.bounds, digits=coordinate_digits),
            )
            for cell in cell_data
        )
        self.gluing = gluing
        self.atol = float(atol)
        self.coordinate_digits = int(coordinate_digits)
        self.maximum_simplex_size = int(maximum_simplex_size)
        self._cell_by_index = MappingProxyType({cell.index: cell for cell in normalized})
        self._audit_cache: dict[AtlasNerveSimplex, QuotientIntersectionAudit2D] = {}
        seam_subcomplex_audits: dict[str, AtlasSeamSubcomplexAudit2D] = {}

        for embedding, name in ((gluing.guard, "guard"), (gluing.reset, "reset")):
            if gluing.handle_chart_id not in chart_ids:
                break  # An induced base-only family is an ordinary rectangle nerve.
            fixed_values = [
                cell.bounds[embedding.fixed_axis]
                for cell in normalized
                if cell.chart_id == gluing.base_chart_id
            ] + [
                cell.bounds[2 + embedding.fixed_axis]
                for cell in normalized
                if cell.chart_id == gluing.base_chart_id
            ]
            minimum = min(fixed_values)
            maximum = max(fixed_values)
            boundary_seam = bool(
                abs(embedding.fixed_value - minimum) <= self.atol
                or abs(embedding.fixed_value - maximum) <= self.atol
            )
            if boundary_seam:
                if name in requested_interior:
                    raise ValueError(
                        f"{name} was declared interior but lies on the boundary "
                        "of the selected base Atlas window"
                    )
                continue
            if name not in requested_interior:
                raise AtlasGoodCoverError(
                    f"{name} seam is not on the boundary of the base Atlas window"
                )
            seam_subcomplex_audits[name] = self._verify_interior_seam_subcomplex(
                normalized,
                embedding=embedding,
                name=name,
            )

        missing_requested = set(requested_interior).difference(seam_subcomplex_audits)
        if missing_requested:
            raise AssertionError(
                "requested interior seams were not verified: "
                f"{sorted(missing_requested)!r}"
            )
        self._seam_subcomplex_audits = MappingProxyType(seam_subcomplex_audits)

        simplices: list[AtlasNerveSimplex] = []
        frontier = [AtlasNerveSimplex((cell.index,)) for cell in normalized]
        while frontier:
            next_frontier: list[AtlasNerveSimplex] = []
            for simplex in frontier:
                audit = self._audit(simplex)
                if not audit.nonempty:
                    continue
                if not audit.contractible:
                    raise AtlasGoodCoverError(
                        "noncontractible finite intersection for actual Atlas cells "
                        f"{simplex.vertices!r}: {audit.reason}"
                    )
                simplices.append(simplex)
                if len(simplex.vertices) >= self.maximum_simplex_size:
                    candidates_exist = any(
                        candidate > simplex.vertices[-1]
                        and self._all_pairwise_intersect(simplex.vertices, candidate)
                        for candidate in self._cell_by_index
                    )
                    if candidates_exist:
                        raise AtlasGoodCoverError(
                            "Atlas overlap exceeds maximum_simplex_size; increase the "
                            "explicit bound before claiming a complete nerve"
                        )
                    continue
                for candidate in self._cell_by_index:
                    if candidate <= simplex.vertices[-1]:
                        continue
                    if not self._all_pairwise_intersect(simplex.vertices, candidate):
                        continue
                    extension = AtlasNerveSimplex((*simplex.vertices, candidate))
                    extension_audit = self._audit(extension)
                    if extension_audit.nonempty:
                        next_frontier.append(extension)
            frontier = next_frontier

        simplices.sort(key=lambda simplex: (simplex.dimension, simplex.vertices))
        dimensions = {simplex: simplex.dimension for simplex in simplices}
        boundaries: dict[AtlasNerveSimplex, dict[AtlasNerveSimplex, int]] = {}
        for simplex in simplices:
            if simplex.dimension == 0:
                boundaries[simplex] = {}
                continue
            boundaries[simplex] = {
                AtlasNerveSimplex(simplex.vertices[:index] + simplex.vertices[index + 1 :]): (
                    1 if index % 2 == 0 else -1
                )
                for index in range(len(simplex.vertices))
            }

        self._simplices = tuple(simplices)
        super().__init__(
            dimensions,
            boundaries,
            metadata={
                "kind": "actual-atlas-reset-quotient-nerve",
                "base_chart_id": gluing.base_chart_id,
                "handle_chart_id": gluing.handle_chart_id,
                "atlas_cell_count": len(normalized),
                "chart_cell_counts": {
                    chart_id: sum(cell.chart_id == chart_id for cell in normalized)
                    for chart_id in sorted(chart_ids)
                },
                "finite_intersections_verified_contractible": True,
                "interior_seam_subcomplexes": tuple(
                    sorted(self._seam_subcomplex_audits)
                ),
                "interior_seam_subcomplex_audits": {
                    name: {
                        "fixed_axis": audit.fixed_axis,
                        "fixed_value": audit.fixed_value,
                        "base_face_cells": len(audit.base_face_cell_indices),
                        "negative_side_base_cells": len(
                            audit.negative_side_base_cell_indices
                        ),
                        "positive_side_base_cells": len(
                            audit.positive_side_base_cell_indices
                        ),
                        "handle_face_cells": len(audit.handle_face_cell_indices),
                        "no_base_top_cell_straddles": (
                            audit.no_base_top_cell_straddles
                        ),
                        "complete_handle_face_attachment_cover": (
                            audit.complete_handle_face_attachment_cover
                        ),
                        "all_incident_top_cofaces_present": (
                            audit.all_incident_top_cofaces_present
                        ),
                    }
                    for name, audit in sorted(
                        self._seam_subcomplex_audits.items()
                    )
                },
            },
        )

    @property
    def atlas_cells(self) -> tuple[AtlasRectangleCell2D, ...]:
        return tuple(self._cell_by_index.values())

    @property
    def atlas_indices(self) -> frozenset[int]:
        return frozenset(self._cell_by_index)

    @property
    def seam_subcomplex_audits(
        self,
    ) -> Mapping[str, AtlasSeamSubcomplexAudit2D]:
        """Return immutable constructor-generated interior-seam certificates."""

        return self._seam_subcomplex_audits

    def atlas_cell(self, index: int) -> AtlasRectangleCell2D:
        try:
            return self._cell_by_index[index]
        except KeyError as error:
            raise KeyError(f"unknown Atlas cell index {index}") from error

    def simplex(self, vertices: Collection[int]) -> AtlasNerveSimplex:
        simplex = AtlasNerveSimplex(tuple(sorted(vertices)))
        if simplex not in self.cell_set:
            raise KeyError(f"not a simplex of this quotient nerve: {simplex.vertices!r}")
        return simplex

    def intersection_audit(
        self, simplex: AtlasNerveSimplex
    ) -> QuotientIntersectionAudit2D:
        if simplex not in self.cell_set:
            raise KeyError(f"not a simplex of this quotient nerve: {simplex.vertices!r}")
        return self._audit_cache[simplex]

    def representatives(
        self, simplex: AtlasNerveSimplex
    ) -> tuple[QuotientRectangleRepresentative2D, ...]:
        return self.intersection_audit(simplex).representatives

    def induced_cells(self, vertices: Collection[int]) -> frozenset[AtlasNerveSimplex]:
        selected = frozenset(vertices)
        unknown = selected.difference(self.atlas_indices)
        if unknown:
            raise ValueError(f"unknown Atlas vertices: {sorted(unknown)!r}")
        return frozenset(
            simplex for simplex in self._simplices if set(simplex.vertices) <= selected
        )

    def _all_pairwise_intersect(
        self, vertices: tuple[int, ...], candidate: int
    ) -> bool:
        return all(
            self._audit(AtlasNerveSimplex(tuple(sorted((vertex, candidate))))).nonempty
            for vertex in vertices
        )

    def _verify_interior_seam_subcomplex(
        self,
        cells: Sequence[AtlasRectangleCell2D],
        *,
        embedding: AffineBoundaryEmbedding2D,
        name: str,
    ) -> AtlasSeamSubcomplexAudit2D:
        base_cells = tuple(
            cell
            for cell in cells
            if cell.chart_id == self.gluing.base_chart_id
        )
        fixed_axis = embedding.fixed_axis
        varying_axis = embedding.varying_axis
        fixed_value = embedding.fixed_value
        straddling = tuple(
            cell.index
            for cell in base_cells
            if (
                cell.bounds[fixed_axis] < fixed_value - self.atol
                and fixed_value + self.atol
                < cell.bounds[2 + fixed_axis]
            )
        )
        if straddling:
            raise AtlasGoodCoverError(
                f"{name} interior seam is crossed by base top cells "
                f"{list(straddling[:10])!r}; it is not a cubical subcomplex"
            )

        base_face_cells = tuple(
            cell
            for cell in base_cells
            if (
                abs(cell.bounds[fixed_axis] - fixed_value) <= self.atol
                or abs(cell.bounds[2 + fixed_axis] - fixed_value) <= self.atol
            )
        )
        base_intervals = self._merged_intervals(
            (
                cell.bounds[varying_axis],
                cell.bounds[2 + varying_axis],
            )
            for cell in base_face_cells
        )
        negative_side_cells = tuple(
            cell
            for cell in base_face_cells
            if abs(cell.bounds[2 + fixed_axis] - fixed_value) <= self.atol
        )
        positive_side_cells = tuple(
            cell
            for cell in base_face_cells
            if abs(cell.bounds[fixed_axis] - fixed_value) <= self.atol
        )
        negative_side_intervals = self._merged_intervals(
            (
                cell.bounds[varying_axis],
                cell.bounds[2 + varying_axis],
            )
            for cell in negative_side_cells
        )
        positive_side_intervals = self._merged_intervals(
            (
                cell.bounds[varying_axis],
                cell.bounds[2 + varying_axis],
            )
            for cell in positive_side_cells
        )

        phase_coordinate = 1 if name == "guard" else 3
        phase_value = (
            self.gluing.phase_bounds[0]
            if name == "guard"
            else self.gluing.phase_bounds[1]
        )
        handle_face_cells = tuple(
            cell
            for cell in cells
            if cell.chart_id == self.gluing.handle_chart_id
            and abs(cell.bounds[phase_coordinate] - phase_value) <= self.atol
        )
        for handle_cell in handle_face_cells:
            segment = embedding.segment(
                handle_cell.bounds[0],
                handle_cell.bounds[2],
                digits=self.coordinate_digits,
            )
            lower = segment[varying_axis]
            upper = segment[2 + varying_axis]
            if not self._interval_covered(lower, upper, base_intervals):
                raise AtlasGoodCoverError(
                    f"{name} handle face of Atlas cell {handle_cell.index} is not "
                    "completely covered by base-cell faces on the interior seam"
                )
            missing_sides = tuple(
                side
                for side, cover in (
                    ("negative", negative_side_intervals),
                    ("positive", positive_side_intervals),
                )
                if not self._interval_covered(lower, upper, cover)
            )
            if missing_sides:
                raise AtlasGoodCoverError(
                    f"{name} handle face of Atlas cell {handle_cell.index} is "
                    "missing incident base top cofaces on the "
                    f"{', '.join(missing_sides)} side of the interior seam"
                )

        return AtlasSeamSubcomplexAudit2D(
            name=name,
            fixed_axis=fixed_axis,
            fixed_value=float(fixed_value),
            base_face_cell_indices=tuple(cell.index for cell in base_face_cells),
            negative_side_base_cell_indices=tuple(
                cell.index for cell in negative_side_cells
            ),
            positive_side_base_cell_indices=tuple(
                cell.index for cell in positive_side_cells
            ),
            handle_face_cell_indices=tuple(cell.index for cell in handle_face_cells),
            no_base_top_cell_straddles=True,
            complete_handle_face_attachment_cover=True,
            all_incident_top_cofaces_present=True,
        )

    def _merged_intervals(
        self,
        intervals: Iterable[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        ordered = sorted(
            (min(float(lower), float(upper)), max(float(lower), float(upper)))
            for lower, upper in intervals
        )
        merged: list[tuple[float, float]] = []
        for lower, upper in ordered:
            if not merged or lower > merged[-1][1] + self.atol:
                merged.append((lower, upper))
                continue
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        return tuple(merged)

    def _interval_covered(
        self,
        lower: float,
        upper: float,
        cover: Sequence[tuple[float, float]],
    ) -> bool:
        target_lower = min(float(lower), float(upper))
        target_upper = max(float(lower), float(upper))
        for cover_lower, cover_upper in cover:
            if cover_upper < target_lower - self.atol:
                continue
            if cover_lower > target_lower + self.atol:
                return False
            return cover_upper >= target_upper - self.atol
        return False

    def _seam_options(self, cell: AtlasRectangleCell2D) -> tuple[_SeamOption, ...]:
        if cell.chart_id != self.gluing.handle_chart_id:
            return ()
        phase_lower, phase_upper = self.gluing.phase_bounds
        u_lower, u_upper = cell.bounds[0], cell.bounds[2]
        result: list[_SeamOption] = []
        for kind, phase, embedding, coordinate in (
            ("guard", phase_lower, self.gluing.guard, cell.bounds[1]),
            ("reset", phase_upper, self.gluing.reset, cell.bounds[3]),
        ):
            if abs(coordinate - phase) > self.atol:
                continue
            result.append(
                _SeamOption(
                    kind=kind,
                    phase=phase,
                    embedding=embedding,
                    base_bounds=embedding.segment(
                        u_lower, u_upper, digits=self.coordinate_digits
                    ),
                    handle_bounds=_normalize_bounds(
                        (u_lower, phase, u_upper, phase),
                        digits=self.coordinate_digits,
                    ),
                )
            )
        return tuple(result)

    def _audit(self, simplex: AtlasNerveSimplex) -> QuotientIntersectionAudit2D:
        cached = self._audit_cache.get(simplex)
        if cached is not None:
            return cached
        try:
            cells = tuple(self._cell_by_index[index] for index in simplex.vertices)
        except KeyError as error:
            raise ValueError(f"simplex refers to unknown Atlas cell {error.args[0]}") from error
        base_cells = tuple(
            cell for cell in cells if cell.chart_id == self.gluing.base_chart_id
        )
        handle_cells = tuple(
            cell for cell in cells if cell.chart_id == self.gluing.handle_chart_id
        )
        base_common = (
            _intersection(
                [cell.bounds for cell in base_cells],
                atol=self.atol,
                digits=self.coordinate_digits,
            )
            if base_cells
            else None
        )
        handle_common = (
            _intersection(
                [cell.bounds for cell in handle_cells],
                atol=self.atol,
                digits=self.coordinate_digits,
            )
            if handle_cells
            else None
        )

        if base_cells and base_common is None:
            audit = QuotientIntersectionAudit2D(simplex, False, False, (), "base cells are disjoint")
            self._audit_cache[simplex] = audit
            return audit
        if not handle_cells:
            representatives = (
                QuotientRectangleRepresentative2D(
                    self.gluing.base_chart_id, base_common
                ),
            )
            audit = QuotientIntersectionAudit2D(
                simplex, True, True, representatives, "convex base-chart intersection"
            )
            self._audit_cache[simplex] = audit
            return audit

        option_families = tuple(self._seam_options(cell) for cell in handle_cells)
        seam_hits: list[tuple[Bounds2D, tuple[_SeamOption, ...]]] = []
        if all(option_families):
            for choices in itertools.product(*option_families):
                rectangles = [choice.base_bounds for choice in choices]
                if base_common is not None:
                    rectangles.append(base_common)
                common = _intersection(
                    rectangles,
                    atol=self.atol,
                    digits=self.coordinate_digits,
                )
                if common is not None:
                    seam_hits.append((common, tuple(choices)))

        unique_hits: dict[Bounds2D, tuple[_SeamOption, ...]] = {}
        for bounds, choices in seam_hits:
            unique_hits.setdefault(bounds, choices)

        if base_cells:
            if not seam_hits:
                audit = QuotientIntersectionAudit2D(
                    simplex, False, False, (), "base/handle cells miss both quotient seams"
                )
                self._audit_cache[simplex] = audit
                return audit
            connected = self._seam_union_is_one_interval(tuple(unique_hits))
            # Retain every representative choice even when two distinct seam
            # choices have the same base image.  The common quotient subset is
            # deduplicated only for its connectedness test; face evaluation
            # still has to visit every base/handle representative.
            representatives = self._seam_representatives(seam_hits)
            audit = QuotientIntersectionAudit2D(
                simplex,
                True,
                connected,
                representatives,
                (
                    "one convex quotient-seam interval"
                    if connected
                    else "mixed-chart intersection has disconnected seam components"
                ),
            )
            self._audit_cache[simplex] = audit
            return audit

        # Handle-only intersections normally live in the handle chart.  Any
        # additional common point created solely by choosing different seams
        # would be a quotient self-identification not certified as a good-cover
        # intersection by this initial adapter.
        if handle_common is not None:
            phase_lower, phase_upper = self.gluing.phase_bounds
            spans_both_seams = (
                abs(handle_common[1] - phase_lower) <= self.atol
                and abs(handle_common[3] - phase_upper) <= self.atol
            )
            if spans_both_seams:
                guard_image = self.gluing.guard.segment(
                    handle_common[0],
                    handle_common[2],
                    digits=self.coordinate_digits,
                )
                reset_image = self.gluing.reset.segment(
                    handle_common[0],
                    handle_common[2],
                    digits=self.coordinate_digits,
                )
                if _intersection(
                    (guard_image, reset_image),
                    atol=self.atol,
                    digits=self.coordinate_digits,
                ) is not None:
                    audit = QuotientIntersectionAudit2D(
                        simplex,
                        True,
                        False,
                        (),
                        "one handle intersection spans both seams and is self-identified",
                    )
                    self._audit_cache[simplex] = audit
                    return audit
            mixed_seam_hit = any(
                len({choice.kind for choice in choices}) > 1
                for _bounds, choices in seam_hits
            )
            if mixed_seam_hit:
                audit = QuotientIntersectionAudit2D(
                    simplex,
                    True,
                    False,
                    (),
                    "handle intersection acquires an extra guard/reset identification",
                )
                self._audit_cache[simplex] = audit
                return audit
            representatives = (
                QuotientRectangleRepresentative2D(
                    self.gluing.handle_chart_id, handle_common
                ),
            )
            audit = QuotientIntersectionAudit2D(
                simplex,
                True,
                True,
                representatives,
                "convex handle-chart intersection with no extra seam identification",
            )
            self._audit_cache[simplex] = audit
            return audit

        if seam_hits:
            connected = self._seam_union_is_one_interval(tuple(unique_hits))
            audit = QuotientIntersectionAudit2D(
                simplex,
                True,
                connected,
                self._seam_representatives(seam_hits),
                (
                    "one convex quotient-seam interval"
                    if connected
                    else "disjoint handle cells meet in disconnected quotient seams"
                ),
            )
            self._audit_cache[simplex] = audit
            return audit

        audit = QuotientIntersectionAudit2D(
            simplex, False, False, (), "handle cells are disjoint in chart and quotient"
        )
        self._audit_cache[simplex] = audit
        return audit

    def _seam_union_is_one_interval(self, bounds: tuple[Bounds2D, ...]) -> bool:
        if not bounds:
            return False
        remaining = set(range(len(bounds)))
        components = 0
        while remaining:
            components += 1
            root = remaining.pop()
            frontier = [root]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    candidate
                    for candidate in remaining
                    if _intersects(bounds[current], bounds[candidate], self.atol)
                }
                remaining.difference_update(neighbors)
                frontier.extend(neighbors)
        if components != 1:
            return False
        nondegenerate_axes = {
            0 if bounds_value[2] - bounds_value[0] > self.atol else 1
            for bounds_value in bounds
            if (
                bounds_value[2] - bounds_value[0] > self.atol
                or bounds_value[3] - bounds_value[1] > self.atol
            )
        }
        return len(nondegenerate_axes) <= 1

    def _seam_representatives(
        self,
        seam_hits: Sequence[tuple[Bounds2D, tuple[_SeamOption, ...]]],
    ) -> tuple[QuotientRectangleRepresentative2D, ...]:
        representatives: set[QuotientRectangleRepresentative2D] = set()
        for base_bounds, choices in seam_hits:
            representatives.add(
                QuotientRectangleRepresentative2D(
                    self.gluing.base_chart_id, base_bounds
                )
            )
            for choice in choices:
                intrinsic_lower, intrinsic_upper = choice.embedding.inverse_interval(
                    base_bounds
                )
                representatives.add(
                    QuotientRectangleRepresentative2D(
                        self.gluing.handle_chart_id,
                        _normalize_bounds(
                            (
                                intrinsic_lower,
                                choice.phase,
                                intrinsic_upper,
                                choice.phase,
                            ),
                            digits=self.coordinate_digits,
                        ),
                    )
                )
        return tuple(sorted(representatives))

def atlas_cells_from_phase_space(phase_space: Any) -> tuple[AtlasRectangleCell2D, ...]:
    """Extract actual indexed rectangles from a Python ``CMGDB.Atlas``."""

    if not hasattr(phase_space, "size") or not hasattr(phase_space, "cell"):
        raise TypeError("phase_space must expose size() and cell(index)")
    result = []
    for index in range(int(phase_space.size())):
        tagged = phase_space.cell(index)
        if hasattr(tagged, "chart_id") and hasattr(tagged, "bounds"):
            chart_id = int(tagged.chart_id)
            bounds = tuple(float(value) for value in tagged.bounds)
        else:
            chart_id, raw_bounds = tagged
            chart_id = int(chart_id)
            bounds = tuple(float(value) for value in raw_bounds)
        result.append(AtlasRectangleCell2D(index, chart_id, bounds))
    return tuple(result)


def atlas_cells_from_morse_graph(
    morse_graph: Any, cell_count: int
) -> tuple[AtlasRectangleCell2D, ...]:
    """Extract actual final-grid rectangles using ``phase_space_chart_box``."""

    result = []
    for index in range(int(cell_count)):
        chart_id, bounds = morse_graph.phase_space_chart_box(index)
        result.append(
            AtlasRectangleCell2D(
                index,
                int(chart_id),
                tuple(float(value) for value in bounds),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class AtlasMappingCylinderTopCell:
    """One actual Atlas chart cell identified with a mapping-cylinder top cell."""

    atlas_index: int
    chart_id: int
    complex_cell: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.atlas_index, bool)
            or not isinstance(self.atlas_index, int)
            or self.atlas_index < 0
        ):
            raise ValueError("atlas_index must be a nonnegative integer")
        if (
            isinstance(self.chart_id, bool)
            or not isinstance(self.chart_id, int)
            or self.chart_id < 0
        ):
            raise ValueError("chart_id must be a nonnegative integer")


class AtlasMappingCylinderRegistry:
    """Preserve Atlas ids while using a scalable double mapping cylinder.

    The registry is deliberately explicit: it does not infer CMGDB's global
    numbering or silently manufacture inactive cells.  Base-chart top cells
    must map to :class:`SuspensionBaseCell` objects and handle-chart top cells
    to :class:`GuardPrismCell` objects of the supplied complex.
    """

    def __init__(
        self,
        complex_: DoubleMappingCylinderComplex,
        top_cells: Iterable[AtlasMappingCylinderTopCell],
        *,
        base_chart_id: int,
        handle_chart_id: int,
    ) -> None:
        records = tuple(sorted(top_cells, key=lambda record: record.atlas_index))
        if not records:
            raise ValueError("an Atlas mapping-cylinder registry needs top cells")
        if base_chart_id == handle_chart_id:
            raise ValueError("base and handle chart ids must be different")
        if len({record.atlas_index for record in records}) != len(records):
            raise ValueError("Atlas top-cell indices must be unique")
        if len({record.complex_cell for record in records}) != len(records):
            raise ValueError("each Atlas cell must identify a distinct complex top cell")

        by_index: dict[int, AtlasMappingCylinderTopCell] = {}
        for record in records:
            if record.chart_id not in {base_chart_id, handle_chart_id}:
                raise ValueError(
                    f"Atlas cell {record.atlas_index} uses undeclared chart "
                    f"{record.chart_id}"
                )
            if record.complex_cell not in complex_.cell_set:
                raise ValueError(
                    f"Atlas cell {record.atlas_index} identifies an unknown "
                    "mapping-cylinder cell"
                )
            if complex_.dimension(record.complex_cell) != complex_.max_dimension:
                raise ValueError(
                    f"Atlas cell {record.atlas_index} does not identify a "
                    "top-dimensional mapping-cylinder cell"
                )
            if record.chart_id == base_chart_id and not isinstance(
                record.complex_cell, SuspensionBaseCell
            ):
                raise ValueError("base-chart Atlas cells must identify base-copy cells")
            if record.chart_id == handle_chart_id and not isinstance(
                record.complex_cell, GuardPrismCell
            ):
                raise ValueError("handle-chart Atlas cells must identify phase prisms")
            by_index[record.atlas_index] = record

        self.complex = complex_
        self.base_chart_id = int(base_chart_id)
        self.handle_chart_id = int(handle_chart_id)
        self._by_index = MappingProxyType(by_index)

    @property
    def atlas_indices(self) -> frozenset[int]:
        return frozenset(self._by_index)

    @property
    def top_cells(self) -> tuple[AtlasMappingCylinderTopCell, ...]:
        return tuple(self._by_index.values())

    def record(self, atlas_index: int) -> AtlasMappingCylinderTopCell:
        try:
            return self._by_index[int(atlas_index)]
        except KeyError as error:
            raise KeyError(f"unknown Atlas cell index {atlas_index}") from error

    def complex_cell(self, atlas_index: int) -> Any:
        return self.record(atlas_index).complex_cell


class AtlasMappingCylinderRelativePair:
    """Relative pair generated by selected actual Atlas top-cell ids."""

    def __init__(
        self,
        registry: AtlasMappingCylinderRegistry,
        p1_atlas_cells: Collection[int],
        p0_atlas_cells: Collection[int] = (),
        *,
        index_pair_certified: bool = False,
        certificate_description: str | None = None,
    ) -> None:
        p1_vertices = frozenset(int(value) for value in p1_atlas_cells)
        p0_vertices = frozenset(int(value) for value in p0_atlas_cells)
        if not p1_vertices:
            raise ValueError("P1 needs at least one actual Atlas cell")
        unknown = p1_vertices.union(p0_vertices).difference(registry.atlas_indices)
        if unknown:
            raise ValueError(f"relative pair refers to unknown Atlas cells: {sorted(unknown)!r}")
        if not p0_vertices <= p1_vertices:
            raise ValueError("P0 Atlas cells must be a subset of P1 Atlas cells")
        if index_pair_certified and not certificate_description:
            raise ValueError("a certified index pair requires certificate_description")

        ambient = registry.complex
        p1_cells = ambient.closure(
            registry.complex_cell(index) for index in p1_vertices
        )
        p0_cells = ambient.closure(
            registry.complex_cell(index) for index in p0_vertices
        ) if p0_vertices else frozenset()
        restricted = FiniteCellComplex(
            {cell: ambient.dimension(cell) for cell in ambient.cells if cell in p1_cells},
            {cell: ambient.boundary(cell) for cell in ambient.cells if cell in p1_cells},
            metadata={
                "kind": "atlas-double-mapping-cylinder-relative-pair-candidate",
                "p1_atlas_cells": tuple(sorted(p1_vertices)),
                "p0_atlas_cells": tuple(sorted(p0_vertices)),
                "index_pair_certified": bool(index_pair_certified),
            },
        )
        self.registry = registry
        self.p1_atlas_cells = p1_vertices
        self.p0_atlas_cells = p0_vertices
        self.complex = restricted
        self.relative_pair = RelativeCellPair(
            restricted,
            restricted.cell_set,
            p0_cells,
        )
        self.index_pair_certified = bool(index_pair_certified)
        self.certificate_description = certificate_description


class AtlasRelativeIndexPair2D:
    """Relative cellular pair induced by two actual Atlas top-cell families.

    ``index_pair_certified`` is provenance supplied by the caller; the class
    validates only the finite subcomplex and relation-preservation obligations.
    It deliberately does not infer isolation from an SCC or Morse label.
    """

    def __init__(
        self,
        nerve: AtlasQuotientNerveComplex2D,
        p1_atlas_cells: Collection[int],
        p0_atlas_cells: Collection[int] = (),
        *,
        index_pair_certified: bool = False,
        certificate_description: str | None = None,
    ) -> None:
        p1_vertices = frozenset(int(value) for value in p1_atlas_cells)
        p0_vertices = frozenset(int(value) for value in p0_atlas_cells)
        if not p1_vertices:
            raise ValueError("P1 needs at least one actual Atlas cell")
        unknown = p1_vertices.union(p0_vertices).difference(nerve.atlas_indices)
        if unknown:
            raise ValueError(f"relative pair refers to unknown Atlas cells: {sorted(unknown)!r}")
        if not p0_vertices <= p1_vertices:
            raise ValueError("P0 Atlas cells must be a subset of P1 Atlas cells")
        if index_pair_certified and not certificate_description:
            raise ValueError("a certified index pair requires certificate_description")

        p1_cells = nerve.induced_cells(p1_vertices)
        p0_cells = nerve.induced_cells(p0_vertices)
        restricted = FiniteCellComplex(
            {cell: nerve.dimension(cell) for cell in nerve.cells if cell in p1_cells},
            {cell: nerve.boundary(cell) for cell in nerve.cells if cell in p1_cells},
            metadata={
                "kind": "actual-atlas-relative-index-pair-candidate",
                "p1_atlas_cells": tuple(sorted(p1_vertices)),
                "p0_atlas_cells": tuple(sorted(p0_vertices)),
                "index_pair_certified": bool(index_pair_certified),
            },
        )
        self.nerve = nerve
        self.p1_atlas_cells = p1_vertices
        self.p0_atlas_cells = p0_vertices
        self.complex = restricted
        self.relative_pair = RelativeCellPair(restricted, restricted.cell_set, p0_cells)
        self.index_pair_certified = bool(index_pair_certified)
        self.certificate_description = certificate_description

    def require_top_relation_preserves_pair(
        self, relation: Mapping[int, Collection[int]]
    ) -> None:
        missing = self.p1_atlas_cells.difference(relation)
        if missing:
            raise ValueError(f"top-cell relation is missing P1 sources: {sorted(missing)!r}")
        for source in self.p1_atlas_cells:
            targets = frozenset(int(value) for value in relation[source])
            if not targets <= self.p1_atlas_cells:
                raise ValueError(
                    f"top-cell relation leaves P1 from source {source}: "
                    f"{sorted(targets - self.p1_atlas_cells)!r}"
                )
            if source in self.p0_atlas_cells and not targets <= self.p0_atlas_cells:
                raise ValueError(
                    f"top-cell relation does not preserve P0 from source {source}: "
                    f"{sorted(targets - self.p0_atlas_cells)!r}"
                )


@dataclass(frozen=True)
class RawAtlasCellCover2D:
    source: AtlasNerveSimplex
    representatives: tuple[QuotientRectangleRepresentative2D, ...]
    returned_pieces: int
    target_atlas_cells: frozenset[int]


@dataclass(frozen=True)
class AtlasPhysicalConleyPreparation2D:
    """Algebraically complete preparation with explicit certification gates."""

    pair: AtlasRelativeIndexPair2D
    carrier: FixedTimeCarrier
    chain_map: CellularChainMap
    payload: CMGDBRelativeHomologyPayload
    raw_covers: Mapping[AtlasNerveSimplex, RawAtlasCellCover2D]
    outer_enclosure_certified: bool
    carrier_construction: str = "face-evaluated raw Atlas covers"
    relation_vertex_images: Mapping[int, frozenset[int]] = field(
        default_factory=dict
    )

    @property
    def finite_relation_algebra_validated(self) -> bool:
        """The constructor has discharged every finite-complex algebra gate."""

        return bool(
            self.carrier.acyclicity_validated
            and self.carrier.preserves_pair(self.pair.relative_pair)
            and self.carrier.carries(self.chain_map)
            and self.chain_map.preserves(self.pair.relative_pair.p1_cells)
            and self.chain_map.preserves(self.pair.relative_pair.p0_cells)
        )

    @property
    def continuous_system_conley_index_certified(self) -> bool:
        return bool(
            self.outer_enclosure_certified and self.pair.index_pair_certified
        )

    def compute_finite_relation_shift_class(self) -> Mapping[str, object]:
        """Compute the shift class of the validated finite relation/pair.

        This endpoint deliberately makes no continuous-system certification
        claim.  It is the CMGDB-style combinatorial result after the quotient
        nerve, carrier, pair, and chain-map gates have passed.
        """

        if not self.finite_relation_algebra_validated:
            raise PhysicalConleyCertificationError(
                "finite relation shift class blocked: an algebraic carrier/pair "
                "gate no longer validates"
            )
        try:
            import CMGDB
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("the local CMGDB explicit-chain bridge is not installed") from error
        if not hasattr(CMGDB, "ComputeRelativeHomologyShiftClass"):
            raise RuntimeError("installed CMGDB lacks ComputeRelativeHomologyShiftClass")
        result = dict(
            CMGDB.ComputeRelativeHomologyShiftClass(*self.payload.as_compute_args())
        )
        result["result_scope"] = "finite_reset_quotient_relation"
        result["continuous_system_conley_index_certified"] = False
        result["finite_relation_algebra_validated"] = True
        return result

    def compute_cmgdb_shift_class(self) -> Mapping[str, object]:
        if not self.outer_enclosure_certified:
            raise PhysicalConleyCertificationError(
                "CMGDB shift class blocked: the face-evaluated box map is not a "
                "certified whole-cell outer enclosure"
            )
        if not self.pair.index_pair_certified:
            raise PhysicalConleyCertificationError(
                "CMGDB shift class blocked: the relative pair is not certified as "
                "an index pair for the fixed-time suspension map"
            )
        result = dict(self.compute_finite_relation_shift_class())
        result["result_scope"] = "certified_continuous_fixed_time_suspension_map"
        result["continuous_system_conley_index_certified"] = True
        return result


@dataclass(frozen=True)
class AtlasMappingCylinderConleyPreparation(AtlasPhysicalConleyPreparation2D):
    """Validated finite algebra on an independent-guard mapping cylinder."""

    pair: AtlasMappingCylinderRelativePair


def _parse_tagged_piece(piece: Any) -> tuple[int, tuple[float, ...]]:
    if hasattr(piece, "chart_id") and hasattr(piece, "bounds"):
        return int(piece.chart_id), tuple(float(value) for value in piece.bounds)
    chart_id, bounds = piece
    return int(chart_id), tuple(float(value) for value in bounds)


def prepare_atlas_physical_conley_2d(
    pair: AtlasRelativeIndexPair2D,
    *,
    box_map: Callable[[int, Sequence[float]], Iterable[Any]],
    atlas: Any,
    expected_top_relation: Mapping[int, Collection[int]] | None = None,
    outer_enclosure_certified: bool = False,
    modulus: int = 5,
) -> AtlasPhysicalConleyPreparation2D:
    """Evaluate every source simplex and build a carried relative chain map.

    If ``expected_top_relation`` is supplied, every vertex evaluation must
    reproduce its MapGraph adjacency exactly.  This is the provenance gate
    connecting the face-level carrier construction to the actual CMGDB run.
    """

    if modulus != 5:
        raise ValueError("the CMGDB explicit-chain bridge currently uses GF(5)")
    if not hasattr(atlas, "cover"):
        raise TypeError("atlas must expose cover(chart_id, bounds)")
    if expected_top_relation is not None:
        pair.require_top_relation_preserves_pair(expected_top_relation)

    nerve = pair.nerve
    relative_complex = pair.complex
    raw_records: dict[AtlasNerveSimplex, RawAtlasCellCover2D] = {}
    raw_image_cells: dict[AtlasNerveSimplex, frozenset[AtlasNerveSimplex]] = {}
    for source in relative_complex.cells:
        representatives = nerve.representatives(source)
        target_vertices: set[int] = set()
        returned_pieces = 0
        for representative in representatives:
            pieces = tuple(box_map(representative.chart_id, representative.bounds))
            returned_pieces += len(pieces)
            for piece in pieces:
                chart_id, bounds = _parse_tagged_piece(piece)
                target_vertices.update(int(value) for value in atlas.cover(chart_id, bounds))
        unknown = target_vertices.difference(nerve.atlas_indices)
        if unknown:
            raise ValueError(
                f"raw cover of {source!r} returned unknown Atlas cells: {sorted(unknown)!r}"
            )
        outside = target_vertices.difference(pair.p1_atlas_cells)
        if outside:
            raise ValueError(
                f"raw cover of {source!r} leaves P1 through Atlas cells "
                f"{sorted(outside)!r}"
            )
        if not target_vertices:
            raise ValueError(f"raw cover of {source!r} is empty")

        if source.dimension == 0 and expected_top_relation is not None:
            atlas_source = source.vertices[0]
            expected = frozenset(int(value) for value in expected_top_relation[atlas_source])
            if frozenset(target_vertices) != expected:
                raise ValueError(
                    "face evaluator does not reproduce MapGraph adjacency for Atlas "
                    f"cell {atlas_source}: raw={sorted(target_vertices)!r}, "
                    f"mapgraph={sorted(expected)!r}"
                )

        target_subcomplex = frozenset(
            cell
            for cell in relative_complex.cells
            if set(cell.vertices) <= target_vertices
        )
        if not target_subcomplex:
            raise ValueError(f"raw target cover of {source!r} has empty nerve subcomplex")
        raw_image_cells[source] = target_subcomplex
        raw_records[source] = RawAtlasCellCover2D(
            source=source,
            representatives=representatives,
            returned_pieces=returned_pieces,
            target_atlas_cells=frozenset(target_vertices),
        )

    carrier_generators: dict[AtlasNerveSimplex, frozenset[AtlasNerveSimplex]] = {}
    for source in relative_complex.cells:
        generators: set[AtlasNerveSimplex] = set()
        for face in relative_complex.closure((source,)):
            generators.update(raw_image_cells[face])
        carrier_generators[source] = frozenset(generators)

    carrier = FixedTimeCarrier(
        relative_complex,
        carrier_generators,
        modulus=modulus,
        validate_acyclic=True,
    )
    carrier.require_preserves_pair(pair.relative_pair)
    chain_map = carrier.construct_chain_map(pair=pair.relative_pair)
    carrier.require_carries(chain_map)
    payload = chain_map.to_cmgdb_payload(pair.relative_pair)
    return AtlasPhysicalConleyPreparation2D(
        pair=pair,
        carrier=carrier,
        chain_map=chain_map,
        payload=payload,
        raw_covers=MappingProxyType(raw_records),
        outer_enclosure_certified=bool(outer_enclosure_certified),
    )


def prepare_atlas_mapping_cylinder_conley(
    pair: AtlasMappingCylinderRelativePair,
    *,
    top_relation: Mapping[int, Collection[int]],
    cell_carrier_generators: Mapping[Any, Collection[Any]],
    outer_enclosure_certified: bool = False,
    modulus: int = 5,
) -> AtlasMappingCylinderConleyPreparation:
    """Select a relative chain map on a scalable Atlas mapping cylinder.

    ``top_relation`` is the actual CMGDB relation on Atlas top cells.  Targets
    of a ``P0`` source may lie outside the registered ``P1`` family: those are
    retained as exit provenance, while the required relative-pair gate is
    ``F(P0) intersection P1 <= P0``.  ``cell_carrier_generators`` must be
    supplied for every cell of the
    relative mapping-cylinder complex, including faces.  The latter is the
    indispensable face-level geometric input: this function never derives it
    from a pairwise graph or from top-cell adjacency alone.

    For non-exit top cells, the carrier value must contain every recorded
    relation target.  Sources in ``P0`` are instead governed by explicit
    carrier preservation of the collapsed exit subcomplex.  Acyclicity, face
    nesting, pair preservation, subordinate selection, and ``dF = Fd`` are
    all hard constructor gates.
    """

    if modulus != 5:
        raise ValueError("the CMGDB explicit-chain bridge currently uses GF(5)")
    missing = pair.p1_atlas_cells.difference(top_relation)
    if missing:
        raise ValueError(f"top-cell relation is missing P1 sources: {sorted(missing)!r}")

    registry = pair.registry
    normalized_relation: dict[int, frozenset[int]] = {}
    for source in pair.p1_atlas_cells:
        targets = frozenset(int(value) for value in top_relation[source])
        if source in pair.p0_atlas_cells:
            enters_pair_interior = (
                targets.intersection(pair.p1_atlas_cells)
                .difference(pair.p0_atlas_cells)
            )
            if enters_pair_interior:
                raise ValueError(
                    f"top-cell relation from exit source {source} enters "
                    "P1\\P0 through Atlas cells "
                    f"{sorted(enters_pair_interior)!r}"
                )
        else:
            outside = targets.difference(pair.p1_atlas_cells)
            if outside:
                raise ValueError(
                    f"top-cell relation leaves P1 from non-exit source {source}: "
                    f"{sorted(outside)!r}"
                )
            if not targets:
                raise ValueError(
                    f"top-cell relation is empty on non-exit source {source}"
                )
        normalized_relation[source] = targets

    carrier = FixedTimeCarrier(
        pair.complex,
        cell_carrier_generators,
        modulus=modulus,
        validate_acyclic=True,
    )
    carrier.require_preserves_pair(pair.relative_pair)

    for source in pair.p1_atlas_cells.difference(pair.p0_atlas_cells):
        source_cell = registry.complex_cell(source)
        carrier_value = carrier.image(source_cell)
        missing_targets = {
            target
            for target in normalized_relation[source]
            if registry.complex_cell(target) not in carrier_value
        }
        if missing_targets:
            raise ValueError(
                "face-level carrier does not contain the actual MapGraph targets "
                f"from Atlas cell {source}: {sorted(missing_targets)!r}"
            )

    chain_map = carrier.construct_chain_map(pair=pair.relative_pair)
    carrier.require_carries(chain_map)
    payload = chain_map.to_cmgdb_payload(pair.relative_pair)
    return AtlasMappingCylinderConleyPreparation(
        pair=pair,
        carrier=carrier,
        chain_map=chain_map,
        payload=payload,
        raw_covers=MappingProxyType({}),
        outer_enclosure_certified=bool(outer_enclosure_certified),
        carrier_construction=(
            "face-evaluated independent-guard mapping-cylinder carrier "
            "subordinate to the actual Atlas top-cell relation off P0, with "
            "an explicit pair-preserving algebraic extension on P0"
        ),
        relation_vertex_images=MappingProxyType(normalized_relation),
    )


def prepare_atlas_relation_conley_2d(
    pair: AtlasRelativeIndexPair2D,
    *,
    top_relation: Mapping[int, Collection[int]],
    outer_enclosure_certified: bool = False,
    use_exit_component_carrier: bool = True,
    modulus: int = 5,
) -> AtlasPhysicalConleyPreparation2D:
    """Build an acyclic carrier directly from an actual Atlas MapGraph relation.

    For a simplex ``sigma``, the raw carrier vertices are the union of the
    recorded images of its Atlas vertices.  This contains the image of the
    corresponding finite intersection whenever each closed top-cell relation
    value is a valid outer enclosure, and it is automatically face nested.

    A relative exit set needs a pair-preserving extension before taking the
    quotient.  With ``use_exit_component_carrier=True``, every vertex in ``P0``
    is carried by the entire connected component of the induced ``P0`` nerve
    containing it.  This is not an identity or a physical-map claim: all of
    ``P0`` is zero in ``C(P1)/C(P0)``.  Component acyclicity and the resulting
    carrier's pair preservation are still checked explicitly.
    """

    if modulus != 5:
        raise ValueError("the CMGDB explicit-chain bridge currently uses GF(5)")
    missing = pair.p1_atlas_cells.difference(top_relation)
    if missing:
        raise ValueError(f"top-cell relation is missing P1 sources: {sorted(missing)!r}")

    normalized_relation: dict[int, frozenset[int]] = {}
    for source in pair.p1_atlas_cells:
        targets = frozenset(int(value) for value in top_relation[source])
        unknown = targets.difference(pair.nerve.atlas_indices)
        if unknown:
            raise ValueError(
                f"top-cell relation from {source} contains unknown Atlas cells: "
                f"{sorted(unknown)!r}"
            )
        if source not in pair.p0_atlas_cells:
            outside = targets.difference(pair.p1_atlas_cells)
            if outside:
                raise ValueError(
                    f"top-cell relation leaves P1 from non-exit source {source}: "
                    f"{sorted(outside)!r}"
                )
            if not targets:
                raise ValueError(
                    f"top-cell relation is empty on non-exit source {source}"
                )
        normalized_relation[source] = targets

    exit_component_by_vertex: dict[int, frozenset[int]] = {}
    if pair.p0_atlas_cells and use_exit_component_carrier:
        adjacency = {vertex: set() for vertex in pair.p0_atlas_cells}
        for simplex in pair.relative_pair.p0_cells:
            if simplex.dimension != 1:
                continue
            first, second = simplex.vertices
            adjacency[first].add(second)
            adjacency[second].add(first)
        remaining = set(pair.p0_atlas_cells)
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            frontier = [root]
            while frontier:
                source = frontier.pop()
                neighbors = adjacency[source].intersection(remaining)
                remaining.difference_update(neighbors)
                component.update(neighbors)
                frontier.extend(neighbors)
            frozen_component = frozenset(component)
            for vertex in component:
                exit_component_by_vertex[vertex] = frozen_component

    vertex_images: dict[int, frozenset[int]] = {}
    for source in pair.p1_atlas_cells:
        if source in pair.p0_atlas_cells:
            if not use_exit_component_carrier:
                targets = normalized_relation[source]
                outside = targets.difference(pair.p0_atlas_cells)
                if outside:
                    raise ValueError(
                        f"top-cell relation does not preserve P0 from {source}: "
                        f"{sorted(outside)!r}"
                    )
                if not targets:
                    raise ValueError(f"top-cell relation is empty on P0 source {source}")
                vertex_images[source] = targets
            else:
                vertex_images[source] = exit_component_by_vertex[source]
        else:
            vertex_images[source] = normalized_relation[source]

    complex_ = pair.complex
    carrier_generators: dict[AtlasNerveSimplex, frozenset[AtlasNerveSimplex]] = {}
    for source in complex_.cells:
        target_vertices: set[int] = set()
        for vertex in source.vertices:
            target_vertices.update(vertex_images[vertex])
        target_subcomplex = frozenset(
            cell for cell in complex_.cells if set(cell.vertices) <= target_vertices
        )
        if not target_subcomplex:
            raise ValueError(f"relation carrier of {source!r} is empty")
        carrier_generators[source] = target_subcomplex

    carrier = FixedTimeCarrier(
        complex_,
        carrier_generators,
        modulus=modulus,
        validate_acyclic=True,
    )
    carrier.require_preserves_pair(pair.relative_pair)
    chain_map = carrier.construct_chain_map(pair=pair.relative_pair)
    carrier.require_carries(chain_map)
    payload = chain_map.to_cmgdb_payload(pair.relative_pair)
    return AtlasPhysicalConleyPreparation2D(
        pair=pair,
        carrier=carrier,
        chain_map=chain_map,
        payload=payload,
        raw_covers=MappingProxyType({}),
        outer_enclosure_certified=bool(outer_enclosure_certified),
        carrier_construction=(
            "actual Atlas MapGraph vertex union with connected P0-component extension"
            if use_exit_component_carrier
            else "actual Atlas MapGraph vertex union"
        ),
        relation_vertex_images=MappingProxyType(vertex_images),
    )


__all__ = [
    "AffineBoundaryEmbedding2D",
    "AtlasGoodCoverError",
    "AtlasMappingCylinderConleyPreparation",
    "AtlasMappingCylinderRegistry",
    "AtlasMappingCylinderRelativePair",
    "AtlasMappingCylinderTopCell",
    "AtlasNerveSimplex",
    "AtlasPhysicalConleyPreparation2D",
    "AtlasQuotientNerveComplex2D",
    "AtlasRectangleCell2D",
    "AtlasRelativeIndexPair2D",
    "AtlasResetGluing2D",
    "AtlasSeamSubcomplexAudit2D",
    "PhysicalConleyCertificationError",
    "QuotientIntersectionAudit2D",
    "QuotientRectangleRepresentative2D",
    "RawAtlasCellCover2D",
    "atlas_cells_from_morse_graph",
    "atlas_cells_from_phase_space",
    "prepare_atlas_physical_conley_2d",
    "prepare_atlas_mapping_cylinder_conley",
    "prepare_atlas_relation_conley_2d",
]
