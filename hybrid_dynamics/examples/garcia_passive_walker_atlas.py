"""Garcia passive-walker plumbing through ``CMGDB.AtlasModel``.

CMGDB owns the Atlas grid, map graph, SCC decomposition, and Morse order.  The
hybrid-specific callback is :class:`CMGDBSuspensionBoxMap`, whose target is a
finite union of rectangles tagged by either

* the physical base chart ``(theta, theta_dot, phi, phi_dot)``; or
* the intrinsic outgoing-guard chart ``(theta, theta_dot, rho, s)``.

Here ``rho`` parametrizes the admissible interval of ``phi_dot``.  Consequently
the handle embedding enforces both ``phi = 2*theta`` and the transverse
heel-strike inequality on the whole rectangular chart.  This is preferable to
using a rectangular ``(theta, theta_dot, phi_dot)`` box containing states that
are not in the outgoing guard.

The default depth-four run is deliberately only a plumbing acceptance run.  An
optional native active-subgrid mode constructs a positive-radius union of full
four-dimensional dyadic cells around the two stored strides and two handle
traversals.  It records every active-boundary exit and rejects radius zero as
an orbit-only family.  The bounded depth/radius screen currently fails closure
and image-connectedness, so neither mode is a recovered Morse decomposition,
a global attractor-lattice computation, a whole-cell enclosure proof, or a
Conley-index computation.
"""

from __future__ import annotations

import itertools
import threading
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar

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
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    GUARD_ALIGNED_DOMAIN_BOUNDS,
    PERIOD_TWO_POINT_A,
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
    guard_aligned_post_impact_state,
    post_impact_state,
)


BASE_CHART_ID = 0
HANDLE_CHART_ID = 1
DEFAULT_T_STAR = 0.5
DEFAULT_BASE_BOUNDS = (
    (-0.32, 0.32),
    (-0.35, 0.05),
    (-0.65, 0.65),
    (-0.55, 0.15),
)
# The largest theta-dot for which the admissible phi-dot interval is
# nondegenerate is (phi_dot_max - eta)/2 = 0.025.  The small margin avoids a
# collapsed coordinate at that single endpoint.
DEFAULT_GUARD_PARAMETER_MARGIN = 1.0e-3


@dataclass(frozen=True, order=True)
class AtlasWalkerCell:
    """One tagged, closed CMGDB Atlas cell."""

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


@dataclass(frozen=True)
class GarciaWalkerActiveFamily:
    """A full-dimensional dyadic neighborhood of the stored period-two gait.

    Seed cells meet either of the two continuous stride arcs or either handle
    traversal.  The active family contains every same-chart dyadic cell within
    the declared Chebyshev index radius of a seed.  It is therefore a finite
    union of closed four-dimensional boxes, not an orbit-only vertex list.
    """

    axis_depth: int
    stencil_radius: int
    base_seed_cells: tuple[tuple[int, ...], ...]
    handle_seed_cells: tuple[tuple[int, ...], ...]
    tagged_cells: tuple[tuple[int, int, tuple[int, ...]], ...]

    @property
    def subdivisions_per_axis(self) -> int:
        return 2**self.axis_depth

    def indices_for_chart(self, chart_id: int) -> frozenset[tuple[int, ...]]:
        return frozenset(
            coordinates
            for tag, _depth, coordinates in self.tagged_cells
            if tag == chart_id
        )

    def chart_counts(self) -> dict[int, int]:
        return dict(Counter(tag for tag, _depth, _coordinates in self.tagged_cells))


@dataclass(frozen=True)
class ActiveSourceCoverage:
    """Coverage of one nonempty geometric box-map value by the active family."""

    source_chart_id: int
    source_bounds: tuple[float, ...]
    returned_pieces: int
    active_target_cells: int
    missing_target_dyadic_cells: int
    pieces_without_active_intersection: int
    pieces_crossing_ambient_chart_boundary: int

    @property
    def explicit_empty_image(self) -> bool:
        return self.returned_pieces == 0

    @property
    def active_boundary_exit(self) -> bool:
        return bool(
            self.returned_pieces > 0
            and (
                self.missing_target_dyadic_cells > 0
                or self.pieces_crossing_ambient_chart_boundary > 0
            )
        )

    @property
    def wholly_outside_active_family(self) -> bool:
        return bool(self.returned_pieces > 0 and self.active_target_cells == 0)


@dataclass(frozen=True)
class ActiveSubgridCoverageDiagnostics:
    """Unique-source active-boundary diagnostics from the native callback."""

    records: tuple[ActiveSourceCoverage, ...]

    @property
    def explicit_empty_sources(self) -> int:
        return sum(record.explicit_empty_image for record in self.records)

    @property
    def boundary_exit_sources(self) -> int:
        return sum(record.active_boundary_exit for record in self.records)

    @property
    def wholly_exiting_sources(self) -> int:
        return sum(record.wholly_outside_active_family for record in self.records)

    @property
    def pieces_without_active_intersection(self) -> int:
        return sum(record.pieces_without_active_intersection for record in self.records)

    @property
    def missing_target_dyadic_cells(self) -> int:
        return sum(record.missing_target_dyadic_cells for record in self.records)

    @property
    def ambient_boundary_pieces(self) -> int:
        return sum(
            record.pieces_crossing_ambient_chart_boundary for record in self.records
        )


class _ActiveSubgridTrackingBoxMap:
    """Wrap the physical callback and audit its values against an active Atlas."""

    def __init__(
        self,
        box_map: CMGDBSuspensionBoxMap,
        family: GarciaWalkerActiveFamily,
    ) -> None:
        self.box_map = box_map
        self.charts = box_map.charts
        self.family = family
        self._atlas: Any | None = None
        self._records: dict[tuple[object, ...], ActiveSourceCoverage] = {}
        self._lock = threading.Lock()
        self._active_indices = {
            chart_id: family.indices_for_chart(chart_id)
            for chart_id in (self.charts.base_chart_id, self.charts.handle_chart_id)
        }

    def bind_atlas(self, atlas: Any) -> None:
        self._atlas = atlas

    def reset_diagnostics(self) -> None:
        with self._lock:
            self._records = {}

    def diagnostics(self) -> ActiveSubgridCoverageDiagnostics:
        with self._lock:
            return ActiveSubgridCoverageDiagnostics(
                tuple(self._records[key] for key in sorted(self._records))
            )

    def __call__(
        self,
        source_chart_id: int,
        source_bounds: Sequence[float],
    ) -> list[tuple[int, list[float]]]:
        if self._atlas is None:
            raise RuntimeError("active-subgrid tracker has no bound Atlas")
        pieces = self.box_map(source_chart_id, source_bounds)
        active_targets: set[int] = set()
        missing_targets: set[tuple[int, tuple[int, ...]]] = set()
        uncovered_pieces = 0
        ambient_pieces = 0
        for target_chart_id, target_bounds in pieces:
            covered = set(
                int(index)
                for index in self._atlas.cover(target_chart_id, target_bounds)
            )
            active_targets.update(covered)
            if not covered:
                uncovered_pieces += 1
            dyadic, crosses_ambient = self._dyadic_cover(
                target_chart_id,
                target_bounds,
            )
            missing_targets.update(
                (target_chart_id, coordinates)
                for coordinates in dyadic - self._active_indices[target_chart_id]
            )
            ambient_pieces += int(crosses_ambient)

        record = ActiveSourceCoverage(
            source_chart_id=int(source_chart_id),
            source_bounds=tuple(float(value) for value in source_bounds),
            returned_pieces=len(pieces),
            active_target_cells=len(active_targets),
            missing_target_dyadic_cells=len(missing_targets),
            pieces_without_active_intersection=uncovered_pieces,
            pieces_crossing_ambient_chart_boundary=ambient_pieces,
        )
        key = (
            int(source_chart_id),
            *(round(float(value), 14) for value in source_bounds),
        )
        with self._lock:
            self._records[key] = record
        return pieces

    def _dyadic_cover(
        self,
        chart_id: int,
        flat_bounds: Sequence[float],
    ) -> tuple[set[tuple[int, ...]], bool]:
        bounds = np.asarray(self.charts.bounds_for(chart_id), dtype=np.float64)
        dimension = len(bounds)
        values = np.asarray(flat_bounds, dtype=np.float64)
        lower = values[:dimension]
        upper = values[dimension:]
        scale = bounds[:, 1] - bounds[:, 0]
        normalized_lower = (lower - bounds[:, 0]) / scale
        normalized_upper = (upper - bounds[:, 0]) / scale
        tolerance = 1.0e-12
        crosses_ambient = bool(
            np.any(normalized_lower < -tolerance)
            or np.any(normalized_upper > 1.0 + tolerance)
        )
        normalized_lower = np.maximum(normalized_lower, 0.0)
        normalized_upper = np.minimum(normalized_upper, 1.0)
        if np.any(normalized_lower > normalized_upper + tolerance):
            return set(), crosses_ambient

        subdivisions = self.family.subdivisions_per_axis
        coordinate_ranges = []
        for low, high in zip(normalized_lower, normalized_upper):
            first = max(0, int(np.ceil(subdivisions * low - tolerance)) - 1)
            last = min(
                subdivisions - 1,
                int(np.floor(subdivisions * high + tolerance)),
            )
            coordinate_ranges.append(range(first, last + 1))
        return {
            tuple(int(value) for value in coordinates)
            for coordinates in itertools.product(*coordinate_ranges)
        }, crosses_ambient


@dataclass(frozen=True)
class WalkerReferenceFailure:
    """A stored-gait source point whose endpoint could not be evaluated."""

    chart_id: int
    coordinates: tuple[float, ...]
    label: str
    message: str


class GarciaWalkerQuotientIncidence:
    """Closed-cell incidence after the two passive-walker seam gluings.

    Same-chart incidence is exact rectangle intersection.  Cross-chart
    incidence solves the one-dimensional feasibility problem induced by the
    nonlinear guard parametrization or reset map.  It is a geometric audit of
    the declared closed cells, not a replacement for a quotient cell complex.
    """

    def __init__(
        self,
        charts: SuspensionAtlasCharts,
        *,
        phi_dot_min: float,
        phi_dot_max: float,
        transversality_eta: float,
        atol: float = 1e-10,
    ) -> None:
        self.charts = charts
        self.phi_dot_min = float(phi_dot_min)
        self.phi_dot_max = float(phi_dot_max)
        self.transversality_eta = float(transversality_eta)
        self.atol = float(atol)

    @staticmethod
    def _rectangles_intersect(
        first_lower: np.ndarray,
        first_upper: np.ndarray,
        second_lower: np.ndarray,
        second_upper: np.ndarray,
        atol: float,
    ) -> bool:
        return bool(
            np.all(first_lower <= second_upper + atol)
            and np.all(second_lower <= first_upper + atol)
        )

    @staticmethod
    def _closed_interval(
        *intervals: tuple[float, float],
        atol: float,
    ) -> tuple[float, float] | None:
        lower = max(interval[0] for interval in intervals)
        upper = min(interval[1] for interval in intervals)
        if lower > upper + atol:
            return None
        return float(lower), float(upper)

    def _lower_guard_phi_dot(self, theta_dot: float) -> float:
        return max(
            self.phi_dot_min,
            2.0 * float(theta_dot) + self.transversality_eta,
        )

    def _scalar_feasible(
        self,
        interval: tuple[float, float],
        gap: Callable[[float], float],
    ) -> bool:
        """Test whether a continuous interval-overlap gap is nonpositive."""

        lower, upper = interval
        if upper - lower <= self.atol:
            return gap(0.5 * (lower + upper)) <= self.atol
        nodes = np.linspace(lower, upper, 33)
        values = np.asarray([gap(float(value)) for value in nodes])
        if float(np.min(values)) <= self.atol:
            return True
        # The gap is a max/min of smooth monotone expressions on this guard
        # patch.  Bounded minimization is a useful extra check between nodes.
        for left, right in zip(nodes[:-1], nodes[1:]):
            result = minimize_scalar(
                gap,
                bounds=(float(left), float(right)),
                method="bounded",
                options={"xatol": max(1e-14, self.atol * 0.1)},
            )
            if result.success and float(result.fun) <= self.atol:
                return True
        return False

    def _monotone_constraints_feasible(
        self,
        interval: tuple[float, float],
        constraints: Collection[tuple[Callable[[float], float], int]],
    ) -> bool:
        """Intersect scalar monotone inequalities ``gap(value) <= atol``.

        ``direction`` is ``1`` for a nondecreasing gap, ``-1`` for a
        nonincreasing gap, and ``0`` for a constant gap.  Each feasible set is
        therefore a closed interval.  Bisection locates its one possible
        endpoint; intersecting those intervals is both substantially cheaper
        and less heuristic than minimizing a max/min gap on 32 subintervals.
        """

        lower, upper = interval
        for gap, direction in constraints:
            low_value = float(gap(lower))
            high_value = float(gap(upper))
            if not np.isfinite(low_value) or not np.isfinite(high_value):
                return False
            if max(low_value, high_value) <= self.atol:
                continue
            if min(low_value, high_value) > self.atol or direction == 0:
                return False
            left, right = lower, upper
            if direction > 0:
                if low_value > self.atol:
                    return False
                for _ in range(60):
                    midpoint = 0.5 * (left + right)
                    if gap(midpoint) <= self.atol:
                        left = midpoint
                    else:
                        right = midpoint
                upper = left
            else:
                if high_value > self.atol:
                    return False
                for _ in range(60):
                    midpoint = 0.5 * (left + right)
                    if gap(midpoint) <= self.atol:
                        right = midpoint
                    else:
                        left = midpoint
                lower = right
            if lower > upper + self.atol:
                return False
        return True

    def _guard_face_intersects(
        self,
        base: AtlasWalkerCell,
        handle: AtlasWalkerCell,
    ) -> bool:
        base_lower, base_upper = base.lower, base.upper
        handle_lower, handle_upper = handle.lower, handle.upper
        theta = self._closed_interval(
            (float(handle_lower[0]), float(handle_upper[0])),
            (float(base_lower[0]), float(base_upper[0])),
            (0.5 * float(base_lower[2]), 0.5 * float(base_upper[2])),
            atol=self.atol,
        )
        if theta is None:
            return False
        theta_dot = self._closed_interval(
            (float(handle_lower[1]), float(handle_upper[1])),
            (float(base_lower[1]), float(base_upper[1])),
            atol=self.atol,
        )
        if theta_dot is None:
            return False
        rho_lower = float(handle_lower[2])
        rho_upper = float(handle_upper[2])

        def image_lower(value: float) -> float:
            lower_phi = self._lower_guard_phi_dot(value)
            width = self.phi_dot_max - lower_phi
            return lower_phi + rho_lower * width

        def image_upper(value: float) -> float:
            lower_phi = self._lower_guard_phi_dot(value)
            width = self.phi_dot_max - lower_phi
            return lower_phi + rho_upper * width

        return self._monotone_constraints_feasible(
            theta_dot,
            (
                (
                    lambda value: image_lower(value) - float(base_upper[3]),
                    1,
                ),
                (
                    lambda value: float(base_lower[3]) - image_upper(value),
                    -1,
                ),
            ),
        )

    def _reset_face_intersects(
        self,
        base: AtlasWalkerCell,
        handle: AtlasWalkerCell,
    ) -> bool:
        base_lower, base_upper = base.lower, base.upper
        handle_lower, handle_upper = handle.lower, handle.upper
        theta = self._closed_interval(
            (float(handle_lower[0]), float(handle_upper[0])),
            (-float(base_upper[0]), -float(base_lower[0])),
            (-0.5 * float(base_upper[2]), -0.5 * float(base_lower[2])),
            atol=self.atol,
        )
        if theta is None:
            return False
        # On the physical Garcia guard patch theta lies in (-pi/4, 0), hence
        # c=cos(2 theta) increases and q=c(1-c) decreases.  The reset images
        # of the handle-velocity interval are [c*hL,c*hU] and [q*hL,q*hU].
        # Three closed intervals on R have a common point iff all three pairs
        # intersect.  Multiplication by the positive c and q turns those six
        # pairwise conditions into monotone scalar inequalities in c.
        theta_lower, theta_upper = theta
        c_lower = float(np.cos(2.0 * theta_lower))
        c_upper = float(np.cos(2.0 * theta_upper))
        if not (
            0.0 < c_lower <= c_upper < 1.0
            and -0.25 * np.pi < theta_lower <= theta_upper < 0.0
        ):
            # This fallback is outside the declared physical chart but keeps
            # the incidence helper usable for an explicitly enlarged chart.
            velocity_lower = float(handle_lower[1])
            velocity_upper = float(handle_upper[1])

            def gap(value: float) -> float:
                collision = float(np.cos(2.0 * value))
                transverse = collision * (1.0 - collision)
                intervals = (
                    (velocity_lower, velocity_upper),
                    (
                        float(base_lower[1]) / collision,
                        float(base_upper[1]) / collision,
                    ),
                    (
                        float(base_lower[3]) / transverse,
                        float(base_upper[3]) / transverse,
                    ),
                )
                return max(interval[0] for interval in intervals) - min(
                    interval[1] for interval in intervals
                )

            return self._scalar_feasible(theta, gap)

        handle_velocity_lower = float(handle_lower[1])
        handle_velocity_upper = float(handle_upper[1])
        base_velocity_lower = float(base_lower[1])
        base_velocity_upper = float(base_upper[1])
        base_phi_dot_lower = float(base_lower[3])
        base_phi_dot_upper = float(base_upper[3])

        def transverse(collision: float) -> float:
            return collision * (1.0 - collision)

        constraints = (
            (
                lambda collision: (
                    handle_velocity_lower * collision - base_velocity_upper
                ),
                1 if handle_velocity_lower > 0.0 else -1,
            ),
            (
                lambda collision: (
                    base_velocity_lower - handle_velocity_upper * collision
                ),
                -1 if handle_velocity_upper > 0.0 else 1,
            ),
            (
                lambda collision: (
                    handle_velocity_lower * transverse(collision)
                    - base_phi_dot_upper
                ),
                -1 if handle_velocity_lower > 0.0 else 1,
            ),
            (
                lambda collision: (
                    base_phi_dot_lower
                    - handle_velocity_upper * transverse(collision)
                ),
                1 if handle_velocity_upper > 0.0 else -1,
            ),
            (
                lambda collision: (
                    base_velocity_lower * (1.0 - collision)
                    - base_phi_dot_upper
                ),
                -1 if base_velocity_lower > 0.0 else 1,
            ),
            (
                lambda collision: (
                    base_phi_dot_lower
                    - base_velocity_upper * (1.0 - collision)
                ),
                1 if base_velocity_upper > 0.0 else -1,
            ),
        )
        return self._monotone_constraints_feasible(
            (c_lower, c_upper),
            constraints,
        )

    def intersects(self, first: AtlasWalkerCell, second: AtlasWalkerCell) -> bool:
        if first == second:
            return True
        if first.chart_id == second.chart_id:
            return self._rectangles_intersect(
                first.lower,
                first.upper,
                second.lower,
                second.upper,
                self.atol,
            )
        if {first.chart_id, second.chart_id} != {
            self.charts.base_chart_id,
            self.charts.handle_chart_id,
        }:
            return False
        base = first if first.chart_id == self.charts.base_chart_id else second
        handle = first if first.chart_id == self.charts.handle_chart_id else second
        if handle.lower[-1] <= self.atol and self._guard_face_intersects(base, handle):
            return True
        return bool(
            handle.upper[-1] >= 1.0 - self.atol
            and self._reset_face_intersects(base, handle)
        )

    def attachment_hull(
        self,
        handle: AtlasWalkerCell,
        side: int,
    ) -> tuple[float, ...]:
        """Return a conservative base-chart hull for one handle attachment."""

        if handle.chart_id != self.charts.handle_chart_id or side not in (-1, 1):
            raise ValueError("attachment hull requires a handle cell and side +/-1")
        lower, upper = handle.lower, handle.upper
        base_bounds = np.asarray(self.charts.base_bounds, dtype=np.float64)
        if side < 0:
            guard_velocity_values = tuple(
                (
                    (1.0 - rho)
                    * max(
                        float(base_bounds[3, 0]),
                        2.0 * theta_dot + self.transversality_eta,
                    )
                    + rho * float(base_bounds[3, 1])
                )
                for theta_dot in (float(lower[1]), float(upper[1]))
                for rho in (float(lower[2]), float(upper[2]))
            )
            return (
                float(lower[0]),
                float(lower[1]),
                float(2.0 * lower[0]),
                min(guard_velocity_values),
                float(upper[0]),
                float(upper[1]),
                float(2.0 * upper[0]),
                max(guard_velocity_values),
            )
        collision_values = tuple(
            float(np.cos(2.0 * theta))
            for theta in (float(lower[0]), float(upper[0]))
        )
        reset_theta_dot_values = tuple(
            collision * theta_dot
            for collision in collision_values
            for theta_dot in (float(lower[1]), float(upper[1]))
        )
        transverse_values = tuple(
            collision * (1.0 - collision) for collision in collision_values
        )
        reset_phi_dot_values = tuple(
            transverse * theta_dot
            for transverse in transverse_values
            for theta_dot in (float(lower[1]), float(upper[1]))
        )
        return (
            float(-upper[0]),
            min(reset_theta_dot_values),
            float(-2.0 * upper[0]),
            min(reset_phi_dot_values),
            float(-lower[0]),
            max(reset_theta_dot_values),
            float(-2.0 * lower[0]),
            max(reset_phi_dot_values),
        )

    def connected_components(
        self,
        cells: Collection[AtlasWalkerCell],
    ) -> tuple[frozenset[AtlasWalkerCell], ...]:
        remaining = set(cells)
        components: list[frozenset[AtlasWalkerCell]] = []
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


class GuardAlignedGarciaWalkerQuotientIncidence(GarciaWalkerQuotientIncidence):
    """Closed-cell quotient incidence for the ``(theta,omega,q,nu)`` chart.

    Both attachments lie in the interior cubical hyperplane ``q=0``.  The
    ``s=0`` attachment is the identity on ``(theta,omega,nu)``; the ``s=1``
    attachment uses ``(-theta,c*omega,-c*(1+c)*omega)`` with
    ``c=cos(2*theta)``.  The latter feasibility test is one-dimensional after
    intersecting the admissible omega intervals.
    """

    def __init__(
        self,
        charts: SuspensionAtlasCharts,
        *,
        transversality_eta: float,
        atol: float = 1e-10,
    ) -> None:
        # Parent velocity parameters are unused by the overridden predicates,
        # but retaining the interface lets the generic local-relation tracker
        # consume either incidence object without branching.
        super().__init__(
            charts,
            phi_dot_min=charts.base_bounds[3][0],
            phi_dot_max=charts.base_bounds[3][1],
            transversality_eta=transversality_eta,
            atol=atol,
        )

    def _base_meets_internal_guard_face(self, base: AtlasWalkerCell) -> bool:
        return bool(base.lower[2] <= self.atol and base.upper[2] >= -self.atol)

    def _guard_face_intersects(
        self,
        base: AtlasWalkerCell,
        handle: AtlasWalkerCell,
    ) -> bool:
        if not self._base_meets_internal_guard_face(base):
            return False
        return bool(
            self._closed_interval(
                (float(base.lower[0]), float(base.upper[0])),
                (float(handle.lower[0]), float(handle.upper[0])),
                atol=self.atol,
            )
            is not None
            and self._closed_interval(
                (float(base.lower[1]), float(base.upper[1])),
                (float(handle.lower[1]), float(handle.upper[1])),
                atol=self.atol,
            )
            is not None
            and self._closed_interval(
                (float(base.lower[3]), float(base.upper[3])),
                (float(handle.lower[2]), float(handle.upper[2])),
                atol=self.atol,
            )
            is not None
        )

    def _reset_face_intersects(
        self,
        base: AtlasWalkerCell,
        handle: AtlasWalkerCell,
    ) -> bool:
        if not self._base_meets_internal_guard_face(base):
            return False
        theta = self._closed_interval(
            (float(handle.lower[0]), float(handle.upper[0])),
            (-float(base.upper[0]), -float(base.lower[0])),
            atol=self.atol,
        )
        if theta is None:
            return False
        handle_omega = (float(handle.lower[1]), float(handle.upper[1]))
        base_omega = (float(base.lower[1]), float(base.upper[1]))
        base_nu = (float(base.lower[3]), float(base.upper[3]))
        theta_lower, theta_upper = theta
        c_lower = float(np.cos(2.0 * theta_lower))
        c_upper = float(np.cos(2.0 * theta_upper))
        if not 0.0 < c_lower <= c_upper:
            return False
        h_lower, h_upper = handle_omega
        w_lower, w_upper = base_omega
        n_lower, n_upper = base_nu

        # At fixed c the feasible pre-impact omega is the intersection of
        # [hL,hU], [wL/c,wU/c], and
        # [-nU/(c(1+c)),-nL/(c(1+c))].  Three closed intervals on R have a
        # common point iff every pair intersects.  After multiplication by
        # the positive denominators those six conditions are monotone in c,
        # so the inherited bisection routine is exact up to ``atol`` and avoids
        # a costly optimization for every candidate cell.
        constraints = (
            (lambda c: h_lower * c - w_upper, 1 if h_lower > 0.0 else -1),
            (lambda c: w_lower - h_upper * c, -1 if h_upper > 0.0 else 1),
            (
                lambda c: n_lower + h_lower * c * (1.0 + c),
                1 if h_lower > 0.0 else -1,
            ),
            (
                lambda c: -h_upper * c * (1.0 + c) - n_upper,
                -1 if h_upper > 0.0 else 1,
            ),
            (lambda c: w_lower * (1.0 + c) + n_lower, 1 if w_lower > 0.0 else -1),
            (lambda c: -n_upper - w_upper * (1.0 + c), -1 if w_upper > 0.0 else 1),
        )
        return self._monotone_constraints_feasible(
            (c_lower, c_upper),
            constraints,
        )

    def attachment_hull(
        self,
        handle: AtlasWalkerCell,
        side: int,
    ) -> tuple[float, ...]:
        if handle.chart_id != self.charts.handle_chart_id or side not in (-1, 1):
            raise ValueError("attachment hull requires a handle cell and side +/-1")
        lower, upper = handle.lower, handle.upper
        if side < 0:
            return (
                float(lower[0]),
                float(lower[1]),
                0.0,
                float(lower[2]),
                float(upper[0]),
                float(upper[1]),
                0.0,
                float(upper[2]),
            )
        collision = tuple(
            float(np.cos(2.0 * theta))
            for theta in (float(lower[0]), float(upper[0]))
        )
        reset_omega = tuple(
            factor * omega
            for factor in collision
            for omega in (float(lower[1]), float(upper[1]))
        )
        reset_nu = tuple(
            -factor * (1.0 + factor) * omega
            for factor in collision
            for omega in (float(lower[1]), float(upper[1]))
        )
        return (
            float(-upper[0]),
            min(reset_omega),
            0.0,
            min(reset_nu),
            float(-lower[0]),
            max(reset_omega),
            0.0,
            max(reset_nu),
        )


def garcia_guard_aligned_atlas_charts(
    *,
    base_bounds: Sequence[Sequence[float]] = GUARD_ALIGNED_DOMAIN_BOUNDS,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
) -> SuspensionAtlasCharts:
    """Return the guard-aligned base chart and intrinsic reset handle.

    Base coordinates are ``(theta, omega, q, nu)`` and handle coordinates are
    ``(theta_guard, omega_guard, nu_guard, s)``.  Because ``q=0`` is centered
    in the default symmetric q interval, it is a grid face at every positive
    uniform dyadic depth, with active base cells retained on both sides.
    """

    bounds = tuple(
        (float(interval[0]), float(interval[1])) for interval in base_bounds
    )
    if len(bounds) != 4:
        raise ValueError("the guard-aligned Garcia base chart must be four-dimensional")
    theta_bounds, omega_bounds, q_bounds, nu_bounds = bounds
    if not q_bounds[0] < 0.0 < q_bounds[1]:
        raise ValueError("the aligned base chart must place q=0 in its interior")
    theta_upper = min(theta_bounds[1], -float(guard_delta))
    nu_lower = max(nu_bounds[0], float(transversality_eta))
    if theta_bounds[0] >= theta_upper or nu_lower >= nu_bounds[1]:
        raise ValueError("aligned base bounds contain no transverse guard patch")

    def guard_coordinates(guard: np.ndarray) -> np.ndarray:
        theta, omega, q, nu = np.asarray(guard, dtype=np.float64)
        if abs(float(q)) > 1.0e-7:
            raise ValueError("guard-aligned handle encoding requires q=0")
        return np.asarray((theta, omega, nu), dtype=np.float64)

    def guard_embedding(intrinsic: np.ndarray) -> np.ndarray:
        theta, omega, nu = np.asarray(intrinsic, dtype=np.float64)
        return np.asarray((theta, omega, 0.0, nu), dtype=np.float64)

    return SuspensionAtlasCharts(
        base_bounds=bounds,
        guard_bounds=(
            (theta_bounds[0], theta_upper),
            omega_bounds,
            (nu_lower, nu_bounds[1]),
        ),
        guard_coordinates=guard_coordinates,
        guard_embedding=guard_embedding,
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )


@dataclass(frozen=True)
class GarciaWalkerAtlasSetup:
    """Objects needed for one native CMGDB run."""

    walker: GarciaPassiveWalker | GuardAlignedGarciaPassiveWalker
    charts: SuspensionAtlasCharts
    box_map: CMGDBSuspensionBoxMap
    model: Any
    active_family: GarciaWalkerActiveFamily | None = None
    active_tracker: _ActiveSubgridTrackingBoxMap | None = None


@dataclass(frozen=True)
class GarciaWalkerAtlasAcceptance:
    """Coarse result and explicit falsification diagnostics."""

    depth: int
    subdivisions: int
    t_star: float
    morse_graph: Any
    map_graph: Any
    cell_tokens: tuple[AtlasWalkerCell, ...]
    relation: Mapping[AtlasWalkerCell, frozenset[AtlasWalkerCell]]
    gait_reference_node_hits: Mapping[int, int]
    reference_endpoint_audit: EndpointCoverageAudit
    reference_evaluation_failures: tuple[WalkerReferenceFailure, ...]
    image_connectivity_audit: RelationConnectivityAudit
    empty_image_sources: tuple[AtlasWalkerCell, ...]
    box_map_diagnostics: SuspensionBoxMapDiagnostics
    active_family: GarciaWalkerActiveFamily | None = None
    active_coverage_diagnostics: ActiveSubgridCoverageDiagnostics | None = None

    @property
    def reference_covered(self) -> bool:
        return bool(
            self.reference_endpoint_audit.passed
            and not self.reference_evaluation_failures
        )

    @property
    def image_values_connected(self) -> bool:
        return self.image_connectivity_audit.passed

    @property
    def plumbing_gates_passed(self) -> bool:
        """Finite plumbing checks, deliberately weaker than scientific acceptance."""

        return bool(
            self.reference_covered
            and self.image_values_connected
            and self.box_map_diagnostics.unresolved_stage_edges == 0
        )

    @property
    def sampled_self_map_complete(self) -> bool:
        return bool(
            not self.empty_image_sources
            and self.box_map_diagnostics.failed_samples == 0
            and (
                self.active_coverage_diagnostics is None
                or self.active_coverage_diagnostics.boundary_exit_sources == 0
            )
        )

    def _failure_summary(self) -> dict[str, object]:
        records = self.box_map_diagnostics.retained_source_records
        failures = [failure for record in records for failure in record.failures]
        reasons = Counter(failure.reason for failure in failures)
        exit_count = sum(
            count
            for reason, count in reasons.items()
            if "leaves the declared state-space bounds" in reason
            or "trajectory leaves the declared state space" in reason
            or "exits the declared state space" in reason
            or "reset leaves the declared state-space bounds" in reason
        )
        chart_count = sum(
            count
            for reason, count in reasons.items()
            if "outside target Atlas chart" in reason
        )
        inadmissible_guard_count = sum(
            count
            for reason, count in reasons.items()
            if "InadmissibleHeelstrikeError" in reason
        )
        incomplete_trajectory_count = sum(
            count
            for reason, count in reasons.items()
            if "recorded trajectory does not cover suspension time" in reason
        )
        return {
            "retained_callback_records": len(records),
            "callback_records_with_failures": sum(bool(record.failures) for record in records),
            "callback_records_with_only_failures": sum(
                bool(record.failures) and record.successful_samples == 0
                for record in records
            ),
            "domain_exit_sample_failures": exit_count,
            "target_chart_sample_failures": chart_count,
            "inadmissible_guard_sample_failures": inadmissible_guard_count,
            "incomplete_trajectory_sample_failures": incomplete_trajectory_count,
            "other_sample_failures": (
                len(failures)
                - exit_count
                - chart_count
                - inadmissible_guard_count
                - incomplete_trajectory_count
            ),
            "failure_reasons": dict(sorted(reasons.items())),
        }

    def summary(self) -> dict[str, object]:
        active = self.active_family
        active_diagnostics = self.active_coverage_diagnostics
        result: dict[str, object] = {
            "depth": self.depth,
            "subdivisions_per_axis": self.subdivisions,
            "t_star": self.t_star,
            "run_role": (
                "gait-local active-subgrid fixed-time study"
                if active is not None
                else "coarse fixed-depth plumbing acceptance"
            ),
            "relation_scope": (
                "full-dimensional dyadic gait neighborhood with explicit boundary exits"
                if active is not None
                else "full rectangular charts with explicit partial-image failures"
            ),
            "map_cells": int(self.map_graph.num_vertices()),
            "morse_nodes": int(self.morse_graph.num_vertices()),
            "morse_edges": [
                [int(source), int(target)]
                for source, target in sorted(self.morse_graph.edges())
            ],
            "gait_reference_node_hits": dict(self.gait_reference_node_hits),
            "reference_probes": len(self.reference_endpoint_audit.witnesses),
            "reference_misses": len(self.reference_endpoint_audit.missed),
            "reference_evaluation_failures": len(self.reference_evaluation_failures),
            "reference_covered": self.reference_covered,
            "nonempty_images": len(self.image_connectivity_audit.images),
            "disconnected_images": len(self.image_connectivity_audit.disconnected),
            "empty_images": len(self.empty_image_sources),
            "empty_sources_excluded_from_connectivity_audit": True,
            "box_map_source_boxes_across_cmgdb_passes": self.box_map_diagnostics.source_boxes,
            "box_map_sampled_points_across_cmgdb_passes": self.box_map_diagnostics.sampled_points,
            "box_map_failed_samples": self.box_map_diagnostics.failed_samples,
            "box_map_empty_images_across_cmgdb_passes": self.box_map_diagnostics.empty_images,
            "box_map_unresolved_stage_edges": self.box_map_diagnostics.unresolved_stage_edges,
            "plumbing_gates_passed": self.plumbing_gates_passed,
            "sampled_self_map_complete": self.sampled_self_map_complete,
            "active_subgrid_used": active is not None,
            "active_subgrid_api_available": True,
            "scientific_morse_recovery_claimed": False,
            "whole_cell_outer_enclosure_certified": False,
            "global_attractor_lattice_interpretation": False,
            "conley_index_computed": False,
        }
        if active is not None:
            chart_counts = active.chart_counts()
            result.update(
                {
                    "active_family_definition": (
                        "all same-chart dyadic cells within the declared Chebyshev "
                        "index radius of cells met by the two stored stride arcs "
                        "or their two handle traversals"
                    ),
                    "active_family_is_orbit_only": False,
                    "active_axis_depth": active.axis_depth,
                    "active_stencil_radius_cells": active.stencil_radius,
                    "active_subdivisions_per_axis": active.subdivisions_per_axis,
                    "active_base_seed_cells": len(active.base_seed_cells),
                    "active_handle_seed_cells": len(active.handle_seed_cells),
                    "active_base_cells": chart_counts.get(BASE_CHART_ID, 0),
                    "active_handle_cells": chart_counts.get(HANDLE_CHART_ID, 0),
                    "ambient_full_cells_per_chart_at_same_depth": (
                        active.subdivisions_per_axis**4
                    ),
                    "active_source_records": (
                        len(active_diagnostics.records)
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_boundary_exit_sources": (
                        active_diagnostics.boundary_exit_sources
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_wholly_exiting_sources": (
                        active_diagnostics.wholly_exiting_sources
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_explicit_empty_sources": (
                        active_diagnostics.explicit_empty_sources
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_missing_target_dyadic_cells": (
                        active_diagnostics.missing_target_dyadic_cells
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_target_pieces_without_active_intersection": (
                        active_diagnostics.pieces_without_active_intersection
                        if active_diagnostics is not None
                        else 0
                    ),
                    "active_target_pieces_crossing_ambient_chart_boundary": (
                        active_diagnostics.ambient_boundary_pieces
                        if active_diagnostics is not None
                        else 0
                    ),
                }
            )
        result.update(self._failure_summary())
        return result


def _guard_phi_dot_lower(
    theta_dot: float,
    *,
    phi_dot_min: float,
    transversality_eta: float,
) -> float:
    return max(
        float(phi_dot_min),
        2.0 * float(theta_dot) + float(transversality_eta),
    )


def garcia_passive_walker_atlas_charts(
    *,
    base_bounds: Sequence[Sequence[float]] = DEFAULT_BASE_BOUNDS,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    parameter_margin: float = DEFAULT_GUARD_PARAMETER_MARGIN,
) -> SuspensionAtlasCharts:
    """Return the physical chart and an admissible intrinsic guard chart.

    Intrinsic coordinates are ``(theta, theta_dot, rho)``.  At fixed
    ``theta_dot``, ``rho`` linearly parametrizes ``phi_dot`` from the smallest
    value allowed by the base chart and transversality inequality to the base
    chart's upper ``phi_dot`` bound.
    """

    bounds = tuple(
        (float(interval[0]), float(interval[1])) for interval in base_bounds
    )
    if len(bounds) != 4:
        raise ValueError("the Garcia base chart must be four-dimensional")
    theta_bounds, theta_dot_bounds, phi_bounds, phi_dot_bounds = bounds
    theta_lower = max(theta_bounds[0], 0.5 * phi_bounds[0])
    theta_upper = min(
        theta_bounds[1],
        0.5 * phi_bounds[1],
        -float(guard_delta),
    )
    theoretical_velocity_upper = 0.5 * (
        phi_dot_bounds[1] - float(transversality_eta)
    )
    velocity_upper = min(
        theta_dot_bounds[1],
        theoretical_velocity_upper - float(parameter_margin),
    )
    velocity_lower = theta_dot_bounds[0]
    if theta_lower >= theta_upper:
        raise ValueError("base bounds contain no outgoing heel-strike theta interval")
    if velocity_lower >= velocity_upper:
        raise ValueError("base bounds contain no nondegenerate transverse guard patch")

    def guard_coordinates(guard: np.ndarray) -> np.ndarray:
        theta, theta_dot, _phi, phi_dot = np.asarray(guard, dtype=np.float64)
        lower_phi_dot = _guard_phi_dot_lower(
            theta_dot,
            phi_dot_min=phi_dot_bounds[0],
            transversality_eta=transversality_eta,
        )
        width = phi_dot_bounds[1] - lower_phi_dot
        if width <= 0.0:
            raise ValueError("guard state lies at a collapsed transverse coordinate")
        rho = (phi_dot - lower_phi_dot) / width
        return np.asarray((theta, theta_dot, rho), dtype=np.float64)

    def guard_embedding(intrinsic: np.ndarray) -> np.ndarray:
        theta, theta_dot, rho = np.asarray(intrinsic, dtype=np.float64)
        lower_phi_dot = _guard_phi_dot_lower(
            theta_dot,
            phi_dot_min=phi_dot_bounds[0],
            transversality_eta=transversality_eta,
        )
        phi_dot = lower_phi_dot + rho * (phi_dot_bounds[1] - lower_phi_dot)
        return np.asarray((theta, theta_dot, 2.0 * theta, phi_dot), dtype=np.float64)

    return SuspensionAtlasCharts(
        base_bounds=bounds,
        guard_bounds=(
            (theta_lower, theta_upper),
            (velocity_lower, velocity_upper),
            (0.0, 1.0),
        ),
        guard_coordinates=guard_coordinates,
        guard_embedding=guard_embedding,
        base_chart_id=BASE_CHART_ID,
        handle_chart_id=HANDLE_CHART_ID,
    )


def build_garcia_gait_active_family(
    walker: GarciaPassiveWalker,
    charts: SuspensionAtlasCharts,
    *,
    axis_depth: int,
    stencil_radius: int = 1,
    base_samples_per_stride: int | None = None,
    handle_samples: int | None = None,
    reference_max_step: float = 0.005,
) -> GarciaWalkerActiveFamily:
    """Construct a disclosed full-dimensional gait-local active family.

    The seed is the set of uniform dyadic boxes met by two stored continuous
    stride arcs and the two corresponding handle traversals.  The returned set
    is the clipped Chebyshev ``stencil_radius`` neighborhood of those seeds in
    each four-dimensional chart.  A positive radius is required so this cannot
    silently degrade into an orbit-only graph.
    """

    if isinstance(axis_depth, bool) or not isinstance(axis_depth, (int, np.integer)):
        raise ValueError("axis_depth must be an integer")
    if axis_depth < 1:
        raise ValueError("axis_depth must be positive")
    if isinstance(stencil_radius, bool) or not isinstance(
        stencil_radius, (int, np.integer)
    ):
        raise ValueError("stencil_radius must be an integer")
    if stencil_radius < 1:
        raise ValueError(
            "the gait-local active family requires stencil_radius >= 1; "
            "an orbit-only seed is not accepted"
        )
    subdivisions = 2**int(axis_depth)
    if base_samples_per_stride is None:
        base_samples_per_stride = max(33, 8 * subdivisions + 1)
    if handle_samples is None:
        handle_samples = max(17, 4 * subdivisions + 1)
    if base_samples_per_stride < 3 or handle_samples < 3:
        raise ValueError("active-family sampling counts must be at least three")

    initial = (
        guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A)
        if isinstance(walker, GuardAlignedGarciaPassiveWalker)
        else post_impact_state(*PERIOD_TWO_POINT_A)
    )
    trajectory = walker.system.simulate(
        initial,
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=reference_max_step,
    )
    if len(trajectory.jump_states) < 2:
        raise RuntimeError("stored Garcia gait did not complete two heel strikes")

    base_points: list[np.ndarray] = []
    for segment in trajectory.segments[:2]:
        if segment.solution is None:
            raise RuntimeError("stored gait trajectory lacks dense output")
        times = np.linspace(
            segment.t_start,
            segment.t_end,
            int(base_samples_per_stride),
        )
        base_points.extend(
            np.asarray(segment.solution(times), dtype=np.float64).T
        )
    handle_points = [
        np.asarray(charts.encode_handle(guard, float(phase)), dtype=np.float64)
        for guard, _reset in trajectory.jump_states[:2]
        for phase in np.linspace(0.0, 1.0, int(handle_samples))
    ]

    def dyadic_index(chart_id: int, point: Sequence[float]) -> tuple[int, ...]:
        bounds = np.asarray(charts.bounds_for(chart_id), dtype=np.float64)
        values = np.asarray(point, dtype=np.float64)
        normalized = (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        if np.any(normalized < -1.0e-9) or np.any(normalized > 1.0 + 1.0e-9):
            raise ValueError(f"active-family seed lies outside chart {chart_id}")
        normalized = np.minimum(np.maximum(normalized, 0.0), 1.0)
        return tuple(
            min(subdivisions - 1, int(np.floor(subdivisions * coordinate)))
            for coordinate in normalized
        )

    base_seeds = frozenset(
        dyadic_index(charts.base_chart_id, point) for point in base_points
    )
    handle_seeds = frozenset(
        dyadic_index(charts.handle_chart_id, point) for point in handle_points
    )

    def thicken(seeds: Collection[tuple[int, ...]]) -> frozenset[tuple[int, ...]]:
        result: set[tuple[int, ...]] = set()
        offsets = tuple(range(-int(stencil_radius), int(stencil_radius) + 1))
        for seed in seeds:
            for delta in itertools.product(offsets, repeat=len(seed)):
                candidate = tuple(value + offset for value, offset in zip(seed, delta))
                if all(0 <= value < subdivisions for value in candidate):
                    result.add(candidate)
        return frozenset(result)

    base_active = thicken(base_seeds)
    handle_active = thicken(handle_seeds)
    tagged = tuple(
        sorted(
            (
                (charts.base_chart_id, int(axis_depth), coordinates)
                for coordinates in base_active
            ),
        )
        + sorted(
            (
                (charts.handle_chart_id, int(axis_depth), coordinates)
                for coordinates in handle_active
            ),
        )
    )
    return GarciaWalkerActiveFamily(
        axis_depth=int(axis_depth),
        stencil_radius=int(stencil_radius),
        base_seed_cells=tuple(sorted(base_seeds)),
        handle_seed_cells=tuple(sorted(handle_seeds)),
        tagged_cells=tagged,
    )


def build_garcia_passive_walker_atlas_model(
    *,
    depth: int = 4,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    max_step: float = 0.02,
    diagnostics_limit: int = 25_000,
    active_stencil_radius: int | None = None,
    active_base_samples_per_stride: int | None = None,
    active_handle_samples: int | None = None,
    active_reference_max_step: float = 0.005,
) -> GarciaWalkerAtlasSetup:
    """Install the Garcia fixed-time callback in a native Atlas.

    With ``active_stencil_radius=None`` this retains the original full-chart
    run.  A positive radius instead installs a directly constructed dyadic
    neighborhood of the stored gait at per-axis depth ``depth / 4``; CMGDB does
    not construct the corresponding full rectangular grid.
    """

    _validate_four_dimensional_depth(depth)
    walker = GarciaPassiveWalker(
        gamma=gamma,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
        domain_bounds=list(DEFAULT_BASE_BOUNDS),
        max_jumps=20,
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
    )
    box_map = CMGDBSuspensionBoxMap(
        walker.system,
        charts,
        t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        max_jumps=20,
        max_step=max_step,
        require_domain_path=True,
        diagnostics_limit=diagnostics_limit,
    )
    if active_stencil_radius is None:
        return GarciaWalkerAtlasSetup(
            walker=walker,
            charts=charts,
            box_map=box_map,
            model=build_cmgdb_atlas_model(box_map, depth=depth),
        )

    active_family = build_garcia_gait_active_family(
        walker,
        charts,
        axis_depth=depth // 4,
        stencil_radius=active_stencil_radius,
        base_samples_per_stride=active_base_samples_per_stride,
        handle_samples=active_handle_samples,
        reference_max_step=active_reference_max_step,
    )
    tracker = _ActiveSubgridTrackingBoxMap(box_map, active_family)
    model = build_cmgdb_atlas_model(
        tracker,  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=active_family.tagged_cells,
    )
    tracker.bind_atlas(model.phaseSpace())
    return GarciaWalkerAtlasSetup(
        walker=walker,
        charts=charts,
        box_map=box_map,
        model=model,
        active_family=active_family,
        active_tracker=tracker,
    )


def compute_garcia_passive_walker_atlas_acceptance(
    *,
    depth: int = 4,
    t_star: float = DEFAULT_T_STAR,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    reference_base_samples_per_stride: int = 17,
    reference_handle_samples: int = 9,
    reference_max_step: float = 0.005,
    active_stencil_radius: int | None = None,
    active_base_samples_per_stride: int | None = None,
    active_handle_samples: int | None = None,
) -> GarciaWalkerAtlasAcceptance:
    """Run the coarse native computation and its stored-gait audit gates."""

    setup = build_garcia_passive_walker_atlas_model(
        depth=depth,
        t_star=t_star,
        samples_per_axis=samples_per_axis,
        padding_cells=padding_cells,
        active_stencil_radius=active_stencil_radius,
        active_base_samples_per_stride=active_base_samples_per_stride,
        active_handle_samples=active_handle_samples,
    )
    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB fork is not installed") from error

    setup.box_map.reset_diagnostics()
    if setup.active_tracker is not None:
        setup.active_tracker.reset_diagnostics()
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(setup.model)
    tokens = _atlas_cells(morse_graph, int(map_graph.num_vertices()))
    relation = _atlas_relation(map_graph, tokens)
    incidence = GarciaWalkerQuotientIncidence(
        setup.charts,
        phi_dot_min=setup.charts.base_bounds[3][0],
        phi_dot_max=setup.charts.base_bounds[3][1],
        transversality_eta=setup.walker.transversality_eta,
    )
    nonempty = {source: targets for source, targets in relation.items() if targets}
    empty_sources = tuple(source for source, targets in relation.items() if not targets)
    connectivity = _audit_relation_connectivity(nonempty, incidence)

    probes, failures, gait_source_tokens = _stored_gait_probes(
        setup,
        tokens,
        base_samples_per_stride=reference_base_samples_per_stride,
        handle_samples=reference_handle_samples,
        reference_max_step=reference_max_step,
    )
    reference_audit = _audit_atlas_endpoint_probes(
        relation,
        probes,
        tokens,
        setup.box_map,
    )
    node_by_cell = _morse_node_by_phase_cell(morse_graph)
    gait_hits = Counter(
        node_by_cell[index]
        for index, token in enumerate(tokens)
        if token in gait_source_tokens and index in node_by_cell
    )
    return GarciaWalkerAtlasAcceptance(
        depth=depth,
        subdivisions=_subdivisions_for_depth(depth),
        t_star=float(t_star),
        morse_graph=morse_graph,
        map_graph=map_graph,
        cell_tokens=tokens,
        relation=relation,
        gait_reference_node_hits=dict(sorted(gait_hits.items())),
        reference_endpoint_audit=reference_audit,
        reference_evaluation_failures=failures,
        image_connectivity_audit=connectivity,
        empty_image_sources=empty_sources,
        box_map_diagnostics=setup.box_map.diagnostics(),
        active_family=setup.active_family,
        active_coverage_diagnostics=(
            setup.active_tracker.diagnostics()
            if setup.active_tracker is not None
            else None
        ),
    )


def _validate_four_dimensional_depth(depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise ValueError("depth must be an integer")
    if depth < 4 or depth % 4:
        raise ValueError("walker depth must be a positive multiple of four")


def _subdivisions_for_depth(depth: int) -> int:
    _validate_four_dimensional_depth(depth)
    return 2 ** (depth // 4)


def _atlas_cells(morse_graph: Any, count: int) -> tuple[AtlasWalkerCell, ...]:
    return tuple(
        AtlasWalkerCell(
            index=index,
            chart_id=int(morse_graph.phase_space_chart_box(index)[0]),
            bounds=tuple(
                float(value) for value in morse_graph.phase_space_chart_box(index)[1]
            ),
        )
        for index in range(count)
    )


def _atlas_relation(
    map_graph: Any,
    tokens: Sequence[AtlasWalkerCell],
) -> dict[AtlasWalkerCell, frozenset[AtlasWalkerCell]]:
    return {
        source: frozenset(tokens[int(target)] for target in map_graph.adjacencies(index))
        for index, source in enumerate(tokens)
    }


def _cells_containing_point(
    cells: Sequence[AtlasWalkerCell],
    chart_id: int,
    coordinates: Sequence[float],
    *,
    atol: float = 1e-10,
) -> frozenset[AtlasWalkerCell]:
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


def _stored_gait_probes(
    setup: GarciaWalkerAtlasSetup,
    cells: Sequence[AtlasWalkerCell],
    *,
    base_samples_per_stride: int,
    handle_samples: int,
    reference_max_step: float,
) -> tuple[
    tuple[EndpointProbe, ...],
    tuple[WalkerReferenceFailure, ...],
    frozenset[AtlasWalkerCell],
]:
    if base_samples_per_stride < 3 or handle_samples < 3:
        raise ValueError("stored-gait reference sampling counts are too small")
    initial = (
        guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A)
        if isinstance(setup.walker, GuardAlignedGarciaPassiveWalker)
        else post_impact_state(*PERIOD_TWO_POINT_A)
    )
    trajectory = setup.walker.system.simulate(
        initial,
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=reference_max_step,
    )
    if len(trajectory.jump_states) < 2:
        raise RuntimeError("stored Garcia gait did not complete two heel strikes")
    reference_map = CMGDBSuspensionBoxMap(
        setup.walker.system,
        setup.charts,
        setup.box_map.t_star,
        samples_per_axis=3,
        padding_cells=0.0,
        max_jumps=20,
        max_step=reference_max_step,
        require_domain_path=True,
        diagnostics_limit=0,
    )
    sources: list[tuple[int, tuple[float, ...], str]] = []
    for stride, segment in enumerate(trajectory.segments[:2]):
        if segment.solution is None:
            raise RuntimeError("stored gait trajectory lacks dense output")
        times = np.linspace(
            segment.t_start,
            segment.t_end,
            base_samples_per_stride,
        )
        states = np.asarray(segment.solution(times), dtype=np.float64).T
        sources.extend(
            (
                BASE_CHART_ID,
                tuple(float(value) for value in state),
                f"gait-base-stride-{stride}-sample-{sample}",
            )
            for sample, state in enumerate(states)
        )
    phases = np.linspace(0.0, 1.0, handle_samples)
    for jump, (guard, _reset) in enumerate(trajectory.jump_states[:2]):
        sources.extend(
            (
                HANDLE_CHART_ID,
                setup.charts.encode_handle(guard, float(phase)),
                f"gait-handle-{jump}-sample-{sample}",
            )
            for sample, phase in enumerate(phases)
        )

    probes: list[EndpointProbe] = []
    failures: list[WalkerReferenceFailure] = []
    source_tokens: set[AtlasWalkerCell] = set()
    for chart_id, coordinates, label in sources:
        try:
            containing_sources = _cells_containing_point(cells, chart_id, coordinates)
            endpoint = reference_map.evaluate_point(chart_id, coordinates)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failures.append(
                WalkerReferenceFailure(
                    chart_id=chart_id,
                    coordinates=coordinates,
                    label=label,
                    message=f"{type(error).__name__}: {error}",
                )
            )
            continue
        source_tokens.update(containing_sources)
        probes.extend(
            EndpointProbe(
                source=source,
                endpoint=endpoint,
                initial_state=coordinates,
                label=label,
            )
            for source in containing_sources
        )
    return tuple(probes), tuple(failures), frozenset(source_tokens)


def _audit_atlas_endpoint_probes(
    relation: Mapping[AtlasWalkerCell, Collection[AtlasWalkerCell]],
    probes: Sequence[EndpointProbe],
    cells: Sequence[AtlasWalkerCell],
    box_map: CMGDBSuspensionBoxMap,
) -> EndpointCoverageAudit:
    witnesses = []
    for probe in probes:
        chart_id, coordinates, _stratum = box_map.encode_endpoint(probe.endpoint)
        witnesses.append(
            EndpointCoverageWitness(
                probe=probe,
                containing_target_cells=_cells_containing_point(
                    cells,
                    chart_id,
                    coordinates,
                ),
                recorded_targets=frozenset(relation[probe.source]),
            )
        )
    return EndpointCoverageAudit(tuple(witnesses))


def _audit_relation_connectivity(
    relation: Mapping[AtlasWalkerCell, Collection[AtlasWalkerCell]],
    incidence: GarciaWalkerQuotientIncidence,
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


def _morse_node_by_phase_cell(morse_graph: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in morse_graph.vertices():
        for index in morse_graph.morse_set(node):
            if int(index) in result:
                raise ValueError("a phase-space cell belongs to two Morse sets")
            result[int(index)] = int(node)
    return result


__all__ = [
    "BASE_CHART_ID",
    "HANDLE_CHART_ID",
    "DEFAULT_T_STAR",
    "DEFAULT_BASE_BOUNDS",
    "DEFAULT_GUARD_PARAMETER_MARGIN",
    "AtlasWalkerCell",
    "GarciaWalkerActiveFamily",
    "ActiveSourceCoverage",
    "ActiveSubgridCoverageDiagnostics",
    "WalkerReferenceFailure",
    "GarciaWalkerQuotientIncidence",
    "GuardAlignedGarciaWalkerQuotientIncidence",
    "GarciaWalkerAtlasSetup",
    "GarciaWalkerAtlasAcceptance",
    "garcia_passive_walker_atlas_charts",
    "garcia_guard_aligned_atlas_charts",
    "build_garcia_gait_active_family",
    "build_garcia_passive_walker_atlas_model",
    "compute_garcia_passive_walker_atlas_acceptance",
]
