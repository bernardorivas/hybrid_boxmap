"""Falsification diagnostics for sampled fixed-time suspension relations.

The routines in this module do **not** certify a whole-cell outer enclosure.
They provide two reusable necessary checks for the exploratory relation built
by :mod:`fixed_time_suspension_grid`:

* endpoint probes obtained from interior points or a reference orbit must land
  in the union of the recorded target cells; and
* a proposed image value can be tested for connectedness in the augmented
  base-plus-handle quotient incidence graph.

The second check uses the reset gluing.  A terminal phase slab is incident to
the base cells containing its reset image, and an initial phase slab is
incident to its guard base cell.  Thus it does not confuse separation in a
plotting chart with separation in the suspension quotient.

Passing either diagnostic is only evidence.  Finite point probes cannot prove
whole-cell coverage, and connected image values are weaker than the acyclic
carrier property required by a Conley-index calculation.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from .fixed_time_suspension_grid import (
    GridSuspensionIngredients,
    SuspensionCemeteryCell,
)
from .grid import Grid
from .sampled_suspension import (
    BaseCell,
    PhaseCell,
    SuspensionSample,
    locate_augmented_cells,
)


AuditedCell: TypeAlias = BaseCell | PhaseCell | SuspensionCemeteryCell
EndpointEvaluator: TypeAlias = Callable[[npt.NDArray[np.float64]], SuspensionSample]


class RelationAuditFailure(RuntimeError):
    """Raised when an exploratory relation fails a requested publication gate."""


def _cell_key(cell: AuditedCell) -> tuple[object, ...]:
    if isinstance(cell, BaseCell):
        return ("base", cell.index)
    if isinstance(cell, PhaseCell):
        return ("phase", cell.guard_cell, cell.slab)
    return ("cemetery", cell.label)


@dataclass(frozen=True)
class EndpointProbe:
    """One independently evaluated endpoint of a declared source cell."""

    source: AuditedCell
    endpoint: SuspensionSample
    initial_state: tuple[float, ...] | None = None
    label: str = ""


@dataclass(frozen=True)
class EndpointCoverageWitness:
    """Coverage result for one endpoint probe."""

    probe: EndpointProbe
    containing_target_cells: frozenset[AuditedCell]
    recorded_targets: frozenset[AuditedCell]

    @property
    def covered(self) -> bool:
        """Whether the recorded target union contains the endpoint."""

        return bool(self.containing_target_cells.intersection(self.recorded_targets))

    @property
    def full_closed_cover_recorded(self) -> bool:
        """Whether every incident closed target cell was recorded.

        This is stronger than point containment and is useful when comparing
        with the minimal closed-cell cover.  It is not required for an
        arbitrary outer relation, because one incident closed cell already
        contains a boundary point.
        """

        return self.containing_target_cells <= self.recorded_targets


@dataclass(frozen=True)
class EndpointEvaluationFailure:
    """A dense source probe for which no endpoint could be evaluated."""

    source: BaseCell
    initial_state: tuple[float, ...]
    message: str


@dataclass(frozen=True)
class EndpointCoverageAudit:
    """Finite endpoint-probe audit of one relation."""

    witnesses: tuple[EndpointCoverageWitness, ...]
    evaluation_failures: tuple[EndpointEvaluationFailure, ...] = ()

    @property
    def missed(self) -> tuple[EndpointCoverageWitness, ...]:
        return tuple(witness for witness in self.witnesses if not witness.covered)

    @property
    def incomplete_closed_covers(self) -> tuple[EndpointCoverageWitness, ...]:
        return tuple(
            witness
            for witness in self.witnesses
            if not witness.full_closed_cover_recorded
        )

    @property
    def passed(self) -> bool:
        return not self.missed and not self.evaluation_failures

    def require_passed(self, *, context: str = "fixed-time relation") -> None:
        """Raise unless every evaluated endpoint is covered and none failed.

        This is a strict falsification gate, not an outer-enclosure
        certification: finitely many passing probes say nothing about points
        that were not evaluated.
        """

        if self.passed:
            return
        raise RelationAuditFailure(
            f"{context} failed endpoint coverage audit: "
            f"{len(self.missed)} missed endpoints and "
            f"{len(self.evaluation_failures)} evaluation failures"
        )


def audit_endpoint_probes(
    relation: Mapping[AuditedCell, Collection[AuditedCell]],
    probes: Iterable[EndpointProbe],
    grid: Grid,
    *,
    handle_slabs: int,
    atol: float = 1e-10,
) -> EndpointCoverageAudit:
    """Check independently evaluated endpoints against recorded cell images.

    The test is point containment in the union of target cells.  At a closed
    grid boundary, an endpoint can lie in several target cells; recording any
    one of them contains the point.  ``full_closed_cover_recorded`` separately
    reports agreement with the complete minimal closed-cell cover.
    """

    witnesses: list[EndpointCoverageWitness] = []
    for probe in probes:
        if probe.source not in relation:
            raise ValueError(f"probe source is absent from relation: {probe.source!r}")
        located = frozenset(
            locate_augmented_cells(
                probe.endpoint,
                grid,
                handle_slabs,
                atol=atol,
            )
        )
        recorded = frozenset(relation[probe.source])
        witnesses.append(
            EndpointCoverageWitness(
                probe=probe,
                containing_target_cells=located,
                recorded_targets=recorded,
            )
        )
    return EndpointCoverageAudit(tuple(witnesses))


def audit_dense_base_endpoints(
    relation: Mapping[AuditedCell, Collection[AuditedCell]],
    grid: Grid,
    endpoint_evaluator: EndpointEvaluator,
    *,
    handle_slabs: int,
    source_cells: Collection[int] | None = None,
    subdivision_level: int = 2,
    atol: float = 1e-10,
) -> EndpointCoverageAudit:
    """Probe a tensor subdivision of each requested base source cell.

    ``subdivision_level=L`` uses ``2**L + 1`` points per coordinate, including
    the source boundary.  This is a falsification test analogous to increasing
    the sampling density of a CMGDB-style box map.  It deliberately makes no
    assertion about unsampled points.
    """

    if (
        isinstance(subdivision_level, bool)
        or not isinstance(subdivision_level, (int, np.integer))
        or subdivision_level < 0
    ):
        raise ValueError("subdivision_level must be a non-negative integer")
    if source_cells is None:
        indices = tuple(
            source.index for source in relation if isinstance(source, BaseCell)
        )
    else:
        indices = tuple(int(index) for index in source_cells)

    probes: list[EndpointProbe] = []
    failures: list[EndpointEvaluationFailure] = []
    for cell_index in indices:
        source = BaseCell(cell_index)
        if source not in relation:
            raise ValueError(f"base source is absent from relation: {source!r}")
        points = grid.get_sample_points(
            cell_index,
            mode="subdivision",
            subdivision_level=subdivision_level,
        )
        for sample_index, raw_point in enumerate(points):
            point = np.asarray(raw_point, dtype=np.float64)
            initial_state = tuple(float(value) for value in point)
            try:
                endpoint = endpoint_evaluator(point)
                if endpoint is None:  # type: ignore[comparison-overlap]
                    raise ValueError("endpoint evaluator returned None")
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failures.append(
                    EndpointEvaluationFailure(
                        source=source,
                        initial_state=initial_state,
                        message=str(error),
                    )
                )
                continue
            probes.append(
                EndpointProbe(
                    source=source,
                    endpoint=endpoint,
                    initial_state=initial_state,
                    label=f"base-{cell_index}-sample-{sample_index}",
                )
            )

    audit = audit_endpoint_probes(
        relation,
        probes,
        grid,
        handle_slabs=handle_slabs,
        atol=atol,
    )
    return EndpointCoverageAudit(audit.witnesses, tuple(failures))


class AugmentedQuotientIncidence:
    """Top-cell intersection model for the implicit suspension quotient.

    Base-to-base incidence is the ordinary closed cubical-grid incidence.
    Phase-to-phase incidence uses neighboring guard-grid cells and neighboring
    phase slabs.  This is conservative when the metadata stores only ambient
    guard-intersecting boxes rather than an explicit guard trace complex.
    Bottom and top gluing use the declared guard and reset cells exactly.
    """

    def __init__(
        self,
        grid: Grid,
        ingredients: GridSuspensionIngredients,
    ) -> None:
        if tuple(int(value) for value in grid.subdivisions) != tuple(
            ingredients.base_subdivisions
        ):
            raise ValueError("grid subdivisions disagree with suspension metadata")
        expected_bounds = np.asarray(ingredients.base_bounds, dtype=float)
        if expected_bounds.shape != grid.bounds.shape or not np.allclose(
            expected_bounds,
            grid.bounds,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("grid bounds disagree with suspension metadata")
        self.grid = grid
        self.ingredients = ingredients
        self._handle_by_guard_cell = {
            handle.guard_cell.index: handle for handle in ingredients.handles
        }
        if len(self._handle_by_guard_cell) != len(ingredients.handles):
            raise ValueError("more than one implicit handle uses a guard-cell token")

    def _base_coordinates(self, cell: BaseCell | int) -> tuple[int, ...]:
        index = cell.index if isinstance(cell, BaseCell) else cell
        if not 0 <= index < self.grid.total_boxes:
            raise ValueError(f"base-cell index is outside the grid: {index}")
        return tuple(
            int(value)
            for value in np.unravel_index(index, tuple(self.grid.subdivisions))
        )

    def _base_cells_intersect(self, first: BaseCell | int, second: BaseCell | int) -> bool:
        a = self._base_coordinates(first)
        b = self._base_coordinates(second)
        return all(abs(left - right) <= 1 for left, right in zip(a, b))

    def intersects(self, first: AuditedCell, second: AuditedCell) -> bool:
        """Return whether two declared closed top cells meet in the quotient."""

        if first == second:
            return True
        if isinstance(first, SuspensionCemeteryCell) or isinstance(
            second, SuspensionCemeteryCell
        ):
            return False
        if isinstance(first, BaseCell) and isinstance(second, BaseCell):
            return self._base_cells_intersect(first, second)
        if isinstance(first, PhaseCell) and isinstance(second, PhaseCell):
            return (
                abs(first.slab - second.slab) <= 1
                and self._base_cells_intersect(
                    first.guard_cell,
                    second.guard_cell,
                )
            )

        base = first if isinstance(first, BaseCell) else second
        phase = first if isinstance(first, PhaseCell) else second
        if not isinstance(base, BaseCell) or not isinstance(phase, PhaseCell):
            raise TypeError("unsupported augmented cell type")
        handle = self._handle_by_guard_cell.get(phase.guard_cell)
        if handle is None:
            return False
        if phase.slab == 0 and base == handle.guard_cell:
            return True
        return (
            phase.slab == self.ingredients.phase_slabs - 1
            and base in handle.reset_cells
        )

    def connected_components(
        self,
        cells: Collection[AuditedCell],
    ) -> tuple[frozenset[AuditedCell], ...]:
        """Return topological components using the quotient incidence graph."""

        remaining = set(cells)
        components: list[frozenset[AuditedCell]] = []
        while remaining:
            root = min(remaining, key=_cell_key)
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
        components.sort(key=lambda value: min(_cell_key(cell) for cell in value))
        return tuple(components)


@dataclass(frozen=True)
class CellSetConnectivity:
    """Connected-component result for one finite augmented cell set."""

    label: object
    cells: frozenset[AuditedCell]
    components: tuple[frozenset[AuditedCell], ...]

    @property
    def connected(self) -> bool:
        return len(self.components) == 1

    def require_connected(self, *, context: str | None = None) -> None:
        """Raise unless this cell union is connected in quotient incidence."""

        if self.connected:
            return
        description = context or str(self.label)
        raise RelationAuditFailure(
            f"{description} has {len(self.components)} quotient-incidence components"
        )


@dataclass(frozen=True)
class RelationConnectivityAudit:
    """Connectivity audit for every image value of a finite relation."""

    images: tuple[CellSetConnectivity, ...]

    @property
    def disconnected(self) -> tuple[CellSetConnectivity, ...]:
        return tuple(image for image in self.images if not image.connected)

    @property
    def passed(self) -> bool:
        return not self.disconnected

    def require_passed(self, *, context: str = "fixed-time relation") -> None:
        """Raise unless every stored image value is quotient-connected.

        Use this gate when the construction claims connected image values,
        such as the cell cover of a connected neighborhood of the image of a
        connected source cell.  Connectivity is not a universal requirement
        on an arbitrary multivalued outer relation.
        """

        if self.passed:
            return
        raise RelationAuditFailure(
            f"{context} failed image-connectivity audit: "
            f"{len(self.disconnected)} disconnected image values"
        )


def audit_relation_image_connectivity(
    relation: Mapping[AuditedCell, Collection[AuditedCell]],
    incidence: AugmentedQuotientIncidence,
) -> RelationConnectivityAudit:
    """Check each stored image union in the quotient incidence graph."""

    images = []
    for source in sorted(relation, key=_cell_key):
        cells = frozenset(relation[source])
        images.append(
            CellSetConnectivity(
                label=source,
                cells=cells,
                components=incidence.connected_components(cells),
            )
        )
    return RelationConnectivityAudit(tuple(images))


def audit_cell_sets_connectivity(
    cell_sets: Iterable[Collection[AuditedCell]],
    incidence: AugmentedQuotientIncidence,
    *,
    label_prefix: str = "set",
) -> tuple[CellSetConnectivity, ...]:
    """Audit arbitrary cell sets, such as recurrent SCC supports.

    Spatial disconnectedness of an SCC is a diagnostic, not a general graph
    theorem violation: mutual reachability under a fixed-time map need not
    imply that the SCC support is connected.
    """

    results = []
    for index, raw_cells in enumerate(cell_sets):
        cells = frozenset(raw_cells)
        results.append(
            CellSetConnectivity(
                label=f"{label_prefix}-{index}",
                cells=cells,
                components=incidence.connected_components(cells),
            )
        )
    return tuple(results)


__all__ = [
    "AuditedCell",
    "RelationAuditFailure",
    "EndpointProbe",
    "EndpointCoverageWitness",
    "EndpointEvaluationFailure",
    "EndpointCoverageAudit",
    "audit_endpoint_probes",
    "audit_dense_base_endpoints",
    "AugmentedQuotientIncidence",
    "CellSetConnectivity",
    "RelationConnectivityAudit",
    "audit_relation_image_connectivity",
    "audit_cell_sets_connectivity",
]
