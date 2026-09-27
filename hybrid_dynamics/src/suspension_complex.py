"""Finite reset-glued cell complexes for the fixed-time suspension map.

This module supplies the topology that a suspension adjacency graph does not
contain.  A base cell complex is augmented by mapping-cylinder prisms over a
guard complex.  ``SuspensionCellComplex`` handles the special case where the
guard already is a base subcomplex.  ``DoubleMappingCylinderComplex`` accepts
an independent guard mesh and separate integral bottom/reset attachment chain
maps, which is the scalable route for nonlinear high-dimensional seams.  Phase
cells are interned by ``(handle_id, guard_cell, slab)`` and hence are global
cells of the resulting complex, not copies local to individual image
calculations.

All stored boundary coefficients are oriented integers.  Validation checks
``boundary ** 2 == 0`` over the integers.  A fixed-time carrier can additionally
be checked for the carrier nesting condition and acyclicity over a chosen
prime field.  Choosing a chain approximation subordinate to that carrier is a
separate, explicit step: :class:`CellularChainMap` validates ``d F = F d`` and
exports the relative map in the exact sparse format consumed by
``CMGDB.ComputeRelativeHomologyShiftClass``.

The construction is combinatorial.  It does not prove that sampled images are
outer enclosures, that a supplied reset-chain map represents a particular
smooth reset, or that an index pair is isolating.  Those are mathematical or
validated-enclosure obligations of the caller.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import networkx as nx

from .sampled_suspension import BaseCell, PhaseCell


Cell = Hashable
IntegerChain = Mapping[Cell, int]
SparseEntry = Tuple[int, int, int]


def _require_hashable(value: object, description: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{description} must be hashable: {value!r}") from error


def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 1
    return True


def _require_prime(modulus: int) -> int:
    if isinstance(modulus, bool) or not isinstance(modulus, int):
        raise TypeError("modulus must be an integer prime")
    if not _is_prime(modulus):
        raise ValueError("modulus must be prime")
    return modulus


def _normalize_chain(
    chain: Mapping[Cell, int] | Iterable[Tuple[Cell, int]],
) -> Dict[Cell, int]:
    items = chain.items() if isinstance(chain, ABCMapping) else chain
    result: Dict[Cell, int] = {}
    for cell, coefficient in items:
        _require_hashable(cell, "chain cell")
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise TypeError(
                f"chain coefficient for {cell!r} must be an integer"
            )
        updated = result.get(cell, 0) + coefficient
        if updated:
            result[cell] = updated
        else:
            result.pop(cell, None)
    return result


def _add_scaled_chain(
    accumulator: Dict[Cell, int], chain: IntegerChain, scale: int
) -> None:
    if scale == 0:
        return
    for cell, coefficient in chain.items():
        updated = accumulator.get(cell, 0) + scale * coefficient
        if updated:
            accumulator[cell] = updated
        else:
            accumulator.pop(cell, None)


def _rank_mod_prime(columns: Iterable[Iterable[Tuple[int, int]]], modulus: int) -> int:
    """Return the rank of sparse column vectors over ``GF(modulus)``."""

    modulus = _require_prime(modulus)
    pivots: Dict[int, Dict[int, int]] = {}
    for entries in columns:
        vector: Dict[int, int] = {}
        for row, coefficient in entries:
            value = (vector.get(row, 0) + coefficient) % modulus
            if value:
                vector[row] = value
            else:
                vector.pop(row, None)

        while vector:
            pivot = max(vector)
            if pivot not in pivots:
                inverse = pow(vector[pivot], -1, modulus)
                vector = {
                    row: (coefficient * inverse) % modulus
                    for row, coefficient in vector.items()
                    if coefficient % modulus
                }
                pivots[pivot] = vector
                break

            pivot_vector = pivots[pivot]
            scale = vector[pivot]
            for row, coefficient in pivot_vector.items():
                value = (vector.get(row, 0) - scale * coefficient) % modulus
                if value:
                    vector[row] = value
                else:
                    vector.pop(row, None)

    return len(pivots)


def _solve_linear_system_mod_prime(
    rows: Sequence[Cell],
    columns: Sequence[Cell],
    column_entries: Mapping[Cell, Mapping[Cell, int]],
    right_hand_side: Mapping[Cell, int],
    modulus: int,
) -> Dict[Cell, int]:
    """Solve a finite sparse linear system over ``GF(modulus)``.

    The matrix columns are named by ``columns`` and their row coefficients by
    ``column_entries``.  Free variables are deterministically set to zero.  A
    missing solution is an internal contradiction when this helper is used by
    the acyclic-carrier construction, so it is reported explicitly rather than
    replaced by a guessed chain.
    """

    modulus = _require_prime(modulus)
    row_index = {cell: index for index, cell in enumerate(rows)}
    unknown_rows = set(right_hand_side).difference(row_index)
    if unknown_rows:
        raise ValueError(
            "right-hand side contains rows outside the proposed carrier: "
            f"{sorted(unknown_rows, key=repr)!r}"
        )

    def add_scaled(
        accumulator: Dict[int, int],
        vector: Mapping[int, int],
        scale: int,
    ) -> None:
        scale %= modulus
        if not scale:
            return
        for coordinate, coefficient in vector.items():
            value = (
                accumulator.get(coordinate, 0) + scale * coefficient
            ) % modulus
            if value:
                accumulator[coordinate] = value
            else:
                accumulator.pop(coordinate, None)

    # Store a normalized sparse boundary vector and the corresponding sparse
    # combination of original columns at every pivot.  Cellular boundary
    # columns have very few nonzero entries; retaining that sparsity avoids
    # allocating and repeatedly reducing a dense rows-by-columns matrix for
    # every carrier value.
    pivots: Dict[int, Tuple[Dict[int, int], Dict[int, int]]] = {}
    for column_number, column in enumerate(columns):
        vector: Dict[int, int] = {}
        for row, coefficient in column_entries.get(column, {}).items():
            if row not in row_index:
                raise ValueError(
                    f"column {column!r} leaves the proposed carrier at {row!r}"
                )
            coordinate = row_index[row]
            value = (vector.get(coordinate, 0) + coefficient) % modulus
            if value:
                vector[coordinate] = value
            else:
                vector.pop(coordinate, None)
        combination = {column_number: 1}

        while vector:
            pivot = min(vector)
            if pivot in pivots:
                pivot_vector, pivot_combination = pivots[pivot]
                scale = -vector[pivot]
                add_scaled(vector, pivot_vector, scale)
                add_scaled(combination, pivot_combination, scale)
                continue

            inverse = pow(vector[pivot], -1, modulus)
            vector = {
                coordinate: coefficient * inverse % modulus
                for coordinate, coefficient in vector.items()
                if coefficient % modulus
            }
            combination = {
                coordinate: coefficient * inverse % modulus
                for coordinate, coefficient in combination.items()
                if coefficient % modulus
            }
            pivots[pivot] = (vector, combination)
            break

    residual = {
        row_index[row]: coefficient % modulus
        for row, coefficient in right_hand_side.items()
        if coefficient % modulus
    }
    solution: Dict[int, int] = {}
    while residual:
        pivot = min(residual)
        if pivot not in pivots:
            raise ValueError("the boundary equation has no solution in the carrier")
        pivot_vector, pivot_combination = pivots[pivot]
        scale = residual[pivot]
        add_scaled(residual, pivot_vector, -scale)
        add_scaled(solution, pivot_combination, scale)

    return {
        columns[column_number]: coefficient
        for column_number, coefficient in solution.items()
        if coefficient % modulus
    }


class FiniteCellComplex:
    """A finite oriented cellular chain complex over the integers.

    ``dimensions`` fixes both the cell set and its deterministic order.
    ``boundaries[cell]`` is a sparse integer chain on cells one dimension
    lower.  Missing boundary entries are interpreted as zero.
    """

    def __init__(
        self,
        dimensions: Mapping[Cell, int],
        boundaries: Mapping[Cell, IntegerChain],
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not dimensions:
            raise ValueError("a finite cell complex must contain at least one cell")

        ordered_cells = tuple(dimensions)
        normalized_dimensions: Dict[Cell, int] = {}
        for cell in ordered_cells:
            _require_hashable(cell, "cell")
            dimension = dimensions[cell]
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"dimension of {cell!r} must be an integer")
            if dimension < 0:
                raise ValueError(f"dimension of {cell!r} must be non-negative")
            normalized_dimensions[cell] = dimension

        unknown_boundary_sources = set(boundaries).difference(normalized_dimensions)
        if unknown_boundary_sources:
            raise ValueError(
                "boundaries contain undeclared source cells: "
                f"{sorted(unknown_boundary_sources, key=repr)!r}"
            )

        normalized_boundaries: Dict[Cell, Mapping[Cell, int]] = {}
        for cell in ordered_cells:
            boundary = _normalize_chain(boundaries.get(cell, {}))
            dimension = normalized_dimensions[cell]
            if dimension == 0 and boundary:
                raise ValueError(f"zero-cell {cell!r} must have zero boundary")
            for face in boundary:
                if face not in normalized_dimensions:
                    raise ValueError(
                        f"boundary of {cell!r} references undeclared cell {face!r}"
                    )
                if normalized_dimensions[face] != dimension - 1:
                    raise ValueError(
                        f"boundary face {face!r} of {cell!r} has dimension "
                        f"{normalized_dimensions[face]}, expected {dimension - 1}"
                    )
            normalized_boundaries[cell] = MappingProxyType(boundary)

        for cell in ordered_cells:
            second_boundary: Dict[Cell, int] = {}
            for face, coefficient in normalized_boundaries[cell].items():
                _add_scaled_chain(
                    second_boundary,
                    normalized_boundaries[face],
                    coefficient,
                )
            if second_boundary:
                raise ValueError(
                    f"boundary squared is nonzero on {cell!r}: {second_boundary!r}"
                )

        by_dimension: List[List[Cell]] = [
            [] for _ in range(max(normalized_dimensions.values()) + 1)
        ]
        for cell in ordered_cells:
            by_dimension[normalized_dimensions[cell]].append(cell)

        self._cells = ordered_cells
        self._cell_set = frozenset(ordered_cells)
        self._position = {cell: position for position, cell in enumerate(ordered_cells)}
        self._dimensions = MappingProxyType(normalized_dimensions)
        self._boundaries = MappingProxyType(normalized_boundaries)
        self._by_dimension = tuple(tuple(group) for group in by_dimension)
        self._metadata = MappingProxyType(dict(metadata or {}))

    @property
    def cells(self) -> Tuple[Cell, ...]:
        return self._cells

    @property
    def cell_set(self) -> FrozenSet[Cell]:
        return self._cell_set

    @property
    def max_dimension(self) -> int:
        return len(self._by_dimension) - 1

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    def dimension(self, cell: Cell) -> int:
        try:
            return self._dimensions[cell]
        except KeyError as error:
            raise KeyError(f"unknown cell: {cell!r}") from error

    def cells_of_dimension(self, dimension: int) -> Tuple[Cell, ...]:
        if dimension < 0 or dimension >= len(self._by_dimension):
            return ()
        return self._by_dimension[dimension]

    def boundary(self, cell: Cell) -> Mapping[Cell, int]:
        try:
            return self._boundaries[cell]
        except KeyError as error:
            raise KeyError(f"unknown cell: {cell!r}") from error

    def closure(self, generators: Iterable[Cell]) -> FrozenSet[Cell]:
        pending = list(generators)
        result = set()
        while pending:
            cell = pending.pop()
            if cell not in self._dimensions:
                raise ValueError(f"unknown closure generator: {cell!r}")
            if cell in result:
                continue
            result.add(cell)
            pending.extend(self._boundaries[cell])
        return frozenset(result)

    def is_subcomplex(self, cells: Collection[Cell]) -> bool:
        subset = set(cells)
        return subset <= self.cell_set and all(
            set(self._boundaries[cell]) <= subset for cell in subset
        )

    def require_subcomplex(
        self, cells: Collection[Cell], *, description: str = "cells"
    ) -> FrozenSet[Cell]:
        subset = frozenset(cells)
        unknown = subset.difference(self.cell_set)
        if unknown:
            raise ValueError(
                f"{description} contain unknown cells: {sorted(unknown, key=repr)!r}"
            )
        missing_faces = {
            face
            for cell in subset
            for face in self._boundaries[cell]
            if face not in subset
        }
        if missing_faces:
            raise ValueError(
                f"{description} are not a subcomplex; missing boundary cells: "
                f"{sorted(missing_faces, key=repr)!r}"
            )
        return subset

    def betti_numbers(
        self,
        cells: Optional[Collection[Cell]] = None,
        *,
        modulus: int = 5,
    ) -> Tuple[int, ...]:
        """Compute cellular Betti numbers of a subcomplex over a prime field."""

        modulus = _require_prime(modulus)
        subset = (
            self.cell_set
            if cells is None
            else self.require_subcomplex(cells, description="homology cells")
        )
        if not subset:
            return ()

        # Basis of each chain group in the complex's cell order, assembled
        # from the subset itself rather than by scanning every cell.
        grouped: List[List[Cell]] = [[] for _ in self._by_dimension]
        for cell in subset:
            grouped[self._dimensions[cell]].append(cell)
        position = self._position
        basis = tuple(
            tuple(sorted(group, key=position.__getitem__)) for group in grouped
        )
        ranks = [0] * len(basis)
        for dimension in range(1, len(basis)):
            row_of = {cell: row for row, cell in enumerate(basis[dimension - 1])}
            columns = []
            for cell in basis[dimension]:
                columns.append(
                    (
                        (row_of[face], coefficient)
                        for face, coefficient in self._boundaries[cell].items()
                    )
                )
            ranks[dimension] = _rank_mod_prime(columns, modulus)

        return tuple(
            len(basis[dimension])
            - ranks[dimension]
            - (ranks[dimension + 1] if dimension + 1 < len(ranks) else 0)
            for dimension in range(len(basis))
        )

    def is_acyclic(
        self,
        cells: Collection[Cell],
        *,
        modulus: int = 5,
    ) -> bool:
        """Return whether a nonempty subcomplex is acyclic over ``GF(modulus)``."""

        subset = self.require_subcomplex(cells, description="acyclicity cells")
        if not subset:
            return False
        betti = self.betti_numbers(subset, modulus=modulus)
        return bool(betti) and betti[0] == 1 and all(value == 0 for value in betti[1:])


@dataclass(frozen=True)
class CubicalCell:
    """One cell of an axis-aligned cubical grid.

    ``anchor[i]`` is the lattice coordinate in axis ``i``.  If
    ``spanning[i]`` is true, the cell spans from that coordinate to the next;
    otherwise it is fixed at the indicated vertex coordinate.
    """

    anchor: Tuple[int, ...]
    spanning: Tuple[bool, ...]

    def __post_init__(self) -> None:
        anchor = tuple(self.anchor)
        spanning = tuple(self.spanning)
        if not anchor or len(anchor) != len(spanning):
            raise ValueError("anchor and spanning must have the same positive length")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in anchor):
            raise TypeError("cubical-cell anchor coordinates must be integers")
        if any(value < 0 for value in anchor):
            raise ValueError("cubical-cell anchor coordinates must be non-negative")
        if any(not isinstance(value, bool) for value in spanning):
            raise TypeError("cubical-cell spanning flags must be booleans")
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "spanning", spanning)

    @property
    def dimension(self) -> int:
        return sum(self.spanning)


class CubicalGridComplex(FiniteCellComplex):
    """The complete face complex of a regular rectangular grid."""

    def __init__(self, subdivisions: Sequence[int]) -> None:
        sizes = tuple(subdivisions)
        if not sizes:
            raise ValueError("subdivisions must contain at least one axis")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in sizes
        ):
            raise ValueError("all subdivisions must be positive integers")

        dimensions: Dict[Cell, int] = {}
        boundaries: Dict[Cell, Dict[Cell, int]] = {}
        masks = sorted(
            itertools.product((False, True), repeat=len(sizes)),
            key=lambda mask: (sum(mask), mask),
        )
        for spanning in masks:
            coordinate_ranges = [
                range(size) if spans else range(size + 1)
                for size, spans in zip(sizes, spanning)
            ]
            for anchor in itertools.product(*coordinate_ranges):
                cell = CubicalCell(tuple(anchor), tuple(spanning))
                dimensions[cell] = cell.dimension
                boundary: Dict[Cell, int] = {}
                active_axis_index = 0
                for axis, spans in enumerate(spanning):
                    if not spans:
                        continue
                    sign = -1 if active_axis_index % 2 == 0 else 1
                    lower_spanning = list(spanning)
                    lower_spanning[axis] = False
                    lower = CubicalCell(tuple(anchor), tuple(lower_spanning))
                    upper_anchor = list(anchor)
                    upper_anchor[axis] += 1
                    upper = CubicalCell(tuple(upper_anchor), tuple(lower_spanning))
                    boundary[lower] = sign
                    boundary[upper] = -sign
                    active_axis_index += 1
                boundaries[cell] = boundary

        self.subdivisions = sizes
        super().__init__(
            dimensions,
            boundaries,
            metadata={"kind": "cubical-grid", "subdivisions": sizes},
        )

    def top_cell(self, index: int) -> CubicalCell:
        """Return the top cube with the package's C-order linear index."""

        total = 1
        for size in self.subdivisions:
            total *= size
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total:
            raise ValueError(f"top-cell index must lie in [0, {total})")

        coordinates = [0] * len(self.subdivisions)
        remainder = index
        for axis in range(len(self.subdivisions) - 1, -1, -1):
            coordinates[axis] = remainder % self.subdivisions[axis]
            remainder //= self.subdivisions[axis]
        return CubicalCell(tuple(coordinates), (True,) * len(coordinates))

    def top_cell_index(self, cell: CubicalCell) -> int:
        if cell not in self.cell_set or cell.dimension != self.max_dimension:
            raise ValueError(f"not a top cell of this grid: {cell!r}")
        result = 0
        for coordinate, size in zip(cell.anchor, self.subdivisions):
            result = result * size + coordinate
        return result


def build_cubical_grid_complex(subdivisions: Sequence[int]) -> CubicalGridComplex:
    """Build the complete cubical face complex for ``subdivisions``."""

    return CubicalGridComplex(subdivisions)


class SparseCubicalGridComplex(FiniteCellComplex):
    """Cellular closure of selected cubes in a common rectangular grid.

    This avoids materializing the full tensor grid for a local Atlas family.
    The selected cubes must all belong to one uniform grid; mixed-depth Atlas
    data must first provide a uniform-depth candidate subfamily or a separate
    conforming adaptive-cell construction.
    """

    def __init__(
        self,
        subdivisions: Sequence[int],
        top_coordinates: Iterable[Sequence[int]],
    ) -> None:
        sizes = tuple(subdivisions)
        if not sizes:
            raise ValueError("subdivisions must contain at least one axis")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in sizes
        ):
            raise ValueError("all subdivisions must be positive integers")
        coordinates = tuple(
            sorted({tuple(int(value) for value in item) for item in top_coordinates})
        )
        if not coordinates:
            raise ValueError("a sparse cubical complex needs selected top cubes")
        for item in coordinates:
            if len(item) != len(sizes):
                raise ValueError("top-cube coordinates have the wrong dimension")
            if any(value < 0 or value >= size for value, size in zip(item, sizes)):
                raise ValueError("top-cube coordinates lie outside the common grid")

        cell_set: set[CubicalCell] = set()
        for top_anchor in coordinates:
            for spanning in itertools.product((False, True), repeat=len(sizes)):
                fixed_axes = tuple(
                    axis for axis, spans in enumerate(spanning) if not spans
                )
                for endpoint_choices in itertools.product(
                    (0, 1), repeat=len(fixed_axes)
                ):
                    anchor = list(top_anchor)
                    for axis, endpoint in zip(fixed_axes, endpoint_choices):
                        anchor[axis] += endpoint
                    cell_set.add(CubicalCell(tuple(anchor), tuple(spanning)))

        cells = tuple(
            sorted(
                cell_set,
                key=lambda cell: (cell.dimension, cell.spanning, cell.anchor),
            )
        )
        dimensions: Dict[Cell, int] = {cell: cell.dimension for cell in cells}
        boundaries: Dict[Cell, Dict[Cell, int]] = {}
        for cell in cells:
            boundary: Dict[Cell, int] = {}
            active_axis_index = 0
            for axis, spans in enumerate(cell.spanning):
                if not spans:
                    continue
                sign = -1 if active_axis_index % 2 == 0 else 1
                face_spanning = list(cell.spanning)
                face_spanning[axis] = False
                lower = CubicalCell(cell.anchor, tuple(face_spanning))
                upper_anchor = list(cell.anchor)
                upper_anchor[axis] += 1
                upper = CubicalCell(tuple(upper_anchor), tuple(face_spanning))
                boundary[lower] = sign
                boundary[upper] = -sign
                active_axis_index += 1
            boundaries[cell] = boundary

        self.subdivisions = sizes
        self.top_coordinates = coordinates
        self.top_coordinate_set = frozenset(coordinates)
        self._top_by_coordinates = MappingProxyType(
            {
                coordinates_: CubicalCell(coordinates_, (True,) * len(sizes))
                for coordinates_ in coordinates
            }
        )
        super().__init__(
            dimensions,
            boundaries,
            metadata={
                "kind": "sparse-cubical-grid",
                "subdivisions": sizes,
                "selected_top_cells": len(coordinates),
            },
        )

    def top_cell_at(self, coordinates: Sequence[int]) -> CubicalCell:
        key = tuple(int(value) for value in coordinates)
        try:
            return self._top_by_coordinates[key]
        except KeyError as error:
            raise KeyError(f"not a selected top cube: {key!r}") from error


class CellularResetMap:
    """An oriented cellular chain map from a guard subcomplex to the base."""

    def __init__(
        self,
        base_complex: FiniteCellComplex,
        guard_cells: Collection[Cell],
        images: Mapping[Cell, IntegerChain],
    ) -> None:
        guard = base_complex.require_subcomplex(
            guard_cells, description="guard cells"
        )
        if not guard:
            raise ValueError("a reset map requires a nonempty guard subcomplex")
        missing = guard.difference(images)
        extra = set(images).difference(guard)
        if missing or extra:
            raise ValueError(
                "reset images must be supplied exactly for the guard cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )

        normalized: Dict[Cell, Mapping[Cell, int]] = {}
        for source in guard:
            image = _normalize_chain(images[source])
            source_dimension = base_complex.dimension(source)
            for target in image:
                if target not in base_complex.cell_set:
                    raise ValueError(
                        f"reset image of {source!r} references unknown cell {target!r}"
                    )
                if base_complex.dimension(target) != source_dimension:
                    raise ValueError(
                        f"reset image of {source!r} changes cellular degree at {target!r}"
                    )
            if source_dimension == 0 and sum(image.values()) != 1:
                raise ValueError(
                    "a reset must send each guard zero-cell to a zero-chain of "
                    "augmentation one"
                )
            normalized[source] = MappingProxyType(image)

        for source in guard:
            boundary_after_reset: Dict[Cell, int] = {}
            for target, coefficient in normalized[source].items():
                _add_scaled_chain(
                    boundary_after_reset,
                    base_complex.boundary(target),
                    coefficient,
                )
            reset_after_boundary: Dict[Cell, int] = {}
            for face, coefficient in base_complex.boundary(source).items():
                _add_scaled_chain(
                    reset_after_boundary,
                    normalized[face],
                    coefficient,
                )
            if boundary_after_reset != reset_after_boundary:
                raise ValueError(
                    f"reset images do not define a chain map on {source!r}: "
                    f"dR={boundary_after_reset!r}, Rd={reset_after_boundary!r}"
                )

        self.base_complex = base_complex
        self.guard_cells = guard
        self._images = MappingProxyType(normalized)

    @classmethod
    def from_cell_map(
        cls,
        base_complex: FiniteCellComplex,
        guard_cells: Collection[Cell],
        cell_map: Mapping[Cell, Cell],
    ) -> "CellularResetMap":
        return cls(
            base_complex,
            guard_cells,
            {source: {target: 1} for source, target in cell_map.items()},
        )

    def image(self, guard_cell: Cell) -> Mapping[Cell, int]:
        try:
            return self._images[guard_cell]
        except KeyError as error:
            raise KeyError(f"not a guard cell: {guard_cell!r}") from error


class CellularAttachmentMap:
    """An integral cellular chain map between two independent complexes.

    This is the attachment datum needed for a genuine algebraic mapping
    cylinder.  Unlike :class:`CellularResetMap`, its source need not already
    be a subcomplex of its target.  The constructor verifies degree,
    augmentation on vertices, and ``d F = F d`` over the integers.
    """

    def __init__(
        self,
        source_complex: FiniteCellComplex,
        target_complex: FiniteCellComplex,
        images: Mapping[Cell, IntegerChain],
    ) -> None:
        missing = source_complex.cell_set.difference(images)
        extra = set(images).difference(source_complex.cell_set)
        if missing or extra:
            raise ValueError(
                "attachment images must be supplied exactly for all source cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )

        normalized: Dict[Cell, Mapping[Cell, int]] = {}
        for source in source_complex.cells:
            source_dimension = source_complex.dimension(source)
            image = _normalize_chain(images[source])
            for target in image:
                if target not in target_complex.cell_set:
                    raise ValueError(
                        f"attachment image of {source!r} references unknown "
                        f"target cell {target!r}"
                    )
                if target_complex.dimension(target) != source_dimension:
                    raise ValueError(
                        f"attachment image of {source!r} changes cellular degree "
                        f"at {target!r}"
                    )
            if source_dimension == 0 and sum(image.values()) != 1:
                raise ValueError(
                    "a cellular attachment must send every source vertex to a "
                    "zero-chain of augmentation one"
                )
            normalized[source] = MappingProxyType(image)

        for source in source_complex.cells:
            boundary_after_map: Dict[Cell, int] = {}
            for target, coefficient in normalized[source].items():
                _add_scaled_chain(
                    boundary_after_map,
                    target_complex.boundary(target),
                    coefficient,
                )
            map_after_boundary: Dict[Cell, int] = {}
            for face, coefficient in source_complex.boundary(source).items():
                _add_scaled_chain(
                    map_after_boundary,
                    normalized[face],
                    coefficient,
                )
            if boundary_after_map != map_after_boundary:
                raise ValueError(
                    f"attachment images do not define an integral chain map on "
                    f"{source!r}: dF={boundary_after_map!r}, "
                    f"Fd={map_after_boundary!r}"
                )

        self.source_complex = source_complex
        self.target_complex = target_complex
        self._images = MappingProxyType(normalized)

    def image(self, source_cell: Cell) -> Mapping[Cell, int]:
        try:
            return self._images[source_cell]
        except KeyError as error:
            raise KeyError(f"not a source cell: {source_cell!r}") from error


class CellularMapBetweenComplexes:
    """A verified degree-zero chain map over a prime field."""

    def __init__(
        self,
        source_complex: FiniteCellComplex,
        target_complex: FiniteCellComplex,
        images: Mapping[Cell, IntegerChain],
        *,
        modulus: int = 5,
    ) -> None:
        modulus = _require_prime(modulus)
        missing = source_complex.cell_set.difference(images)
        extra = set(images).difference(source_complex.cell_set)
        if missing or extra:
            raise ValueError(
                "chain-map images must be supplied exactly for all source cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )

        normalized: Dict[Cell, Mapping[Cell, int]] = {}
        for source in source_complex.cells:
            source_dimension = source_complex.dimension(source)
            image = {
                target: coefficient % modulus
                for target, coefficient in _normalize_chain(images[source]).items()
                if coefficient % modulus
            }
            for target in image:
                if target not in target_complex.cell_set:
                    raise ValueError(
                        f"chain image of {source!r} references unknown target "
                        f"cell {target!r}"
                    )
                if target_complex.dimension(target) != source_dimension:
                    raise ValueError(
                        f"chain image of {source!r} changes degree at {target!r}"
                    )
            if source_dimension == 0 and sum(image.values()) % modulus != 1:
                raise ValueError(
                    "a cellular map must send every source vertex to augmentation "
                    f"one over GF({modulus})"
                )
            normalized[source] = MappingProxyType(image)

        for source in source_complex.cells:
            boundary_after_map: Dict[Cell, int] = {}
            for target, coefficient in normalized[source].items():
                for face, incidence in target_complex.boundary(target).items():
                    value = (
                        boundary_after_map.get(face, 0) + coefficient * incidence
                    ) % modulus
                    if value:
                        boundary_after_map[face] = value
                    else:
                        boundary_after_map.pop(face, None)
            map_after_boundary: Dict[Cell, int] = {}
            for face, incidence in source_complex.boundary(source).items():
                for target, coefficient in normalized[face].items():
                    value = (
                        map_after_boundary.get(target, 0) + incidence * coefficient
                    ) % modulus
                    if value:
                        map_after_boundary[target] = value
                    else:
                        map_after_boundary.pop(target, None)
            if boundary_after_map != map_after_boundary:
                raise ValueError(
                    f"dF != Fd on {source!r} over GF({modulus}): "
                    f"dF={boundary_after_map!r}, Fd={map_after_boundary!r}"
                )

        self.source_complex = source_complex
        self.target_complex = target_complex
        self.modulus = modulus
        self._images = MappingProxyType(normalized)

    def image(self, source_cell: Cell) -> Mapping[Cell, int]:
        try:
            return self._images[source_cell]
        except KeyError as error:
            raise KeyError(f"not a source cell: {source_cell!r}") from error


class CrossComplexAcyclicCarrier:
    """A face-compatible acyclic carrier from one complex to another.

    The selector is constructed by the same acyclic-carrier induction used for
    fixed-time endomorphisms, but source boundaries and target fillings now
    belong to independent complexes.  Selection is over ``GF(modulus)``.
    ``construct_integral_attachment_map`` succeeds only when the selected map
    has an exact integral lift; it never silently uses a mod-p map as a CW
    attachment.
    """

    def __init__(
        self,
        source_complex: FiniteCellComplex,
        target_complex: FiniteCellComplex,
        image_generators: Mapping[Cell, Collection[Cell]],
        *,
        modulus: int = 5,
        validate_acyclic: bool = True,
    ) -> None:
        modulus = _require_prime(modulus)
        missing = source_complex.cell_set.difference(image_generators)
        extra = set(image_generators).difference(source_complex.cell_set)
        if missing or extra:
            raise ValueError(
                "carrier images must be supplied exactly for all source cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )
        images: Dict[Cell, FrozenSet[Cell]] = {}
        acyclicity_cache: Dict[FrozenSet[Cell], bool] = {}
        for source in source_complex.cells:
            generators = tuple(image_generators[source])
            if not generators:
                raise ValueError(f"carrier image of {source!r} must be nonempty")
            image = target_complex.closure(generators)
            if validate_acyclic:
                acyclic = acyclicity_cache.get(image)
                if acyclic is None:
                    acyclic = target_complex.is_acyclic(image, modulus=modulus)
                    acyclicity_cache[image] = acyclic
                if not acyclic:
                    raise ValueError(
                        f"carrier image of {source!r} is not acyclic over "
                        f"GF({modulus})"
                    )
            images[source] = image

        for source in source_complex.cells:
            for face in source_complex.boundary(source):
                if not images[face] <= images[source]:
                    raise ValueError(
                        f"carrier nesting fails for face {face!r} of {source!r}"
                    )

        self.source_complex = source_complex
        self.target_complex = target_complex
        self.modulus = modulus
        self.acyclicity_validated = bool(validate_acyclic)
        self._images = MappingProxyType(images)

    def image(self, source_cell: Cell) -> FrozenSet[Cell]:
        try:
            return self._images[source_cell]
        except KeyError as error:
            raise KeyError(f"not a source cell: {source_cell!r}") from error

    def carries(self, chain_map: CellularMapBetweenComplexes) -> bool:
        return bool(
            chain_map.source_complex is self.source_complex
            and chain_map.target_complex is self.target_complex
            and chain_map.modulus == self.modulus
            and all(
                set(chain_map.image(source)) <= self._images[source]
                for source in self.source_complex.cells
            )
        )

    def construct_chain_map(self) -> CellularMapBetweenComplexes:
        if not self.acyclicity_validated:
            raise ValueError(
                "constructing a chain selector requires validated acyclic "
                "carrier values"
            )

        images: Dict[Cell, Dict[Cell, int]] = {}
        for dimension in range(self.source_complex.max_dimension + 1):
            for source in self.source_complex.cells_of_dimension(dimension):
                carrier_value = self._images[source]
                if dimension == 0:
                    vertices = tuple(
                        cell
                        for cell in self.target_complex.cells_of_dimension(0)
                        if cell in carrier_value
                    )
                    if not vertices:
                        raise ValueError(
                            f"carrier value of {source!r} contains no target vertex"
                        )
                    images[source] = {vertices[0]: 1}
                    continue

                right_hand_side: Dict[Cell, int] = {}
                for face, incidence in self.source_complex.boundary(source).items():
                    for target, coefficient in images[face].items():
                        value = (
                            right_hand_side.get(target, 0)
                            + incidence * coefficient
                        ) % self.modulus
                        if value:
                            right_hand_side[target] = value
                        else:
                            right_hand_side.pop(target, None)

                target_rows = tuple(
                    cell
                    for cell in self.target_complex.cells_of_dimension(dimension - 1)
                    if cell in carrier_value
                )
                target_columns = tuple(
                    cell
                    for cell in self.target_complex.cells_of_dimension(dimension)
                    if cell in carrier_value
                )
                try:
                    images[source] = _solve_linear_system_mod_prime(
                        target_rows,
                        target_columns,
                        {
                            cell: self.target_complex.boundary(cell)
                            for cell in target_columns
                        },
                        right_hand_side,
                        self.modulus,
                    )
                except ValueError as error:
                    raise ValueError(
                        "cross-complex acyclic-carrier selection failed on "
                        f"{source!r}: {error}"
                    ) from error

        chain_map = CellularMapBetweenComplexes(
            self.source_complex,
            self.target_complex,
            images,
            modulus=self.modulus,
        )
        if not self.carries(chain_map):
            raise AssertionError("constructed chain map is not carrier subordinate")
        return chain_map

    def construct_integral_attachment_map(self) -> CellularAttachmentMap:
        """Lift the selected mod-p map only when the lift is exactly cellular."""

        selected = self.construct_chain_map()
        midpoint = self.modulus // 2
        lifted = {
            source: {
                target: (
                    coefficient
                    if coefficient <= midpoint
                    else coefficient - self.modulus
                )
                for target, coefficient in selected.image(source).items()
            }
            for source in self.source_complex.cells
        }
        try:
            return CellularAttachmentMap(
                self.source_complex,
                self.target_complex,
                lifted,
            )
        except ValueError as error:
            raise ValueError(
                "the subordinate GF(p) selector has no verified integral lift "
                "under the canonical coefficient representatives; provide an "
                "explicit integral attachment or a certified integral selector"
            ) from error


@dataclass(frozen=True)
class CubicalHyperplaneAttachmentAudit:
    """Hard-gate record for an attachment into one cubical hyperplane.

    This audit complements, rather than replaces, the algebraic constructors.
    ``CellularAttachmentMap`` has already verified an integral chain map and
    ``CrossComplexAcyclicCarrier`` has already verified face nesting and
    acyclicity.  The audit additionally verifies geometric hyperplane support,
    nonempty handle-face images, and retention of every base top-cell coface
    incident to the attachment support.
    """

    axis: int
    coordinate: int
    interior: bool
    source_cell_count: int
    source_top_cell_count: int
    target_support_cells: tuple[CubicalCell, ...]
    target_facets: tuple[CubicalCell, ...]
    incident_base_top_cells: tuple[CubicalCell, ...]
    no_base_top_cell_straddles: bool
    complete_source_cell_images: bool
    all_incident_top_cofaces_present: bool
    carrier_face_nesting_and_acyclicity_verified: bool
    integral_chain_map_verified: bool
    carrier_subordination_verified: bool


def audit_cubical_hyperplane_attachment(
    carrier: CrossComplexAcyclicCarrier,
    attachment: CellularAttachmentMap,
    *,
    axis: int,
    coordinate: int,
) -> CubicalHyperplaneAttachmentAudit:
    """Verify an acyclic-carrier attachment into a cubical grid hyperplane.

    The target may be a complete or sparse cubical grid.  For every
    top-dimensional target facet used by the attachment, all ambient incident
    top cubes (two for an interior hyperplane, one at an outer boundary) must
    be present in the selected target complex.  Consequently this function
    does not close a local seam transitively by adding cofaces: missing cofaces
    are a hard failure that the caller must resolve in its predeclared active
    family.
    """

    if not isinstance(carrier, CrossComplexAcyclicCarrier):
        raise TypeError("carrier must be a CrossComplexAcyclicCarrier")
    if not isinstance(attachment, CellularAttachmentMap):
        raise TypeError("attachment must be a CellularAttachmentMap")
    if not carrier.acyclicity_validated:
        raise ValueError("hyperplane attachment requires validated carrier acyclicity")
    if (
        carrier.source_complex is not attachment.source_complex
        or carrier.target_complex is not attachment.target_complex
    ):
        raise ValueError("carrier and integral attachment must use identical complexes")

    target = attachment.target_complex
    if not isinstance(target, (CubicalGridComplex, SparseCubicalGridComplex)):
        raise TypeError(
            "hyperplane attachment target must be a complete or sparse cubical grid"
        )
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("hyperplane axis must be an integer")
    if not 0 <= axis < len(target.subdivisions):
        raise ValueError("hyperplane axis lies outside the target dimension")
    if isinstance(coordinate, bool) or not isinstance(coordinate, int):
        raise TypeError("hyperplane coordinate must be an integer grid vertex")
    if not 0 <= coordinate <= target.subdivisions[axis]:
        raise ValueError("hyperplane coordinate lies outside the target grid")
    source = attachment.source_complex
    if source.max_dimension + 1 != target.max_dimension:
        raise ValueError(
            "a hyperplane attachment source must have codimension one in its target"
        )

    support: set[CubicalCell] = set()
    complete_images = True
    for source_cell in source.cells:
        image = attachment.image(source_cell)
        if not image:
            complete_images = False
            break
        for target_cell in image:
            if not isinstance(target_cell, CubicalCell):
                raise ValueError("hyperplane attachment contains a non-cubical target")
            if target_cell.spanning[axis] or target_cell.anchor[axis] != coordinate:
                raise ValueError(
                    "attachment image leaves the declared cubical hyperplane at "
                    f"target {target_cell!r}"
                )
            support.add(target_cell)
    if not complete_images:
        raise ValueError("every handle-face cell needs a nonempty attachment image")

    source_top_cells = source.cells_of_dimension(source.max_dimension)
    target_facets: set[CubicalCell] = set()
    for source_cell in source_top_cells:
        image = attachment.image(source_cell)
        if any(abs(coefficient) != 1 for coefficient in image.values()):
            raise ValueError(
                "top-dimensional hyperplane attachment coefficients must be unit "
                f"incidences on source {source_cell!r}"
            )
        target_facets.update(image)
    if not target_facets:
        raise ValueError("hyperplane attachment has no top-dimensional target facets")

    incident_top_cells: set[CubicalCell] = set()
    missing_cofaces: set[CubicalCell] = set()
    top_spanning = (True,) * target.max_dimension
    for facet in target_facets:
        expected_anchors: list[tuple[int, ...]] = []
        if coordinate > 0:
            lower = list(facet.anchor)
            lower[axis] = coordinate - 1
            expected_anchors.append(tuple(lower))
        if coordinate < target.subdivisions[axis]:
            upper = list(facet.anchor)
            upper[axis] = coordinate
            expected_anchors.append(tuple(upper))
        for anchor in expected_anchors:
            coface = CubicalCell(anchor, top_spanning)
            if coface in target.cell_set:
                incident_top_cells.add(coface)
            else:
                missing_cofaces.add(coface)
    if missing_cofaces:
        raise ValueError(
            "attachment hyperplane support is missing incident base top cofaces: "
            f"{sorted(missing_cofaces, key=repr)[:10]!r}"
        )

    straddling = tuple(
        cell
        for cell in target.cells_of_dimension(target.max_dimension)
        if cell.anchor[axis] < coordinate < cell.anchor[axis] + 1
    )
    if straddling:  # Integer cubical coordinates make this an invariant check.
        raise ValueError(
            "base top cells straddle the declared hyperplane: "
            f"{list(straddling[:10])!r}"
        )

    modular_attachment = CellularMapBetweenComplexes(
        source,
        target,
        {cell: attachment.image(cell) for cell in source.cells},
        modulus=carrier.modulus,
    )
    if not carrier.carries(modular_attachment):
        raise ValueError("integral hyperplane attachment is not carrier subordinate")

    return CubicalHyperplaneAttachmentAudit(
        axis=axis,
        coordinate=coordinate,
        interior=0 < coordinate < target.subdivisions[axis],
        source_cell_count=len(source.cells),
        source_top_cell_count=len(source_top_cells),
        target_support_cells=tuple(
            sorted(support, key=lambda cell: (cell.dimension, cell.spanning, cell.anchor))
        ),
        target_facets=tuple(
            sorted(
                target_facets,
                key=lambda cell: (cell.dimension, cell.spanning, cell.anchor),
            )
        ),
        incident_base_top_cells=tuple(
            sorted(
                incident_top_cells,
                key=lambda cell: (cell.dimension, cell.spanning, cell.anchor),
            )
        ),
        no_base_top_cell_straddles=True,
        complete_source_cell_images=True,
        all_incident_top_cofaces_present=True,
        carrier_face_nesting_and_acyclicity_verified=True,
        integral_chain_map_verified=True,
        carrier_subordination_verified=True,
    )


@dataclass(frozen=True)
class DoubleMappingCylinderHandle:
    """Independent guard complex with bottom and reset attachments to a base."""

    handle_id: Hashable
    guard_complex: FiniteCellComplex
    guard_attachment: CellularAttachmentMap
    reset_attachment: CellularAttachmentMap
    slabs: int

    def __post_init__(self) -> None:
        _require_hashable(self.handle_id, "handle_id")
        if self.guard_attachment.source_complex is not self.guard_complex:
            raise ValueError("guard_attachment must have guard_complex as its source")
        if self.reset_attachment.source_complex is not self.guard_complex:
            raise ValueError("reset_attachment must have guard_complex as its source")
        if (
            self.guard_attachment.target_complex
            is not self.reset_attachment.target_complex
        ):
            raise ValueError("bottom and reset attachments must share one base complex")
        if isinstance(self.slabs, bool) or not isinstance(self.slabs, int) or self.slabs <= 0:
            raise ValueError("slabs must be a positive integer")


@dataclass(frozen=True)
class ResetHandle:
    """One unit reset handle subdivided into uniform symbolic phase slabs."""

    handle_id: Hashable
    reset: CellularResetMap
    slabs: int

    def __post_init__(self) -> None:
        _require_hashable(self.handle_id, "handle_id")
        if isinstance(self.slabs, bool) or not isinstance(self.slabs, int) or self.slabs <= 0:
            raise ValueError("slabs must be a positive integer")


@dataclass(frozen=True)
class SuspensionBaseCell:
    """A cell in the canonical base copy of the suspension quotient."""

    base_cell: Cell

    def __post_init__(self) -> None:
        _require_hashable(self.base_cell, "base cell")


@dataclass(frozen=True)
class GuardPrismCell:
    """The product of one guard cell and one closed phase slab."""

    handle_id: Hashable
    guard_cell: Cell
    slab: int

    def __post_init__(self) -> None:
        _require_hashable(self.handle_id, "handle_id")
        _require_hashable(self.guard_cell, "guard cell")
        if isinstance(self.slab, bool) or not isinstance(self.slab, int) or self.slab < 0:
            raise ValueError("phase slab must be a non-negative integer")


@dataclass(frozen=True)
class PhaseSliceCell:
    """An interior phase-level copy of one guard cell."""

    handle_id: Hashable
    guard_cell: Cell
    level: int

    def __post_init__(self) -> None:
        _require_hashable(self.handle_id, "handle_id")
        _require_hashable(self.guard_cell, "guard cell")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level <= 0:
            raise ValueError("interior phase level must be a positive integer")


class PhaseCellRegistry:
    """Canonical registry for phase cells of all reset handles in one complex."""

    def __init__(self) -> None:
        self._prisms: Dict[Tuple[Hashable, Cell, int], GuardPrismCell] = {}
        self._slices: Dict[Tuple[Hashable, Cell, int], PhaseSliceCell] = {}
        self._frozen = False

    def prism(self, handle_id: Hashable, guard_cell: Cell, slab: int) -> GuardPrismCell:
        key = (handle_id, guard_cell, slab)
        if key not in self._prisms:
            if self._frozen:
                raise KeyError(f"unregistered phase prism: {key!r}")
            self._prisms[key] = GuardPrismCell(*key)
        return self._prisms[key]

    def slice(self, handle_id: Hashable, guard_cell: Cell, level: int) -> PhaseSliceCell:
        key = (handle_id, guard_cell, level)
        if key not in self._slices:
            if self._frozen:
                raise KeyError(f"unregistered phase slice: {key!r}")
            self._slices[key] = PhaseSliceCell(*key)
        return self._slices[key]

    def freeze(self) -> None:
        """Prevent accidental creation of cells outside the built complex."""

        self._frozen = True

    @property
    def prism_cells(self) -> Tuple[GuardPrismCell, ...]:
        return tuple(self._prisms.values())

    @property
    def slice_cells(self) -> Tuple[PhaseSliceCell, ...]:
        return tuple(self._slices.values())


class SuspensionCellComplex(FiniteCellComplex):
    """A base complex with reset mapping cylinders attached cellwise."""

    def __init__(
        self,
        base_complex: FiniteCellComplex,
        handles: Iterable[ResetHandle],
    ) -> None:
        handle_data = tuple(handles)
        handle_ids = [handle.handle_id for handle in handle_data]
        if len(set(handle_ids)) != len(handle_ids):
            raise ValueError("reset handle identifiers must be unique")
        for handle in handle_data:
            if handle.reset.base_complex is not base_complex:
                raise ValueError(
                    f"reset handle {handle.handle_id!r} was built on a different "
                    "base complex object"
                )

        registry = PhaseCellRegistry()
        base_refs = {
            cell: SuspensionBaseCell(cell) for cell in base_complex.cells
        }
        dimensions: Dict[Cell, int] = {
            base_refs[cell]: base_complex.dimension(cell) for cell in base_complex.cells
        }
        boundaries: Dict[Cell, Dict[Cell, int]] = {
            base_refs[cell]: {
                base_refs[face]: coefficient
                for face, coefficient in base_complex.boundary(cell).items()
            }
            for cell in base_complex.cells
        }

        for handle in handle_data:
            guard_in_order = tuple(
                cell
                for cell in base_complex.cells
                if cell in handle.reset.guard_cells
            )
            for guard_cell in guard_in_order:
                guard_dimension = base_complex.dimension(guard_cell)
                for level in range(1, handle.slabs):
                    phase_slice = registry.slice(
                        handle.handle_id, guard_cell, level
                    )
                    dimensions[phase_slice] = guard_dimension
                    boundaries[phase_slice] = {
                        registry.slice(handle.handle_id, face, level): coefficient
                        for face, coefficient in base_complex.boundary(guard_cell).items()
                    }

                for slab in range(handle.slabs):
                    prism = registry.prism(handle.handle_id, guard_cell, slab)
                    dimensions[prism] = guard_dimension + 1
                    boundary: Dict[Cell, int] = {
                        registry.prism(handle.handle_id, face, slab): coefficient
                        for face, coefficient in base_complex.boundary(guard_cell).items()
                    }
                    endpoint_sign = -1 if guard_dimension % 2 == 0 else 1

                    if slab == 0:
                        bottom = {base_refs[guard_cell]: 1}
                    else:
                        bottom = {
                            registry.slice(handle.handle_id, guard_cell, slab): 1
                        }
                    _add_scaled_chain(boundary, bottom, endpoint_sign)

                    if slab + 1 == handle.slabs:
                        top = {
                            base_refs[target]: coefficient
                            for target, coefficient in handle.reset.image(guard_cell).items()
                        }
                    else:
                        top = {
                            registry.slice(handle.handle_id, guard_cell, slab + 1): 1
                        }
                    _add_scaled_chain(boundary, top, -endpoint_sign)
                    boundaries[prism] = boundary

        registry.freeze()
        self.base_complex = base_complex
        self.handles = handle_data
        self.phase_registry = registry
        self._base_refs = MappingProxyType(base_refs)
        super().__init__(
            dimensions,
            boundaries,
            metadata={
                "kind": "reset-glued-suspension",
                "handle_ids": tuple(handle_ids),
            },
        )

    @classmethod
    def from_base(
        cls,
        base_complex: FiniteCellComplex,
        handles: Iterable[ResetHandle],
    ) -> "SuspensionCellComplex":
        return cls(base_complex, handles)

    def base_cell(self, cell: Cell) -> SuspensionBaseCell:
        try:
            return self._base_refs[cell]
        except KeyError as error:
            raise KeyError(f"not a cell of the base complex: {cell!r}") from error

    def prism_cell(
        self, handle_id: Hashable, guard_cell: Cell, slab: int
    ) -> GuardPrismCell:
        cell = self.phase_registry.prism(handle_id, guard_cell, slab)
        if cell not in self.cell_set:
            raise KeyError(f"not a phase prism of this suspension complex: {cell!r}")
        return cell

    def slice_cell(
        self, handle_id: Hashable, guard_cell: Cell, level: int
    ) -> PhaseSliceCell:
        cell = self.phase_registry.slice(handle_id, guard_cell, level)
        if cell not in self.cell_set:
            raise KeyError(f"not a phase slice of this suspension complex: {cell!r}")
        return cell

    @property
    def base_cells(self) -> Tuple[SuspensionBaseCell, ...]:
        return tuple(self._base_refs.values())

    @property
    def phase_cells(self) -> Tuple[Cell, ...]:
        return self.phase_registry.slice_cells + self.phase_registry.prism_cells


class DoubleMappingCylinderComplex(FiniteCellComplex):
    """Reset suspension built from an independent guard complex.

    For a guard cell ``g`` of dimension ``d``, each phase prism has boundary

    ``prism(d g) + (-1)^d (top(g) - bottom(g))``.

    The first bottom is attached by ``guard_attachment`` and the last top by
    ``reset_attachment``.  Interior phase levels are literal copies of the
    guard complex.  Both attachment maps are verified integral chain maps, so
    the resulting boundary is checked over the integers by
    :class:`FiniteCellComplex`.
    """

    def __init__(
        self,
        base_complex: FiniteCellComplex,
        handles: Iterable[DoubleMappingCylinderHandle],
    ) -> None:
        handle_data = tuple(handles)
        handle_ids = [handle.handle_id for handle in handle_data]
        if len(set(handle_ids)) != len(handle_ids):
            raise ValueError("double mapping-cylinder handle ids must be unique")
        for handle in handle_data:
            if handle.guard_attachment.target_complex is not base_complex:
                raise ValueError(
                    f"handle {handle.handle_id!r} attachments target a different "
                    "base complex"
                )

        registry = PhaseCellRegistry()
        base_refs = {
            cell: SuspensionBaseCell(cell) for cell in base_complex.cells
        }
        dimensions: Dict[Cell, int] = {
            base_refs[cell]: base_complex.dimension(cell) for cell in base_complex.cells
        }
        boundaries: Dict[Cell, Dict[Cell, int]] = {
            base_refs[cell]: {
                base_refs[face]: coefficient
                for face, coefficient in base_complex.boundary(cell).items()
            }
            for cell in base_complex.cells
        }

        for handle in handle_data:
            guard = handle.guard_complex
            for guard_cell in guard.cells:
                guard_dimension = guard.dimension(guard_cell)
                for level in range(1, handle.slabs):
                    phase_slice = registry.slice(
                        handle.handle_id, guard_cell, level
                    )
                    dimensions[phase_slice] = guard_dimension
                    boundaries[phase_slice] = {
                        registry.slice(handle.handle_id, face, level): coefficient
                        for face, coefficient in guard.boundary(guard_cell).items()
                    }

                for slab in range(handle.slabs):
                    prism = registry.prism(handle.handle_id, guard_cell, slab)
                    dimensions[prism] = guard_dimension + 1
                    boundary: Dict[Cell, int] = {
                        registry.prism(handle.handle_id, face, slab): coefficient
                        for face, coefficient in guard.boundary(guard_cell).items()
                    }

                    endpoint_sign = -1 if guard_dimension % 2 == 0 else 1
                    if slab == 0:
                        bottom = {
                            base_refs[target]: coefficient
                            for target, coefficient in handle.guard_attachment.image(
                                guard_cell
                            ).items()
                        }
                    else:
                        bottom = {
                            registry.slice(handle.handle_id, guard_cell, slab): 1
                        }
                    _add_scaled_chain(boundary, bottom, endpoint_sign)

                    if slab + 1 == handle.slabs:
                        top = {
                            base_refs[target]: coefficient
                            for target, coefficient in handle.reset_attachment.image(
                                guard_cell
                            ).items()
                        }
                    else:
                        top = {
                            registry.slice(handle.handle_id, guard_cell, slab + 1): 1
                        }
                    _add_scaled_chain(boundary, top, -endpoint_sign)
                    boundaries[prism] = boundary

        registry.freeze()
        self.base_complex = base_complex
        self.handles = handle_data
        self.phase_registry = registry
        self._base_refs = MappingProxyType(base_refs)
        self._handle_by_id = MappingProxyType(
            {handle.handle_id: handle for handle in handle_data}
        )
        super().__init__(
            dimensions,
            boundaries,
            metadata={
                "kind": "independent-guard-double-mapping-cylinder",
                "handle_ids": tuple(handle_ids),
                "integral_attachment_maps_verified": True,
            },
        )

    def base_cell(self, cell: Cell) -> SuspensionBaseCell:
        try:
            return self._base_refs[cell]
        except KeyError as error:
            raise KeyError(f"not a cell of the base complex: {cell!r}") from error

    def prism_cell(
        self, handle_id: Hashable, guard_cell: Cell, slab: int
    ) -> GuardPrismCell:
        try:
            handle = self._handle_by_id[handle_id]
        except KeyError as error:
            raise KeyError(f"unknown double mapping-cylinder handle {handle_id!r}") from error
        if guard_cell not in handle.guard_complex.cell_set:
            raise KeyError(f"not a cell of handle {handle_id!r}'s guard complex")
        cell = self.phase_registry.prism(handle_id, guard_cell, slab)
        if cell not in self.cell_set:
            raise KeyError(f"not a phase prism of this complex: {cell!r}")
        return cell

    def slice_cell(
        self, handle_id: Hashable, guard_cell: Cell, level: int
    ) -> PhaseSliceCell:
        try:
            handle = self._handle_by_id[handle_id]
        except KeyError as error:
            raise KeyError(f"unknown double mapping-cylinder handle {handle_id!r}") from error
        if guard_cell not in handle.guard_complex.cell_set:
            raise KeyError(f"not a cell of handle {handle_id!r}'s guard complex")
        cell = self.phase_registry.slice(handle_id, guard_cell, level)
        if cell not in self.cell_set:
            raise KeyError(f"not a phase slice of this complex: {cell!r}")
        return cell

    @property
    def base_cells(self) -> Tuple[SuspensionBaseCell, ...]:
        return tuple(self._base_refs.values())

    @property
    def phase_cells(self) -> Tuple[Cell, ...]:
        return self.phase_registry.slice_cells + self.phase_registry.prism_cells


class RelativeCellPair:
    """A finite cellular pair ``(P1, P0)`` with ``P0 <= P1``."""

    def __init__(
        self,
        complex_: FiniteCellComplex,
        p1_cells: Collection[Cell],
        p0_cells: Collection[Cell] = (),
    ) -> None:
        p1 = complex_.require_subcomplex(p1_cells, description="P1 cells")
        p0 = complex_.require_subcomplex(p0_cells, description="P0 cells")
        if not p0 <= p1:
            raise ValueError("P0 must be a subcomplex of P1")
        self.complex = complex_
        self.p1_cells = p1
        self.p0_cells = p0
        quotient = p1.difference(p0)
        self._basis = tuple(
            tuple(cell for cell in complex_.cells_of_dimension(dimension) if cell in quotient)
            for dimension in range(complex_.max_dimension + 1)
        )

    @classmethod
    def generated(
        cls,
        complex_: FiniteCellComplex,
        p1_generators: Iterable[Cell],
        p0_generators: Iterable[Cell] = (),
    ) -> "RelativeCellPair":
        return cls(
            complex_,
            complex_.closure(p1_generators),
            complex_.closure(p0_generators),
        )

    @property
    def basis_by_dimension(self) -> Tuple[Tuple[Cell, ...], ...]:
        return self._basis

    @property
    def cell_counts(self) -> Tuple[int, ...]:
        return tuple(len(group) for group in self._basis)

    def boundary_entries(self, *, modulus: int = 5) -> Tuple[Tuple[SparseEntry, ...], ...]:
        modulus = _require_prime(modulus)
        result: List[Tuple[SparseEntry, ...]] = [()]
        for dimension in range(1, len(self._basis)):
            row_of = {
                cell: row for row, cell in enumerate(self._basis[dimension - 1])
            }
            entries: List[SparseEntry] = []
            for column, cell in enumerate(self._basis[dimension]):
                for face, coefficient in self.complex.boundary(cell).items():
                    if face not in row_of:
                        continue  # The face is zero in C(P1) / C(P0).
                    value = coefficient % modulus
                    if value:
                        entries.append((row_of[face], column, value))
            result.append(tuple(entries))
        return tuple(result)


class FixedTimeCellRelation:
    """A finite multivalued fixed-time relation on selected suspension cells."""

    def __init__(
        self,
        complex_: FiniteCellComplex,
        cells: Collection[Cell],
        images: Mapping[Cell, Collection[Cell]],
    ) -> None:
        ordered = tuple(cell for cell in complex_.cells if cell in set(cells))
        if not ordered:
            raise ValueError("a fixed-time relation needs at least one cell")
        if set(ordered) != set(cells):
            unknown = set(cells).difference(complex_.cell_set)
            raise ValueError(f"relation contains unknown cells: {unknown!r}")
        missing = set(ordered).difference(images)
        extra = set(images).difference(ordered)
        if missing or extra:
            raise ValueError(
                "relation images must be supplied exactly for its cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )
        universe = set(ordered)
        normalized: Dict[Cell, FrozenSet[Cell]] = {}
        for source in ordered:
            targets = frozenset(images[source])
            unknown_targets = targets.difference(universe)
            if unknown_targets:
                raise ValueError(
                    f"image of {source!r} leaves the relation cell family: "
                    f"{sorted(unknown_targets, key=repr)!r}"
                )
            normalized[source] = targets
        self.complex = complex_
        self.cells = ordered
        self._images = MappingProxyType(normalized)

    def image(self, cell: Cell) -> FrozenSet[Cell]:
        try:
            return self._images[cell]
        except KeyError as error:
            raise KeyError(f"not a source cell of this relation: {cell!r}") from error

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.graph["semantics"] = "fixed-time suspension multivalued relation"
        graph.add_nodes_from(self.cells)
        for source in self.cells:
            graph.add_edges_from((source, target) for target in self._images[source])
        return graph

    def to_integer_relation(self) -> "CMGDBCellRelationPayload":
        index_of = {cell: index for index, cell in enumerate(self.cells)}
        adjacency = tuple(
            tuple(sorted(index_of[target] for target in self._images[source]))
            for source in self.cells
        )
        return CMGDBCellRelationPayload(self.cells, adjacency)


class SampledSuspensionCellAdapter:
    """Map legacy sampled ``BaseCell``/``PhaseCell`` tokens to quotient cells.

    The pointwise sampler labels a phase slab by the index of a
    guard-intersecting *base box*.  A cellular suspension instead needs the
    actual guard-trace cell.  ``guard_cell_by_box`` supplies that missing
    correspondence.  It must be single-valued; if a base box contains several
    independently meshed guard cells, use the quotient cell identifiers
    directly instead of the legacy token type.
    """

    def __init__(
        self,
        suspension: SuspensionCellComplex,
        base_grid_complex: CubicalGridComplex,
        *,
        handle_id: Hashable,
        guard_cell_by_box: Mapping[int, Cell],
    ) -> None:
        if suspension.base_complex is not base_grid_complex:
            raise ValueError(
                "the suspension and sampled-token adapter must share the same "
                "base grid complex object"
            )
        handle_by_id = {
            handle.handle_id: handle for handle in suspension.handles
        }
        if handle_id not in handle_by_id:
            raise ValueError(f"unknown reset handle identifier: {handle_id!r}")
        handle = handle_by_id[handle_id]

        normalized: Dict[int, Cell] = {}
        for box_index, guard_cell in guard_cell_by_box.items():
            if isinstance(box_index, bool) or not isinstance(box_index, int):
                raise TypeError("guard-cell box indices must be integers")
            base_grid_complex.top_cell(box_index)  # range validation
            if guard_cell not in base_grid_complex.cell_set:
                raise ValueError(
                    f"guard trace for box {box_index} is not a base-complex cell"
                )
            if guard_cell not in handle.reset.guard_cells:
                raise ValueError(
                    f"cell {guard_cell!r} for box {box_index} is not in the "
                    f"guard of reset handle {handle_id!r}"
                )
            normalized[box_index] = guard_cell

        self.suspension = suspension
        self.base_grid_complex = base_grid_complex
        self.handle_id = handle_id
        self.guard_cell_by_box = MappingProxyType(normalized)

    def cell_for_token(self, token: BaseCell | PhaseCell) -> Cell:
        if isinstance(token, BaseCell):
            return self.suspension.base_cell(
                self.base_grid_complex.top_cell(token.index)
            )
        if isinstance(token, PhaseCell):
            try:
                guard_cell = self.guard_cell_by_box[token.guard_cell]
            except KeyError as error:
                raise KeyError(
                    "no guard-trace cell was supplied for sampled base box "
                    f"{token.guard_cell}"
                ) from error
            return self.suspension.prism_cell(
                self.handle_id, guard_cell, token.slab
            )
        raise TypeError("token must be BaseCell or PhaseCell")

    def relation_from_tokens(
        self,
        cells: Collection[BaseCell | PhaseCell],
        images: Mapping[
            BaseCell | PhaseCell, Collection[BaseCell | PhaseCell]
        ],
    ) -> FixedTimeCellRelation:
        source_tokens = set(cells)
        missing = source_tokens.difference(images)
        extra = set(images).difference(source_tokens)
        if missing or extra:
            raise ValueError(
                "token relation images must be supplied exactly for its cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )
        converted_cells = tuple(self.cell_for_token(cell) for cell in cells)
        # Several guard-intersecting full-dimensional boxes can legitimately
        # name the same guard-trace cell.  Their legacy PhaseCell tokens then
        # collapse to one registered prism, whose image is the union of all
        # corresponding token images.  A dict comprehension would silently
        # retain only the last source token and drop graph edges.
        converted_images: Dict[Cell, set[Cell]] = {}
        for source, targets in images.items():
            converted_source = self.cell_for_token(source)
            converted_images.setdefault(converted_source, set()).update(
                self.cell_for_token(target) for target in targets
            )
        return FixedTimeCellRelation(
            self.suspension, converted_cells, converted_images
        )

    def relation_from_graph(self, graph: nx.DiGraph) -> FixedTimeCellRelation:
        if not graph.is_directed():
            raise TypeError("sampled suspension graph must be directed")
        images = {
            source: tuple(graph.successors(source)) for source in graph.nodes
        }
        return self.relation_from_tokens(tuple(graph.nodes), images)


class FixedTimeCarrier:
    """A validated acyclic cellular carrier for one fixed-time map.

    Each supplied image is interpreted as a set of generators and replaced by
    its cellular closure.  The constructor verifies the face-nesting carrier
    condition.  With ``validate_acyclic=True`` it also verifies that every
    carrier value has the homology of a point over ``GF(modulus)``.
    """

    def __init__(
        self,
        complex_: FiniteCellComplex,
        image_generators: Mapping[Cell, Collection[Cell]],
        *,
        modulus: int = 5,
        validate_acyclic: bool = True,
    ) -> None:
        modulus = _require_prime(modulus)
        missing = complex_.cell_set.difference(image_generators)
        extra = set(image_generators).difference(complex_.cell_set)
        if missing or extra:
            raise ValueError(
                "carrier images must be supplied exactly for all complex cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )
        images: Dict[Cell, FrozenSet[Cell]] = {}
        # Physical Atlas nerves often give many source simplices the same
        # induced target subcomplex.  Homology depends only on that closed
        # target cell set, so validate each distinct carrier value once while
        # still retaining and checking the value for every source cell.
        acyclicity_cache: Dict[FrozenSet[Cell], bool] = {}
        for source in complex_.cells:
            generators = tuple(image_generators[source])
            if not generators:
                raise ValueError(f"carrier image of {source!r} must be nonempty")
            image = complex_.closure(generators)
            if validate_acyclic:
                acyclic = acyclicity_cache.get(image)
                if acyclic is None:
                    acyclic = complex_.is_acyclic(image, modulus=modulus)
                    acyclicity_cache[image] = acyclic
                if not acyclic:
                    raise ValueError(
                        f"carrier image of {source!r} is not acyclic over GF({modulus})"
                    )
            images[source] = image

        for source in complex_.cells:
            for face in complex_.boundary(source):
                if not images[face] <= images[source]:
                    raise ValueError(
                        f"carrier nesting fails for face {face!r} of {source!r}"
                    )

        self.complex = complex_
        self.modulus = modulus
        self.acyclicity_validated = validate_acyclic
        self._images = MappingProxyType(images)

    @classmethod
    def identity(
        cls, complex_: FiniteCellComplex, *, modulus: int = 5
    ) -> "FixedTimeCarrier":
        return cls(
            complex_,
            {cell: (cell,) for cell in complex_.cells},
            modulus=modulus,
            validate_acyclic=True,
        )

    def image(self, cell: Cell) -> FrozenSet[Cell]:
        try:
            return self._images[cell]
        except KeyError as error:
            raise KeyError(f"not a source cell of this carrier: {cell!r}") from error

    def carries(self, chain_map: "CellularChainMap") -> bool:
        if (
            chain_map.complex is not self.complex
            or chain_map.modulus != self.modulus
        ):
            return False
        return all(
            set(chain_map.image(cell)) <= self._images[cell]
            for cell in self.complex.cells
        )

    def require_carries(self, chain_map: "CellularChainMap") -> None:
        if not self.carries(chain_map):
            raise ValueError("the cellular chain map is not subordinate to this carrier")

    def preserves_pair(self, pair: RelativeCellPair) -> bool:
        """Return whether every carrier value respects ``P1`` and ``P0``.

        This is the topological pair-preservation gate.  Checking only a final
        matrix would be too late: a selector can preserve the pair even when
        the multivalued carrier from which it was chosen does not.
        """

        if pair.complex is not self.complex:
            return False
        return all(
            self._images[source] <= pair.p1_cells for source in pair.p1_cells
        ) and all(
            self._images[source] <= pair.p0_cells for source in pair.p0_cells
        )

    def require_preserves_pair(self, pair: RelativeCellPair) -> None:
        if not self.preserves_pair(pair):
            raise ValueError("the acyclic carrier does not preserve the relative pair")

    def construct_chain_map(
        self,
        *,
        pair: RelativeCellPair | None = None,
    ) -> "CellularChainMap":
        """Construct a deterministic chain selector subordinate to the carrier.

        The construction is the finite acyclic-carrier induction.  A vertex is
        sent to one vertex of its carrier value.  In degree ``d > 0`` the image
        of the boundary has already been constructed; a ``d``-chain inside the
        carrier value is found whose boundary equals that image.  Acyclicity
        guarantees existence, while the explicit linear solve and the
        :class:`CellularChainMap` constructor independently verify it.
        """

        if not self.acyclicity_validated:
            raise ValueError(
                "constructing a chain selector requires validated acyclic carrier values"
            )
        if pair is not None:
            self.require_preserves_pair(pair)

        images: Dict[Cell, Dict[Cell, int]] = {}
        for dimension in range(self.complex.max_dimension + 1):
            for source in self.complex.cells_of_dimension(dimension):
                carrier_value = self._images[source]
                if dimension == 0:
                    vertices = tuple(
                        cell
                        for cell in self.complex.cells_of_dimension(0)
                        if cell in carrier_value
                    )
                    if not vertices:
                        raise ValueError(
                            f"carrier value of {source!r} contains no zero-cell"
                        )
                    images[source] = {vertices[0]: 1}
                    continue

                right_hand_side: Dict[Cell, int] = {}
                for face, incidence in self.complex.boundary(source).items():
                    for target, coefficient in images[face].items():
                        value = (
                            right_hand_side.get(target, 0)
                            + incidence * coefficient
                        ) % self.modulus
                        if value:
                            right_hand_side[target] = value
                        else:
                            right_hand_side.pop(target, None)

                target_rows = tuple(
                    cell
                    for cell in self.complex.cells_of_dimension(dimension - 1)
                    if cell in carrier_value
                )
                target_columns = tuple(
                    cell
                    for cell in self.complex.cells_of_dimension(dimension)
                    if cell in carrier_value
                )
                try:
                    images[source] = _solve_linear_system_mod_prime(
                        target_rows,
                        target_columns,
                        {
                            cell: self.complex.boundary(cell)
                            for cell in target_columns
                        },
                        right_hand_side,
                        self.modulus,
                    )
                except ValueError as error:
                    raise ValueError(
                        "acyclic-carrier chain selection failed on "
                        f"{source!r}: {error}"
                    ) from error

        chain_map = CellularChainMap(
            self.complex,
            images,
            modulus=self.modulus,
        )
        self.require_carries(chain_map)
        if pair is not None:
            if not chain_map.preserves(pair.p1_cells):
                raise AssertionError("constructed chain map does not preserve P1")
            if not chain_map.preserves(pair.p0_cells):
                raise AssertionError("constructed chain map does not preserve P0")
        return chain_map


class CellularChainMap:
    """A degree-zero cellular chain map over a prime field."""

    def __init__(
        self,
        complex_: FiniteCellComplex,
        images: Mapping[Cell, IntegerChain],
        *,
        modulus: int = 5,
    ) -> None:
        modulus = _require_prime(modulus)
        missing = complex_.cell_set.difference(images)
        extra = set(images).difference(complex_.cell_set)
        if missing or extra:
            raise ValueError(
                "chain-map images must be supplied exactly for all complex cells; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )

        normalized: Dict[Cell, Mapping[Cell, int]] = {}
        for source in complex_.cells:
            source_dimension = complex_.dimension(source)
            image = {
                target: coefficient % modulus
                for target, coefficient in _normalize_chain(images[source]).items()
                if coefficient % modulus
            }
            for target in image:
                if target not in complex_.cell_set:
                    raise ValueError(
                        f"chain image of {source!r} references unknown cell {target!r}"
                    )
                if complex_.dimension(target) != source_dimension:
                    raise ValueError(
                        f"chain image of {source!r} changes degree at {target!r}"
                    )
            normalized[source] = MappingProxyType(image)

        for source in complex_.cells:
            boundary_after_map: Dict[Cell, int] = {}
            for target, coefficient in normalized[source].items():
                for face, incidence in complex_.boundary(target).items():
                    value = (
                        boundary_after_map.get(face, 0) + coefficient * incidence
                    ) % modulus
                    if value:
                        boundary_after_map[face] = value
                    else:
                        boundary_after_map.pop(face, None)

            map_after_boundary: Dict[Cell, int] = {}
            for face, incidence in complex_.boundary(source).items():
                for target, coefficient in normalized[face].items():
                    value = (
                        map_after_boundary.get(target, 0) + incidence * coefficient
                    ) % modulus
                    if value:
                        map_after_boundary[target] = value
                    else:
                        map_after_boundary.pop(target, None)

            if boundary_after_map != map_after_boundary:
                raise ValueError(
                    f"dF != Fd on {source!r} over GF({modulus}): "
                    f"dF={boundary_after_map!r}, Fd={map_after_boundary!r}"
                )

        self.complex = complex_
        self.modulus = modulus
        self._images = MappingProxyType(normalized)

    @classmethod
    def identity(
        cls, complex_: FiniteCellComplex, *, modulus: int = 5
    ) -> "CellularChainMap":
        return cls(
            complex_,
            {cell: {cell: 1} for cell in complex_.cells},
            modulus=modulus,
        )

    def image(self, cell: Cell) -> Mapping[Cell, int]:
        try:
            return self._images[cell]
        except KeyError as error:
            raise KeyError(f"not a source cell of this chain map: {cell!r}") from error

    def preserves(self, cells: Collection[Cell]) -> bool:
        subset = set(cells)
        return all(set(self._images[cell]) <= subset for cell in subset)

    def to_cmgdb_payload(
        self, pair: RelativeCellPair
    ) -> "CMGDBRelativeHomologyPayload":
        if pair.complex is not self.complex:
            raise ValueError("relative pair and chain map use different complexes")
        if self.modulus != 5:
            raise ValueError(
                "CMGDB relative-homology bridge currently requires coefficient field 5"
            )
        if not self.preserves(pair.p1_cells):
            raise ValueError("chain map does not preserve P1")
        if not self.preserves(pair.p0_cells):
            raise ValueError("chain map does not preserve P0")

        chain_entries: List[Tuple[SparseEntry, ...]] = []
        for basis in pair.basis_by_dimension:
            row_of = {cell: row for row, cell in enumerate(basis)}
            entries: List[SparseEntry] = []
            for column, source in enumerate(basis):
                for target, coefficient in self._images[source].items():
                    if target not in row_of:
                        continue  # The target is zero in C(P1) / C(P0).
                    value = coefficient % 5
                    if value:
                        entries.append((row_of[target], column, value))
            chain_entries.append(tuple(entries))

        return CMGDBRelativeHomologyPayload(
            cell_counts=pair.cell_counts,
            boundary_entries=pair.boundary_entries(modulus=5),
            chain_map_entries=tuple(chain_entries),
            basis_by_dimension=pair.basis_by_dimension,
        )


@dataclass(frozen=True)
class CMGDBCellRelationPayload:
    """Integer relabeling of a cell relation for graph/SCC interchange."""

    cells: Tuple[Cell, ...]
    adjacency: Tuple[Tuple[int, ...], ...]

    def as_adjacency_dict(self) -> Dict[int, List[int]]:
        return {index: list(targets) for index, targets in enumerate(self.adjacency)}

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "cells": [repr(cell) for cell in self.cells],
            "adjacency": [list(targets) for targets in self.adjacency],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class CMGDBRelativeHomologyPayload:
    """Sparse relative chain complex/map input for the generalized CMGDB API.

    For degree ``d``, boundary triples are ``(row_in_degree_d_minus_1,
    column_in_degree_d, coefficient_mod_5)`` and chain-map triples are square
    degree-``d`` entries.  Degree zero has an empty boundary list.  Basis labels
    are retained as provenance but are not passed to CMGDB.
    """

    cell_counts: Tuple[int, ...]
    boundary_entries: Tuple[Tuple[SparseEntry, ...], ...]
    chain_map_entries: Tuple[Tuple[SparseEntry, ...], ...]
    basis_by_dimension: Tuple[Tuple[Cell, ...], ...]
    coefficient_field: int = 5

    def __post_init__(self) -> None:
        if self.coefficient_field != 5:
            raise ValueError(
                "CMGDB relative-homology payloads currently require "
                "coefficient_field=5"
            )
        degree_count = len(self.cell_counts)
        if degree_count == 0:
            raise ValueError("cell_counts must contain degree zero")
        if not (
            len(self.boundary_entries)
            == len(self.chain_map_entries)
            == len(self.basis_by_dimension)
            == degree_count
        ):
            raise ValueError("all CMGDB payload degree lists must have equal length")
        if any(count < 0 for count in self.cell_counts):
            raise ValueError("cell counts must be non-negative")
        if tuple(len(group) for group in self.basis_by_dimension) != self.cell_counts:
            raise ValueError("basis sizes do not match cell_counts")
        if self.boundary_entries[0]:
            raise ValueError("degree-zero boundary entries must be empty")

        for degree in range(degree_count):
            boundary_shape = (
                0 if degree == 0 else self.cell_counts[degree - 1],
                self.cell_counts[degree],
            )
            self._validate_sparse(
                self.boundary_entries[degree], boundary_shape, f"boundary[{degree}]"
            )
            self._validate_sparse(
                self.chain_map_entries[degree],
                (self.cell_counts[degree], self.cell_counts[degree]),
                f"chain_map[{degree}]",
            )

    @staticmethod
    def _validate_sparse(
        entries: Sequence[SparseEntry], shape: Tuple[int, int], description: str
    ) -> None:
        coordinates = set()
        for row, column, coefficient in entries:
            if not 0 <= row < shape[0] or not 0 <= column < shape[1]:
                raise ValueError(
                    f"{description} entry {(row, column)!r} is outside shape {shape!r}"
                )
            if not 0 < coefficient < 5:
                raise ValueError(
                    f"{description} coefficient must be reduced to 1..4"
                )
            if (row, column) in coordinates:
                raise ValueError(f"{description} contains duplicate coordinates")
            coordinates.add((row, column))

    def as_compute_args(
        self,
    ) -> Tuple[List[int], List[List[SparseEntry]], List[List[SparseEntry]]]:
        """Return exact arguments for ``ComputeRelativeHomologyShiftClass``."""

        return (
            list(self.cell_counts),
            [list(entries) for entries in self.boundary_entries],
            [list(entries) for entries in self.chain_map_entries],
        )

    def to_json_dict(self, *, include_basis: bool = True) -> Dict[str, object]:
        result: Dict[str, object] = {
            "coefficient_field": self.coefficient_field,
            "cell_counts": list(self.cell_counts),
            "boundary_entries": [
                [list(entry) for entry in degree_entries]
                for degree_entries in self.boundary_entries
            ],
            "chain_map_entries": [
                [list(entry) for entry in degree_entries]
                for degree_entries in self.chain_map_entries
            ],
        }
        if include_basis:
            result["basis_by_dimension"] = [
                [repr(cell) for cell in basis]
                for basis in self.basis_by_dimension
            ]
        return result

    def to_json(self, *, include_basis: bool = True, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_json_dict(include_basis=include_basis),
            indent=indent,
            sort_keys=True,
        )


__all__ = [
    "CMGDBCellRelationPayload",
    "CMGDBRelativeHomologyPayload",
    "CellularAttachmentMap",
    "CellularChainMap",
    "CellularMapBetweenComplexes",
    "CellularResetMap",
    "CrossComplexAcyclicCarrier",
    "CubicalHyperplaneAttachmentAudit",
    "CubicalCell",
    "CubicalGridComplex",
    "DoubleMappingCylinderComplex",
    "DoubleMappingCylinderHandle",
    "FiniteCellComplex",
    "FixedTimeCarrier",
    "FixedTimeCellRelation",
    "GuardPrismCell",
    "PhaseCellRegistry",
    "PhaseSliceCell",
    "RelativeCellPair",
    "ResetHandle",
    "SampledSuspensionCellAdapter",
    "SparseCubicalGridComplex",
    "SuspensionBaseCell",
    "SuspensionCellComplex",
    "build_cubical_grid_complex",
    "audit_cubical_hyperplane_attachment",
]
