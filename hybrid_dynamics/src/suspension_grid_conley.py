"""Finite-relation Conley labels for Morse sets on the paper suspension grid.

For a Morse set ``S`` of the sampled relation ``F_n`` on ``Xi_n`` this module
forms the standard combinatorial pair

``X = S cup F(S)``,   ``A = F(S) minus S``

of atoms, covers ``|X|`` by the elementary pieces of its atoms, and hands the
pieces to the verified reset-quotient nerve of :mod:`atlas_conley`
(:class:`AtlasQuotientNerveComplex2D`).  Every piece is an actual closed
rectangle in the base chart ``(x, y)`` or in the handle chart ``(u, s)``, and
the identifications ``(u, 0) ~ gamma(u)`` and ``(u, 1) ~ r(gamma(u))`` are
supplied as affine seam embeddings.  The relation on pieces sends a piece of
the atom ``xi`` to every piece of ``F(xi)``; the carrier, the acyclicity,
pair, and chain-map checks, and the CMGDB shift-class computation are those
of :func:`atlas_conley.prepare_atlas_relation_conley_2d`.

The relative homology ``H_*(X, A; GF(5))`` of the pair is always computed
once the nerve is built; it needs no carrier.  If it is zero, the shift
class is zero in every degree whatever the index map, so the label is
trivial and the index map is not formed.  Otherwise the index map is formed
and its shift class is the label; if a carrier, pair, or chain-map check
fails, the homology is still reported and only the label is missing.

The index map is built in one of two ways (``index_map``).

``"exit-components"`` (the default) is the construction above, in which the
image of every atom of ``A`` (which may map partly outside ``X``) is
replaced by its connected component of ``A``, zero in ``C(X)/C(A)``.  The
carrier of an atom of ``A`` is then its whole component, and the map is
refused when a component is not acyclic (an annulus, for example).

``"excision"`` keeps the true images.  With ``Xbar = X cup F(X)`` and
``Abar = Xbar minus S`` (atoms), the pieces of ``F(xi)`` lie in ``Xbar`` for
every atom ``xi`` of ``X``, and ``F(A)`` misses ``S`` when ``S`` is a
strongly connected component (an atom of ``A`` with an image in ``S`` would
lie on a cycle through ``S``).  The chain map ``F_#: C(X, A) -> C(Xbar,
Abar)`` is chosen from the acyclic carrier that sends a nerve simplex of
``X`` to the subcomplex of the nerve of ``Xbar`` spanned by the images of
its vertices.  Every carrier must be acyclic and the carrier of a simplex
of ``A`` must lie in ``Abar``; an atom of ``X`` with an empty image (its
samples left the window) has an empty carrier, and the map is refused.  The
inclusion ``i: (X, A) -> (Xbar, Abar)`` is an excision (``Xbar = X cup
Abar`` and ``X cap Abar = A`` as atoms), and that it induces an isomorphism
on ``H_*( ; GF(5))`` is checked: equal dimensions in every degree and an
invertible matrix.  The index map is ``i_*^{-1} F_*`` on ``H_*(X, A;
GF(5))``, computed exactly over ``GF(5)``
(:class:`suspension_complex.RelativeHomologyBasis`); its shift class, degree
by degree, is read by ``CMGDB.ComputeRelativeHomologyShiftClass`` from the
matrices of the map on homology.  When ``F(X)`` lies in ``X`` (so that
``F(A)`` lies in ``A``), ``Xbar = X``, ``Abar = A``, and ``i`` is the
identity.

The result is a finite-relation shift class over ``GF(5)``.  It is not a
certified Conley index of the continuous fixed-time map, because the
relation is sampled.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .atlas_conley import (
    AffineBoundaryEmbedding2D,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    _InducedCarrierGenerators,
    prepare_atlas_relation_conley_2d,
)
from .suspension_complex import (
    CrossComplexAcyclicCarrier,
    RelativeHomologyBasis,
    _rank_mod_prime,
)
from .suspension_grid import SuspensionGrid, UnsupportedSuspensionGridError, build_suspension_grid
from .suspension_grid_relation import SuspensionGridProblem, SuspensionGridRelation


BASE_CHART = 0
HANDLE_CHART = 1

#: Constructions of the index map; the first is the default.
INDEX_MAPS = ("exit-components", "excision")


class IndexSizeLimitError(RuntimeError):
    """The pair of a Morse set has more elementary pieces than the caller allows."""


class ExcisionError(RuntimeError):
    """The inclusion ``(X, A) -> (Xbar, Abar)`` is not an isomorphism on homology."""


def _affine_seam(
    curve,
    u_bounds: tuple[float, float],
    *,
    name: str,
) -> AffineBoundaryEmbedding2D:
    """Recover an axis-aligned affine seam ``u -> (fixed, a u + b)``."""

    u = np.linspace(u_bounds[0], u_bounds[1], 65)
    points = curve(u)
    constant = [
        axis
        for axis in range(2)
        if np.ptp(points[:, axis]) <= 1.0e-12 * max(1.0, float(np.max(np.abs(points[:, axis]))))
    ]
    if len(constant) != 1:
        raise UnsupportedSuspensionGridError(
            f"the {name} seam is not an axis-aligned segment; the quotient nerve "
            "of atlas_conley needs an axis-aligned affine seam"
        )
    fixed_axis = constant[0]
    varying = 1 - fixed_axis
    slope, offset = np.polyfit(u, points[:, varying], 1)
    residual = np.max(np.abs(slope * u + offset - points[:, varying]))
    if residual > 1.0e-9 * max(1.0, float(np.max(np.abs(points[:, varying])))):
        raise UnsupportedSuspensionGridError(f"the {name} seam is not affine in u")
    return AffineBoundaryEmbedding2D(
        fixed_axis,
        float(points[0, fixed_axis]),
        float(slope),
        offset=float(offset),
    )


def suspension_grid_gluing(grid: SuspensionGrid) -> AtlasResetGluing2D:
    """Seam data ``(u,0) ~ gamma(u)`` and ``(u,1) ~ r(gamma(u))``."""

    return AtlasResetGluing2D(
        BASE_CHART,
        HANDLE_CHART,
        guard=_affine_seam(grid.guard.gamma, grid.guard.u_bounds, name="guard"),
        reset=_affine_seam(grid.guard.reset, grid.guard.u_bounds, name="reset"),
    )


def piece_rectangles(grid: SuspensionGrid, pieces: npt.ArrayLike) -> list[AtlasRectangleCell2D]:
    """Actual closed rectangles of pieces (base chart or handle chart)."""

    values = np.asarray(pieces, dtype=np.int64)
    base = values[values < grid.n_base]
    handle = values[values >= grid.n_base]
    cells = [
        AtlasRectangleCell2D(int(piece), BASE_CHART, tuple(float(v) for v in bounds))
        for piece, bounds in zip(base, grid.base_bounds(base))
    ]
    cells.extend(
        AtlasRectangleCell2D(int(piece), HANDLE_CHART, tuple(float(v) for v in bounds))
        for piece, bounds in zip(handle, grid.handle_bounds(handle))
    )
    return cells


def _interior_seams(
    gluing: AtlasResetGluing2D,
    cells: list[AtlasRectangleCell2D],
) -> tuple[str, ...]:
    base = [cell for cell in cells if cell.chart_id == BASE_CHART]
    handle = [cell for cell in cells if cell.chart_id == HANDLE_CHART]
    if not base or not handle:
        return ()
    result = []
    for name, embedding in (("guard", gluing.guard), ("reset", gluing.reset)):
        axis = embedding.fixed_axis
        lower = min(cell.bounds[axis] for cell in base)
        upper = max(cell.bounds[2 + axis] for cell in base)
        if lower + 1.0e-10 < embedding.fixed_value < upper - 1.0e-10:
            result.append(name)
    return tuple(result)


@dataclass
class SuspensionGridConleyResult:
    """Finite-relation shift class of one Morse set, or the blocker.

    ``homology_computed`` and ``homology_dimensions`` record the relative
    homology of the pair, which is known whenever the nerve was built.
    ``computed`` says whether the label ``shift_class`` is known, and
    ``label_source`` how: from the index map, or from zero relative homology.
    ``blocker`` is the reason the label is missing, and
    ``index_map_blocker`` the reason the index map could not be formed.
    ``index_map`` names the construction of the index map (see
    ``INDEX_MAPS``).  For ``"excision"``, ``excision`` records the pair
    ``(Xbar, Abar)`` (atoms, pieces, nerve cell counts, and homology
    dimensions, as far as they were computed) and the matrices of the index
    map on ``H_k(X, A; GF(5))``, one list of rows per degree.  ``to_dict``
    writes ``index_map`` and ``excision`` only for ``"excision"``, so the
    record of the default construction is unchanged.
    """

    morse_node: int
    computed: bool
    shift_class: tuple[str, ...] = ()
    homology_dimensions: tuple[int, ...] = ()
    homology_computed: bool = False
    label_source: str = ""
    pair_atoms: dict[str, int] = field(default_factory=dict)
    pair_pieces: dict[str, int] = field(default_factory=dict)
    nerve_cell_counts: tuple[int, ...] = ()
    blocker: str = ""
    index_map_blocker: str = ""
    seconds: float = 0.0
    index_map: str = INDEX_MAPS[0]
    excision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = {
            "morse_node": self.morse_node,
            "computed": self.computed,
            "shift_class": list(self.shift_class),
            "homology_dimensions": list(self.homology_dimensions),
            "homology_computed": self.homology_computed,
            "label_source": self.label_source,
            "pair_atoms": self.pair_atoms,
            "pair_pieces": self.pair_pieces,
            "nerve_cell_counts": list(self.nerve_cell_counts),
            "coefficient_field": 5,
            "result_scope": "finite_reset_quotient_relation" if self.computed else "",
            "continuous_system_conley_index_certified": False,
            "blocker": self.blocker,
            "index_map_blocker": self.index_map_blocker,
            "seconds": self.seconds,
        }
        # The record of the default construction keeps its earlier keys.
        if self.index_map != INDEX_MAPS[0]:
            record["index_map"] = self.index_map
            record["excision"] = self.excision
        return record


def relative_homology_dimensions(
    pair: AtlasRelativeIndexPair2D,
    modulus: int = 5,
) -> tuple[int, ...]:
    """Dimensions of ``H_k(X, A; GF(modulus))`` from the relative boundary matrices."""

    relative = pair.relative_pair
    counts = relative.cell_counts
    entries = relative.boundary_entries(modulus=modulus)
    ranks = [0] * (len(counts) + 1)
    for dimension in range(1, len(counts)):
        columns: list[list[tuple[int, int]]] = [[] for _ in range(counts[dimension])]
        for row, column, value in entries[dimension]:
            columns[column].append((row, value))
        ranks[dimension] = _rank_mod_prime(columns, modulus)
    return tuple(
        int(counts[dimension] - ranks[dimension] - ranks[dimension + 1])
        for dimension in range(len(counts))
    )


def _solve_mod_prime(
    matrix: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    modulus: int,
) -> list[list[int]] | None:
    """``matrix^{-1} right`` over ``GF(modulus)``; ``None`` if ``matrix`` is singular."""

    size = len(matrix)
    width = len(right[0]) if right else 0
    rows = [
        [int(value) % modulus for value in matrix[row]]
        + [int(value) % modulus for value in right[row]]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        inverse = pow(rows[column][column], -1, modulus)
        rows[column] = [value * inverse % modulus for value in rows[column]]
        for row in range(size):
            scale = rows[row][column]
            if row != column and scale:
                rows[row] = [
                    (value - scale * lead) % modulus
                    for value, lead in zip(rows[row], rows[column])
                ]
    return [row[size : size + width] for row in rows]


def _shift_class_of_matrices(matrices: Sequence[Sequence[Sequence[int]]]) -> tuple[str, ...]:
    """Shift class, degree by degree, of the maps given by square matrices over ``GF(5)``.

    The matrices are handed to ``CMGDB.ComputeRelativeHomologyShiftClass``
    as a chain complex with zero boundary, so its homology is the space
    itself and the induced map is the matrix; the strings are those of the
    labels of the default construction.
    """

    import CMGDB

    counts = [len(matrix) for matrix in matrices]
    entries = [
        [
            (row, column, int(value) % 5)
            for row, values in enumerate(matrix)
            for column, value in enumerate(values)
            if int(value) % 5
        ]
        for matrix in matrices
    ]
    payload = CMGDB.ComputeRelativeHomologyShiftClass(counts, [[] for _ in counts], entries)
    if [int(value) for value in payload["homology_dimensions"]] != counts:
        raise AssertionError("CMGDB changed the dimensions of a complex with zero boundary")
    return tuple(str(entry) for entry in payload["shift_class"])


def _excision_index_map(
    relation: SuspensionGridRelation,
    gluing: AtlasResetGluing2D,
    s_atoms: npt.NDArray[np.int64],
    x_atoms: npt.NDArray[np.int64],
    x_pieces: npt.NDArray[np.int64],
    a_pieces: npt.NDArray[np.int64],
    pair: AtlasRelativeIndexPair2D,
    dimensions: tuple[int, ...],
    record: dict[str, Any],
    *,
    maximum_simplex_size: int,
    max_pieces: int | None,
) -> tuple[str, ...]:
    """Shift class of ``i_*^{-1} F_*`` on ``H_*(X, A; GF(5))``; see the module notes.

    ``pair`` is the pair ``(X, A)`` on the nerve of ``X`` and ``dimensions``
    its relative homology.  ``record`` receives the data of ``(Xbar, Abar)``
    and the matrices of the index map as they are computed.
    """

    grid = relation.grid
    xbar_atoms = np.union1d(x_atoms, relation.image_of(x_atoms))
    abar_atoms = np.setdiff1d(xbar_atoms, s_atoms)
    xbar_pieces = np.concatenate([grid.atom(atom) for atom in xbar_atoms])
    abar_pieces = (
        np.concatenate([grid.atom(atom) for atom in abar_atoms])
        if abar_atoms.size
        else np.zeros(0, dtype=np.int64)
    )
    record["pair_atoms"] = {"Xbar": int(xbar_atoms.size), "Abar": int(abar_atoms.size)}
    record["pair_pieces"] = {"Xbar": int(xbar_pieces.size), "Abar": int(abar_pieces.size)}
    if max_pieces is not None and xbar_pieces.size > int(max_pieces):
        raise IndexSizeLimitError(
            f"Xbar = X cup F(X) has {xbar_pieces.size} elementary pieces, more than "
            f"max_pieces={int(max_pieces)}; the index map was not attempted"
        )

    cycle_degrees = [degree for degree, value in enumerate(dimensions) if value]
    if np.array_equal(xbar_atoms, x_atoms):
        # F(X) lies in X: then Xbar = X, Abar = X minus S = A, and i is the
        # identity of (X, A), whose nerve and homology are reused.
        nerve = pair.nerve
        source = target = pair
        source_homology = target_homology = RelativeHomologyBasis(
            pair.relative_pair, cycle_degrees=cycle_degrees
        )
    else:
        cells = piece_rectangles(grid, xbar_pieces)
        nerve = AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            maximum_simplex_size=maximum_simplex_size,
            interior_seam_subcomplexes=_interior_seams(gluing, cells),
        )
        source = AtlasRelativeIndexPair2D(nerve, x_pieces.tolist(), a_pieces.tolist())
        if source.complex.cell_set != pair.complex.cell_set:
            raise AssertionError(
                "the nerve of Xbar restricted to the pieces of X is not the nerve of X"
            )
        target = AtlasRelativeIndexPair2D(nerve, xbar_pieces.tolist(), abar_pieces.tolist())
        source_homology = None
        target_homology = RelativeHomologyBasis(target.relative_pair)
    counts: dict[int, int] = {}
    for cell in nerve.cells:
        counts[nerve.dimension(cell)] = counts.get(nerve.dimension(cell), 0) + 1
    record["nerve_cell_counts"] = [counts.get(d, 0) for d in range(max(counts) + 1)]
    record["homology_dimensions"] = list(target_homology.dimensions)
    degrees = max(len(dimensions), len(target_homology.dimensions))
    padded = tuple(dimensions) + (0,) * (degrees - len(dimensions))
    padded_target = target_homology.dimensions + (0,) * (
        degrees - len(target_homology.dimensions)
    )
    if padded != padded_target:
        raise ExcisionError(
            f"H_*(Xbar, Abar) has dimensions {padded_target!r} and H_*(X, A) has "
            f"{padded!r}; the inclusion is not an isomorphism"
        )

    # True images: a piece of the atom xi goes to every piece of F(xi).
    xbar_set = set(xbar_pieces.tolist())
    vertex_images: dict[int, frozenset[int]] = {}
    for atom in x_atoms.tolist():
        targets = frozenset(
            int(piece) for target_atom in relation.image(atom) for piece in grid.atom(target_atom)
        )
        if not targets:
            raise ValueError(f"the atom {atom} of X has an empty image; its carrier is empty")
        if not targets <= xbar_set:
            raise AssertionError(f"the image of the atom {atom} leaves Xbar")
        for piece in grid.atom(atom).tolist():
            vertex_images[int(piece)] = targets
    carrier = CrossComplexAcyclicCarrier(
        source.complex,
        target.complex,
        _InducedCarrierGenerators(target.complex, vertex_images, source_complex=source.complex),
        modulus=5,
        validate_acyclic=True,
    )
    abar_cells = target.relative_pair.p0_cells
    for cell in source.relative_pair.p0_cells:
        if not carrier.image(cell) <= abar_cells:
            raise ValueError(f"the carrier of {cell!r} in A does not lie in Abar")
    chain_map = carrier.construct_chain_map()

    if source_homology is None:
        source_homology = RelativeHomologyBasis(source.relative_pair, cycle_degrees=cycle_degrees)
    if source_homology.dimensions != tuple(dimensions):
        raise AssertionError(
            f"the homology basis of (X, A) has dimensions {source_homology.dimensions!r}, "
            f"the relative boundary matrices give {tuple(dimensions)!r}"
        )
    matrices: list[list[list[int]]] = []
    for degree, size in enumerate(dimensions):
        if not size:
            matrices.append([])
            continue
        cycles = source_homology.cycles(degree)
        own = [source_homology.coordinates(degree, cycle) for cycle in cycles]
        if _solve_mod_prime([list(row) for row in zip(*own)], [[0]] * size, 5) is None:
            raise AssertionError(
                f"the cycles of degree {degree} are not a basis of H_{degree}(X, A)"
            )
        included = [target_homology.coordinates(degree, cycle) for cycle in cycles]
        mapped = []
        for cycle in cycles:
            image: dict[Any, int] = {}
            for cell, coefficient in cycle.items():
                for target_cell, value in chain_map.image(cell).items():
                    total = (image.get(target_cell, 0) + coefficient * value) % 5
                    if total:
                        image[target_cell] = total
                    else:
                        image.pop(target_cell, None)
            mapped.append(target_homology.coordinates(degree, image))
        # Columns are the coordinates of the images of the basis cycles.
        inclusion = [list(row) for row in zip(*included)]
        matrix = _solve_mod_prime(inclusion, [list(row) for row in zip(*mapped)], 5)
        if matrix is None:
            raise ExcisionError(
                f"the inclusion (X, A) -> (Xbar, Abar) is not an isomorphism on H_{degree}"
            )
        matrices.append(matrix)
    record["index_matrices"] = matrices
    return _shift_class_of_matrices(matrices)


def compute_suspension_grid_conley_index(
    relation: SuspensionGridRelation,
    morse_set: npt.ArrayLike,
    *,
    morse_node: int = 0,
    maximum_simplex_size: int = 16,
    max_pieces: int | None = None,
    gluing: AtlasResetGluing2D | None = None,
    index_map: str = INDEX_MAPS[0],
) -> SuspensionGridConleyResult:
    """Shift class of the index map on ``(S cup F(S), F(S) minus S)``.

    The relative homology of the pair is computed first.  If it is zero, the
    label is zero in every degree and the index map is not formed.  A failed
    gate of the index map (carrier acyclicity, pair preservation, chain map)
    is recorded as ``index_map_blocker``; the label is then missing, and the
    homology is still reported.  If the quotient nerve is not a good cover,
    neither is computed.  With ``max_pieces``, a pair whose ``X`` has more
    elementary pieces is not attempted: the blocker is
    ``IndexSizeLimitError``.  The quotient nerve has about ten simplices per
    piece and is held in memory, so this bounds the time and memory of the
    computation on fine grids.  ``gluing`` is ``suspension_grid_gluing`` of
    the grid, built here when it is not given.

    ``index_map`` chooses the construction of the index map (see the module
    notes): ``"exit-components"`` (the default) or ``"excision"``.  With
    ``"excision"`` and ``max_pieces``, the index map is also not attempted
    when ``Xbar = X cup F(X)`` has more pieces; the homology of ``(X, A)``
    is still reported.
    """

    if index_map not in INDEX_MAPS:
        raise ValueError(f"index_map must be one of {INDEX_MAPS!r}; got {index_map!r}")
    started = time.perf_counter()
    grid = relation.grid
    s_atoms = np.unique(np.asarray(morse_set, dtype=np.int64))
    image = relation.image_of(s_atoms)
    x_atoms = np.union1d(s_atoms, image)
    a_atoms = np.setdiff1d(image, s_atoms)
    x_pieces = np.concatenate([grid.atom(atom) for atom in x_atoms])
    a_pieces = (
        np.concatenate([grid.atom(atom) for atom in a_atoms])
        if a_atoms.size
        else np.zeros(0, dtype=np.int64)
    )
    result = SuspensionGridConleyResult(
        morse_node=morse_node,
        computed=False,
        pair_atoms={"S": int(s_atoms.size), "X": int(x_atoms.size), "A": int(a_atoms.size)},
        pair_pieces={"X": int(x_pieces.size), "A": int(a_pieces.size)},
        index_map=index_map,
    )
    try:
        if max_pieces is not None and x_pieces.size > int(max_pieces):
            raise IndexSizeLimitError(
                f"X = S cup F(S) has {x_pieces.size} elementary pieces, more than "
                f"max_pieces={int(max_pieces)}; the index was not attempted"
            )
        if gluing is None:
            gluing = suspension_grid_gluing(grid)
        cells = piece_rectangles(grid, x_pieces)
        nerve = AtlasQuotientNerveComplex2D(
            cells,
            gluing,
            maximum_simplex_size=maximum_simplex_size,
            interior_seam_subcomplexes=_interior_seams(gluing, cells),
        )
        counts: dict[int, int] = {}
        for cell in nerve.cells:
            counts[nerve.dimension(cell)] = counts.get(nerve.dimension(cell), 0) + 1
        result.nerve_cell_counts = tuple(counts.get(d, 0) for d in range(max(counts) + 1))
        pair = AtlasRelativeIndexPair2D(nerve, x_pieces.tolist(), a_pieces.tolist())
        result.homology_dimensions = relative_homology_dimensions(pair)
        result.homology_computed = True
        if not any(result.homology_dimensions):
            result.computed = True
            result.shift_class = ("0",) * len(result.homology_dimensions)
            result.label_source = "zero relative homology"
            result.seconds = time.perf_counter() - started
            return result
        if index_map == "excision":
            try:
                result.shift_class = _excision_index_map(
                    relation,
                    gluing,
                    s_atoms,
                    x_atoms,
                    x_pieces,
                    a_pieces,
                    pair,
                    result.homology_dimensions,
                    result.excision,
                    maximum_simplex_size=maximum_simplex_size,
                    max_pieces=max_pieces,
                )
            except Exception as error:  # the homology stays; only the label is missing
                result.index_map_blocker = f"{type(error).__name__}: {error}"
                raise
            result.computed = True
            result.label_source = "index map"
            result.seconds = time.perf_counter() - started
            return result
        x_set = set(x_pieces.tolist())
        exit_atoms = set(a_atoms.tolist())
        piece_image: dict[int, list[int]] = {}
        atom_images: dict[int, list[int]] = {}
        for atom in x_atoms.tolist():
            targets = relation.image(atom)
            pieces = [int(piece) for target in targets for piece in grid.atom(target)]
            if atom in exit_atoms:
                # Exit sources are carried by their connected A component;
                # their raw targets outside X are recorded exits.
                pieces = [piece for piece in pieces if piece in x_set]
            atom_images[atom] = pieces
        for piece in x_pieces.tolist():
            piece_image[piece] = atom_images[int(grid.atom_of_piece[piece])]
        try:
            preparation = prepare_atlas_relation_conley_2d(pair, top_relation=piece_image)
            payload = preparation.compute_finite_relation_shift_class()
        except Exception as error:  # the homology stays; only the label is missing
            result.index_map_blocker = f"{type(error).__name__}: {error}"
            raise
        dimensions = tuple(int(v) for v in payload["homology_dimensions"])
        if dimensions != result.homology_dimensions:
            raise AssertionError(
                f"the index map reports homology dimensions {dimensions!r}, but the "
                f"relative boundary matrices give {result.homology_dimensions!r}"
            )
        result.computed = True
        result.shift_class = tuple(str(entry) for entry in payload["shift_class"])
        result.label_source = "index map"
    except Exception as error:  # the blocker is reported, never replaced by a label
        result.blocker = f"{type(error).__name__}: {error}"
    result.seconds = time.perf_counter() - started
    return result


# Worker state for the process-parallel index computation.
_INDEX_WORKER: dict[str, Any] = {}


def _initialize_index_worker(
    factory: Callable[[], SuspensionGridProblem],
    level: int,
    summary: dict[str, Any],
    tau: float,
) -> None:
    problem = factory()
    grid = build_suspension_grid(problem.window, problem.guard, level)
    if grid.summary() != summary:
        raise RuntimeError("the grid rebuilt in an index worker differs from the grid of the run")
    _INDEX_WORKER["grid"] = grid
    _INDEX_WORKER["gluing"] = suspension_grid_gluing(grid)
    _INDEX_WORKER["tau"] = float(tau)


def _index_task(
    morse_node: int,
    morse_set: npt.NDArray[np.int64],
    rows: sparse.csr_matrix,
    maximum_simplex_size: int,
    max_pieces: int | None,
    index_map: str,
) -> SuspensionGridConleyResult:
    relation = SuspensionGridRelation(
        grid=_INDEX_WORKER["grid"],
        tau=_INDEX_WORKER["tau"],
        matrix=rows,
        sampled=rows,
        statistics={},
    )
    return compute_suspension_grid_conley_index(
        relation,
        morse_set,
        morse_node=morse_node,
        maximum_simplex_size=maximum_simplex_size,
        max_pieces=max_pieces,
        gluing=_INDEX_WORKER["gluing"],
        index_map=index_map,
    )


def _rows_for_index(relation: SuspensionGridRelation, morse_set: npt.ArrayLike) -> sparse.csr_matrix:
    """The relation restricted to the rows of ``S cup F(S)``, the rows the index reads.

    Both constructions of the index map read only these rows: the excision
    pair ``Xbar = X cup F(X)`` needs the images of the atoms of ``X``.
    """

    s_atoms = np.unique(np.asarray(morse_set, dtype=np.int64))
    x_atoms = np.union1d(s_atoms, relation.image_of(s_atoms))
    keep = np.zeros(relation.n_atoms, dtype=np.float64)
    keep[x_atoms] = 1.0
    return (sparse.diags(keep, format="csr") @ relation.matrix).tocsr()


def compute_suspension_grid_conley_indices(
    relation: SuspensionGridRelation,
    morse_sets: Sequence[npt.ArrayLike],
    *,
    workers: int = 1,
    problem_factory: Callable[[], SuspensionGridProblem] | None = None,
    maximum_simplex_size: int = 16,
    max_pieces: int | None = None,
    index_map: str = INDEX_MAPS[0],
) -> list[SuspensionGridConleyResult]:
    """:func:`compute_suspension_grid_conley_index` of every Morse set, in node order.

    With ``workers > 1`` the Morse sets are handled in worker processes,
    largest first.  Each worker rebuilds the grid from the picklable
    ``problem_factory`` (and checks it against the grid of ``relation``) and
    receives, per Morse set ``S``, only the rows of ``S cup F(S)``, which are
    the rows the index reads; the results are those of the serial loop.
    Each worker holds its own copy of the grid (about 1 GB at ``2**10`` and
    4 GB at ``2**11`` base cells per axis) besides the quotient nerve of the
    pair it works on.
    """

    grid = relation.grid
    sets = [np.asarray(morse_set, dtype=np.int64) for morse_set in morse_sets]
    workers = min(int(workers), len(sets))
    if workers <= 1:
        gluing = suspension_grid_gluing(grid)
        return [
            compute_suspension_grid_conley_index(
                relation,
                morse_set,
                morse_node=node,
                maximum_simplex_size=maximum_simplex_size,
                max_pieces=max_pieces,
                gluing=gluing,
                index_map=index_map,
            )
            for node, morse_set in enumerate(sets)
        ]
    if problem_factory is None:
        raise ValueError("a parallel index computation requires a picklable problem_factory")
    order = sorted(range(len(sets)), key=lambda node: -sets[node].size)
    results: list[SuspensionGridConleyResult | None] = [None] * len(sets)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_index_worker,
        initargs=(problem_factory, grid.level, grid.summary(), relation.tau),
    ) as executor:
        futures = {
            node: executor.submit(
                _index_task,
                node,
                sets[node],
                _rows_for_index(relation, sets[node]),
                maximum_simplex_size,
                max_pieces,
                index_map,
            )
            for node in order
        }
        for node, future in futures.items():
            results[node] = future.result()
    return [result for result in results if result is not None]


__all__ = [
    "ExcisionError",
    "INDEX_MAPS",
    "IndexSizeLimitError",
    "SuspensionGridConleyResult",
    "compute_suspension_grid_conley_index",
    "compute_suspension_grid_conley_indices",
    "piece_rectangles",
    "relative_homology_dimensions",
    "suspension_grid_gluing",
]
