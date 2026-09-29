"""Finite-relation Conley labels for Morse sets on the paper suspension grid.

For a Morse set ``S`` of the sampled relation ``F_n`` on ``Xi_n`` this module
forms, by default, the standard combinatorial pair

``X = S cup F(S)``,   ``A = F(S) minus S``

of atoms (``index_pair="image"``), covers ``|X|`` by the elementary pieces of
its atoms, and hands the pieces to the verified reset-quotient nerve of
:mod:`atlas_conley` (:class:`AtlasQuotientNerveComplex2D`).  Every piece is
an actual closed rectangle in the base chart ``(x, y)`` or in the handle
chart ``(u, s)``, and the identifications ``(u, 0) ~ gamma(u)`` and
``(u, 1) ~ r(gamma(u))`` are supplied as affine seam embeddings.  The
relation on pieces sends a piece of the atom ``xi`` to every piece of
``F(xi)``; the carrier, the acyclicity, pair, and chain-map checks, and the
CMGDB shift-class computation are those of
:func:`atlas_conley.prepare_atlas_relation_conley_2d`.  The atoms of ``A``
map partly outside ``X``, so a piece of ``A`` is carried by its connected
component of ``A`` instead of its image.

For the image pair the index map is built in one of two ways
(``index_map``), or by the first with the second as fallback (``"auto"``).

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

``"auto"`` forms the exit-components map and, only when that map fails
(a carrier that is not acyclic, for example) and the relative homology is
nonzero, the excision map on the same nerve of ``X``.  Both are index maps
of the same pair, so either gives the label.  ``label_source`` names the
construction that gave it, ``"index map (exit components)"`` or ``"index
map (excision pair)"``, and ``exit_components_blocker`` keeps the reason the
first one failed.

With ``index_pair="forward-closure"`` the pair is that of
Proposition ``prop:grid-conley-index`` of the manuscript,

``U = S cup F(S) cup F^2(S) cup ...``,   ``V = U minus S``,

the forward closure of ``S`` and its complement of ``S``.  Both are forward
invariant: ``F(U)`` lies in ``U`` by construction, and ``F(V)`` lies in
``V`` because an atom of ``V`` that reaches ``S`` would belong to the
strongly connected component ``S``.  Every piece is then carried by the
pieces of its true image, with no exit to handle, so ``index_map`` does not
apply to it.  ``U`` contains everything downstream of ``S`` and can be much
larger than ``X``.

With ``excise=True`` the forward-closure pair is computed on a neighborhood
of ``S`` in ``U`` instead of on all of ``U``.  The chain complex
``C(U) / C(V)`` of the nerve has one generator for each simplex with a
vertex in ``S``.  The vertices of such a simplex meet a piece of ``S``, so
they lie in ``W = S cup (atoms of U whose closures meet |S|)``, and
``C(U) / C(V) = C(W) / C(W minus S)`` with the same basis and boundary.
The index map on it needs the chain map only on the simplices of the nerve
of ``W``, and their carriers lie in the nerve of ``W cup F(W)``.  So the
nerve is built on ``W cup F(W)``, the chain map is formed by the
acyclic-carrier induction on the simplices of the nerve of ``W``, and the
carriers of the other simplices of the nerve of ``U``, which all lie in
``V``, are neither formed nor checked.  ``W`` is formed from the atom
adjacency of the grid.  Before the nerve is built, the rectangles and seams
of the pieces of ``U`` outside ``W`` are checked not to meet ``S``, the
test the nerve itself would make, and the nerve is checked to join each
piece of ``S`` only to pieces of ``W``.

The relative homology ``H_*(X, A; GF(5))`` of the pair is always computed
once the nerve is built; it needs no carrier.  If it is zero, the shift
class is zero in every degree whatever the index map, so the label is
trivial and the index map is not formed.  Otherwise the index map is formed
and its shift class is the label; if a carrier, pair, or chain-map check
fails, the homology is still reported and only the label is missing.

The carrier and chain map of the exit-components construction (also the
first attempt of ``"auto"``) and of the forward-closure pair without
``excise`` are formed either by the Python construction of
:func:`atlas_conley.prepare_atlas_relation_conley_2d` or by the native
kernel ``CMGDB.ComputeCarrierChainMap`` (:mod:`carrier_kernel`), which forms
the same chain map and hands the same payload to CMGDB.  ``backend``
chooses: ``"python"``, ``"native"``, or ``"auto"`` (the default), the native
kernel when the installed CMGDB provides it.  The excision construction and
the excised forward-closure pair use the Python construction with every
backend.

The result is a finite-relation shift class over ``GF(5)``.  It is not a
certified Conley index of the continuous fixed-time map, because the
relation is sampled.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Collection, Mapping, Sequence
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
from .carrier_kernel import (
    CONLEY_BACKENDS,
    native_relation_shift_class,
    resolve_conley_backend,
)
from .suspension_complex import (
    CMGDBRelativeHomologyPayload,
    CrossComplexAcyclicCarrier,
    RelativeHomologyBasis,
    _check_right_hand_side,
    _eliminate_columns_mod_prime,
    _rank_mod_prime,
    _solve_with_pivots,
)
from .suspension_grid import SuspensionGrid, UnsupportedSuspensionGridError, build_suspension_grid
from .suspension_grid_relation import SuspensionGridProblem, SuspensionGridRelation


BASE_CHART = 0
HANDLE_CHART = 1

#: The pairs of a Morse set ``S``: ``"image"`` is ``(S cup F(S), F(S) minus S)``
#: and ``"forward-closure"`` is ``(U, U minus S)`` with ``U`` the forward
#: closure of ``S`` (Proposition ``prop:grid-conley-index``).
INDEX_PAIRS = ("image", "forward-closure")

#: Names of the two sets ``(P1, P0)`` of each pair in the records.
_PAIR_NAMES = {"image": ("X", "A"), "forward-closure": ("U", "V")}

#: Constructions of the index map of the image pair; the first is the default.
#: ``"auto"`` uses the first and falls back to ``"excision"`` when it fails.
INDEX_MAPS = ("exit-components", "excision", "auto")


class IndexSizeLimitError(RuntimeError):
    """The pair of a Morse set has more elementary pieces than the caller allows."""


class ExcisionError(RuntimeError):
    """The inclusion ``(X, A) -> (Xbar, Abar)`` is not an isomorphism on homology."""


def _pieces_of_atoms(grid: SuspensionGrid, atoms: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    if not atoms.size:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([grid.atom(atom) for atom in atoms])


def index_pair_atoms(
    relation: SuspensionGridRelation,
    morse_set: npt.ArrayLike,
    index_pair: str = "image",
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Atoms ``(S, P1, P0)`` of the pair of the Morse set ``S``.

    ``index_pair="image"`` gives ``P1 = S cup F(S)`` and ``P0 = F(S) minus
    S``.  ``index_pair="forward-closure"`` gives the forward closure ``P1 =
    U`` of ``S`` and ``P0 = V = U minus S``; it checks that ``F(U)`` lies in
    ``U`` and ``F(V)`` in ``V`` and raises ``ValueError`` otherwise (``F(V)``
    meets ``S`` only if ``S`` is not a strongly connected component of ``F``).
    """

    atoms = _pair_atoms(relation, morse_set, index_pair)
    if index_pair == "forward-closure":
        _check_forward_invariance(relation, *atoms)
    return atoms


def _pair_atoms(
    relation: SuspensionGridRelation,
    morse_set: npt.ArrayLike,
    index_pair: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The atoms of :func:`index_pair_atoms`, without the checks of the forward-closure pair."""

    if index_pair not in INDEX_PAIRS:
        raise ValueError(f"index_pair must be one of {INDEX_PAIRS!r}; got {index_pair!r}")
    s_atoms = np.unique(np.asarray(morse_set, dtype=np.int64))
    if index_pair == "image":
        image = relation.image_of(s_atoms)
        return s_atoms, np.union1d(s_atoms, image), np.setdiff1d(image, s_atoms)
    u_atoms = relation.forward_closure(s_atoms)
    return s_atoms, u_atoms, np.setdiff1d(u_atoms, s_atoms)


def _check_forward_invariance(
    relation: SuspensionGridRelation,
    s_atoms: npt.NDArray[np.int64],
    u_atoms: npt.NDArray[np.int64],
    v_atoms: npt.NDArray[np.int64],
) -> None:
    """Raise ``ValueError`` unless ``F(U)`` lies in ``U`` and ``F(V)`` in ``V``."""

    outside = np.setdiff1d(relation.image_of(u_atoms), u_atoms)
    if outside.size:
        raise ValueError(f"F(U) is not contained in U: {outside.size} atoms of F(U) lie outside U")
    into_s = np.intersect1d(relation.image_of(v_atoms), s_atoms)
    if into_s.size:
        raise ValueError(
            f"F(V) meets S in {into_s.size} atoms, so S is not a strongly connected "
            "component of F"
        )


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
    ``index_pair`` is the pair (see :data:`INDEX_PAIRS`).  ``pair_atoms``
    and ``pair_pieces`` count the sets ``X`` and ``A`` of the image pair, or
    ``U`` and ``V`` of the forward-closure pair and, when it is excised,
    ``W`` and ``Y = W cup F(W)``.  ``index_map`` names the construction of
    the index map of the image pair (see ``INDEX_MAPS``).  For
    ``"excision"``, ``excision`` records the pair ``(Xbar, Abar)`` (atoms,
    pieces, nerve cell counts, and homology dimensions, as far as they were
    computed) and the matrices of the index map on ``H_k(X, A; GF(5))``, one
    list of rows per degree.  For ``"auto"``, ``label_source`` names the
    construction that gave the label and ``exit_components_blocker`` is the
    reason the exit-components construction failed (empty if it did not).
    ``to_dict`` writes ``index_pair`` only for the forward-closure pair,
    ``index_map`` and ``excision`` only for a construction other than the
    default, and ``exit_components_blocker`` only for ``"auto"``, so the
    record of the default pair and construction keeps the keys it had
    before.
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
    index_pair: str = "image"
    index_map: str = INDEX_MAPS[0]
    excision: dict[str, Any] = field(default_factory=dict)
    exit_components_blocker: str = ""

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
        # The record of the default pair and construction keeps its earlier keys.
        if self.index_pair != "image":
            record["index_pair"] = self.index_pair
        if self.index_map != INDEX_MAPS[0]:
            record["index_map"] = self.index_map
            record["excision"] = self.excision
        if self.index_map == "auto":
            record["exit_components_blocker"] = self.exit_components_blocker
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


def _footprints(
    grid: SuspensionGrid,
    gluing: AtlasResetGluing2D,
    pieces: npt.NDArray[np.int64],
    atol: float,
) -> dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
    """Closed footprints of pieces in each chart: the pieces and the bounds ``(x0, y0, x1, y1)``.

    A base piece has its rectangle in the base chart.  A handle piece has its
    rectangle ``(u0, s0, u1, s1)`` in the handle chart and, on a seam
    (``s0 = 0`` or ``s1 = 1``), the segment of the base chart its face is
    glued to.  Two pieces meet in the quotient exactly when footprints of
    both in one chart meet; the quotient nerve finds the candidate
    neighbors of a cell in the same way.
    """

    base = pieces[pieces < grid.n_base]
    handle = pieces[pieces >= grid.n_base]
    handle_bounds = grid.handle_bounds(handle).reshape(-1, 4)
    owners = {BASE_CHART: [base], HANDLE_CHART: [handle]}
    bounds = {
        BASE_CHART: [grid.base_bounds(base).reshape(-1, 4)],
        HANDLE_CHART: [handle_bounds],
    }
    lower, upper = gluing.phase_bounds
    for embedding, on_seam in (
        (gluing.guard, np.abs(handle_bounds[:, 1] - lower) <= atol),
        (gluing.reset, np.abs(handle_bounds[:, 3] - upper) <= atol),
    ):
        ends = embedding.scale * handle_bounds[on_seam][:, [0, 2]] + embedding.offset
        segments = np.empty((ends.shape[0], 4))
        fixed, varying = embedding.fixed_axis, embedding.varying_axis
        segments[:, fixed] = embedding.fixed_value
        segments[:, 2 + fixed] = embedding.fixed_value
        segments[:, varying] = ends.min(axis=1)
        segments[:, 2 + varying] = ends.max(axis=1)
        owners[BASE_CHART].append(handle[on_seam])
        bounds[BASE_CHART].append(segments)
    return {
        chart: (np.concatenate(owners[chart]), np.concatenate(bounds[chart]))
        for chart in owners
    }


def _bucket_entries(
    bounds: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    size: npt.NDArray[np.float64],
    shape: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pairs ``(row of bounds, bucket)`` for the buckets each rectangle meets.

    The buckets are the cells of the uniform grid of cell size ``size`` with
    ``shape`` cells from ``origin``; ranges outside it are clipped.
    """

    first = np.clip(np.floor((bounds[:, :2] - origin) / size), 0, shape - 1).astype(np.int64)
    last = np.clip(np.floor((bounds[:, 2:] - origin) / size), 0, shape - 1).astype(np.int64)
    counts = last - first + 1
    total = counts[:, 0] * counts[:, 1]
    rows = np.repeat(np.arange(bounds.shape[0]), total)
    offset = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
    column = first[rows, 0] + offset // counts[rows, 1]
    row = first[rows, 1] + offset % counts[rows, 1]
    return rows, column * shape[1] + row


def _pieces_meeting(
    grid: SuspensionGrid,
    gluing: AtlasResetGluing2D,
    pieces: npt.NDArray[np.int64],
    others: npt.NDArray[np.int64],
    *,
    atol: float = 1.0e-9,
) -> npt.NDArray[np.int64]:
    """The pieces of ``pieces`` whose closures meet the closure of a piece of ``others``.

    This is decided from the rectangles and the seams of the gluing (see
    :func:`_footprints`), with the tolerance ``atol``, and not from
    ``grid.piece_adjacency``.  In each chart the footprints of ``others``
    are put in the buckets of a uniform grid; a footprint of ``pieces``,
    widened by ``atol``, is compared with the footprints of ``others`` only
    when it meets one of their buckets.
    """

    first = _footprints(grid, gluing, np.asarray(pieces, dtype=np.int64), atol)
    second = _footprints(grid, gluing, np.asarray(others, dtype=np.int64), atol)
    found: list[npt.NDArray[np.int64]] = []
    for chart, (owner, bounds) in first.items():
        other_bounds = second[chart][1]
        if not owner.size or not other_bounds.shape[0]:
            continue
        widened = bounds + np.array([-atol, -atol, atol, atol])
        origin = other_bounds[:, :2].min(axis=0)
        extent = other_bounds[:, 2:].max(axis=0) - origin
        near = np.all((widened[:, :2] <= origin + extent) & (widened[:, 2:] >= origin), axis=1)
        owner, widened = owner[near], widened[near]
        if not owner.size:
            continue
        sides = other_bounds[:, 2:] - other_bounds[:, :2]
        size = np.array(
            [
                float(np.median(side[side > atol])) if np.any(side > atol) else max(span, atol)
                for side, span in zip(sides.T, extent)
            ]
        )
        shape = (np.floor(extent / size) + 1).astype(np.int64)
        other_rows, other_keys = _bucket_entries(other_bounds, origin, size, shape)
        rows, keys = _bucket_entries(widened, origin, size, shape)
        # Only the footprints that share a bucket with ``others`` can meet them.
        rows = np.unique(rows[np.isin(keys, other_keys)])
        if not rows.size:
            continue
        owner, widened = owner[rows], widened[rows]
        rows, keys = _bucket_entries(widened, origin, size, shape)
        order = np.argsort(other_keys, kind="stable")
        sorted_keys = other_keys[order]
        start = np.searchsorted(sorted_keys, keys, "left")
        hits = np.searchsorted(sorted_keys, keys, "right") - start
        offset = np.arange(int(hits.sum())) - np.repeat(np.cumsum(hits) - hits, hits)
        left = np.repeat(rows, hits)
        right = other_rows[order[np.repeat(start, hits) + offset]]
        a, b = widened[left], other_bounds[right]
        meet = (a[:, 0] <= b[:, 2]) & (b[:, 0] <= a[:, 2]) & (a[:, 1] <= b[:, 3]) & (b[:, 1] <= a[:, 3])
        found.append(owner[left[meet]])
    if not found:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(found))


def _check_neighborhood(
    grid: SuspensionGrid,
    nerve: AtlasQuotientNerveComplex2D,
    s_pieces: Collection[int],
    w_pieces: Collection[int],
) -> None:
    """Check that the nerve joins each piece of ``S`` only to pieces of ``W``.

    ``W`` is formed from ``grid.atom_adjacency``, that is, from the pieces
    whose closures meet in ``Sigma X`` (``grid.piece_adjacency``).  The
    nerve decides the same question from the rectangles and the seams, so
    every edge of the nerve is checked to be a pair of meeting pieces, and
    every edge at a piece of ``S`` to end in ``W``.
    """

    edges = np.array([cell.vertices for cell in nerve.cells_of_dimension(1)], dtype=np.int64)
    if not edges.size:
        return
    meeting = np.asarray(grid.piece_adjacency[edges[:, 0], edges[:, 1]]).ravel() != 0
    if not np.all(meeting):
        first, second = edges[np.flatnonzero(~meeting)[0]]
        raise AssertionError(
            f"the nerve joins pieces {int(first)} and {int(second)}, whose closures "
            "do not meet in the grid adjacency"
        )
    s_set = np.isin(edges, np.fromiter(s_pieces, dtype=np.int64))
    w_set = np.isin(edges, np.fromiter(w_pieces, dtype=np.int64))
    leaves = (s_set[:, 0] & ~w_set[:, 1]) | (s_set[:, 1] & ~w_set[:, 0])
    if np.any(leaves):
        first, second = edges[np.flatnonzero(leaves)[0]]
        raise AssertionError(
            f"the nerve joins pieces {int(first)} and {int(second)}, one in S and "
            "the other outside W"
        )


def _add_chain(
    accumulator: dict[Any, int], chain: Mapping[Any, int], scale: int, modulus: int
) -> None:
    for cell, coefficient in chain.items():
        value = (accumulator.get(cell, 0) + scale * coefficient) % modulus
        if value:
            accumulator[cell] = value
        else:
            accumulator.pop(cell, None)


def _excised_shift_class(
    pair: AtlasRelativeIndexPair2D,
    source_pieces: Collection[int],
    piece_image: Mapping[int, Collection[int]],
    modulus: int = 5,
) -> dict[str, Any]:
    """Shift class of the index map of ``pair`` from a chain map on a subcomplex.

    ``source_pieces`` spans the subcomplex ``K_W`` (the simplices of the
    nerve with every vertex in ``source_pieces``), which must contain every
    generator of ``C(P1) / C(P0)`` (every simplex with a vertex outside
    ``P0``).  The carrier of a simplex of ``K_W`` is the subcomplex induced
    on the union of the images of its vertices, as in
    :func:`atlas_conley.prepare_atlas_relation_conley_2d`.  Each carrier is
    checked to be acyclic, to contain the carriers of the faces, and to lie
    in ``P0`` when the simplex does.  The chain map is formed on ``K_W`` by
    the acyclic-carrier induction of
    :meth:`suspension_complex.FixedTimeCarrier.construct_chain_map`, checked
    to commute with the boundary, and read on the basis of
    ``C(P1) / C(P0)``; its CMGDB shift class is returned.
    """

    complex_ = pair.complex
    relative = pair.relative_pair
    sources = pair.nerve.induced_cells(source_pieces)
    basis = relative.basis_by_dimension
    if any(cell not in sources for group in basis for cell in group):
        raise ValueError("the neighborhood misses a generator of C(P1) / C(P0)")
    vertex_images: dict[int, frozenset[int]] = {}
    for piece in source_pieces:
        targets = frozenset(int(value) for value in piece_image[piece])
        if not targets:
            raise ValueError(f"the image of piece {piece} is empty")
        outside = targets.difference(pair.p1_atlas_cells)
        if outside:
            raise ValueError(f"the image of piece {piece} leaves P1: {sorted(outside)!r}")
        vertex_images[int(piece)] = targets
    generators = _InducedCarrierGenerators(complex_, vertex_images)
    p0_cells = relative.p0_cells
    p0_vertices = pair.p0_atlas_cells
    position = complex_._position
    distinct: dict[frozenset, frozenset] = {}
    acyclic: dict[int, bool] = {}
    carriers: dict[Any, frozenset] = {}
    images: dict[Any, dict[Any, int]] = {}
    eliminated: dict[tuple[int, int], tuple[Any, Any, Any]] = {}

    def cells_in_order(value: frozenset, dimension: int) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (cell for cell in value if complex_.dimension(cell) == dimension),
                key=position.__getitem__,
            )
        )

    for dimension in range(complex_.max_dimension + 1):
        for source in complex_.cells_of_dimension(dimension):
            if source not in sources:
                continue
            value = generators[source]
            value = distinct.setdefault(value, value)
            if id(value) not in acyclic:
                acyclic[id(value)] = complex_.is_acyclic(value, modulus=modulus)
            if not acyclic[id(value)]:
                raise ValueError(
                    f"carrier image of {source!r} is not acyclic over GF({modulus})"
                )
            if p0_vertices.issuperset(source.vertices) and not value <= p0_cells:
                raise ValueError(f"the carrier of {source!r} does not preserve P0")
            for face in complex_.boundary(source):
                if not carriers[face] <= value:
                    raise ValueError(f"carrier nesting fails for face {face!r} of {source!r}")
            carriers[source] = value
            if dimension == 0:
                images[source] = {cells_in_order(value, 0)[0]: 1}
                continue
            right_hand_side: dict[Any, int] = {}
            for face, incidence in complex_.boundary(source).items():
                _add_chain(right_hand_side, images[face], incidence, modulus)
            key = (id(value), dimension)
            system = eliminated.get(key)
            if system is None:
                rows = cells_in_order(value, dimension - 1)
                columns = cells_in_order(value, dimension)
                row_index = {cell: index for index, cell in enumerate(rows)}
                pivots = _eliminate_columns_mod_prime(
                    row_index,
                    columns,
                    {cell: complex_.boundary(cell) for cell in columns},
                    modulus,
                )
                system = (row_index, pivots, columns)
                if len(eliminated) >= 256:
                    eliminated.pop(next(iter(eliminated)))
                eliminated[key] = system
            row_index, pivots, columns = system
            try:
                _check_right_hand_side(row_index, right_hand_side)
                images[source] = _solve_with_pivots(
                    row_index, pivots, columns, right_hand_side, modulus
                )
            except ValueError as error:
                raise ValueError(
                    f"acyclic-carrier chain selection failed on {source!r}: {error}"
                ) from error

    for source, image in images.items():
        boundary_after_map: dict[Any, int] = {}
        for target, coefficient in image.items():
            _add_chain(boundary_after_map, complex_.boundary(target), coefficient, modulus)
        map_after_boundary: dict[Any, int] = {}
        for face, incidence in complex_.boundary(source).items():
            _add_chain(map_after_boundary, images[face], incidence, modulus)
        if boundary_after_map != map_after_boundary:
            raise ValueError(f"dF != Fd on {source!r} over GF({modulus})")

    chain_entries = []
    for group in basis:
        row_of = {cell: row for row, cell in enumerate(group)}
        entries = []
        for column, source in enumerate(group):
            for target, coefficient in images[source].items():
                if target in row_of:
                    entries.append((row_of[target], column, coefficient % modulus))
                elif target not in p0_cells:
                    raise AssertionError(f"the chain map sends {source!r} outside P1")
        chain_entries.append(tuple(entries))
    payload = CMGDBRelativeHomologyPayload(
        cell_counts=relative.cell_counts,
        boundary_entries=relative.boundary_entries(modulus=modulus),
        chain_map_entries=tuple(chain_entries),
        basis_by_dimension=basis,
    )
    import CMGDB

    return dict(CMGDB.ComputeRelativeHomologyShiftClass(*payload.as_compute_args()))


def compute_suspension_grid_conley_index(
    relation: SuspensionGridRelation,
    morse_set: npt.ArrayLike,
    *,
    morse_node: int = 0,
    maximum_simplex_size: int = 16,
    max_pieces: int | None = None,
    gluing: AtlasResetGluing2D | None = None,
    index_pair: str = "image",
    excise: bool = False,
    index_map: str = INDEX_MAPS[0],
    backend: str = CONLEY_BACKENDS[0],
) -> SuspensionGridConleyResult:
    """Shift class of the index map on the pair ``index_pair`` of ``S``.

    The default pair is ``(S cup F(S), F(S) minus S)``;
    ``index_pair="forward-closure"`` gives the pair ``(U, U minus S)`` of
    the forward closure ``U`` of ``S`` (see :func:`index_pair_atoms`); when
    ``F(V)`` meets ``S`` (``S`` is not a strongly connected component), the
    index is not attempted and the reason is the blocker.
    The relative homology of the pair is computed first.  If it is zero, the
    label is zero in every degree and the index map is not formed.  A failed
    gate of the index map (carrier acyclicity, pair preservation, chain map)
    is recorded as ``index_map_blocker``; the label is then missing, and the
    homology is still reported.  If the quotient nerve is not a good cover,
    neither is computed.  With ``max_pieces``, a pair whose ``X`` (or ``U``)
    has more elementary pieces is not attempted: the blocker is
    ``IndexSizeLimitError``.  The quotient nerve has about ten simplices per
    piece and is held in memory, so this bounds the time and memory of the
    computation on fine grids.  ``gluing`` is ``suspension_grid_gluing`` of
    the grid, built here when it is not given.

    ``excise=True`` (forward-closure pair only) computes the same chain
    complex ``C(U) / C(V)`` and index map on the nerve of ``Y = W cup
    F(W)``, where ``W`` is ``S`` with the atoms of ``U`` whose closures meet
    ``|S|`` (see the module notes).  A piece of ``U`` outside ``W`` whose
    rectangle or seam meets ``S`` is reported as the blocker.
    ``pair_pieces`` then also records the pieces of ``W`` and of ``Y``,
    ``nerve_cell_counts`` is that of the nerve of ``Y``, and ``max_pieces``
    bounds the pieces of ``Y``.

    ``index_map`` (image pair only) chooses the construction of the index
    map (see the module notes): ``"exit-components"`` (the default),
    ``"excision"``, or ``"auto"``, the first with the second as fallback when
    the first fails and the relative homology is nonzero.  With the
    excision construction and ``max_pieces``, the index map is also not
    attempted when ``Xbar = X cup F(X)`` has more pieces; the homology of
    ``(X, A)`` is still reported.

    ``backend`` (see :data:`carrier_kernel.CONLEY_BACKENDS`) chooses how the
    carrier and chain map of the exit-components construction and of the
    forward-closure pair without ``excise`` are formed: ``"python"``,
    ``"native"`` (``CMGDB.ComputeCarrierChainMap``; ``RuntimeError`` if the
    installed CMGDB lacks it), or ``"auto"``, the native kernel when it is
    available.  The result does not depend on it.
    """

    started = time.perf_counter()
    grid = relation.grid
    if excise and index_pair != "forward-closure":
        raise ValueError("excise applies to the forward-closure pair")
    if index_map not in INDEX_MAPS:
        raise ValueError(f"index_map must be one of {INDEX_MAPS!r}; got {index_map!r}")
    if index_map != INDEX_MAPS[0] and index_pair == "forward-closure":
        raise ValueError("index_map applies to the image pair")
    backend = resolve_conley_backend(backend)
    s_atoms, x_atoms, a_atoms = _pair_atoms(relation, morse_set, index_pair)
    x_pieces = _pieces_of_atoms(grid, x_atoms)
    a_pieces = _pieces_of_atoms(grid, a_atoms)
    x_name, a_name = _PAIR_NAMES[index_pair]
    result = SuspensionGridConleyResult(
        morse_node=morse_node,
        computed=False,
        pair_atoms={"S": int(s_atoms.size), x_name: int(x_atoms.size), a_name: int(a_atoms.size)},
        pair_pieces={x_name: int(x_pieces.size), a_name: int(a_pieces.size)},
        index_pair=index_pair,
        index_map=index_map,
    )
    # The atoms whose pieces carry the nerve, and the atoms whose pieces are
    # the vertices of the simplices on which the chain map is formed.
    nerve_atoms = source_atoms = x_atoms
    description = "X = S cup F(S)" if index_pair == "image" else "U = the forward closure of S"
    if excise:
        near = np.unique(grid.atom_adjacency[s_atoms].indices).astype(np.int64)
        source_atoms = np.union1d(s_atoms, np.intersect1d(near, x_atoms))
        nerve_atoms = np.union1d(source_atoms, relation.image_of(source_atoms))
        result.pair_atoms.update(W=int(source_atoms.size), Y=int(nerve_atoms.size))
        description = "Y = W cup F(W)"
    nerve_pieces = _pieces_of_atoms(grid, nerve_atoms)
    exit_pieces = nerve_pieces[np.isin(grid.atom_of_piece[nerve_pieces], a_atoms)]
    source_pieces = _pieces_of_atoms(grid, source_atoms)
    if excise:
        result.pair_pieces.update(W=int(source_pieces.size), Y=int(nerve_pieces.size))
    try:
        if index_pair == "forward-closure":
            # A set that is not a strongly connected component is reported
            # here, as the blocker, before any homology is computed.
            _check_forward_invariance(relation, s_atoms, x_atoms, a_atoms)
        if max_pieces is not None and nerve_pieces.size > int(max_pieces):
            raise IndexSizeLimitError(
                f"{description} has {nerve_pieces.size} elementary pieces, more than "
                f"max_pieces={int(max_pieces)}; the index was not attempted"
            )
        if gluing is None:
            gluing = suspension_grid_gluing(grid)
        if excise:
            # W is formed from grid.atom_adjacency; the rectangles and seams
            # of the pieces decide independently that no piece of U outside
            # W meets S.
            outside = x_pieces[~np.isin(grid.atom_of_piece[x_pieces], source_atoms)]
            meeting = _pieces_meeting(grid, gluing, outside, _pieces_of_atoms(grid, s_atoms))
            if meeting.size:
                raise AssertionError(
                    f"{meeting.size} pieces of U outside W meet S (the first is piece "
                    f"{int(meeting[0])} of atom {int(grid.atom_of_piece[meeting[0]])}); "
                    "W does not contain the neighborhood of S"
                )
        cells = piece_rectangles(grid, nerve_pieces)
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
        if excise:
            _check_neighborhood(
                grid,
                nerve,
                _pieces_of_atoms(grid, s_atoms).tolist(),
                source_pieces.tolist(),
            )
        pair = AtlasRelativeIndexPair2D(nerve, nerve_pieces.tolist(), exit_pieces.tolist())
        result.homology_dimensions = relative_homology_dimensions(pair)
        result.homology_computed = True
        if not any(result.homology_dimensions):
            result.computed = True
            result.shift_class = ("0",) * len(result.homology_dimensions)
            result.label_source = "zero relative homology"
            result.seconds = time.perf_counter() - started
            return result
        if index_map != "excision":
            x_set = set(x_pieces.tolist())
            # In the forward-closure pair every atom maps into U and every atom
            # of V into V, so each piece is carried by its true image.
            exit_atoms = set(a_atoms.tolist()) if index_pair == "image" else set()
            piece_image: dict[int, list[int]] = {}
            atom_images: dict[int, list[int]] = {}
            for atom in source_atoms.tolist():
                targets = relation.image(atom)
                pieces = [int(piece) for target in targets for piece in grid.atom(target)]
                if atom in exit_atoms:
                    # Exit sources are carried by their connected A component;
                    # their raw targets outside X are recorded exits.
                    pieces = [piece for piece in pieces if piece in x_set]
                atom_images[atom] = pieces
            for piece in source_pieces.tolist():
                piece_image[piece] = atom_images[int(grid.atom_of_piece[piece])]
            payload = None
            try:
                if index_pair == "forward-closure":
                    empty = sum(1 for pieces in atom_images.values() if not pieces)
                    if empty:
                        # Samples whose image points lie outside the window are
                        # discarded, so an atom of U can have an empty image.
                        raise ValueError(
                            f"{empty} of the {len(atom_images)} atoms of "
                            f"{'W' if excise else 'U'} have an empty image (every image "
                            "point left the window), so their pieces have no carrier"
                        )
                if excise:
                    payload = _excised_shift_class(pair, source_pieces.tolist(), piece_image)
                elif backend == "native":
                    payload = native_relation_shift_class(
                        pair,
                        top_relation=piece_image,
                        use_exit_component_carrier=index_pair == "image",
                    )
                else:
                    preparation = prepare_atlas_relation_conley_2d(
                        pair,
                        top_relation=piece_image,
                        use_exit_component_carrier=index_pair == "image",
                    )
                    payload = preparation.compute_finite_relation_shift_class()
            except Exception as error:  # the homology stays; only the label is missing
                blocker = f"{type(error).__name__}: {error}"
                if index_map != "auto":
                    result.index_map_blocker = blocker
                    raise
                # The excision construction is tried below.
                result.exit_components_blocker = blocker
            if payload is not None:
                dimensions = tuple(int(v) for v in payload["homology_dimensions"])
                if dimensions != result.homology_dimensions:
                    raise AssertionError(
                        f"the index map reports homology dimensions {dimensions!r}, but the "
                        f"relative boundary matrices give {result.homology_dimensions!r}"
                    )
                result.shift_class = tuple(str(entry) for entry in payload["shift_class"])
                result.label_source = (
                    "index map (exit components)" if index_map == "auto" else "index map"
                )
        if not result.label_source:
            # "excision", or "auto" after the exit-components construction failed.
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
            result.label_source = (
                "index map (excision pair)" if index_map == "auto" else "index map"
            )
        result.computed = True
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
    index_pair: str,
    excise: bool,
    index_map: str,
    backend: str,
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
        index_pair=index_pair,
        excise=excise,
        index_map=index_map,
        backend=backend,
    )


def _rows_for_index(
    relation: SuspensionGridRelation, pair_atoms: npt.NDArray[np.int64]
) -> sparse.csr_matrix:
    """The relation restricted to the rows of the pair (``S cup F(S)`` or ``U``).

    These are the rows the index reads: ``S cup F(S)`` and the forward
    closure ``U`` are formed from the rows of ``S`` and of ``U``, and the
    excision pair ``Xbar = X cup F(X)`` needs the images of the atoms of
    ``X``.
    """

    keep = np.zeros(relation.n_atoms, dtype=np.float64)
    keep[pair_atoms] = 1.0
    return (sparse.diags(keep, format="csr") @ relation.matrix).tocsr()


def compute_suspension_grid_conley_indices(
    relation: SuspensionGridRelation,
    morse_sets: Sequence[npt.ArrayLike],
    *,
    workers: int = 1,
    problem_factory: Callable[[], SuspensionGridProblem] | None = None,
    maximum_simplex_size: int = 16,
    max_pieces: int | None = None,
    index_pair: str = "image",
    excise: bool = False,
    index_map: str = INDEX_MAPS[0],
    backend: str = CONLEY_BACKENDS[0],
) -> list[SuspensionGridConleyResult]:
    """:func:`compute_suspension_grid_conley_index` of every Morse set, in node order.

    With ``workers > 1`` the Morse sets are handled in worker processes,
    largest pair first.  Each worker rebuilds the grid from the picklable
    ``problem_factory`` (and checks it against the grid of ``relation``) and
    receives, per Morse set ``S``, only the rows of ``S cup F(S)`` (of ``U``
    for the forward-closure pair), which are the rows the index reads; the
    results are those of the serial loop.
    Each worker holds its own copy of the grid (about 1 GB at ``2**10`` and
    4 GB at ``2**11`` base cells per axis) besides the quotient nerve of the
    pair it works on.  ``backend`` is checked here (``"native"`` raises
    ``RuntimeError`` when the installed CMGDB lacks the native kernel) and
    passed to every index.
    """

    resolve_conley_backend(backend)
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
                index_pair=index_pair,
                excise=excise,
                index_map=index_map,
                backend=backend,
            )
            for node, morse_set in enumerate(sets)
        ]
    if problem_factory is None:
        raise ValueError("a parallel index computation requires a picklable problem_factory")
    pair_atoms = [_pair_atoms(relation, morse_set, index_pair)[1] for morse_set in sets]
    order = sorted(range(len(sets)), key=lambda node: -pair_atoms[node].size)
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
                _rows_for_index(relation, pair_atoms[node]),
                maximum_simplex_size,
                max_pieces,
                index_pair,
                excise,
                index_map,
                backend,
            )
            for node in order
        }
        for node, future in futures.items():
            results[node] = future.result()
    return [result for result in results if result is not None]


__all__ = [
    "CONLEY_BACKENDS",
    "ExcisionError",
    "INDEX_MAPS",
    "INDEX_PAIRS",
    "IndexSizeLimitError",
    "SuspensionGridConleyResult",
    "compute_suspension_grid_conley_index",
    "compute_suspension_grid_conley_indices",
    "index_pair_atoms",
    "piece_rectangles",
    "relative_homology_dimensions",
    "suspension_grid_gluing",
]
