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

The result is a finite-relation shift class over ``GF(5)``.  It is not a
certified Conley index of the continuous fixed-time map, because the
relation is sampled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .atlas_conley import (
    AffineBoundaryEmbedding2D,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    prepare_atlas_relation_conley_2d,
)
from .suspension_grid import SuspensionGrid, UnsupportedSuspensionGridError
from .suspension_grid_relation import SuspensionGridRelation


BASE_CHART = 0
HANDLE_CHART = 1


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
    """Finite-relation shift class of one Morse set, or the blocker."""

    morse_node: int
    computed: bool
    shift_class: tuple[str, ...] = ()
    homology_dimensions: tuple[int, ...] = ()
    pair_atoms: dict[str, int] = field(default_factory=dict)
    pair_pieces: dict[str, int] = field(default_factory=dict)
    nerve_cell_counts: tuple[int, ...] = ()
    blocker: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "morse_node": self.morse_node,
            "computed": self.computed,
            "shift_class": list(self.shift_class),
            "homology_dimensions": list(self.homology_dimensions),
            "pair_atoms": self.pair_atoms,
            "pair_pieces": self.pair_pieces,
            "nerve_cell_counts": list(self.nerve_cell_counts),
            "coefficient_field": 5,
            "result_scope": "finite_reset_quotient_relation" if self.computed else "",
            "continuous_system_conley_index_certified": False,
            "blocker": self.blocker,
            "seconds": self.seconds,
        }


def compute_suspension_grid_conley_index(
    relation: SuspensionGridRelation,
    morse_set: npt.ArrayLike,
    *,
    morse_node: int = 0,
    maximum_simplex_size: int = 16,
) -> SuspensionGridConleyResult:
    """Shift class of the index map on ``(S cup F(S), F(S) minus S)``.

    Any failed gate (good cover of the quotient nerve, carrier acyclicity,
    pair preservation, chain map) is returned as ``blocker`` instead of a
    label.
    """

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
    )
    try:
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
        preparation = prepare_atlas_relation_conley_2d(pair, top_relation=piece_image)
        payload = preparation.compute_finite_relation_shift_class()
        result.computed = True
        result.shift_class = tuple(str(entry) for entry in payload["shift_class"])
        result.homology_dimensions = tuple(int(v) for v in payload["homology_dimensions"])
    except Exception as error:  # the blocker is reported, never replaced by a label
        result.blocker = f"{type(error).__name__}: {error}"
    result.seconds = time.perf_counter() - started
    return result


__all__ = [
    "SuspensionGridConleyResult",
    "compute_suspension_grid_conley_index",
    "piece_rectangles",
    "suspension_grid_gluing",
]
