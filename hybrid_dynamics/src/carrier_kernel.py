"""The relation carrier of the index map as arrays for the native kernel of CMGDB.

The carrier of :func:`atlas_conley.prepare_atlas_relation_conley_2d` sends a
simplex ``sigma`` of the nerve of ``P1`` to the subcomplex induced on the
union ``T(sigma)`` of the images of its vertices.  Every carrier is checked
to be acyclic over ``GF(5)``, and the chain map is chosen by the
acyclic-carrier induction of
:meth:`suspension_complex.FixedTimeCarrier.construct_chain_map`: a vertex
goes to the smallest vertex of its carrier, and a ``d``-simplex to the unique
``d``-chain of its carrier whose boundary is the image of the boundary of the
simplex and which is supported on the greedy independent ``d``-cells of the
carrier (those whose boundary is not in the span of the boundaries of the
``d``-cells before them, in the order of the complex).

``CMGDB.ComputeCarrierChainMap``, in the CMGDB builds that provide it, forms
the same carriers, checks, and chain map natively.  This module writes a pair
and its relation as the arrays that function reads:

* the simplices of the complex, one ``int32`` array of shape
  ``(n_d, d + 1)`` per degree ``d``, in the order of the complex (by
  dimension, then by vertex tuple); the vertices are Atlas cell indices;
* the images of the vertices, by row of the array of degree 0, as a CSR
  pair (``int64`` offsets, ``int32`` vertex labels), with the image of an
  exit vertex already replaced by its component of ``P0`` when the carrier
  uses exit components;
* the exit vertices as a ``uint8`` mask; ``P0`` is the subcomplex they
  induce.

:func:`native_relation_shift_class` calls the function, raises for a failure
the exception that the Python construction raises in the same situation,
and hands the relative payload of the kernel to
``CMGDB.ComputeRelativeHomologyShiftClass``.  That payload has the entries,
in the same order, of
:meth:`suspension_complex.CellularChainMap.to_cmgdb_payload`, so the shift
class, the dimensions, and the blockers are those of the Python
construction.  The Python construction stays the reference and is used
whenever the installed CMGDB lacks the native function
(:func:`resolve_conley_backend`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import atlas_conley
from .atlas_conley import AtlasRelativeIndexPair2D, _relation_vertex_images
from .suspension_complex import CMGDBRelativeHomologyPayload, FiniteCellComplex


#: The native function of the CMGDB fork.
KERNEL_FUNCTION = "ComputeCarrierChainMap"

#: Implementations of the carrier and chain map of the index map: ``"auto"``
#: is the native kernel when the installed CMGDB provides it, and the Python
#: construction otherwise.
CONLEY_BACKENDS = ("auto", "python", "native")

#: Statuses of ``ComputeCarrierChainMap`` other than ``"ok"``.
KERNEL_FAILURES = (
    "empty_carrier",
    "not_acyclic",
    "no_solution",
    "pair_violation",
    "chain_map_invalid",
)

_INT32 = np.iinfo(np.int32)


def native_carrier_kernel_available() -> bool:
    """Whether the installed CMGDB provides ``ComputeCarrierChainMap``."""

    try:
        import CMGDB
    except ImportError:
        return False
    return hasattr(CMGDB, KERNEL_FUNCTION)


def resolve_conley_backend(backend: str) -> str:
    """The implementation, ``"python"`` or ``"native"``, that ``backend`` selects.

    ``"auto"`` selects the native kernel when the installed CMGDB provides
    it and the Python construction otherwise.  ``"native"`` raises
    ``RuntimeError`` when the installed CMGDB does not provide it.
    """

    if backend not in CONLEY_BACKENDS:
        raise ValueError(f"backend must be one of {CONLEY_BACKENDS!r}; got {backend!r}")
    if backend == "python":
        return "python"
    if native_carrier_kernel_available():
        return "native"
    if backend == "native":
        raise RuntimeError(
            f"backend='native' needs CMGDB.{KERNEL_FUNCTION}, which the installed CMGDB "
            "does not provide; install a CMGDB build that has it, or use backend='auto' "
            "or backend='python'"
        )
    return "python"


def _require_int32(values: npt.NDArray[np.int64], description: str) -> None:
    if values.size and (values.min() < _INT32.min or values.max() > _INT32.max):
        raise ValueError(f"{description} do not fit in int32")


def simplex_arrays(complex_: FiniteCellComplex) -> tuple[npt.NDArray[np.int32], ...]:
    """The simplices of ``complex_``, one ``int32`` array per degree, in the order of the complex.

    Row ``i`` of the array of degree ``d`` is the vertex tuple of
    ``complex_.cells_of_dimension(d)[i]``.  The complex must be the oriented
    simplicial complex that the kernel assumes: cells ordered by dimension
    and then by vertex tuple, each with sorted distinct vertices, and the
    boundary of ``[v_0, ..., v_d]`` the sum of ``(-1)^i`` times the face
    without ``v_i``, listed by ``i``.  The quotient nerves of
    :mod:`atlas_conley` and the complexes of their pairs are of this kind;
    any other complex raises ``ValueError``.
    """

    groups = [complex_.cells_of_dimension(d) for d in range(complex_.max_dimension + 1)]
    if complex_.cells != tuple(itertools.chain.from_iterable(groups)):
        raise ValueError("the cells of the complex are not ordered by dimension")
    arrays: list[npt.NDArray[np.int32]] = []
    for dimension, group in enumerate(groups):
        rows: list[tuple[int, ...]] = []
        for cell in group:
            vertices = getattr(cell, "vertices", None)
            if vertices is None or len(vertices) != dimension + 1:
                raise ValueError(
                    f"the cell {cell!r} of dimension {dimension} is not a simplex with "
                    f"{dimension + 1} vertices"
                )
            rows.append(tuple(int(vertex) for vertex in vertices))
        values = np.array(rows, dtype=np.int64).reshape(len(rows), dimension + 1)
        _require_int32(values, f"the vertices of the cells of dimension {dimension}")
        unsorted = np.flatnonzero(np.any(values[:, 1:] <= values[:, :-1], axis=1))
        if unsorted.size:
            raise ValueError(
                f"the vertices of the cell {group[int(unsorted[0])]!r} are not sorted and distinct"
            )
        if len(rows) > 1:
            # Consecutive rows increase lexicographically: their first
            # difference is positive (and exists).
            difference = values[1:] - values[:-1]
            first = np.argmax(difference != 0, axis=1)
            lead = difference[np.arange(len(difference)), first]
            disorder = np.flatnonzero(lead <= 0)
            if disorder.size:
                row = int(disorder[0])
                raise ValueError(
                    f"the cells of dimension {dimension} are not in lexicographic order: "
                    f"row {row} is {group[row]!r} and row {row + 1} is {group[row + 1]!r}"
                )
        if dimension:
            signs = [1 if index % 2 == 0 else -1 for index in range(dimension + 1)]
            for cell, row in zip(group, rows):
                boundary = complex_.boundary(cell)
                faces = [row[:index] + row[index + 1 :] for index in range(dimension + 1)]
                if list(boundary.values()) != signs or [
                    tuple(face.vertices) for face in boundary
                ] != faces:
                    raise ValueError(
                        f"the boundary of {cell!r} is not the alternating sum of its faces "
                        "by removed vertex"
                    )
        arrays.append(values.astype(np.int32))
    return tuple(arrays)


def vertex_image_csr(
    complex_: FiniteCellComplex,
    vertex_images: Mapping[int, Collection[int]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """CSR offsets (``int64``) and labels (``int32``) of the images of the vertices.

    Row ``i`` lists, sorted, the image of the vertex of
    ``complex_.cells_of_dimension(0)[i]``.
    """

    images: list[list[int]] = []
    for cell in complex_.cells_of_dimension(0):
        vertex = cell.vertices[0]
        if vertex not in vertex_images:
            raise ValueError(f"the vertex {vertex} of the complex has no image")
        images.append(sorted({int(target) for target in vertex_images[vertex]}))
    indptr = np.zeros(len(images) + 1, dtype=np.int64)
    np.cumsum([len(image) for image in images], out=indptr[1:])
    indices = np.fromiter(
        itertools.chain.from_iterable(images), dtype=np.int64, count=int(indptr[-1])
    )
    _require_int32(indices, "the vertex images")
    return indptr, indices.astype(np.int32)


def exit_mask(complex_: FiniteCellComplex, exit_vertices: Collection[int]) -> npt.NDArray[np.uint8]:
    """``uint8`` mask of the exit vertices, by row of the vertices of ``complex_``."""

    vertices = [cell.vertices[0] for cell in complex_.cells_of_dimension(0)]
    exits = frozenset(int(vertex) for vertex in exit_vertices)
    unknown = exits.difference(vertices)
    if unknown:
        raise ValueError(f"exit vertices outside the complex: {sorted(unknown)!r}")
    return np.fromiter(
        (vertex in exits for vertex in vertices), dtype=np.uint8, count=len(vertices)
    )


@dataclass(frozen=True)
class CarrierKernelArrays:
    """Inputs of ``CMGDB.ComputeCarrierChainMap`` for a complex that is its own target."""

    source_simplices: tuple[npt.NDArray[np.int32], ...]
    vertex_image_indptr: npt.NDArray[np.int64]
    vertex_image_indices: npt.NDArray[np.int32]
    source_exit: npt.NDArray[np.uint8]

    def compute(self, *, return_carriers: bool = False) -> dict[str, Any]:
        """The result dictionary of ``CMGDB.ComputeCarrierChainMap`` on these arrays."""

        import CMGDB

        kernel = getattr(CMGDB, KERNEL_FUNCTION)
        return dict(
            kernel(
                list(self.source_simplices),
                self.vertex_image_indptr,
                self.vertex_image_indices,
                self.source_exit,
                modulus=5,
                return_carriers=return_carriers,
            )
        )


def carrier_kernel_arrays(
    complex_: FiniteCellComplex,
    vertex_images: Mapping[int, Collection[int]],
    exit_vertices: Collection[int] = (),
) -> CarrierKernelArrays:
    """The arrays of the carrier with the given vertex images on ``complex_``."""

    indptr, indices = vertex_image_csr(complex_, vertex_images)
    return CarrierKernelArrays(
        source_simplices=simplex_arrays(complex_),
        vertex_image_indptr=indptr,
        vertex_image_indices=indices,
        source_exit=exit_mask(complex_, exit_vertices),
    )


def relation_carrier_arrays(
    pair: AtlasRelativeIndexPair2D,
    *,
    top_relation: Mapping[int, Collection[int]],
    use_exit_component_carrier: bool = True,
) -> CarrierKernelArrays:
    """The arrays of the carrier of :func:`atlas_conley.prepare_atlas_relation_conley_2d`.

    The vertex images, and the checks of the relation with their errors, are
    those of the Python construction; the exit vertices are those of ``P0``.
    """

    vertex_images = _relation_vertex_images(
        pair, top_relation, use_exit_component_carrier=use_exit_component_carrier
    )
    return carrier_kernel_arrays(pair.complex, vertex_images, pair.p0_atlas_cells)


def kernel_failure(
    result: Mapping[str, Any],
    pair: AtlasRelativeIndexPair2D,
    vertex_images: Mapping[int, Collection[int]],
    reference: Callable[[], object],
) -> Exception:
    """The exception of the Python construction for a failed kernel result.

    The failing cell is ``pair.complex.cells_of_dimension(failure_degree)
    [failure_row]``.  An empty or non-acyclic carrier is reported at the
    first cell of the complex with such a carrier, as in the Python
    construction, whose message is rebuilt from the cell.  Before it forms
    the chain map, the Python construction checks that the carriers
    preserve the pair (the image of every exit vertex consists of exit
    vertices); then a missing solution is reported at its cell.  The other
    failures of the chain map have Python messages that record chains the
    kernel does not return: ``reference`` (the Python construction) is run
    for them and its exception is returned.  If it succeeds, the kernel and
    the Python construction disagree, and an ``AssertionError`` is returned.
    """

    status = str(result.get("status"))
    degree = int(result.get("failure_degree", -1))
    row = int(result.get("failure_row", -1))
    cells = pair.complex.cells_of_dimension(degree) if degree >= 0 else ()
    if status not in KERNEL_FAILURES:
        return AssertionError(f"the native carrier kernel returned the status {status!r}")
    if not 0 <= row < len(cells):
        return AssertionError(
            f"the native carrier kernel reports {status!r} at degree {degree} and row {row}, "
            "which is not a cell of the complex"
        )
    cell = cells[row]
    if status == "empty_carrier":
        return ValueError(f"relation carrier of {cell!r} is empty")
    if status == "not_acyclic":
        return ValueError(f"carrier image of {cell!r} is not acyclic over GF(5)")
    exits = pair.p0_atlas_cells
    if any(not exits.issuperset(vertex_images[vertex]) for vertex in exits):
        return ValueError("the acyclic carrier does not preserve the relative pair")
    if status == "no_solution":
        return ValueError(
            f"acyclic-carrier chain selection failed on {cell!r}: the boundary equation "
            "has no solution in the carrier"
        )
    try:
        reference()
    except Exception as error:  # the exception of the Python construction
        return error
    return AssertionError(
        f"the native carrier kernel reports {status!r} on {cell!r}, but the Python "
        "construction forms the chain map"
    )


def _entries(degrees: Any) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(
        tuple((int(row), int(column), int(value)) for row, column, value in entries)
        for entries in degrees
    )


def native_relation_payload(
    pair: AtlasRelativeIndexPair2D,
    *,
    top_relation: Mapping[int, Collection[int]],
    use_exit_component_carrier: bool = True,
    modulus: int = 5,
) -> CMGDBRelativeHomologyPayload:
    """The payload of the index map of ``pair`` from ``CMGDB.ComputeCarrierChainMap``.

    It is the payload of :func:`atlas_conley.prepare_atlas_relation_conley_2d`
    (``preparation.payload``): the same checks run in the same order and
    raise the same exceptions (see :func:`kernel_failure`), and the chain
    map is the same.  The cell counts and the relative boundary returned by
    the kernel are checked against those of ``pair``.
    """

    if modulus != 5:
        raise ValueError("the CMGDB explicit-chain bridge currently uses GF(5)")
    vertex_images = _relation_vertex_images(
        pair, top_relation, use_exit_component_carrier=use_exit_component_carrier
    )
    arrays = carrier_kernel_arrays(pair.complex, vertex_images, pair.p0_atlas_cells)
    result = arrays.compute()
    if result.get("status") != "ok":

        def reference() -> object:
            return atlas_conley.prepare_atlas_relation_conley_2d(
                pair,
                top_relation=top_relation,
                use_exit_component_carrier=use_exit_component_carrier,
            )

        raise kernel_failure(result, pair, vertex_images, reference)
    relative = pair.relative_pair
    raw = result["payload"]
    counts = tuple(int(value) for value in raw["cell_counts"])
    if counts != relative.cell_counts:
        raise AssertionError(
            f"the native carrier kernel gives the relative cell counts {counts!r}, "
            f"the pair {relative.cell_counts!r}"
        )
    boundary = _entries(raw["boundary_entries"])
    if boundary != relative.boundary_entries(modulus=5):
        raise AssertionError("the native carrier kernel gives another relative boundary")
    return CMGDBRelativeHomologyPayload(
        cell_counts=counts,
        boundary_entries=boundary,
        chain_map_entries=_entries(raw["chain_map_entries"]),
        basis_by_dimension=relative.basis_by_dimension,
    )


def native_relation_shift_class(
    pair: AtlasRelativeIndexPair2D,
    *,
    top_relation: Mapping[int, Collection[int]],
    use_exit_component_carrier: bool = True,
) -> dict[str, object]:
    """The shift class of the index map of ``pair`` with the native carrier kernel.

    The result is that of
    ``prepare_atlas_relation_conley_2d(...).compute_finite_relation_shift_class()``
    for the same arguments: ``CMGDB.ComputeRelativeHomologyShiftClass`` of
    :func:`native_relation_payload`, with the same added keys.
    """

    payload = native_relation_payload(
        pair,
        top_relation=top_relation,
        use_exit_component_carrier=use_exit_component_carrier,
    )
    import CMGDB

    if not hasattr(CMGDB, "ComputeRelativeHomologyShiftClass"):
        raise RuntimeError("installed CMGDB lacks ComputeRelativeHomologyShiftClass")
    result = dict(CMGDB.ComputeRelativeHomologyShiftClass(*payload.as_compute_args()))
    result["result_scope"] = "finite_reset_quotient_relation"
    result["continuous_system_conley_index_certified"] = False
    result["finite_relation_algebra_validated"] = True
    return result


__all__ = [
    "CONLEY_BACKENDS",
    "CarrierKernelArrays",
    "KERNEL_FAILURES",
    "KERNEL_FUNCTION",
    "carrier_kernel_arrays",
    "exit_mask",
    "kernel_failure",
    "native_carrier_kernel_available",
    "native_relation_payload",
    "native_relation_shift_class",
    "relation_carrier_arrays",
    "resolve_conley_backend",
    "simplex_arrays",
    "vertex_image_csr",
]
