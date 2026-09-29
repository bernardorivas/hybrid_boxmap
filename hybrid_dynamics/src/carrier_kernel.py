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
and hands the relative payload of the kernel to the shift-class function
of CMGDB (:func:`suspension_complex.cmgdb_shift_class_function`).  That
payload has the entries, in the same order, of
:meth:`suspension_complex.CellularChainMap.to_cmgdb_payload`, so the shift
class, the dimensions, and the blockers are those of the Python
construction.  The Python construction stays the reference and is used
whenever the installed CMGDB lacks the native function
(:func:`resolve_conley_backend`).

The function also maps a complex into another one, given as a second set of
arrays with its own exit mask.  Two constructions of
:mod:`suspension_grid_conley` use this form, with the same carriers (the
subcomplex of the target induced on ``T(sigma)``) and the same chain map:

* the excision construction maps the nerve of ``X`` into the nerve of
  ``Xbar = X cup F(X)`` (:class:`suspension_complex.CrossComplexAcyclicCarrier`);
  :func:`native_cross_complex_chain_map` returns its chain map;
* the excised forward-closure pair maps the subcomplex of the nerve of
  ``Y = W cup F(W)`` spanned by ``W`` into that nerve;
  :func:`native_subcomplex_chain_entries` returns the chain-map entries on
  the relative basis of the pair.

Both raise the exceptions of the Python construction, as above.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import operator
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from . import atlas_conley
from .atlas_conley import (
    AtlasNerveSimplex,
    AtlasRelativeIndexPair2D,
    _relation_vertex_images,
)
from .suspension_complex import (
    SHIFT_CLASS_FUNCTION,
    CMGDBRelativeHomologyPayload,
    FiniteCellComplex,
    cmgdb_shift_class_function,
)

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


def check_conley_backend(backend: str) -> None:
    """Raise the error of :func:`resolve_conley_backend` for ``backend``, if any.

    Only ``"native"`` imports CMGDB here, to check that it provides the
    native kernel; ``"auto"`` is left to be resolved when a construction
    needs it.
    """

    if backend not in CONLEY_BACKENDS:
        raise ValueError(f"backend must be one of {CONLEY_BACKENDS!r}; got {backend!r}")
    if backend == "native" and not native_carrier_kernel_available():
        raise RuntimeError(
            f"backend='native' needs CMGDB.{KERNEL_FUNCTION}, which the installed CMGDB "
            "does not provide; install a CMGDB build that has it, or use backend='auto' "
            "or backend='python'"
        )


def resolve_conley_backend(backend: str) -> str:
    """The implementation, ``"python"`` or ``"native"``, that ``backend`` selects.

    ``"auto"`` selects the native kernel when the installed CMGDB provides
    it and the Python construction otherwise.  ``"native"`` raises
    ``RuntimeError`` when the installed CMGDB does not provide it.
    """

    check_conley_backend(backend)
    if backend == "auto":
        return "native" if native_carrier_kernel_available() else "python"
    return backend


def cmgdb_provenance(backend: str | None) -> dict[str, object]:
    """The installed CMGDB and the functions of a Conley run, for its metadata.

    ``version`` is the version of the installed ``cmgdb`` distribution
    (``None`` when it is not installed).  ``shift_class_function`` is the
    name of the CMGDB function that computes the shift classes
    (:func:`suspension_complex.cmgdb_shift_class_function`), and
    ``conley_backend`` the implementation that ``backend`` selects
    (:func:`resolve_conley_backend`).  Both are ``None`` when ``backend`` is
    ``None`` (no index is computed), and the function also when CMGDB is not
    installed.
    """

    try:
        version: str | None = importlib.metadata.version("cmgdb")
    except importlib.metadata.PackageNotFoundError:
        version = None
    function = conley_backend = None
    if backend is not None:
        conley_backend = resolve_conley_backend(backend)
        try:
            import CMGDB
        except ImportError:
            pass
        else:
            function = (
                SHIFT_CLASS_FUNCTION
                if hasattr(CMGDB, SHIFT_CLASS_FUNCTION)
                else "ComputeRelativeHomologyShiftClass"
            )
    return {
        "version": version,
        "shift_class_function": function,
        "conley_backend": conley_backend,
    }


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
        values = _vertex_rows(group, dimension)
        _require_int32(values, f"the vertices of the cells of dimension {dimension}")
        unsorted = np.flatnonzero(np.any(values[:, 1:] <= values[:, :-1], axis=1))
        if unsorted.size:
            raise ValueError(
                f"the vertices of the cell {group[int(unsorted[0])]!r} are not sorted and distinct"
            )
        if len(group) > 1:
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
        if dimension and not _alternating_boundaries(complex_, group, values):
            _check_boundaries_by_cell(complex_, group, values)
        arrays.append(values.astype(np.int32))
    return tuple(arrays)


_VERTICES = operator.attrgetter("vertices")
_VALUES = operator.methodcaller("values")


def _all_of_length(items: Sequence[Any], length: int) -> bool:
    return bool(np.all(np.fromiter(map(len, items), dtype=np.int64, count=len(items)) == length))


def _vertex_rows(cells: Sequence[Any], dimension: int) -> npt.NDArray[np.int64]:
    """The vertex tuples of ``cells`` as the rows of an ``int64`` array.

    Every cell must be a simplex with ``dimension + 1`` vertices; the first
    that is not raises ``ValueError``.
    """

    size = dimension + 1
    try:
        vertices = list(map(_VERTICES, cells))
        regular = _all_of_length(vertices, size)
    except (AttributeError, TypeError):
        regular = False
    if not regular:
        for cell in cells:
            tuple_ = getattr(cell, "vertices", None)
            if tuple_ is None or len(tuple_) != size:
                raise ValueError(
                    f"the cell {cell!r} of dimension {dimension} is not a simplex with "
                    f"{size} vertices"
                )
        vertices = list(map(_VERTICES, cells))
    return np.fromiter(
        itertools.chain.from_iterable(vertices), dtype=np.int64, count=len(cells) * size
    ).reshape(len(cells), size)


def _alternating_boundaries(
    complex_: FiniteCellComplex, cells: Sequence[Any], values: npt.NDArray[np.int64]
) -> bool:
    """Whether the boundary of every cell is the alternating sum of its faces by removed vertex.

    ``values`` holds the vertex tuples of ``cells``, which have dimension
    ``d >= 1``: the boundary of ``[v_0, ..., v_d]`` must list, for ``i = 0,
    ..., d``, the face without ``v_i`` with the coefficient ``(-1)^i``.
    """

    count, size = values.shape
    faces_per_cell = size * (size - 1)
    try:
        boundaries = list(map(complex_.boundary, cells))
        if not _all_of_length(boundaries, size):
            return False
        faces = list(map(_VERTICES, itertools.chain.from_iterable(boundaries)))
        if not _all_of_length(faces, size - 1):
            return False
        coefficients = np.fromiter(
            itertools.chain.from_iterable(map(_VALUES, boundaries)),
            dtype=np.int64,
            count=count * size,
        ).reshape(count, size)
        face_vertices = np.fromiter(
            itertools.chain.from_iterable(faces), dtype=np.int64, count=count * faces_per_cell
        ).reshape(count, size, size - 1)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    signs = np.where(np.arange(size) % 2 == 0, 1, -1)
    # Row i of columns lists the positions of the vertices of the face without v_i.
    columns = np.array([[j for j in range(size) if j != i] for i in range(size)], dtype=np.intp)
    return bool(
        np.array_equal(coefficients, np.broadcast_to(signs, coefficients.shape))
        and np.array_equal(face_vertices, values[:, columns])
    )


def _check_boundaries_by_cell(
    complex_: FiniteCellComplex, cells: Sequence[Any], values: npt.NDArray[np.int64]
) -> None:
    """Raise ``ValueError`` at the first cell whose boundary is not the alternating sum.

    This is the test of :func:`_alternating_boundaries`, cell by cell.
    """

    size = values.shape[1]
    signs = [1 if index % 2 == 0 else -1 for index in range(size)]
    for cell, row in zip(cells, values.tolist()):
        boundary = complex_.boundary(cell)
        faces = [row[:index] + row[index + 1 :] for index in range(size)]
        if list(boundary.values()) != signs or [list(face.vertices) for face in boundary] != faces:
            raise ValueError(
                f"the boundary of {cell!r} is not the alternating sum of its faces "
                "by removed vertex"
            )


def vertex_image_csr(
    complex_: FiniteCellComplex,
    vertex_images: Mapping[int, Collection[int]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """CSR offsets (``int64``) and labels (``int32``) of the images of the vertices.

    Row ``i`` lists, sorted, the image of the vertex of
    ``complex_.cells_of_dimension(0)[i]``.
    """

    return _image_csr([cell.vertices[0] for cell in complex_.cells_of_dimension(0)], vertex_images)


def _image_csr(
    vertices: Sequence[int],
    vertex_images: Mapping[int, Collection[int]],
    targets: Collection[int] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """CSR offsets and labels of the images of ``vertices``, restricted to ``targets`` if given.

    Each distinct image object is sorted once: the exit vertices of a
    component of ``P0`` share the component as their image, and its row is
    repeated for each of them.
    """

    positions: dict[int, int] = {}
    # The distinct images in order of first use; holding them keeps their ids unique.
    images: list[Collection[int]] = []
    image_of_row = np.empty(len(vertices), dtype=np.int64)
    for row, vertex in enumerate(vertices):
        if vertex not in vertex_images:
            raise ValueError(f"the vertex {vertex} of the complex has no image")
        image = vertex_images[vertex]
        position = positions.get(id(image))
        if position is None:
            position = positions[id(image)] = len(images)
            images.append(image)
        image_of_row[row] = position
    offsets, labels = _sorted_images(images, targets)
    if len(images) == len(vertices):
        # Every vertex has its own image, so the rows are those of the images.
        return offsets, labels
    indptr = np.zeros(len(vertices) + 1, dtype=np.int64)
    np.cumsum(np.diff(offsets)[image_of_row], out=indptr[1:])
    starts = offsets[image_of_row].tolist()
    ends = offsets[image_of_row + 1].tolist()
    return indptr, np.concatenate([labels[start:end] for start, end in zip(starts, ends)])


def _sorted_images(
    images: Sequence[Collection[int]], targets: Collection[int] | None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """CSR offsets and ``int32`` labels of the sorted distinct elements of each image.

    The images are restricted to ``targets`` if given.
    """

    lengths = np.fromiter(map(len, images), dtype=np.int64, count=len(images))
    values = np.fromiter(
        itertools.chain.from_iterable(images), dtype=np.int64, count=int(lengths.sum())
    )
    owners = np.repeat(np.arange(len(images), dtype=np.int64), lengths)
    if targets is not None:
        keep = np.isin(values, np.fromiter(targets, dtype=np.int64, count=len(targets)))
        values, owners = values[keep], owners[keep]
    _require_int32(values, "the vertex images")
    # One key per element, ordered by image and then by label; it fits in
    # int64, since the labels fit in int32 and there are fewer than 2**31
    # images.
    low = int(values.min()) if values.size else 0
    span = int(values.max()) - low + 1 if values.size else 1
    keys = owners * span + (values - low)
    del values, owners
    keys.sort()
    first = np.ones(keys.size, dtype=bool)
    np.not_equal(keys[1:], keys[:-1], out=first[1:])
    owners, labels = np.divmod(keys[first], span)
    offsets = np.zeros(len(images) + 1, dtype=np.int64)
    np.cumsum(np.bincount(owners, minlength=len(images)), out=offsets[1:])
    return offsets, (labels + low).astype(np.int32)


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
    """Inputs of ``CMGDB.ComputeCarrierChainMap``.

    ``target_simplices`` and ``target_exit`` are ``None`` for a complex that
    is its own target.
    """

    source_simplices: tuple[npt.NDArray[np.int32], ...]
    vertex_image_indptr: npt.NDArray[np.int64]
    vertex_image_indices: npt.NDArray[np.int32]
    source_exit: npt.NDArray[np.uint8]
    target_simplices: tuple[npt.NDArray[np.int32], ...] | None = None
    target_exit: npt.NDArray[np.uint8] | None = None

    def compute(self, *, return_carriers: bool = False) -> dict[str, Any]:
        """The result dictionary of ``CMGDB.ComputeCarrierChainMap`` on these arrays."""

        import CMGDB

        kernel = getattr(CMGDB, KERNEL_FUNCTION)
        target: dict[str, Any] = {}
        if self.target_simplices is not None:
            target = {
                "target_simplices": list(self.target_simplices),
                "target_exit": self.target_exit,
            }
        return dict(
            kernel(
                list(self.source_simplices),
                self.vertex_image_indptr,
                self.vertex_image_indices,
                self.source_exit,
                modulus=5,
                return_carriers=return_carriers,
                **target,
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
    return _reference_failure(status, cell, reference)


def _reference_failure(status: str, cell: object, reference: Callable[[], object]) -> Exception:
    """The exception that ``reference`` (the Python construction) raises.

    If it raises none, the kernel, which reports ``status`` at ``cell``, and
    the Python construction disagree, and an ``AssertionError`` is returned.
    """

    try:
        reference()
    except Exception as error:  # the exception of the Python construction
        return error
    return AssertionError(
        f"the native carrier kernel reports {status!r} on {cell!r}, but the Python "
        "construction forms the chain map"
    )


def _failing_cell(
    result: Mapping[str, Any], cells: Sequence[Sequence[Any]]
) -> tuple[str, int, int, Any]:
    """Status, degree, row, and source cell of a failed kernel result.

    ``cells[d]`` are the source cells of degree ``d`` in the order of the
    arrays.  An unknown status or a position outside the source raises
    ``AssertionError``.
    """

    status = str(result.get("status"))
    degree = int(result.get("failure_degree", -1))
    row = int(result.get("failure_row", -1))
    if status not in KERNEL_FAILURES:
        raise AssertionError(f"the native carrier kernel returned the status {status!r}")
    group = cells[degree] if 0 <= degree < len(cells) else ()
    if not 0 <= row < len(group):
        raise AssertionError(
            f"the native carrier kernel reports {status!r} at degree {degree} and row {row}, "
            "which is not a cell of the source"
        )
    return status, degree, row, group[row]


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
    for the same arguments: the shift class of :func:`native_relation_payload`
    by :func:`suspension_complex.cmgdb_shift_class_function`, with the same
    added keys.
    """

    payload = native_relation_payload(
        pair,
        top_relation=top_relation,
        use_exit_component_carrier=use_exit_component_carrier,
    )
    import CMGDB

    if not hasattr(CMGDB, "ComputeRelativeHomologyShiftClass"):
        raise RuntimeError("installed CMGDB lacks ComputeRelativeHomologyShiftClass")
    result = dict(cmgdb_shift_class_function(CMGDB)(*payload.as_compute_args()))
    result["result_scope"] = "finite_reset_quotient_relation"
    result["continuous_system_conley_index_certified"] = False
    result["finite_relation_algebra_validated"] = True
    return result


# ---------------------------------------------------------------------------
# Carriers into another complex
# ---------------------------------------------------------------------------


class KernelChainMap:
    """The chain map of a successful ``ComputeCarrierChainMap`` call, read by source cell.

    ``source_cells[d]`` and ``target_cells[d]`` are the ``d``-cells of the
    source and of the target in the order of the arrays handed to the
    kernel, and ``chain_map[d]`` holds the ``(source_row, target_row,
    coefficient)`` rows it returned, by source row.  :meth:`image` gives the
    image of a source cell as
    :meth:`suspension_complex.CellularMapBetweenComplexes.image` does: its
    target cells with coefficients in ``1..4``, listed in the order of the
    kernel, which is that of the Python construction.
    """

    def __init__(
        self,
        source_cells: Sequence[Sequence[Any]],
        target_cells: Sequence[Sequence[Any]],
        chain_map: Sequence[npt.ArrayLike],
    ) -> None:
        if len(chain_map) != len(source_cells):
            raise AssertionError(
                f"the native carrier kernel returned {len(chain_map)} degrees of the chain "
                f"map for a source with {len(source_cells)}"
            )
        self._target_cells = target_cells
        self._images: list[list[list[int]]] = []
        self._offsets: list[list[int]] = []
        self._position: dict[Any, tuple[int, int]] = {}
        for degree, cells in enumerate(source_cells):
            entries = np.asarray(chain_map[degree], dtype=np.int64).reshape(-1, 3)
            targets = len(target_cells[degree]) if degree < len(target_cells) else 0
            if entries.size and (
                np.any(np.diff(entries[:, 0]) < 0)
                or entries[0, 0] < 0
                or entries[-1, 0] >= len(cells)
                or entries[:, 1].min() < 0
                or entries[:, 1].max() >= targets
                or entries[:, 2].min() < 1
                or entries[:, 2].max() > 4
            ):
                raise AssertionError(
                    f"the native carrier kernel returned invalid chain-map rows in degree {degree}"
                )
            self._images.append(entries[:, 1:].tolist())
            self._offsets.append(
                np.searchsorted(entries[:, 0], np.arange(len(cells) + 1)).tolist()
            )
            for row, cell in enumerate(cells):
                self._position[cell] = (degree, row)

    def image(self, source_cell: Any) -> Mapping[Any, int]:
        try:
            degree, row = self._position[source_cell]
        except KeyError as error:
            raise KeyError(f"not a source cell: {source_cell!r}") from error
        targets = self._target_cells[degree]
        offsets = self._offsets[degree]
        return MappingProxyType(
            {
                targets[target]: value
                for target, value in self._images[degree][offsets[row] : offsets[row + 1]]
            }
        )


def cross_complex_carrier_arrays(
    source_complex: FiniteCellComplex,
    target_complex: FiniteCellComplex,
    vertex_images: Mapping[int, Collection[int]],
    source_exit_vertices: Collection[int] = (),
    target_exit_vertices: Collection[int] = (),
) -> CarrierKernelArrays:
    """The arrays of the carrier from ``source_complex`` into ``target_complex``.

    The carrier of a source simplex is the subcomplex of ``target_complex``
    induced on the union of the images of its vertices, as with
    :class:`atlas_conley._InducedCarrierGenerators` and ``source_complex``:
    an image vertex that is not a vertex of the target spans no simplex, so
    it is left out of the arrays.  When ``target_complex`` is
    ``source_complex``, the target arrays are ``None`` (the kernel then maps
    the complex into itself) and the two exit sets must be equal.
    """

    source = simplex_arrays(source_complex)
    source_labels = source[0][:, 0].tolist()
    target: tuple[npt.NDArray[np.int32], ...] | None = None
    target_exit: npt.NDArray[np.uint8] | None = None
    if target_complex is source_complex:
        if frozenset(int(v) for v in source_exit_vertices) != frozenset(
            int(v) for v in target_exit_vertices
        ):
            raise ValueError("a complex mapped into itself has one set of exit vertices")
        target_labels = source_labels
    else:
        target = simplex_arrays(target_complex)
        target_labels = target[0][:, 0].tolist()
        target_exit = exit_mask(target_complex, target_exit_vertices)
    indptr, indices = _image_csr(source_labels, vertex_images, frozenset(target_labels))
    return CarrierKernelArrays(
        source_simplices=source,
        vertex_image_indptr=indptr,
        vertex_image_indices=indices,
        source_exit=exit_mask(source_complex, source_exit_vertices),
        target_simplices=target,
        target_exit=target_exit,
    )


def native_cross_complex_chain_map(
    source: AtlasRelativeIndexPair2D,
    target: AtlasRelativeIndexPair2D,
    vertex_images: Mapping[int, Collection[int]],
    reference: Callable[[], object],
) -> KernelChainMap:
    """The chain map of the excision construction, from the native kernel.

    The Python construction (``reference``; see
    ``suspension_grid_conley._excision_index_map``) forms the
    :class:`suspension_complex.CrossComplexAcyclicCarrier` from
    ``source.complex`` to ``target.complex`` whose value at a simplex is the
    subcomplex induced on the union of the images of its vertices
    (:class:`atlas_conley._InducedCarrierGenerators`), checks that the
    carrier of every simplex of the ``P0`` of ``source`` lies in the ``P0``
    of ``target``, and constructs the chain map.  The kernel forms the same
    carriers, makes the same checks in the same order, and constructs the
    same chain map, which is returned.

    A carrier that is not acyclic is reported at the first simplex of the
    source with such a carrier, with the message of the Python construction,
    which checks each simplex in turn for an empty carrier and then for one
    that is not acyclic; the kernel has found no empty carrier before it
    checks acyclicity.  The other failures (an empty carrier, the pair, the
    chain map) are reported by the Python construction in another order or
    with chains in the message, so ``reference`` is run and its exception
    raised (``AssertionError`` if it raises none).
    """

    arrays = cross_complex_carrier_arrays(
        source.complex,
        target.complex,
        vertex_images,
        source.p0_atlas_cells,
        target.p0_atlas_cells,
    )
    result = arrays.compute()
    source_cells = [
        source.complex.cells_of_dimension(degree)
        for degree in range(len(arrays.source_simplices))
    ]
    if result.get("status") != "ok":
        status, _, _, cell = _failing_cell(result, source_cells)
        if status == "not_acyclic":
            raise ValueError(f"carrier image of {cell!r} is not acyclic over GF(5)")
        raise _reference_failure(status, cell, reference)
    target_cells = [
        target.complex.cells_of_dimension(degree)
        for degree in range(target.complex.max_dimension + 1)
    ]
    return KernelChainMap(source_cells, target_cells, result["chain_map"])


def native_subcomplex_chain_entries(
    pair: AtlasRelativeIndexPair2D,
    sources: Collection[AtlasNerveSimplex],
    vertex_images: Mapping[int, Collection[int]],
    reference: Callable[[], object],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """The chain-map entries of the excised forward-closure pair, from the native kernel.

    The Python construction (``reference``; see
    ``suspension_grid_conley._excised_chain_map_entries``) forms the chain
    map on the simplices of ``pair.complex`` that lie in ``sources`` (a
    subcomplex ``K`` that contains the basis of ``C(P1) / C(P0)``), the
    carrier of a simplex being the subcomplex of ``pair.complex`` induced on
    the union of the images of its vertices, and returns its entries on
    that basis, column by column, without the targets in ``P0``.  The kernel
    maps ``K`` into ``pair.complex`` with the same carriers and chain map,
    and the entries are read from it in the same order.

    The Python construction takes the simplices of ``K`` in order and
    checks each (its carrier acyclic, then in ``P0`` when the simplex lies
    in ``P0``) before it forms its image; it raises at the first failure.
    A carrier leaves ``P0`` from a simplex of ``P0`` only when the carrier of
    one of its vertices does, and every image before the first carrier that
    is not acyclic exists, since the earlier carriers are acyclic.  So the
    first failure is at the first vertex of ``P0`` whose carrier leaves
    ``P0`` or at the first simplex whose carrier is not acyclic, whichever
    comes first; these are reported with the messages of the Python
    construction.  For the other failures ``reference`` is run and its
    exception raised (``AssertionError`` if it raises none).
    """

    complex_ = pair.complex
    relative = pair.relative_pair
    target = simplex_arrays(complex_)
    labels = target[0][:, 0]
    cells = [complex_.cells_of_dimension(degree) for degree in range(len(target))]
    # The rows of the simplices of K among those of the complex.
    rows = [
        np.flatnonzero(
            np.fromiter((cell in sources for cell in group), dtype=bool, count=len(group))
        )
        for group in cells
    ]
    source = tuple(np.ascontiguousarray(array[selected]) for array, selected in zip(target, rows))
    source_labels = source[0][:, 0].tolist()
    exits = frozenset(int(vertex) for vertex in pair.p0_atlas_cells)
    target_exit = exit_mask(complex_, exits)
    indptr, indices = _image_csr(source_labels, vertex_images, frozenset(labels.tolist()))
    arrays = CarrierKernelArrays(
        source_simplices=source,
        vertex_image_indptr=indptr,
        vertex_image_indices=indices,
        source_exit=np.fromiter(
            (label in exits for label in source_labels), dtype=np.uint8, count=len(source_labels)
        ),
        target_simplices=target,
        target_exit=target_exit,
    )
    result = arrays.compute()
    if result.get("status") != "ok":
        source_cells = [
            [group[row] for row in selected.tolist()] for group, selected in zip(cells, rows)
        ]
        status, degree, row, cell = _failing_cell(result, source_cells)
        if status == "pair_violation" and degree == 0:
            raise ValueError(f"the carrier of {cell!r} does not preserve P0")
        if status == "not_acyclic":
            # A vertex before the simplex whose carrier leaves P0 fails first.
            before = row if degree == 0 else len(source_labels)
            for vertex_row, label in enumerate(source_labels[:before]):
                image = indices[indptr[vertex_row] : indptr[vertex_row + 1]].tolist()
                if label in exits and not exits.issuperset(image):
                    raise ValueError(
                        f"the carrier of {source_cells[0][vertex_row]!r} does not preserve P0"
                    )
            raise ValueError(f"carrier image of {cell!r} is not acyclic over GF(5)")
        raise _reference_failure(status, cell, reference)

    chain_map = result["chain_map"]
    if len(chain_map) != len(target):
        raise AssertionError(
            f"the native carrier kernel returned {len(chain_map)} degrees of the chain map "
            f"for a source with {len(target)}"
        )
    in_p0 = target_exit.astype(bool)
    entries: list[tuple[tuple[int, int, int], ...]] = []
    for degree, array in enumerate(target):
        # The position of every simplex in the basis of C(P1) / C(P0), or -1 in P0.
        outside = ~np.all(in_p0[np.searchsorted(labels, array)], axis=1)
        position = np.full(len(array), -1, dtype=np.int64)
        position[outside] = np.arange(int(np.count_nonzero(outside)))
        if int(np.count_nonzero(outside)) != relative.cell_counts[degree]:
            raise AssertionError(
                f"the simplices outside P0 are not the basis of C(P1) / C(P0) in degree {degree}"
            )
        found = np.asarray(chain_map[degree], dtype=np.int64).reshape(-1, 3)
        columns = position[rows[degree][found[:, 0]]]
        images = position[found[:, 1]]
        keep = (columns >= 0) & (images >= 0)
        entries.append(
            tuple(zip(images[keep].tolist(), columns[keep].tolist(), found[keep, 2].tolist()))
        )
    return tuple(entries)


__all__ = [
    "CONLEY_BACKENDS",
    "KERNEL_FAILURES",
    "KERNEL_FUNCTION",
    "CarrierKernelArrays",
    "KernelChainMap",
    "carrier_kernel_arrays",
    "check_conley_backend",
    "cmgdb_provenance",
    "cross_complex_carrier_arrays",
    "exit_mask",
    "kernel_failure",
    "native_carrier_kernel_available",
    "native_cross_complex_chain_map",
    "native_relation_payload",
    "native_relation_shift_class",
    "native_subcomplex_chain_entries",
    "relation_carrier_arrays",
    "resolve_conley_backend",
    "simplex_arrays",
    "vertex_image_csr",
]
