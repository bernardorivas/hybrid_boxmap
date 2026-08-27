"""Exit-aware local CMGDB computation for the Garcia passive walker.

This module deliberately studies the fixed-time suspension map, not a stride
return map.  The ambient rectangles are an open computational window: target
pieces and sampled trajectories may leave it, and no cemetery state is added.
Every such loss is retained per source.  A gait result is considered only from
the reference-carrying recurrent component and its local candidate pair.

The discovery workflow starts with a complete coarse cover of both Atlas
charts.  A failure-pruned recurrent graph is used only to locate a refinement
window; it is unioned with every stored-gait source cell, a fixed physical
collar, and its quotient-seam neighbors.  The coarse exterior is then removed:
the refined family is an explicitly open local Atlas window.  Subsequent
rounds use the complete, unpruned relation on that window to add finely
resolved in-domain image support until it stabilizes or a disclosed cap/exit
condition stops the run.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import os
import shutil
import tempfile
import threading
import time
import weakref
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from ..src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    SourceBoxMapDiagnostic,
    SuspensionAtlasCharts,
    build_cmgdb_atlas_model,
)
from .garcia_passive_walker_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    DEFAULT_BASE_BOUNDS,
    AtlasWalkerCell,
    GarciaWalkerAtlasSetup,
    GarciaWalkerQuotientIncidence,
    GuardAlignedGarciaWalkerQuotientIncidence,
    _atlas_cells,
    _audit_atlas_endpoint_probes,
    _morse_node_by_phase_cell,
    _stored_gait_probes,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from .garcia_endpoint_precompute import (
    EndpointCacheEntry,
    GarciaEndpointPrecomputeConfig,
    GarciaEndpointPrecomputeResult,
    PARITY_PROVENANCE,
    precompute_garcia_endpoint_cache,
)
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    GUARD_ALIGNED_DOMAIN_BOUNDS,
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
)


def _source_key(chart_id: int, bounds: Sequence[float]) -> tuple[object, ...]:
    return (
        int(chart_id),
        *(round(float(value), 14) for value in bounds),
    )


@dataclass(frozen=True, order=True)
class GarciaDyadicCell:
    """One tagged four-dimensional dyadic Atlas cell."""

    chart_id: int
    axis_depth: int
    coordinates: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.chart_id not in (BASE_CHART_ID, HANDLE_CHART_ID):
            raise ValueError(f"unknown Garcia chart {self.chart_id}")
        if self.axis_depth < 0:
            raise ValueError("axis_depth must be non-negative")
        if len(self.coordinates) != 4:
            raise ValueError("a Garcia dyadic cell needs four coordinates")
        subdivisions = 2**self.axis_depth
        if any(value < 0 or value >= subdivisions for value in self.coordinates):
            raise ValueError("dyadic coordinates lie outside their declared depth")

    def bounds(self, charts: SuspensionAtlasCharts) -> tuple[float, ...]:
        chart_bounds = charts.bounds_for(self.chart_id)
        subdivisions = 2**self.axis_depth
        lower = tuple(
            interval[0]
            + (interval[1] - interval[0]) * coordinate / subdivisions
            for interval, coordinate in zip(chart_bounds, self.coordinates)
        )
        upper = tuple(
            interval[0]
            + (interval[1] - interval[0]) * (coordinate + 1) / subdivisions
            for interval, coordinate in zip(chart_bounds, self.coordinates)
        )
        return tuple(float(value) for value in (*lower, *upper))

    def is_ancestor_of(self, other: "GarciaDyadicCell") -> bool:
        if self.chart_id != other.chart_id or self.axis_depth > other.axis_depth:
            return False
        shift = other.axis_depth - self.axis_depth
        return self.coordinates == tuple(value >> shift for value in other.coordinates)

    def descendants(self, target_axis_depth: int) -> tuple["GarciaDyadicCell", ...]:
        if target_axis_depth < self.axis_depth:
            raise ValueError("cannot refine a dyadic cell to a shallower depth")
        difference = target_axis_depth - self.axis_depth
        if difference == 0:
            return (self,)
        scale = 2**difference
        return tuple(
            GarciaDyadicCell(
                self.chart_id,
                target_axis_depth,
                tuple(
                    scale * coordinate + offset
                    for coordinate, offset in zip(self.coordinates, offsets)
                ),
            )
            for offsets in itertools.product(range(scale), repeat=4)
        )


@dataclass(frozen=True)
class GarciaMixedDyadicFamily:
    """A finite antichain of tagged dyadic cells, possibly at mixed depths."""

    cells: tuple[GarciaDyadicCell, ...]
    role: str

    def __post_init__(self) -> None:
        ordered = tuple(sorted(set(self.cells)))
        if ordered != self.cells:
            raise ValueError("mixed dyadic family must be sorted and duplicate-free")
        seen: set[GarciaDyadicCell] = set()
        for cell in sorted(
            ordered,
            key=lambda value: (value.chart_id, value.axis_depth, value.coordinates),
        ):
            has_ancestor = False
            for depth in range(cell.axis_depth + 1):
                shift = cell.axis_depth - depth
                ancestor = GarciaDyadicCell(
                    cell.chart_id,
                    depth,
                    tuple(value >> shift for value in cell.coordinates),
                )
                if ancestor in seen:
                    has_ancestor = True
                    break
            if has_ancestor:
                raise ValueError("mixed dyadic family must be an antichain")
            seen.add(cell)

    @cached_property
    def max_axis_depth(self) -> int:
        return max((cell.axis_depth for cell in self.cells), default=0)

    @cached_property
    def min_axis_depth(self) -> int:
        return min((cell.axis_depth for cell in self.cells), default=0)

    def chart_counts(self) -> dict[int, int]:
        return dict(Counter(cell.chart_id for cell in self.cells))

    def tagged_cells(self) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        return tuple(
            (cell.chart_id, cell.axis_depth, cell.coordinates) for cell in self.cells
        )

    def covering_cell(self, fine_cell: GarciaDyadicCell) -> GarciaDyadicCell | None:
        for depth in range(fine_cell.axis_depth, -1, -1):
            shift = fine_cell.axis_depth - depth
            candidate = GarciaDyadicCell(
                fine_cell.chart_id,
                depth,
                tuple(value >> shift for value in fine_cell.coordinates),
            )
            if candidate in self._cell_set:
                return candidate
        return None

    @cached_property
    def _cell_set(self) -> frozenset[GarciaDyadicCell]:
        return frozenset(self.cells)


class GarciaRefinementSizeLimitExceeded(RuntimeError):
    """Raised before a refinement family exceeds its declared cell cap."""

    def __init__(self, *, limit: int, lower_bound: int) -> None:
        self.limit = int(limit)
        self.lower_bound = int(lower_bound)
        super().__init__(
            f"refinement needs at least {self.lower_bound} cells, above cap {self.limit}"
        )


class GarciaRelationSizeLimitExceeded(RuntimeError):
    """Raised after a raw checkpoint but before an oversized global audit."""

    def __init__(self, *, kind: str, limit: int, observed: int) -> None:
        self.kind = str(kind)
        self.limit = int(limit)
        self.observed = int(observed)
        super().__init__(
            f"{self.kind} count {self.observed} exceeds explicit cap {self.limit}"
        )


def garcia_failure_reason_is_explicit_domain_exit(reason: str) -> bool:
    """Classify only failures that explicitly certify departure from domain."""

    if not isinstance(reason, str):
        return False
    return any(
        marker in reason
        for marker in (
            "leaves the declared state-space bounds",
            "exits the declared state space",
            "reset leaves the declared state-space bounds",
        )
    )


def garcia_non_domain_failure_sources(
    provenance: Mapping[int, "GarciaSourceExitProvenance"],
    sources: Collection[int],
) -> frozenset[int]:
    """Return sources with at least one failure not certified as domain exit."""

    return frozenset(
        source
        for source in sources
        if any(
            count and not garcia_failure_reason_is_explicit_domain_exit(reason)
            for reason, count in provenance[source].failure_reasons
        )
    )


_RAW_RELATION_ESTIMATED_BYTES_PER_CELL = 4096
_RAW_RELATION_ESTIMATED_BYTES_PER_EDGE = 192
_SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_CELL = 192
_SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_EDGE = 48


def _estimated_raw_relation_storage_bytes(vertex_count: int, edge_count: int) -> int:
    """Conservative CPython estimate for relation plus source provenance.

    The estimate deliberately includes substantially more than the adjacency
    integers themselves: frozenset hash-table capacity, per-source dictionaries,
    geometry, and open-exit provenance.  It is a pre-audit safety bound, not a
    claim about allocator-exact resident memory.
    """

    return (
        _RAW_RELATION_ESTIMATED_BYTES_PER_CELL * int(vertex_count)
        + _RAW_RELATION_ESTIMATED_BYTES_PER_EDGE * int(edge_count)
    )


@dataclass(frozen=True)
class GarciaSparseConnectivityAudit:
    """Memory-bounded exact connectivity result for a dyadic Atlas relation."""

    disconnected_image_components: tuple[
        tuple[int, tuple[tuple[int, ...], ...]], ...
    ]
    relation_nonempty_images: int
    relation_edges: int
    same_chart_undirected_adjacencies: int
    quotient_undirected_adjacencies: int
    total_undirected_adjacencies: int
    estimated_raw_relation_storage_bytes: int
    estimated_peak_adjacency_storage_bytes: int
    algorithm: str

    @property
    def disconnected_sources(self) -> frozenset[int]:
        return frozenset(source for source, _components in self.disconnected_image_components)


def _relation_strongly_connected_components(
    relation: Mapping[int, Collection[int]],
    vertices: Collection[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Compute SCCs without copying relation edges into a graph object.

    This is an iterative Tarjan traversal.  Apart from the returned components,
    it allocates only arrays and stacks whose size is linear in the number of
    relation vertices.  In particular, the (potentially very large) MapGraph
    edge set remains in its existing adjacency containers.
    """

    count = len(relation)
    if set(relation) != set(range(count)):
        raise ValueError("relation sources must be consecutive integers")
    if vertices is None:
        ordered_vertices: Collection[int] = range(count)
        allowed: frozenset[int] | None = None
    else:
        allowed = frozenset(vertices)
        if any(type(vertex) is not int or vertex < 0 or vertex >= count for vertex in allowed):
            raise ValueError("SCC vertex subset references an unknown relation cell")
        ordered_vertices = sorted(allowed)

    indices = [-1] * count
    lowlinks = [0] * count
    on_stack = bytearray(count)
    tarjan_stack: list[int] = []
    components: list[tuple[int, ...]] = []
    next_index = 0

    # A frame is [vertex, iterator, parent].  Lists keep the frame mutable while
    # avoiding Python recursion at the depth-16/20 relation sizes.
    for root in ordered_vertices:
        if indices[root] >= 0:
            continue
        indices[root] = next_index
        lowlinks[root] = next_index
        next_index += 1
        tarjan_stack.append(root)
        on_stack[root] = 1
        frames: list[list[object]] = [[root, iter(relation[root]), None]]
        while frames:
            vertex = int(frames[-1][0])
            iterator = frames[-1][1]
            parent = frames[-1][2]
            try:
                target = next(iterator)  # type: ignore[arg-type]
            except StopIteration:
                frames.pop()
                if parent is not None:
                    parent_index = int(parent)
                    lowlinks[parent_index] = min(
                        lowlinks[parent_index], lowlinks[vertex]
                    )
                if lowlinks[vertex] == indices[vertex]:
                    component: list[int] = []
                    while True:
                        member = tarjan_stack.pop()
                        on_stack[member] = 0
                        component.append(member)
                        if member == vertex:
                            break
                    components.append(tuple(sorted(component)))
                continue

            if type(target) is not int or target < 0 or target >= count:
                raise ValueError("relation references an unknown target")
            if allowed is not None and target not in allowed:
                continue
            if indices[target] < 0:
                indices[target] = next_index
                lowlinks[target] = next_index
                next_index += 1
                tarjan_stack.append(target)
                on_stack[target] = 1
                frames.append([target, iter(relation[target]), vertex])
            elif on_stack[target]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[target])

    components.sort(key=lambda component: (component[0], component))
    return tuple(components)


def _relation_recurrent_components(
    relation: Mapping[int, Collection[int]],
    vertices: Collection[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return canonical recurrent SCCs of the full or induced relation."""

    recurrent = tuple(
        component
        for component in _relation_strongly_connected_components(relation, vertices)
        if len(component) > 1 or component[0] in relation[component[0]]
    )
    return recurrent


class _DyadicTrieNode:
    """One compact node in a fixed four-dimensional dyadic prefix trie."""

    __slots__ = ("children", "leaf_index")

    def __init__(self) -> None:
        self.children: dict[int, _DyadicTrieNode] = {}
        self.leaf_index: int | None = None


def _dyadic_child_code(cell: GarciaDyadicCell, level: int) -> int:
    shift = cell.axis_depth - level - 1
    code = 0
    for axis, coordinate in enumerate(cell.coordinates):
        code |= ((coordinate >> shift) & 1) << axis
    return code


def _dyadic_coordinate_child_code(
    coordinates: Sequence[int],
    axis_depth: int,
    level: int,
) -> int:
    shift = axis_depth - level - 1
    code = 0
    for axis, coordinate in enumerate(coordinates):
        code |= ((int(coordinate) >> shift) & 1) << axis
    return code


def _allowed_descendant_codes(offset: tuple[int, int, int, int]) -> tuple[int, ...]:
    choices = tuple(
        (0,) if value > 0 else (1,) if value < 0 else (0, 1)
        for value in offset
    )
    return tuple(
        sum(bit << axis for axis, bit in enumerate(bits))
        for bits in itertools.product(*choices)
    )


def _same_chart_sparse_adjacency(
    dyadic_cells: Sequence[GarciaDyadicCell],
    *,
    max_undirected_adjacencies: int | None,
    store_adjacency: bool = True,
) -> tuple[list[list[int]] | None, int, str]:
    """Build exact closed-leaf adjacency without rectangle-pair scans.

    Uniform families use 80 hash probes per leaf.  A mixed antichain uses a
    16-ary dyadic prefix trie.  For a leaf ``C`` at depth ``d``, every touching
    finer leaf lies below one of the 80 neighboring depth-``d`` regions.  On a
    nonzero offset axis its remaining bits are forced to the contacting face;
    zero-offset axes are unconstrained.  Each same-depth/coarser pair is emitted
    from its shallower endpoint, so every undirected edge is produced once.
    """

    count = len(dyadic_cells)
    adjacency: list[list[int]] | None = (
        [[] for _ in range(count)] if store_adjacency else None
    )
    depths = {cell.axis_depth for cell in dyadic_cells}
    undirected = 0

    def add_edge(first: int, second: int) -> None:
        nonlocal undirected
        if first == second:
            return
        # The uniform and trie traversals each emit an undirected pair exactly
        # once.  Lists avoid the several-hundred-MiB hash-table overhead that
        # 4D set adjacency would incur at depth 16.
        if adjacency is not None:
            adjacency[first].append(second)
            adjacency[second].append(first)
        undirected += 1
        if (
            max_undirected_adjacencies is not None
            and undirected > max_undirected_adjacencies
        ):
            raise GarciaRelationSizeLimitExceeded(
                kind="sparse_spatial_adjacency",
                limit=max_undirected_adjacencies,
                observed=undirected,
            )

    offsets = tuple(
        tuple(int(value) for value in offset)
        for offset in itertools.product((-1, 0, 1), repeat=4)
        if any(offset)
    )
    if len(depths) <= 1:
        lookup = {
            (cell.chart_id, cell.axis_depth, cell.coordinates): index
            for index, cell in enumerate(dyadic_cells)
        }
        for index, cell in enumerate(dyadic_cells):
            subdivisions = 2**cell.axis_depth
            for offset in offsets:
                coordinates = tuple(
                    coordinate + delta
                    for coordinate, delta in zip(cell.coordinates, offset, strict=True)
                )
                if any(value < 0 or value >= subdivisions for value in coordinates):
                    continue
                neighbor = lookup.get((cell.chart_id, cell.axis_depth, coordinates))
                if neighbor is not None and neighbor > index:
                    add_edge(index, neighbor)
        return adjacency, undirected, "uniform_dyadic_80_neighbor_hash_v1"

    roots = {BASE_CHART_ID: _DyadicTrieNode(), HANDLE_CHART_ID: _DyadicTrieNode()}
    for index, cell in enumerate(dyadic_cells):
        node = roots[cell.chart_id]
        for level in range(cell.axis_depth):
            if node.leaf_index is not None:
                raise ValueError("dyadic family is not an antichain")
            code = _dyadic_child_code(cell, level)
            node = node.children.setdefault(code, _DyadicTrieNode())
        if node.leaf_index is not None or node.children:
            raise ValueError("dyadic family is not an antichain")
        node.leaf_index = index

    allowed_codes = {offset: _allowed_descendant_codes(offset) for offset in offsets}

    def emit_descendant_leaves(
        node: _DyadicTrieNode,
        source: int,
        codes: tuple[int, ...],
    ) -> None:
        if node.leaf_index is not None:
            add_edge(source, node.leaf_index)
            return
        for code in codes:
            child = node.children.get(code)
            if child is not None:
                emit_descendant_leaves(child, source, codes)

    for index, cell in enumerate(dyadic_cells):
        subdivisions = 2**cell.axis_depth
        for offset in offsets:
            neighboring_coordinates = tuple(
                coordinate + delta
                for coordinate, delta in zip(cell.coordinates, offset, strict=True)
            )
            if any(
                value < 0 or value >= subdivisions
                for value in neighboring_coordinates
            ):
                continue
            node = roots[cell.chart_id]
            coarser_leaf = False
            for level in range(cell.axis_depth):
                if node.leaf_index is not None:
                    # This edge is emitted when the coarser leaf is the source.
                    coarser_leaf = True
                    break
                code = _dyadic_coordinate_child_code(
                    neighboring_coordinates,
                    cell.axis_depth,
                    level,
                )
                child = node.children.get(code)
                if child is None:
                    coarser_leaf = True
                    break
                node = child
            if coarser_leaf:
                continue
            if node.leaf_index is not None:
                if node.leaf_index > index:
                    add_edge(index, node.leaf_index)
                continue
            emit_descendant_leaves(node, index, allowed_codes[offset])
    return adjacency, undirected, "mixed_antichain_dyadic_trie_v1"


def audit_garcia_sparse_relation_connectivity(
    relation: Mapping[int, Collection[int]],
    dyadic_cells: Sequence[GarciaDyadicCell],
    quotient_neighbor_pairs: Collection[tuple[int, int]],
    *,
    max_relation_edges: int | None = None,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
) -> GarciaSparseConnectivityAudit:
    """Audit every nonempty image using exact sparse grid/quotient incidence."""

    for name, cap in (
        ("max_relation_edges", max_relation_edges),
        ("max_relation_storage_bytes", max_relation_storage_bytes),
        ("max_undirected_adjacencies", max_undirected_adjacencies),
        ("max_adjacency_storage_bytes", max_adjacency_storage_bytes),
    ):
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError(f"{name} must be a nonnegative exact integer or None")
    if set(relation) != set(range(len(dyadic_cells))):
        raise ValueError("relation sources must index the complete dyadic family")
    relation_edges = sum(len(targets) for targets in relation.values())
    if max_relation_edges is not None and relation_edges > max_relation_edges:
        raise GarciaRelationSizeLimitExceeded(
            kind="raw_relation_edge",
            limit=max_relation_edges,
            observed=relation_edges,
        )
    relation_metadata = getattr(relation, "metadata", None)
    if isinstance(relation_metadata, Mapping) and type(
        relation_metadata.get("payload_bytes")
    ) is int:
        # The mmap-backed CSR retains no Python container per edge.  Keep the
        # conservative per-source geometry/provenance allowance, but account
        # for adjacency by its exact persisted byte count.
        estimated_relation_bytes = (
            _RAW_RELATION_ESTIMATED_BYTES_PER_CELL * len(dyadic_cells)
            + int(relation_metadata["payload_bytes"])
        )
    else:
        estimated_relation_bytes = _estimated_raw_relation_storage_bytes(
            len(dyadic_cells), relation_edges
        )
    if (
        max_relation_storage_bytes is not None
        and estimated_relation_bytes > max_relation_storage_bytes
    ):
        raise GarciaRelationSizeLimitExceeded(
            kind="raw_relation_estimated_storage_byte",
            limit=max_relation_storage_bytes,
            observed=estimated_relation_bytes,
        )
    base_adjacency_bytes = (
        _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_CELL * len(dyadic_cells)
    )
    memory_edge_limit: int | None = None
    if max_adjacency_storage_bytes is not None:
        if max_adjacency_storage_bytes < base_adjacency_bytes:
            raise GarciaRelationSizeLimitExceeded(
                kind="sparse_adjacency_storage_byte",
                limit=max_adjacency_storage_bytes,
                observed=base_adjacency_bytes,
            )
        memory_edge_limit = (
            max_adjacency_storage_bytes - base_adjacency_bytes
        ) // _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_EDGE
    effective_adjacency_limit = max_undirected_adjacencies
    if memory_edge_limit is not None:
        effective_adjacency_limit = (
            memory_edge_limit
            if effective_adjacency_limit is None
            else min(effective_adjacency_limit, memory_edge_limit)
        )
    try:
        mutable_adjacency, same_chart_edges, algorithm = _same_chart_sparse_adjacency(
            dyadic_cells,
            max_undirected_adjacencies=effective_adjacency_limit,
        )
    except GarciaRelationSizeLimitExceeded as error:
        if memory_edge_limit is not None and error.limit == memory_edge_limit:
            raise GarciaRelationSizeLimitExceeded(
                kind="sparse_adjacency_storage_byte",
                limit=int(max_adjacency_storage_bytes),
                observed=(
                    base_adjacency_bytes
                    + _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_EDGE * error.observed
                ),
            ) from error
        raise
    if mutable_adjacency is None:  # pragma: no cover - internal API invariant
        raise AssertionError("connectivity audit requires stored sparse adjacency")
    quotient_edges = 0
    previous_pair: tuple[int, int] | None = None
    for first_raw, second_raw in quotient_neighbor_pairs:
        first, second = int(first_raw), int(second_raw)
        normalized = (min(first, second), max(first, second))
        if previous_pair is not None and normalized <= previous_pair:
            raise ValueError("quotient-neighbor pairs must be sorted and duplicate-free")
        previous_pair = normalized
        if (
            first < 0
            or second < 0
            or first >= len(mutable_adjacency)
            or second >= len(mutable_adjacency)
        ):
            raise ValueError("quotient-neighbor pair references an unknown cell")
        if first == second:
            raise ValueError("quotient-neighbor pair contains a self-edge")
        # Same-chart and quotient adjacencies are disjoint by construction.
        mutable_adjacency[first].append(second)
        mutable_adjacency[second].append(first)
        quotient_edges += 1
        total = same_chart_edges + quotient_edges
        if (
            effective_adjacency_limit is not None
            and total > effective_adjacency_limit
        ):
            if memory_edge_limit is not None and effective_adjacency_limit == memory_edge_limit:
                raise GarciaRelationSizeLimitExceeded(
                    kind="sparse_adjacency_storage_byte",
                    limit=int(max_adjacency_storage_bytes),
                    observed=(
                        base_adjacency_bytes
                        + _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_EDGE * total
                    ),
                )
            raise GarciaRelationSizeLimitExceeded(
                kind="sparse_spatial_adjacency",
                limit=effective_adjacency_limit,
                observed=total,
            )
    adjacency = tuple(tuple(neighbors) for neighbors in mutable_adjacency)
    del mutable_adjacency

    disconnected: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    nonempty = 0
    for source in sorted(relation):
        targets = frozenset(int(value) for value in relation[source])
        if not targets:
            continue
        nonempty += 1
        if any(target < 0 or target >= len(adjacency) for target in targets):
            raise ValueError(f"relation source {source} references an unknown target")
        remaining = set(targets)
        components: list[tuple[int, ...]] = []
        while remaining:
            root = min(remaining)
            remaining.remove(root)
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                neighbors = remaining.intersection(adjacency[current])
                if neighbors:
                    remaining.difference_update(neighbors)
                    component.update(neighbors)
                    stack.extend(neighbors)
            components.append(tuple(sorted(component)))
        components.sort(key=lambda component: component[0])
        if len(components) != 1:
            disconnected.append((source, tuple(components)))
    return GarciaSparseConnectivityAudit(
        disconnected_image_components=tuple(disconnected),
        relation_nonempty_images=nonempty,
        relation_edges=relation_edges,
        same_chart_undirected_adjacencies=same_chart_edges,
        quotient_undirected_adjacencies=quotient_edges,
        total_undirected_adjacencies=same_chart_edges + quotient_edges,
        estimated_raw_relation_storage_bytes=estimated_relation_bytes,
        estimated_peak_adjacency_storage_bytes=(
            base_adjacency_bytes
            + _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_EDGE
            * (same_chart_edges + quotient_edges)
        ),
        algorithm=algorithm,
    )


def full_garcia_dyadic_family(axis_depth: int) -> GarciaMixedDyadicFamily:
    """Return the complete two-chart uniform cover at ``axis_depth``."""

    if axis_depth < 0:
        raise ValueError("axis_depth must be non-negative")
    subdivisions = 2**axis_depth
    cells = tuple(
        GarciaDyadicCell(chart_id, axis_depth, tuple(coordinates))
        for chart_id in (BASE_CHART_ID, HANDLE_CHART_ID)
        for coordinates in itertools.product(range(subdivisions), repeat=4)
    )
    return GarciaMixedDyadicFamily(cells, role="complete coarse ambient cover")


@dataclass(frozen=True)
class GarciaCoverageRecord:
    """Raw geometric coverage of one callback value before MapGraph clipping."""

    source_chart_id: int
    source_bounds: tuple[float, ...]
    returned_pieces: int
    active_target_cells: int
    missing_in_domain_cells: frozenset[GarciaDyadicCell]
    ambient_boundary_pieces: int
    wholly_outside_active_family_pieces: int
    target_pieces: tuple[tuple[int, tuple[float, ...]], ...]
    callback_invocations: int

    @property
    def explicit_empty_callback(self) -> bool:
        return self.returned_pieces == 0

    @property
    def active_boundary_exit(self) -> bool:
        return bool(self.missing_in_domain_cells)

    @property
    def ambient_boundary_exit(self) -> bool:
        return self.ambient_boundary_pieces > 0


class _AuditedGarciaSuspensionBoxMap(CMGDBSuspensionBoxMap):
    """Capture one compact diagnostic record for every distinct source box."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_lock = threading.Lock()
        self._source_records: dict[tuple[object, ...], SourceBoxMapDiagnostic] = {}
        self._source_invocations: Counter[tuple[object, ...]] = Counter()
        self._source_value_cache: dict[
            tuple[object, ...],
            tuple[tuple[tuple[int, tuple[float, ...]], ...], SourceBoxMapDiagnostic],
        ] = {}
        self._source_cache_hits = 0
        self._source_cache_misses = 0
        self._point_lock = threading.Lock()
        self._point_cache: dict[
            tuple[object, ...],
            tuple[bool, object],
        ] = {}
        self._point_cache_hits = 0
        self._point_cache_misses = 0
        self._precomputed_unclaimed: set[tuple[object, ...]] = set()
        self._endpoint_precompute_result: GarciaEndpointPrecomputeResult | None = None

    def reset_source_records(self, *, preserve_evaluation_cache: bool = False) -> None:
        with self._source_lock:
            self._source_records = {}
            self._source_invocations = Counter()
            self._source_cache_hits = 0
            self._source_cache_misses = 0
            if not preserve_evaluation_cache:
                self._source_value_cache = {}
        with self._point_lock:
            self._point_cache_hits = 0
            self._endpoint_precompute_result = None
            if not preserve_evaluation_cache:
                self._point_cache = {}
                self._point_cache_misses = 0
                self._precomputed_unclaimed = set()

    def source_value_cache_keys(self) -> frozenset[tuple[object, ...]]:
        with self._source_lock:
            return frozenset(self._source_value_cache)

    def source_cache_counts(self) -> tuple[int, int]:
        """Return ``(new source values, reused source values)`` this run."""

        with self._source_lock:
            return self._source_cache_misses, self._source_cache_hits

    def seed_point_cache(
        self,
        entries: Sequence[EndpointCacheEntry],
        *,
        precompute_result: GarciaEndpointPrecomputeResult | None = None,
    ) -> None:
        """Install deterministic precomputed values before callback traversal."""

        seeded: dict[tuple[object, ...], tuple[bool, object]] = {}
        for key, cached in entries:
            normalized_key = tuple(key)
            if normalized_key in seeded:
                raise ValueError("precomputed endpoint cache contains duplicate keys")
            succeeded, value = cached
            if not isinstance(succeeded, bool):
                raise TypeError("precomputed endpoint status must be boolean")
            seeded[normalized_key] = (succeeded, value)
        with self._point_lock:
            if self._point_cache or self._point_cache_hits or self._point_cache_misses:
                raise RuntimeError("endpoint cache must be empty before it is seeded")
            self._point_cache = seeded
            # These points were physically evaluated in the worker processes;
            # count them as unique integrations, not as free cache hits.
            self._point_cache_misses = len(seeded)
            self._point_cache_hits = 0
            # The first ordinary callback access claims the already-counted
            # integration.  Only later accesses are lattice reuses, preserving
            # the serial diagnostics identity ``unique + reused = sampled``.
            self._precomputed_unclaimed = set(seeded)
            self._endpoint_precompute_result = precompute_result

    def extend_point_cache(
        self,
        entries: Sequence[EndpointCacheEntry],
        *,
        precompute_result: GarciaEndpointPrecomputeResult | None = None,
    ) -> None:
        """Merge newly precomputed nodes into a preserved same-model cache."""

        with self._point_lock:
            new_keys: set[tuple[object, ...]] = set()
            for key, cached in entries:
                normalized_key = tuple(key)
                previous = self._point_cache.get(normalized_key)
                if previous is not None:
                    continue
                self._point_cache[normalized_key] = cached
                new_keys.add(normalized_key)
            self._point_cache_misses += len(new_keys)
            self._precomputed_unclaimed.update(new_keys)
            self._endpoint_precompute_result = precompute_result

    def endpoint_precompute_result(self) -> GarciaEndpointPrecomputeResult | None:
        with self._point_lock:
            return self._endpoint_precompute_result

    def point_cache_counts(self) -> tuple[int, int]:
        """Return ``(unique integrations, reused lattice samples)``."""

        with self._point_lock:
            return self._point_cache_misses, self._point_cache_hits

    def release_evaluation_caches(self) -> None:
        """Release ODE/source caches after their atomic raw checkpoint exists.

        Raw counters and endpoint-precompute provenance must be captured first.
        The remaining postprocessing only uses the map's stateless endpoint
        encoder, so retaining millions of endpoint objects would add memory
        pressure without changing any derived audit.
        """

        with self._source_lock:
            self._source_records = {}
            self._source_invocations = Counter()
            self._source_value_cache = {}
            self._source_cache_hits = 0
            self._source_cache_misses = 0
        with self._point_lock:
            self._point_cache = {}
            self._precomputed_unclaimed = set()
            self._point_cache_hits = 0
            self._point_cache_misses = 0
            self._endpoint_precompute_result = None
        self.reset_diagnostics()

    def _evaluate_point(self, source_chart_id: int, point: np.ndarray):
        key = _source_key(source_chart_id, point)
        # Three samples per axis reuse the closed faces of neighboring dyadic
        # boxes.  Cache the deterministic physical endpoint at those shared
        # nodes; source-box hulling and diagnostics are still recomputed from
        # the complete 3^4 logical tensor for every cell.
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
                self._point_cache[key] = (False, (type(error), error.args))
                raise
            self._point_cache[key] = (True, value)
            return value

    def source_records(
        self,
    ) -> tuple[
        dict[tuple[object, ...], SourceBoxMapDiagnostic],
        dict[tuple[object, ...], int],
    ]:
        with self._source_lock:
            return dict(self._source_records), dict(self._source_invocations)

    def _record(self, record: SourceBoxMapDiagnostic) -> None:
        key = _source_key(record.source_chart_id, record.source_bounds)
        with self._source_lock:
            self._source_records[key] = record
            self._source_invocations[key] += 1
        super()._record(record)

    def __call__(
        self,
        source_chart_id: int,
        source_bounds: Sequence[float],
    ) -> list[tuple[int, list[float]]]:
        """Replay deterministic whole-source values across saturation rounds."""

        key = _source_key(source_chart_id, source_bounds)
        with self._source_lock:
            cached = self._source_value_cache.get(key)
            if cached is not None:
                self._source_cache_hits += 1
        if cached is not None:
            pieces, record = cached
            self._record(record)
            return [
                (chart_id, [float(value) for value in bounds])
                for chart_id, bounds in pieces
            ]

        pieces = super().__call__(source_chart_id, source_bounds)
        encoded = tuple(
            (int(chart_id), tuple(float(value) for value in bounds))
            for chart_id, bounds in pieces
        )
        with self._source_lock:
            record = self._source_records[key]
            self._source_value_cache[key] = (encoded, record)
            self._source_cache_misses += 1
        return pieces


class _OpenExitTrackingBoxMap:
    """Record active-family and ambient exits without adding a cemetery cell."""

    def __init__(
        self,
        box_map: _AuditedGarciaSuspensionBoxMap,
        family: GarciaMixedDyadicFamily,
        quotient_incidence: GarciaWalkerQuotientIncidence,
    ) -> None:
        self.box_map = box_map
        self.charts = box_map.charts
        self.family = family
        self.quotient_incidence = quotient_incidence
        self._atlas: Any | None = None
        self._lock = threading.Lock()
        self._records: dict[tuple[object, ...], GarciaCoverageRecord] = {}
        self._attachment_cache: dict[tuple[int, int], frozenset[int]] = {}
        self._quotient_neighbors_by_handle: dict[int, frozenset[int]] = {}
        self.quotient_neighbor_pairs: tuple[tuple[int, int], ...] = ()

    def bind_atlas(self, atlas: Any) -> None:
        self._atlas = atlas
        cells = tuple(
            AtlasWalkerCell(
                index,
                int(atlas.cell(index).chart_id),
                tuple(float(value) for value in atlas.cell(index).bounds),
            )
            for index in range(int(atlas.size()))
        )
        self.quotient_neighbor_pairs = _quotient_cross_chart_pairs(
            cells,
            self.quotient_incidence,
            self.charts,
            atlas,
        )
        neighbors: dict[int, set[int]] = {}
        for first, second in self.quotient_neighbor_pairs:
            handle = first if cells[first].chart_id == self.charts.handle_chart_id else second
            base = second if handle == first else first
            neighbors.setdefault(handle, set()).add(base)
        self._quotient_neighbors_by_handle = {
            handle: frozenset(base_cells)
            for handle, base_cells in neighbors.items()
        }

    def reset_records(self) -> None:
        with self._lock:
            self._records = {}

    def records(self) -> dict[tuple[object, ...], GarciaCoverageRecord]:
        with self._lock:
            return dict(self._records)

    def __call__(
        self,
        source_chart_id: int,
        source_bounds: Sequence[float],
    ) -> list[tuple[int, list[float]]]:
        if self._atlas is None:
            raise RuntimeError("open-exit tracker has no bound Atlas")
        key = _source_key(source_chart_id, source_bounds)
        # Compute each deterministic finite box-map value once.  CMGDB asks for
        # every source again while materializing the returned MapGraph; replaying
        # the exact tagged rectangles avoids a second set of ODE integrations.
        # Holding this lock during evaluation also prevents duplicate work if a
        # future CMGDB traversal invokes the Python callback concurrently.
        with self._lock:
            previous = self._records.get(key)
            if previous is not None:
                self._records[key] = replace(
                    previous,
                    callback_invocations=previous.callback_invocations + 1,
                )
                return [
                    (chart_id, list(bounds))
                    for chart_id, bounds in previous.target_pieces
                ]

            pieces = self.box_map(source_chart_id, source_bounds)
            active_targets: set[int] = set()
            missing: set[GarciaDyadicCell] = set()
            ambient_boundary_pieces = 0
            wholly_outside = 0
            encoded_pieces: list[tuple[int, tuple[float, ...]]] = []
            encoded_pieces.extend(
                (int(target_chart), tuple(float(value) for value in target_bounds))
                for target_chart, target_bounds in pieces
            )
            for target_chart, target_bounds in encoded_pieces:
                covered = set(
                    int(value)
                    for value in self._atlas.cover(target_chart, target_bounds)
                )
                active_targets.update(covered)
                if not covered:
                    wholly_outside += 1
                fine_cover, boundary_crossings = _dyadic_rectangle_cover(
                    self.charts,
                    int(target_chart),
                    target_bounds,
                    self.family.max_axis_depth,
                )
                ambient_boundary_pieces += int(
                    _piece_has_unmatched_ambient_exit(
                        self.charts,
                        self.quotient_incidence,
                        self._atlas,
                        target_chart,
                        target_bounds,
                        boundary_crossings,
                        encoded_pieces,
                        self._attachment_cache,
                        self._quotient_neighbors_by_handle,
                    )
                )
                missing.update(
                    cell
                    for cell in fine_cover
                    if self.family.covering_cell(cell) is None
                )

            self._records[key] = GarciaCoverageRecord(
                source_chart_id=int(source_chart_id),
                source_bounds=tuple(float(value) for value in source_bounds),
                returned_pieces=len(pieces),
                active_target_cells=len(active_targets),
                missing_in_domain_cells=frozenset(missing),
                ambient_boundary_pieces=ambient_boundary_pieces,
                wholly_outside_active_family_pieces=wholly_outside,
                target_pieces=tuple(encoded_pieces),
                callback_invocations=1,
            )
        return pieces


def _dyadic_rectangle_cover(
    charts: SuspensionAtlasCharts,
    chart_id: int,
    flat_bounds: Sequence[float],
    axis_depth: int,
) -> tuple[frozenset[GarciaDyadicCell], tuple[tuple[int, int], ...]]:
    chart_bounds = np.asarray(charts.bounds_for(chart_id), dtype=np.float64)
    values = np.asarray(flat_bounds, dtype=np.float64)
    dimension = len(chart_bounds)
    lower = values[:dimension]
    upper = values[dimension:]
    scale = chart_bounds[:, 1] - chart_bounds[:, 0]
    normalized_lower = (lower - chart_bounds[:, 0]) / scale
    normalized_upper = (upper - chart_bounds[:, 0]) / scale
    tolerance = 1.0e-12
    boundary_crossings = tuple(
        (axis, side)
        for axis, (low, high) in enumerate(
            zip(normalized_lower, normalized_upper, strict=True)
        )
        for side in (
            *((-1,) if low < -tolerance else ()),
            *((1,) if high > 1.0 + tolerance else ()),
        )
    )
    normalized_lower = np.maximum(normalized_lower, 0.0)
    normalized_upper = np.minimum(normalized_upper, 1.0)
    if np.any(normalized_lower > normalized_upper + tolerance):
        return frozenset(), boundary_crossings
    subdivisions = 2**axis_depth
    ranges = []
    for low, high in zip(normalized_lower, normalized_upper):
        first = max(0, int(np.ceil(subdivisions * low - tolerance)) - 1)
        last = min(
            subdivisions - 1,
            int(np.floor(subdivisions * high + tolerance)),
        )
        ranges.append(range(first, last + 1))
    return (
        frozenset(
            GarciaDyadicCell(chart_id, axis_depth, tuple(coordinates))
            for coordinates in itertools.product(*ranges)
        ),
        boundary_crossings,
    )


def _piece_has_unmatched_ambient_exit(
    charts: SuspensionAtlasCharts,
    incidence: GarciaWalkerQuotientIncidence,
    atlas: Any,
    chart_id: int,
    flat_bounds: Sequence[float],
    boundary_crossings: Collection[tuple[int, int]],
    all_pieces: Collection[tuple[int, tuple[float, ...]]],
    attachment_cache: dict[tuple[int, int], frozenset[int]] | None = None,
    quotient_neighbors_by_handle: Mapping[int, frozenset[int]] | None = None,
) -> bool:
    """Distinguish open chart exits from represented handle quotient seams."""

    unmatched = set(boundary_crossings)
    if chart_id != charts.handle_chart_id or not unmatched:
        return bool(unmatched)
    handle_bounds = list(float(value) for value in flat_bounds)
    returned_base_targets = frozenset(
        int(index)
        for piece_chart, piece_bounds in all_pieces
        if piece_chart == charts.base_chart_id
        for index in atlas.cover(piece_chart, piece_bounds)
    )
    for side in (-1, 1):
        crossing = (3, side)
        if crossing not in unmatched:
            continue
        seam_phase = 0.0 if side < 0 else 1.0
        seam_bounds = list(handle_bounds)
        seam_bounds[3] = seam_phase
        seam_bounds[7] = seam_phase
        seam_handle_cells = tuple(
            AtlasWalkerCell(
                int(index),
                charts.handle_chart_id,
                tuple(float(value) for value in atlas.cell(int(index)).bounds),
            )
            for index in atlas.cover(charts.handle_chart_id, seam_bounds)
        )
        attachments: set[int] = set()
        covered = bool(seam_handle_cells)
        for handle_cell in seam_handle_cells:
            cache_key = (handle_cell.index, side)
            if quotient_neighbors_by_handle is not None:
                neighbors = set(
                    quotient_neighbors_by_handle.get(handle_cell.index, frozenset())
                )
                covered = covered and bool(neighbors)
                attachments.update(neighbors)
                continue
            if attachment_cache is not None and cache_key in attachment_cache:
                neighbors = set(attachment_cache[cache_key])
                covered = covered and bool(neighbors)
                attachments.update(neighbors)
                continue
            broad_hull = incidence.attachment_hull(handle_cell, side)
            neighbors = {
                int(index)
                for index in atlas.cover(charts.base_chart_id, broad_hull)
                if incidence.intersects(
                    AtlasWalkerCell(
                        int(index),
                        charts.base_chart_id,
                        tuple(float(value) for value in atlas.cell(int(index)).bounds),
                    ),
                    handle_cell,
                )
            }
            if attachment_cache is not None:
                attachment_cache[cache_key] = frozenset(neighbors)
            covered = covered and bool(neighbors)
            attachments.update(neighbors)
        if covered and attachments.issubset(returned_base_targets):
            unmatched.remove(crossing)
    return bool(unmatched)


@dataclass(frozen=True)
class GarciaSourceExitProvenance:
    source_index: int
    callback_invocations: int
    successful_samples: int
    failed_samples: int
    failure_reasons: tuple[tuple[str, int], ...]
    unresolved_stage_edges: tuple[tuple[int, int], ...]
    returned_pieces: int
    explicit_empty_callback: bool
    mapgraph_empty_image: bool
    active_target_cells: int
    missing_in_domain_target_cells: int
    missing_in_domain_witnesses: tuple[GarciaDyadicCell, ...]
    target_pieces: tuple[tuple[int, tuple[float, ...]], ...]
    ambient_boundary_pieces: int
    wholly_outside_active_family_pieces: int

    @property
    def has_sample_failure(self) -> bool:
        return self.failed_samples > 0

    @property
    def has_open_exit(self) -> bool:
        return bool(
            self.has_sample_failure
            or self.unresolved_stage_edges
            or self.explicit_empty_callback
            or self.mapgraph_empty_image
            or self.missing_in_domain_target_cells
            or self.ambient_boundary_pieces
            or self.wholly_outside_active_family_pieces
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "callback_invocations": self.callback_invocations,
            "successful_samples_per_invocation": self.successful_samples,
            "failed_samples_per_invocation": self.failed_samples,
            "failure_reasons": dict(self.failure_reasons),
            "unresolved_stage_edges": [
                list(edge) for edge in self.unresolved_stage_edges
            ],
            "returned_pieces": self.returned_pieces,
            "explicit_empty_callback": self.explicit_empty_callback,
            "mapgraph_empty_image": self.mapgraph_empty_image,
            "active_target_cells": self.active_target_cells,
            "missing_in_domain_target_cells": self.missing_in_domain_target_cells,
            "missing_in_domain_witnesses": [
                {
                    "chart_id": cell.chart_id,
                    "axis_depth": cell.axis_depth,
                    "coordinates": list(cell.coordinates),
                }
                for cell in self.missing_in_domain_witnesses
            ],
            "target_pieces": [
                {"chart_id": chart_id, "bounds": list(bounds)}
                for chart_id, bounds in self.target_pieces
            ],
            "ambient_boundary_pieces": self.ambient_boundary_pieces,
            "wholly_outside_active_family_pieces": (
                self.wholly_outside_active_family_pieces
            ),
            "has_open_exit": self.has_open_exit,
        }


@dataclass(frozen=True)
class GarciaLocalCandidateAudit:
    morse_node: int | None
    s_cells: frozenset[int]
    x_cells: frozenset[int]
    a_cells: frozenset[int]
    reference_source_cells: frozenset[int]
    reference_source_cells_in_candidate: frozenset[int]
    reference_labels_total: int
    reference_labels_recovered: int
    reference_missing_labels: tuple[str, ...]
    reference_endpoint_misses: int
    reference_evaluation_failures: int
    failed_sources_in_s: tuple[int, ...]
    unresolved_stage_sources_in_s: tuple[int, ...]
    open_exit_sources_in_s: tuple[int, ...]
    failed_sources_in_a: tuple[int, ...]
    empty_sources_in_s: tuple[int, ...]
    disconnected_sources_in_s: tuple[int, ...]
    disconnected_sources_in_a: tuple[int, ...]
    disconnected_image_components: tuple[
        tuple[int, tuple[tuple[int, ...], ...]], ...
    ]
    missing_in_domain_sources_in_s: tuple[int, ...]
    missing_in_domain_sources_in_x: tuple[int, ...]
    a_exit_sources: tuple[int, ...]
    pair_second_condition_violations: tuple[int, ...]
    recurrent_components_in_x: tuple[tuple[int, ...], ...]
    touches_nonglued_ambient_boundary: tuple[int, ...]
    minimum_boundary_margin_by_chart_axis: Mapping[str, tuple[float | None, ...]]

    @property
    def reference_recovered(self) -> bool:
        return bool(
            self.morse_node is not None
            and self.reference_labels_total > 0
            and self.reference_labels_recovered == self.reference_labels_total
            and not self.reference_missing_labels
            and self.reference_endpoint_misses == 0
            and self.reference_evaluation_failures == 0
        )

    @property
    def discovery_gates_passed(self) -> bool:
        return bool(
            self.reference_recovered
            and self.s_cells
            and not self.failed_sources_in_s
            and not self.unresolved_stage_sources_in_s
            and not self.open_exit_sources_in_s
            and not self.empty_sources_in_s
            and not self.disconnected_sources_in_s
            and not self.missing_in_domain_sources_in_s
            and not self.pair_second_condition_violations
            and not self.touches_nonglued_ambient_boundary
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "morse_node": self.morse_node,
            "S": sorted(self.s_cells),
            "X": sorted(self.x_cells),
            "A": sorted(self.a_cells),
            "counts": {
                "S": len(self.s_cells),
                "X": len(self.x_cells),
                "A": len(self.a_cells),
            },
            "reference_source_cells": sorted(self.reference_source_cells),
            "reference_source_cells_in_candidate": sorted(
                self.reference_source_cells_in_candidate
            ),
            "reference_labels_total": self.reference_labels_total,
            "reference_labels_recovered": self.reference_labels_recovered,
            "reference_missing_labels": list(self.reference_missing_labels),
            "reference_endpoint_misses": self.reference_endpoint_misses,
            "reference_evaluation_failures": self.reference_evaluation_failures,
            "reference_recovered": self.reference_recovered,
            "failed_sources_in_S": list(self.failed_sources_in_s),
            "unresolved_stage_sources_in_S": list(
                self.unresolved_stage_sources_in_s
            ),
            "open_exit_sources_in_S": list(self.open_exit_sources_in_s),
            "failed_sources_in_A": list(self.failed_sources_in_a),
            "empty_sources_in_S": list(self.empty_sources_in_s),
            "disconnected_sources_in_S": list(self.disconnected_sources_in_s),
            "disconnected_sources_in_A": list(self.disconnected_sources_in_a),
            "disconnected_image_components": [
                {
                    "source": source,
                    "components": [list(component) for component in components],
                }
                for source, components in self.disconnected_image_components
            ],
            "missing_in_domain_sources_in_X": list(
                self.missing_in_domain_sources_in_x
            ),
            "missing_in_domain_sources_in_S": list(
                self.missing_in_domain_sources_in_s
            ),
            "A_open_exit_sources": list(self.a_exit_sources),
            "pair_second_condition_violations": list(
                self.pair_second_condition_violations
            ),
            "recurrent_components_in_X": [
                list(component) for component in self.recurrent_components_in_x
            ],
            "touches_nonglued_ambient_boundary": list(
                self.touches_nonglued_ambient_boundary
            ),
            "minimum_boundary_margin_by_chart_axis": {
                key: list(value)
                for key, value in self.minimum_boundary_margin_by_chart_axis.items()
            },
            "discovery_gates_passed": self.discovery_gates_passed,
        }


@dataclass(frozen=True)
class GarciaSafeRefinementLocator:
    """A failure-pruned graph locator used only to place adaptive resolution.

    This object is never an invariant-set or acceptance certificate.  The
    actual CMGDB relation remains unpruned; final isolation and pair checks must
    use that relation together with its recorded open exits.
    """

    eligible_cells: frozenset[int]
    disconnected_sources_excluded: frozenset[int]
    recurrent_components: tuple[tuple[int, ...], ...]
    selected_component: frozenset[int]
    reference_source_cells: frozenset[int]
    selected_reference_hits: frozenset[int]
    seed_cells: frozenset[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "garcia-safe-refinement-locator-v1",
            "purpose": "adaptive_resolution_only",
            "scientific_result_accepted": False,
            "acceptance_relation": "complete_unpruned_open_exit_relation",
            "eligibility": (
                "relation evaluated, nonempty image, no recorded open exit, "
                "and quotient-connected image"
            ),
            "eligible_cells": sorted(self.eligible_cells),
            "disconnected_sources_excluded": sorted(
                self.disconnected_sources_excluded
            ),
            "recurrent_components": [
                list(component) for component in self.recurrent_components
            ],
            "selected_component": sorted(self.selected_component),
            "selected_component_cells": len(self.selected_component),
            "reference_source_cells": sorted(self.reference_source_cells),
            "selected_reference_hits": sorted(self.selected_reference_hits),
            "selected_reference_hit_count": len(self.selected_reference_hits),
            "seed_cells": sorted(self.seed_cells),
            "seed_cell_count": len(self.seed_cells),
            "selection_rule": (
                "recurrent SCC maximizing reference-cell hits, then size, "
                "then smallest cell identifier; union every reference source cell"
            ),
        }


@dataclass(frozen=True)
class GarciaExitAwareSupportAudit:
    """Finite support-closure audit for the local candidate pair ``(N, L)``.

    Here ``N = S union F(S)`` and ``L = N - S`` use the actually stored local
    relation.  Only missing in-domain images of ``N - L`` may enlarge the
    active window.  Images and failures on ``L`` remain recorded exits; chasing
    them would turn a local isolating-neighborhood computation back into an
    ambient-domain saturation.
    """

    n_cells: frozenset[int]
    l_cells: frozenset[int]
    n_minus_l_cells: frozenset[int]
    expandable_sources: frozenset[int]
    added_cells: frozenset[GarciaDyadicCell]
    nonexpandable_open_exit_sources: frozenset[int]
    preserved_l_exit_sources: frozenset[int]
    pair_second_condition_violations: frozenset[int]
    terminal_blockers: tuple[str, ...]

    @property
    def support_saturated(self) -> bool:
        return not self.expandable_sources and not self.added_cells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "garcia-open-index-pair-support-audit-v1",
            "policy": (
                "expand only represented in-domain images missing from N for "
                "sources in N\\L; preserve L exits"
            ),
            "N": sorted(self.n_cells),
            "L": sorted(self.l_cells),
            "N_minus_L": sorted(self.n_minus_l_cells),
            "expandable_sources": sorted(self.expandable_sources),
            "added_cells": [
                {
                    "chart_id": cell.chart_id,
                    "axis_depth": cell.axis_depth,
                    "coordinates": list(cell.coordinates),
                }
                for cell in sorted(self.added_cells)
            ],
            "counts": {
                "N": len(self.n_cells),
                "L": len(self.l_cells),
                "N_minus_L": len(self.n_minus_l_cells),
                "expandable_sources": len(self.expandable_sources),
                "added_cells": len(self.added_cells),
                "nonexpandable_open_exit_sources": len(
                    self.nonexpandable_open_exit_sources
                ),
                "preserved_L_exit_sources": len(self.preserved_l_exit_sources),
                "pair_second_condition_violations": len(
                    self.pair_second_condition_violations
                ),
                "terminal_blockers": len(self.terminal_blockers),
            },
            "nonexpandable_open_exit_sources": sorted(
                self.nonexpandable_open_exit_sources
            ),
            "preserved_L_exit_sources": sorted(self.preserved_l_exit_sources),
            "pair_second_condition_violations": sorted(
                self.pair_second_condition_violations
            ),
            "terminal_blockers": list(self.terminal_blockers),
            "support_saturated": self.support_saturated,
            "scientific_result_accepted": False,
        }


@dataclass(frozen=True)
class GarciaLocalRelationRun:
    family: GarciaMixedDyadicFamily
    walker: GarciaPassiveWalker | GuardAlignedGarciaPassiveWalker
    charts: SuspensionAtlasCharts
    box_map: _AuditedGarciaSuspensionBoxMap
    morse_graph: Any
    map_graph: Any
    cells: tuple[AtlasWalkerCell, ...]
    dyadic_cells: tuple[GarciaDyadicCell, ...]
    relation: Mapping[int, Collection[int]]
    node_by_cell: Mapping[int, int]
    quotient_neighbor_pairs: tuple[tuple[int, int], ...]
    source_provenance: Mapping[int, GarciaSourceExitProvenance]
    reference_node_hits: Mapping[int, int]
    reference_probe_count: int
    reference_miss_count: int
    reference_evaluation_failures: tuple[object, ...]
    candidate: GarciaLocalCandidateAudit
    refinement_locator: GarciaSafeRefinementLocator
    elapsed_seconds: float
    t_star: float
    samples_per_axis: int
    padding_cells: float
    max_step: float
    max_jumps: int
    connectivity_audit: GarciaSparseConnectivityAudit | None = None
    raw_checkpoint_path: str | None = None
    raw_compute_metadata: Mapping[str, object] | None = None
    support_saturated: bool | None = None
    support_audit: GarciaExitAwareSupportAudit | None = None
    relation_reference: Mapping[str, object] | None = None
    family_provenance: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        morse_nodes = [int(value) for value in self.morse_graph.vertices()]
        morse_sets = {
            str(node): sorted(int(value) for value in self.morse_graph.morse_set(node))
            for node in morse_nodes
        }
        diagnostics = self.box_map.diagnostics()
        unique_point_evaluations, reused_point_evaluations = (
            self.box_map.point_cache_counts()
        )
        new_source_values, reused_source_values = self.box_map.source_cache_counts()
        precompute = self.box_map.endpoint_precompute_result()
        raw_metrics = dict(self.raw_compute_metadata or {})
        connectivity = self.connectivity_audit
        csr_backed = self.relation_reference is not None
        coordinate_system = _garcia_coordinate_system(self.walker)
        payload: dict[str, object] = {
            "schema": (
                "garcia-walker-open-exit-atlas-relation-v2"
                if csr_backed
                else "garcia-walker-open-exit-atlas-relation-v1"
            ),
            "metadata": {
                "model": "garcia_passive_walker_fixed_time_suspension_atlas",
                "coordinate_system": coordinate_system,
                "relation_scope": "local_open_exit_fixed_time_suspension",
                "legacy_cemetery_used": False,
                "t_star": self.t_star,
                "gamma": self.walker.gamma,
                "guard_delta": self.walker.guard_delta,
                "transversality_eta": self.walker.transversality_eta,
                "samples_per_axis": self.samples_per_axis,
                "padding_cells": self.padding_cells,
                "max_step": self.max_step,
                "max_jumps": self.max_jumps,
                "require_domain_path": True,
                "depth": 4 * self.family.max_axis_depth,
                "coarse_depth": 4 * self.family.min_axis_depth,
                "base_chart_id": self.charts.base_chart_id,
                "handle_chart_id": self.charts.handle_chart_id,
                "family_role": self.family.role,
                "min_axis_depth": self.family.min_axis_depth,
                "max_axis_depth": self.family.max_axis_depth,
                "mixed_depth": self.family.min_axis_depth != self.family.max_axis_depth,
                "active_cells": len(self.family.cells),
                "active_chart_counts": self.family.chart_counts(),
                "elapsed_seconds": self.elapsed_seconds,
                "whole_cell_outer_enclosure_certified": False,
                "continuous_system_conley_index_certified": False,
                "support_saturated": self.support_saturated,
                "support_saturation_policy": "open_index_pair_N_minus_L_v1",
                "cmgdb_callback_invocations": sum(
                    record.callback_invocations
                    for record in self.source_provenance.values()
                ),
                "box_map_unique_source_boxes_evaluated": raw_metrics.get(
                    "box_map_unique_source_boxes_evaluated", diagnostics.source_boxes
                ),
                "new_whole_source_values_evaluated": raw_metrics.get(
                    "new_whole_source_values_evaluated", new_source_values
                ),
                "reused_whole_source_values": raw_metrics.get(
                    "reused_whole_source_values", reused_source_values
                ),
                "box_map_sampled_points_evaluated": raw_metrics.get(
                    "box_map_sampled_points_evaluated", diagnostics.sampled_points
                ),
                "unique_physical_endpoint_evaluations": raw_metrics.get(
                    "unique_physical_endpoint_evaluations", unique_point_evaluations
                ),
                "reused_lattice_endpoint_evaluations": raw_metrics.get(
                    "reused_lattice_endpoint_evaluations", reused_point_evaluations
                ),
                "box_map_failed_samples": raw_metrics.get(
                    "box_map_failed_samples", diagnostics.failed_samples
                ),
                "box_map_empty_images": raw_metrics.get(
                    "box_map_empty_images", diagnostics.empty_images
                ),
                "box_map_unresolved_stage_edges": raw_metrics.get(
                    "box_map_unresolved_stage_edges", diagnostics.unresolved_stage_edges
                ),
                "raw_checkpoint_path": self.raw_checkpoint_path,
                "raw_checkpoint_resumed": bool(raw_metrics.get("raw_checkpoint_resumed")),
                "connectivity_algorithm": (
                    None if connectivity is None else connectivity.algorithm
                ),
                "relation_edges": (
                    sum(len(targets) for targets in self.relation.values())
                    if connectivity is None
                    else connectivity.relation_edges
                ),
                "estimated_raw_relation_storage_bytes": (
                    _estimated_raw_relation_storage_bytes(
                        len(self.relation),
                        sum(len(targets) for targets in self.relation.values()),
                    )
                    if connectivity is None
                    else connectivity.estimated_raw_relation_storage_bytes
                ),
                "relation_storage_representation": (
                    "mmap_csr" if csr_backed else "python_frozenset_rows"
                ),
                "csr_payload_bytes": (
                    None
                    if self.relation_reference is None
                    else self.relation_reference["payload_bytes"]
                ),
                "same_chart_undirected_adjacencies": (
                    None
                    if connectivity is None
                    else connectivity.same_chart_undirected_adjacencies
                ),
                "quotient_undirected_adjacencies": (
                    None
                    if connectivity is None
                    else connectivity.quotient_undirected_adjacencies
                ),
                "estimated_peak_adjacency_storage_bytes": (
                    None
                    if connectivity is None
                    else connectivity.estimated_peak_adjacency_storage_bytes
                ),
                "endpoint_precompute": (
                    raw_metrics.get("endpoint_precompute")
                    if precompute is None
                    else {
                        "logical_tensor_points": precompute.logical_tensor_points,
                        "unique_points": precompute.unique_points,
                        "duplicate_tensor_points": precompute.duplicate_tensor_points,
                        "requested_workers": precompute.requested_workers,
                        "used_workers": precompute.used_workers,
                        "process_count": (
                            precompute.used_workers
                            if precompute.mode == "process_pool"
                            else 0
                        ),
                        "mode": precompute.mode,
                        "elapsed_seconds": precompute.elapsed_seconds,
                        "fallback_reason": precompute.fallback_reason,
                        "parity_provenance": dict(PARITY_PROVENANCE),
                    }
                ),
            },
            "charts": {
                "base": {
                    "chart_id": self.charts.base_chart_id,
                    "coordinates": (
                        ["theta", "omega", "q", "nu"]
                        if coordinate_system == "guard_aligned"
                        else ["theta", "theta_dot", "phi", "phi_dot"]
                    ),
                    "bounds": [list(value) for value in self.charts.base_bounds],
                },
                "handle": {
                    "chart_id": self.charts.handle_chart_id,
                    "coordinates": (
                        ["theta_guard", "omega_guard", "nu_guard", "s"]
                        if coordinate_system == "guard_aligned"
                        else ["theta_guard", "theta_dot_guard", "rho", "s"]
                    ),
                    "bounds": [list(value) for value in self.charts.handle_bounds],
                },
            },
            "reference_audit": {
                "probes": self.reference_probe_count,
                "misses": self.reference_miss_count,
                "evaluation_failures": len(self.reference_evaluation_failures),
                "morse_node_hits": dict(self.reference_node_hits),
            },
            "morse_graph": {
                "nodes": morse_nodes,
                "edges": [
                    [int(source), int(target)]
                    for source, target in sorted(self.morse_graph.edges())
                ],
                "morse_sets": morse_sets,
            },
            # Same-chart rectangle incidence is recoverable directly from the
            # stored bounds.  These are the additional base/handle incidences
            # created by the guard and reset quotient seams.
            "quotient_neighbor_pairs": [
                [first, second] for first, second in self.quotient_neighbor_pairs
            ],
            "candidate": self.candidate.to_dict(),
            "refinement_locator": self.refinement_locator.to_dict(),
            "support_audit": (
                None if self.support_audit is None else self.support_audit.to_dict()
            ),
            "relation_csr": (
                None if self.relation_reference is None else dict(self.relation_reference)
            ),
            "family_provenance": (
                None if self.family_provenance is None else dict(self.family_provenance)
            ),
            "cells": [
                {
                    "index": index,
                    "chart_id": cell.chart_id,
                    "axis_depth": dyadic.axis_depth,
                    "coordinates": list(dyadic.coordinates),
                    "bounds": list(cell.bounds),
                    **(
                        {}
                        if csr_backed
                        else {"image": sorted(self.relation[index])}
                    ),
                    "morse_node": self.node_by_cell.get(index),
                    "relation_evaluated": True,
                    "open_exit": self.source_provenance[index].to_dict(),
                }
                for index, (cell, dyadic) in enumerate(
                    zip(self.cells, self.dyadic_cells, strict=True)
                )
            ],
        }
        attach_garcia_resume_fingerprint(payload)
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            if target.suffix == ".gz":
                with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                    stream.write(payload)
            else:
                temporary.write_text(payload, encoding="utf-8")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o644)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


_GARCIA_LOCAL_ALGORITHM_REVISION = "garcia-local-open-safe-locator-v4"


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _garcia_resume_fingerprint(
    payload: Mapping[str, object],
    *,
    algorithm_revision: str = _GARCIA_LOCAL_ALGORITHM_REVISION,
) -> dict[str, object]:
    metadata = payload["metadata"]
    charts = payload["charts"]
    if not isinstance(metadata, Mapping) or not isinstance(charts, Mapping):
        raise ValueError("cannot fingerprint malformed Garcia metadata/charts")
    relation_content_sha256 = _sha256_json(
        {
            "cells": payload["cells"],
            "quotient_neighbor_pairs": payload["quotient_neighbor_pairs"],
            "refinement_locator": payload["refinement_locator"],
            "candidate": payload["candidate"],
            "reference_audit": payload["reference_audit"],
            "morse_graph": payload["morse_graph"],
            "support_audit": payload.get("support_audit"),
            "relation_csr": payload.get("relation_csr"),
            "family_provenance": payload.get("family_provenance"),
        }
    )
    fields: dict[str, object] = {
        "algorithm_revision": str(algorithm_revision),
        "schema": payload["schema"],
        "model": metadata["model"],
        "base_bounds": charts["base"]["bounds"],
        "handle_bounds": charts["handle"]["bounds"],
        "gamma": metadata["gamma"],
        "guard_delta": metadata["guard_delta"],
        "transversality_eta": metadata["transversality_eta"],
        "t_star": metadata["t_star"],
        "max_step": metadata["max_step"],
        "max_jumps": metadata["max_jumps"],
        "require_domain_path": metadata["require_domain_path"],
        "samples_per_axis": metadata["samples_per_axis"],
        "padding_cells": metadata["padding_cells"],
        "relation_content_sha256": relation_content_sha256,
    }
    fields["sha256"] = _sha256_json(fields)
    return fields


def attach_garcia_resume_fingerprint(
    payload: dict[str, object],
) -> dict[str, object]:
    """Attach a reproducible configuration and relation-content fingerprint."""

    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Garcia payload metadata must be mutable")
    metadata["resume_fingerprint"] = _garcia_resume_fingerprint(payload)
    return payload


def validate_garcia_resume_fingerprint(payload: Mapping[str, object]) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("resume payload has malformed metadata")
    stored = metadata.get("resume_fingerprint")
    if not isinstance(stored, Mapping):
        raise ValueError(
            "resume payload lacks a full fingerprint; refresh its derived audits first"
        )
    revision = stored.get("algorithm_revision")
    supported_revisions = {
        "garcia-local-open-safe-locator-v3",
        _GARCIA_LOCAL_ALGORITHM_REVISION,
    }
    if revision not in supported_revisions:
        raise ValueError("resume payload uses an unsupported derived-audit revision")
    expected = _garcia_resume_fingerprint(
        payload,
        algorithm_revision=str(revision),
    )
    if dict(stored) != expected:
        raise ValueError("resume fingerprint or persisted relation content does not match")


def _family_geometry_map(
    family: GarciaMixedDyadicFamily,
    charts: SuspensionAtlasCharts,
) -> dict[tuple[object, ...], GarciaDyadicCell]:
    result: dict[tuple[object, ...], GarciaDyadicCell] = {}
    for cell in family.cells:
        key = _source_key(cell.chart_id, cell.bounds(charts))
        if key in result:
            raise ValueError("two active dyadic cells have the same geometry")
        result[key] = cell
    return result


def _candidate_boundary_audit(
    s_cells: Collection[int],
    cells: Sequence[AtlasWalkerCell],
    charts: SuspensionAtlasCharts,
    *,
    atol: float = 1.0e-12,
) -> tuple[tuple[int, ...], dict[str, tuple[float | None, ...]]]:
    touches: list[int] = []
    margins: dict[str, list[float | None]] = {
        "base": [None] * 4,
        "handle": [None] * 4,
    }
    for index in sorted(s_cells):
        cell = cells[index]
        chart_bounds = charts.bounds_for(cell.chart_id)
        lower = cell.lower
        upper = cell.upper
        label = "base" if cell.chart_id == charts.base_chart_id else "handle"
        touches_cell = False
        for axis, interval in enumerate(chart_bounds):
            # Handle s=0 and s=1 are quotient seams, not open boundaries.
            if cell.chart_id == charts.handle_chart_id and axis == 3:
                continue
            margin = min(lower[axis] - interval[0], interval[1] - upper[axis])
            previous = margins[label][axis]
            margins[label][axis] = margin if previous is None else min(previous, margin)
            touches_cell = touches_cell or margin <= atol
        if touches_cell:
            touches.append(index)
    return tuple(touches), {
        key: tuple(None if value is None else float(value) for value in values)
        for key, values in margins.items()
    }


def _quotient_cross_chart_pairs(
    cells: Sequence[AtlasWalkerCell],
    incidence: GarciaWalkerQuotientIncidence,
    charts: SuspensionAtlasCharts,
    atlas: Any,
    *,
    atol: float = 1.0e-12,
) -> tuple[tuple[int, int], ...]:
    """Return exact extra leaf incidences across the two quotient seams.

    A conservative theta/phi hull first asks the native Atlas for a short list
    of base candidates.  The nonlinear Garcia incidence predicate then filters
    that list exactly according to the same guard/reset equations used by the
    connectivity audit.
    """

    pairs: set[tuple[int, int]] = set()
    for handle in cells:
        if handle.chart_id != charts.handle_chart_id:
            continue
        lower = handle.lower
        upper = handle.upper
        broad_hulls: list[tuple[float, ...]] = []
        if lower[3] <= atol:
            broad_hulls.append(incidence.attachment_hull(handle, -1))
        if upper[3] >= 1.0 - atol:
            broad_hulls.append(incidence.attachment_hull(handle, 1))
        for hull in broad_hulls:
            for base_index_raw in atlas.cover(charts.base_chart_id, hull):
                base_index = int(base_index_raw)
                base = cells[base_index]
                if base.chart_id != charts.base_chart_id:
                    continue
                if incidence.intersects(base, handle):
                    pairs.add((min(base.index, handle.index), max(base.index, handle.index)))
    return tuple(sorted(pairs))


def _safe_refinement_locator(
    relation: Mapping[int, frozenset[int]],
    provenance: Mapping[int, GarciaSourceExitProvenance],
    disconnected_sources: Collection[int],
    reference_source_cells: Collection[int],
) -> GarciaSafeRefinementLocator:
    """Locate recurrent safe dynamics without changing the stored relation."""

    disconnected = frozenset(disconnected_sources)
    eligible = frozenset(
        source
        for source, targets in relation.items()
        if targets and not provenance[source].has_open_exit and source not in disconnected
    )
    recurrent = list(_relation_recurrent_components(relation, eligible))
    references = frozenset(int(value) for value in reference_source_cells)
    selected = frozenset()
    if recurrent:
        selected = frozenset(
            max(
                recurrent,
                key=lambda component: (
                    len(references.intersection(component)),
                    len(component),
                    -component[0],
                ),
            )
        )
    return GarciaSafeRefinementLocator(
        eligible_cells=eligible,
        disconnected_sources_excluded=disconnected,
        recurrent_components=tuple(recurrent),
        selected_component=selected,
        reference_source_cells=references,
        selected_reference_hits=selected & references,
        seed_cells=selected | references,
    )


_GARCIA_RAW_CHECKPOINT_SCHEMA = (
    "garcia-walker-open-exit-raw-relation-checkpoint-v1"
)
_GARCIA_RAW_RELATION_REVISION = "garcia-raw-mapgraph-relation-v1"
_GARCIA_PHYSICAL_MODEL_REVISION = "garcia-physical-event-reset-v2"
_GARCIA_GUARD_ALIGNED_MODEL_REVISION = "garcia-guard-aligned-event-reset-v1"
_GARCIA_BOXMAP_REVISION = "garcia-fixed-time-suspension-sample-bloat-v2"
_GARCIA_SAMPLE_PROVENANCE_POLICY = (
    "hash_bound_and_internal_count_checked_not_recomputed_without_ode"
)


def _garcia_coordinate_system(
    walker: GarciaPassiveWalker | GuardAlignedGarciaPassiveWalker,
) -> str:
    if isinstance(walker, GuardAlignedGarciaPassiveWalker):
        return "guard_aligned"
    if isinstance(walker, GarciaPassiveWalker):
        return "physical"
    raise TypeError(f"unsupported Garcia walker type {type(walker).__name__}")


def _garcia_model_revision(
    walker: GarciaPassiveWalker | GuardAlignedGarciaPassiveWalker,
) -> str:
    return (
        _GARCIA_GUARD_ALIGNED_MODEL_REVISION
        if _garcia_coordinate_system(walker) == "guard_aligned"
        else _GARCIA_PHYSICAL_MODEL_REVISION
    )


class _CellRelationView(Mapping[AtlasWalkerCell, frozenset[AtlasWalkerCell]]):
    """Zero-copy cell-token view of an integer-indexed relation."""

    def __init__(
        self,
        cells: Sequence[AtlasWalkerCell],
        relation: Mapping[int, Collection[int]],
    ) -> None:
        self.cells = tuple(cells)
        self.relation = relation
        self.index_by_cell = {cell: index for index, cell in enumerate(self.cells)}

    def __getitem__(self, key: AtlasWalkerCell) -> frozenset[AtlasWalkerCell]:
        source = self.index_by_cell[key]
        return frozenset(self.cells[target] for target in self.relation[source])

    def __iter__(self):
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)


class _StoredMapGraph:
    """Minimal read-only MapGraph interface for a strict raw checkpoint."""

    def __init__(self, relation: Mapping[int, Collection[int]]) -> None:
        # Keep a zero-copy view.  Materializing tuple-of-tuples here used to
        # duplicate every deep MapGraph edge solely to emulate this tiny API.
        self._relation = relation

    def num_vertices(self) -> int:
        return len(self._relation)

    def adjacencies(self, index: int) -> list[int]:
        return sorted(int(value) for value in self._relation[int(index)])


class _StoredMorseGraph:
    """Minimal read-only MorseGraph interface for checkpoint finalization."""

    def __init__(self, snapshot: Mapping[str, object]) -> None:
        self._nodes = tuple(int(value) for value in snapshot["nodes"])
        raw_sets = snapshot["morse_sets"]
        if not isinstance(raw_sets, Mapping):
            raise ValueError("raw checkpoint has malformed Morse sets")
        self._sets = {
            int(node): tuple(int(value) for value in cells)
            for node, cells in raw_sets.items()
        }
        self._edges = tuple(
            (int(edge[0]), int(edge[1])) for edge in snapshot["edges"]
        )

    def num_vertices(self) -> int:
        return len(self._nodes)

    def vertices(self) -> list[int]:
        return list(self._nodes)

    def morse_set(self, node: int) -> list[int]:
        return list(self._sets[int(node)])

    def edges(self) -> list[tuple[int, int]]:
        return list(self._edges)


@dataclass(frozen=True)
class _GarciaRawRelationStage:
    family: GarciaMixedDyadicFamily
    walker: GarciaPassiveWalker | GuardAlignedGarciaPassiveWalker
    charts: SuspensionAtlasCharts
    box_map: _AuditedGarciaSuspensionBoxMap
    model: Any
    morse_graph: Any
    map_graph: Any
    cells: tuple[AtlasWalkerCell, ...]
    dyadic_cells: tuple[GarciaDyadicCell, ...]
    relation: Mapping[int, Collection[int]]
    quotient_neighbor_pairs: tuple[tuple[int, int], ...]
    source_provenance: Mapping[int, GarciaSourceExitProvenance]
    raw_elapsed_seconds: float
    compute_metadata: Mapping[str, object]
    checkpoint_path: str | None = None
    connectivity_audit: GarciaSparseConnectivityAudit | None = None
    relation_reference: Mapping[str, object] | None = None
    family_provenance: Mapping[str, object] | None = None


def _morse_snapshot(morse_graph: Any) -> dict[str, object]:
    nodes = [int(value) for value in morse_graph.vertices()]
    return {
        "nodes": nodes,
        "edges": [
            [int(source), int(target)]
            for source, target in sorted(morse_graph.edges())
        ],
        "morse_sets": {
            str(node): sorted(int(value) for value in morse_graph.morse_set(node))
            for node in nodes
        },
    }


def _canonicalize_morse_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Relabel Morse nodes canonically by the minimum phase-cell index."""

    raw_sets = snapshot["morse_sets"]
    if not isinstance(raw_sets, Mapping):
        raise ValueError("Morse snapshot has malformed Morse sets")
    ordered = sorted(
        (
            int(old_node),
            tuple(sorted(int(value) for value in values)),
        )
        for old_node, values in raw_sets.items()
    )
    ordered.sort(key=lambda item: (item[1][0], item[1]))
    relabel = {old_node: new_node for new_node, (old_node, _cells) in enumerate(ordered)}
    return {
        "nodes": list(range(len(ordered))),
        "edges": sorted(
            [relabel[int(source)], relabel[int(target)]]
            for source, target in snapshot["edges"]
        ),
        "morse_sets": {
            str(new_node): list(cells)
            for new_node, (_old_node, cells) in enumerate(ordered)
        },
    }


def _raw_cell_record(
    index: int,
    cell: AtlasWalkerCell,
    dyadic: GarciaDyadicCell,
    relation: Mapping[int, frozenset[int]],
    provenance: Mapping[int, GarciaSourceExitProvenance],
) -> dict[str, object]:
    return {
        "record": "cell",
        "index": int(index),
        "chart_id": int(cell.chart_id),
        "axis_depth": int(dyadic.axis_depth),
        "coordinates": list(dyadic.coordinates),
        "bounds": list(cell.bounds),
        "image": sorted(int(value) for value in relation[index]),
        "relation_evaluated": True,
        "open_exit": provenance[index].to_dict(),
    }


def _raw_checkpoint_configuration(stage: _GarciaRawRelationStage) -> dict[str, object]:
    return {
        "raw_relation_revision": _GARCIA_RAW_RELATION_REVISION,
        "physical_model_revision": _GARCIA_PHYSICAL_MODEL_REVISION,
        "boxmap_revision": _GARCIA_BOXMAP_REVISION,
        "model": "garcia_passive_walker_fixed_time_suspension_atlas",
        "relation_scope": "post_mapgraph_pre_audit_checkpoint",
        "t_star": stage.box_map.t_star,
        "gamma": stage.walker.gamma,
        "guard_delta": stage.walker.guard_delta,
        "transversality_eta": stage.walker.transversality_eta,
        "samples_per_axis": stage.box_map.samples_per_axis,
        "padding_cells": stage.box_map.padding_cells,
        "max_step": stage.box_map.max_step,
        "max_jumps": stage.box_map.max_jumps,
        "require_domain_path": stage.box_map.require_domain_path,
        "family_role": stage.family.role,
        "min_axis_depth": stage.family.min_axis_depth,
        "max_axis_depth": stage.family.max_axis_depth,
        "active_cells": len(stage.family.cells),
        "base_bounds": [list(value) for value in stage.charts.base_bounds],
        "handle_bounds": [list(value) for value in stage.charts.handle_bounds],
    }


def _raw_checkpoint_scope_claims() -> dict[str, object]:
    return {
        "checkpoint_scope": "post_mapgraph_pre_audit",
        "checkpoint_complete": False,
        "audit_complete": False,
        "scientific_result_accepted": False,
        "whole_cell_outer_enclosure_certified": False,
        "sample_failure_provenance_validation": _GARCIA_SAMPLE_PROVENANCE_POLICY,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _update_raw_content_hash(
    digest: Any,
    record_kind: str,
    value: object,
) -> None:
    encoded_kind = record_kind.encode("ascii")
    encoded_value = _canonical_json_bytes(value)
    digest.update(len(encoded_kind).to_bytes(4, "big"))
    digest.update(encoded_kind)
    digest.update(len(encoded_value).to_bytes(8, "big"))
    digest.update(encoded_value)


def _raw_fingerprint(
    *,
    configuration: Mapping[str, object],
    content_sha256: str,
    cell_count: int,
    relation_edges: int,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "raw_relation_revision": _GARCIA_RAW_RELATION_REVISION,
        "configuration_sha256": _sha256_json(dict(configuration)),
        "content_sha256": str(content_sha256),
        "cell_count": int(cell_count),
        "relation_edges": int(relation_edges),
    }
    fields["sha256"] = _sha256_json(fields)
    return fields


def _raw_compute_metadata(box_map: _AuditedGarciaSuspensionBoxMap) -> dict[str, object]:
    diagnostics = box_map.diagnostics()
    unique_points, reused_points = box_map.point_cache_counts()
    new_sources, reused_sources = box_map.source_cache_counts()
    precompute = box_map.endpoint_precompute_result()
    return {
        "box_map_unique_source_boxes_evaluated": diagnostics.source_boxes,
        "new_whole_source_values_evaluated": new_sources,
        "reused_whole_source_values": reused_sources,
        "box_map_sampled_points_evaluated": diagnostics.sampled_points,
        "unique_physical_endpoint_evaluations": unique_points,
        "reused_lattice_endpoint_evaluations": reused_points,
        "box_map_failed_samples": diagnostics.failed_samples,
        "box_map_empty_images": diagnostics.empty_images,
        "box_map_unresolved_stage_edges": diagnostics.unresolved_stage_edges,
        "endpoint_precompute": (
            None
            if precompute is None
            else {
                "logical_tensor_points": precompute.logical_tensor_points,
                "unique_points": precompute.unique_points,
                "duplicate_tensor_points": precompute.duplicate_tensor_points,
                "requested_workers": precompute.requested_workers,
                "used_workers": precompute.used_workers,
                "process_count": (
                    precompute.used_workers if precompute.mode == "process_pool" else 0
                ),
                "mode": precompute.mode,
                "elapsed_seconds": precompute.elapsed_seconds,
                "fallback_reason": precompute.fallback_reason,
                "parity_provenance": dict(PARITY_PROVENANCE),
            }
        ),
    }


def _validate_endpoint_precompute_metadata(
    raw: object,
    *,
    logical_samples: int,
    unique_endpoint_evaluations: int,
    reused_endpoint_evaluations: int,
    new_source_values: int,
    reused_source_values: int,
) -> None:
    """Validate optional parallel-precompute provenance before re-publishing it."""

    if raw is None:
        return
    required = {
        "logical_tensor_points",
        "unique_points",
        "duplicate_tensor_points",
        "requested_workers",
        "used_workers",
        "process_count",
        "mode",
        "elapsed_seconds",
        "fallback_reason",
        "parity_provenance",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("raw checkpoint endpoint-precompute metadata is malformed")
    integer_fields = (
        "logical_tensor_points",
        "unique_points",
        "duplicate_tensor_points",
        "requested_workers",
        "used_workers",
        "process_count",
    )
    if any(type(raw[key]) is not int or raw[key] < 0 for key in integer_fields):
        raise ValueError("raw checkpoint endpoint-precompute counts are invalid")
    logical = raw["logical_tensor_points"]
    unique = raw["unique_points"]
    duplicates = raw["duplicate_tensor_points"]
    requested = raw["requested_workers"]
    used = raw["used_workers"]
    processes = raw["process_count"]
    if (
        logical != logical_samples
        or unique + duplicates != logical
        or unique != unique_endpoint_evaluations
        or duplicates != reused_endpoint_evaluations
        or new_source_values + reused_source_values <= 0
        or reused_source_values != 0
    ):
        raise ValueError("raw checkpoint endpoint-precompute totals are inconsistent")
    mode = raw["mode"]
    fallback = raw["fallback_reason"]
    if mode == "serial":
        valid_workers = requested == used == 1 and processes == 0
        valid_fallback = fallback is None
    elif mode == "serial_fallback":
        valid_workers = requested > 1 and used == 1 and processes == 0
        valid_fallback = isinstance(fallback, str) and bool(fallback.strip())
    elif mode == "process_pool":
        valid_workers = requested == used == processes and used > 1
        valid_fallback = fallback is None
    else:
        raise ValueError("raw checkpoint endpoint-precompute mode is unsupported")
    if not valid_workers or not valid_fallback:
        raise ValueError("raw checkpoint endpoint-precompute worker provenance is invalid")
    elapsed = raw["elapsed_seconds"]
    if (
        type(elapsed) not in (int, float)
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("raw checkpoint endpoint-precompute elapsed time is invalid")
    if raw["parity_provenance"] != dict(PARITY_PROVENANCE):
        raise ValueError("raw checkpoint endpoint-precompute parity provenance differs")


def write_garcia_raw_relation_checkpoint(
    stage: _GarciaRawRelationStage,
    path: str | Path,
) -> Path:
    """Atomically persist a post-MapGraph, pre-audit JSON-lines checkpoint.

    This checkpoint resumes connectivity/reference/candidate postprocessing. It
    is intentionally not a mid-callback journal and therefore cannot resume an
    interrupted ODE/MapGraph construction.  The atomic rename happens only
    after a complete trailer, file fsync, and directory fsync.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    configuration = _raw_checkpoint_configuration(stage)
    scope_claims = _raw_checkpoint_scope_claims()
    header = {
        "record": "header",
        "schema": _GARCIA_RAW_CHECKPOINT_SCHEMA,
        **scope_claims,
        "configuration": configuration,
        "compute_metadata": dict(stage.compute_metadata),
        "raw_elapsed_seconds": float(stage.raw_elapsed_seconds),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    relation_edges = 0
    try:
        opener = gzip.open if target.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
            _update_raw_content_hash(digest, "scope_claims", scope_claims)
            _update_raw_content_hash(digest, "configuration", configuration)
            _update_raw_content_hash(
                digest, "compute_metadata", dict(stage.compute_metadata)
            )
            _update_raw_content_hash(
                digest, "raw_elapsed_seconds", float(stage.raw_elapsed_seconds)
            )
            for index, (cell, dyadic) in enumerate(
                zip(stage.cells, stage.dyadic_cells, strict=True)
            ):
                record = _raw_cell_record(
                    index,
                    cell,
                    dyadic,
                    stage.relation,
                    stage.source_provenance,
                )
                relation_edges += len(stage.relation[index])
                _update_raw_content_hash(digest, "cell", record)
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
            pairs = [
                [int(first), int(second)]
                for first, second in stage.quotient_neighbor_pairs
            ]
            _update_raw_content_hash(digest, "quotient_neighbor_pairs", pairs)
            morse = _morse_snapshot(stage.morse_graph)
            trailer = {
                "record": "trailer",
                "schema": _GARCIA_RAW_CHECKPOINT_SCHEMA,
                "checkpoint_complete": True,
                "audit_complete": False,
                "scientific_result_accepted": False,
                "cell_count": len(stage.cells),
                "relation_edges": relation_edges,
                "quotient_neighbor_pairs": pairs,
                "morse_graph": morse,
                "morse_snapshot_sha256": _sha256_json(morse),
                "raw_relation_fingerprint": _raw_fingerprint(
                    configuration=configuration,
                    content_sha256=digest.hexdigest(),
                    cell_count=len(stage.cells),
                    relation_edges=relation_edges,
                ),
            }
            stream.write(
                json.dumps(trailer, sort_keys=True, allow_nan=False) + "\n"
            )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o644)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _source_provenance_from_dict(
    source_index: int,
    raw: Mapping[str, object],
) -> GarciaSourceExitProvenance:
    def nonnegative_integer(key: str) -> int:
        value = raw[key]
        if type(value) is not int or value < 0:
            raise ValueError(
                f"raw source {source_index} field {key} must be a nonnegative integer"
            )
        return value

    for key in (
        "explicit_empty_callback",
        "mapgraph_empty_image",
        "has_open_exit",
    ):
        if type(raw[key]) is not bool:
            raise ValueError(f"raw source {source_index} field {key} must be boolean")
    failure_reasons = raw["failure_reasons"]
    if not isinstance(failure_reasons, Mapping):
        raise ValueError(f"raw source {source_index} has malformed failure reasons")
    witnesses = raw["missing_in_domain_witnesses"]
    pieces = raw["target_pieces"]
    if not isinstance(witnesses, list) or not isinstance(pieces, list):
        raise ValueError(f"raw source {source_index} lacks canonical target provenance")
    normalized_reasons: list[tuple[str, int]] = []
    for reason, count in failure_reasons.items():
        if not isinstance(reason, str) or type(count) is not int or count < 0:
            raise ValueError(f"raw source {source_index} has invalid failure count")
        normalized_reasons.append((reason, count))
    unresolved: list[tuple[int, int]] = []
    for edge in raw["unresolved_stage_edges"]:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(value) is not int or value < 0 for value in edge)
        ):
            raise ValueError(f"raw source {source_index} has invalid unresolved edge")
        unresolved.append((edge[0], edge[1]))
    normalized_pieces: list[tuple[int, tuple[float, ...]]] = []
    for item in pieces:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"chart_id", "bounds"}
            or type(item["chart_id"]) is not int
        ):
            raise ValueError(f"raw source {source_index} has invalid target piece")
        chart_id = item["chart_id"]
        if chart_id not in (BASE_CHART_ID, HANDLE_CHART_ID):
            raise ValueError(f"raw source {source_index} has unknown target chart")
        raw_bounds = item["bounds"]
        if (
            not isinstance(raw_bounds, list)
            or len(raw_bounds) != 8
            or any(type(value) not in (int, float) for value in raw_bounds)
        ):
            raise ValueError(f"raw source {source_index} has invalid target bounds")
        bounds = tuple(float(value) for value in raw_bounds)
        if not all(np.isfinite(value) for value in bounds):
            raise ValueError(f"raw source {source_index} has nonfinite target bounds")
        if any(lower > upper for lower, upper in zip(bounds[:4], bounds[4:], strict=True)):
            raise ValueError(f"raw source {source_index} has reversed target bounds")
        normalized_pieces.append((chart_id, bounds))
    normalized_witnesses: list[GarciaDyadicCell] = []
    for item in witnesses:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"chart_id", "axis_depth", "coordinates"}
            or type(item["chart_id"]) is not int
            or type(item["axis_depth"]) is not int
            or not isinstance(item["coordinates"], list)
            or len(item["coordinates"]) != 4
            or any(type(value) is not int for value in item["coordinates"])
        ):
            raise ValueError(f"raw source {source_index} has invalid missing witness")
        normalized_witnesses.append(
            GarciaDyadicCell(
                item["chart_id"],
                item["axis_depth"],
                tuple(item["coordinates"]),
            )
        )
    provenance = GarciaSourceExitProvenance(
        source_index=int(source_index),
        callback_invocations=nonnegative_integer("callback_invocations"),
        successful_samples=nonnegative_integer("successful_samples_per_invocation"),
        failed_samples=nonnegative_integer("failed_samples_per_invocation"),
        failure_reasons=tuple(sorted(normalized_reasons)),
        unresolved_stage_edges=tuple(unresolved),
        returned_pieces=nonnegative_integer("returned_pieces"),
        explicit_empty_callback=raw["explicit_empty_callback"],
        mapgraph_empty_image=raw["mapgraph_empty_image"],
        active_target_cells=nonnegative_integer("active_target_cells"),
        missing_in_domain_target_cells=nonnegative_integer(
            "missing_in_domain_target_cells"
        ),
        missing_in_domain_witnesses=tuple(sorted(normalized_witnesses)),
        target_pieces=tuple(normalized_pieces),
        ambient_boundary_pieces=nonnegative_integer("ambient_boundary_pieces"),
        wholly_outside_active_family_pieces=nonnegative_integer(
            "wholly_outside_active_family_pieces"
        ),
    )
    if bool(raw["has_open_exit"]) != provenance.has_open_exit:
        raise ValueError(f"raw source {source_index} has inconsistent open-exit flag")
    return provenance


def _validate_raw_morse_snapshot(
    snapshot: Mapping[str, object],
    relation: Mapping[int, Collection[int]],
) -> None:
    raw_nodes = snapshot["nodes"]
    raw_sets = snapshot["morse_sets"]
    raw_edges = snapshot["edges"]
    if (
        not isinstance(raw_nodes, list)
        or any(type(value) is not int or value < 0 for value in raw_nodes)
        or raw_nodes != sorted(set(raw_nodes))
    ):
        raise ValueError("raw checkpoint has malformed Morse nodes")
    if not isinstance(raw_sets, Mapping) or not isinstance(raw_edges, list):
        raise ValueError("raw checkpoint has malformed Morse sets")
    nodes = tuple(raw_nodes)
    normalized: dict[int, frozenset[int]] = {}
    for raw_node, values in raw_sets.items():
        if (
            not isinstance(raw_node, str)
            or not raw_node.isdigit()
            or str(int(raw_node)) != raw_node
            or not isinstance(values, list)
            or any(type(value) is not int or value < 0 for value in values)
            or values != sorted(set(values))
            or any(value >= len(relation) for value in values)
        ):
            raise ValueError("raw checkpoint has malformed Morse sets")
        normalized[int(raw_node)] = frozenset(values)
    edges_list: list[tuple[int, int]] = []
    for edge in raw_edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(value) is not int or value < 0 for value in edge)
            or edge[0] == edge[1]
        ):
            raise ValueError("raw checkpoint has malformed Morse edges")
        edges_list.append((edge[0], edge[1]))
    if edges_list != sorted(set(edges_list)):
        raise ValueError("raw checkpoint Morse edges are not canonical")
    edges = tuple(edges_list)
    if set(nodes) != set(normalized) or len(nodes) != len(set(nodes)):
        raise ValueError("raw checkpoint Morse nodes disagree with Morse-set keys")
    recurrent = _relation_recurrent_components(relation)
    canonical = {
        node: frozenset(component) for node, component in enumerate(recurrent)
    }
    if normalized != canonical or nodes != tuple(range(len(canonical))):
        raise ValueError("raw checkpoint Morse sets disagree with relation SCCs")
    seen: set[int] = set()
    for component in normalized.values():
        if seen.intersection(component):
            raise ValueError("raw checkpoint Morse sets overlap")
        seen.update(component)
    stored_adjacency: dict[int, list[int]] = {node: [] for node in nodes}
    for source, target in edges:
        if source not in normalized or target not in normalized:
            raise ValueError("raw checkpoint Morse edge references an unknown node")
        stored_adjacency[source].append(target)

    stored_closure: set[tuple[int, int]] = set()
    for source in nodes:
        visited: set[int] = set()
        stack = list(stored_adjacency[source])
        while stack:
            target = stack.pop()
            if target == source:
                raise ValueError("raw checkpoint Morse order is cyclic")
            if target in visited:
                continue
            visited.add(target)
            stack.extend(stored_adjacency[target])
        stored_closure.update((source, target) for target in visited)

    # CMGDB may store either the Hasse diagram or redundant comparable edges.
    # Its transitive closure must nevertheless equal reachability between the
    # recurrent SCCs through the complete (including transient) MapGraph.  A
    # traversal from each recurrent SCC avoids materializing either another
    # copy of every MapGraph edge or a worst-case-large condensation graph.
    recurrent_node_by_cell = [-1] * len(relation)
    for node, component in canonical.items():
        for cell in component:
            recurrent_node_by_cell[cell] = node
    expected_order: set[tuple[int, int]] = set()
    for source, component in canonical.items():
        visited = bytearray(len(relation))
        stack = list(component)
        for cell in component:
            visited[cell] = 1
        while stack:
            cell = stack.pop()
            for target_cell in relation[cell]:
                target_node = recurrent_node_by_cell[target_cell]
                if target_node >= 0 and target_node != source:
                    expected_order.add((source, target_node))
                if not visited[target_cell]:
                    visited[target_cell] = 1
                    stack.append(target_cell)
    if stored_closure != expected_order:
        raise ValueError("raw checkpoint Morse order disagrees with relation reachability")


def read_garcia_raw_relation_checkpoint(
    path: str | Path,
    *,
    expected_family: GarciaMixedDyadicFamily | None = None,
    expected_t_star: float | None = None,
    expected_samples_per_axis: int | None = None,
    expected_padding_cells: float | None = None,
    expected_gamma: float | None = None,
    expected_guard_delta: float | None = None,
    expected_transversality_eta: float | None = None,
    expected_max_step: float | None = None,
    expected_max_jumps: int | None = None,
    max_relation_edges: int | None = None,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
) -> _GarciaRawRelationStage:
    """Strictly load and geometry-validate an atomic raw-relation checkpoint."""

    for name, cap in (
        ("max_relation_edges", max_relation_edges),
        ("max_relation_storage_bytes", max_relation_storage_bytes),
        ("max_undirected_adjacencies", max_undirected_adjacencies),
        ("max_adjacency_storage_bytes", max_adjacency_storage_bytes),
    ):
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError(f"{name} must be a nonnegative exact integer or None")
    for name, value in (
        ("expected_t_star", expected_t_star),
        ("expected_padding_cells", expected_padding_cells),
        ("expected_gamma", expected_gamma),
        ("expected_guard_delta", expected_guard_delta),
        ("expected_transversality_eta", expected_transversality_eta),
        ("expected_max_step", expected_max_step),
    ):
        if value is not None and (
            type(value) not in (int, float) or not np.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite real number or None")
    for name, value in (
        ("expected_samples_per_axis", expected_samples_per_axis),
        ("expected_max_jumps", expected_max_jumps),
    ):
        if value is not None and type(value) is not int:
            raise ValueError(f"{name} must be an exact integer or None")
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        first_line = stream.readline()
        if not first_line:
            raise ValueError("raw checkpoint is empty")
        header = json.loads(first_line)
        if not isinstance(header, Mapping):
            raise ValueError("raw checkpoint header is not an object")
        scope_claims = _raw_checkpoint_scope_claims()
        expected_header_keys = {
            "record",
            "schema",
            *scope_claims,
            "configuration",
            "compute_metadata",
            "raw_elapsed_seconds",
        }
        if (
            set(header) != expected_header_keys
            or header.get("record") != "header"
            or header.get("schema") != _GARCIA_RAW_CHECKPOINT_SCHEMA
            or any(header.get(key) != value for key, value in scope_claims.items())
        ):
            raise ValueError("raw checkpoint header is malformed or mis-scoped")
        configuration = header["configuration"]
        if not isinstance(configuration, Mapping):
            raise ValueError("raw checkpoint configuration is malformed")
        expected_configuration_keys = {
            "raw_relation_revision",
            "physical_model_revision",
            "boxmap_revision",
            "model",
            "relation_scope",
            "t_star",
            "gamma",
            "guard_delta",
            "transversality_eta",
            "samples_per_axis",
            "padding_cells",
            "max_step",
            "max_jumps",
            "require_domain_path",
            "family_role",
            "min_axis_depth",
            "max_axis_depth",
            "active_cells",
            "base_bounds",
            "handle_bounds",
        }
        if set(configuration) != expected_configuration_keys:
            raise ValueError("raw checkpoint configuration is not canonical")
        if configuration["raw_relation_revision"] != _GARCIA_RAW_RELATION_REVISION:
            raise ValueError("unsupported raw-relation revision")
        if configuration["physical_model_revision"] != _GARCIA_PHYSICAL_MODEL_REVISION:
            raise ValueError("raw checkpoint uses a different physical-model revision")
        if configuration["boxmap_revision"] != _GARCIA_BOXMAP_REVISION:
            raise ValueError("raw checkpoint uses a different box-map revision")
        if configuration["model"] != "garcia_passive_walker_fixed_time_suspension_atlas":
            raise ValueError("raw checkpoint identifies a different physical model")
        if configuration["relation_scope"] != "post_mapgraph_pre_audit_checkpoint":
            raise ValueError("raw checkpoint relation scope is inconsistent")
        if type(configuration["require_domain_path"]) is not bool:
            raise ValueError("raw checkpoint require_domain_path must be boolean")
        if configuration["require_domain_path"] is not True:
            raise ValueError("raw checkpoint did not require an in-domain path")
        for key in (
            "samples_per_axis",
            "max_jumps",
            "min_axis_depth",
            "max_axis_depth",
            "active_cells",
        ):
            if type(configuration[key]) is not int or int(configuration[key]) < 0:
                raise ValueError(f"raw checkpoint {key} must be a nonnegative integer")
        for key in (
            "t_star",
            "gamma",
            "guard_delta",
            "transversality_eta",
            "padding_cells",
            "max_step",
        ):
            if type(configuration[key]) not in (int, float) or not np.isfinite(
                float(configuration[key])
            ):
                raise ValueError(f"raw checkpoint {key} must be finite")
        if float(configuration["t_star"]) <= 0.0:
            raise ValueError("raw checkpoint t_star must be positive")
        if int(configuration["samples_per_axis"]) < 3:
            raise ValueError("raw checkpoint sampling count is too small")
        if float(configuration["padding_cells"]) < 0.0:
            raise ValueError("raw checkpoint padding must be nonnegative")
        if float(configuration["max_step"]) <= 0.0:
            raise ValueError("raw checkpoint max_step must be positive")
        if not isinstance(configuration["family_role"], str):
            raise ValueError("raw checkpoint family role must be a string")
        for chart_key in ("base_bounds", "handle_bounds"):
            bounds = configuration[chart_key]
            if not isinstance(bounds, list) or len(bounds) != 4:
                raise ValueError(f"raw checkpoint {chart_key} is malformed")
            for interval in bounds:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or any(
                        type(value) not in (int, float) or not np.isfinite(float(value))
                        for value in interval
                    )
                    or float(interval[0]) >= float(interval[1])
                ):
                    raise ValueError(f"raw checkpoint {chart_key} is malformed")
        digest = hashlib.sha256()
        _update_raw_content_hash(digest, "scope_claims", scope_claims)
        _update_raw_content_hash(digest, "configuration", configuration)
        compute_metadata_header = header["compute_metadata"]
        raw_elapsed_header = header["raw_elapsed_seconds"]
        if not isinstance(compute_metadata_header, Mapping):
            raise ValueError("raw checkpoint compute metadata is malformed")
        if type(raw_elapsed_header) not in (int, float) or not np.isfinite(
            float(raw_elapsed_header)
        ) or float(raw_elapsed_header) < 0.0:
            raise ValueError("raw checkpoint elapsed time must be finite and nonnegative")
        _update_raw_content_hash(
            digest, "compute_metadata", dict(compute_metadata_header)
        )
        _update_raw_content_hash(
            digest, "raw_elapsed_seconds", float(raw_elapsed_header)
        )
        cells: list[AtlasWalkerCell] = []
        dyadic_cells: list[GarciaDyadicCell] = []
        relation: dict[int, frozenset[int]] = {}
        streamed_relation_edges = 0
        provenance: dict[int, GarciaSourceExitProvenance] = {}
        raw_open_exit_flags: dict[int, bool] = {}
        trailer: Mapping[str, object] | None = None
        for line in stream:
            if not line.strip():
                raise ValueError("raw checkpoint contains a blank record")
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("raw checkpoint record is not an object")
            kind = record.get("record")
            if kind == "trailer":
                if trailer is not None:
                    raise ValueError("raw checkpoint contains two trailers")
                trailer = record
                continue
            if trailer is not None:
                raise ValueError("raw checkpoint has data after its trailer")
            if kind != "cell":
                raise ValueError("raw checkpoint contains an unknown record")
            if set(record) != {
                "record",
                "index",
                "chart_id",
                "axis_depth",
                "coordinates",
                "bounds",
                "image",
                "relation_evaluated",
                "open_exit",
            }:
                raise ValueError("raw checkpoint cell record is not canonical")
            expected_index = len(cells)
            for key in ("index", "chart_id", "axis_depth"):
                if type(record[key]) is not int:
                    raise ValueError(f"raw cell field {key} must be an integer")
            if record["index"] != expected_index:
                raise ValueError("raw checkpoint cells are not complete and ordered")
            if type(record["relation_evaluated"]) is not bool or not record[
                "relation_evaluated"
            ]:
                raise ValueError(f"raw relation was not evaluated at source {expected_index}")
            _update_raw_content_hash(digest, "cell", record)
            coordinates = record["coordinates"]
            if (
                not isinstance(coordinates, list)
                or len(coordinates) != 4
                or any(type(value) is not int for value in coordinates)
            ):
                raise ValueError(f"raw cell {expected_index} has invalid coordinates")
            raw_bounds = record["bounds"]
            if (
                not isinstance(raw_bounds, list)
                or len(raw_bounds) != 8
                or any(
                    type(value) not in (int, float) or not np.isfinite(float(value))
                    for value in raw_bounds
                )
                or any(
                    float(lower) >= float(upper)
                    for lower, upper in zip(raw_bounds[:4], raw_bounds[4:], strict=True)
                )
            ):
                raise ValueError(f"raw cell {expected_index} has invalid bounds")
            image = record["image"]
            if (
                not isinstance(image, list)
                or any(type(value) is not int or value < 0 for value in image)
                or image != sorted(set(image))
            ):
                raise ValueError(f"raw cell {expected_index} has noncanonical image")
            dyadic = GarciaDyadicCell(
                record["chart_id"],
                record["axis_depth"],
                tuple(coordinates),
            )
            cell = AtlasWalkerCell(
                expected_index,
                dyadic.chart_id,
                tuple(float(value) for value in raw_bounds),
            )
            open_exit = record["open_exit"]
            if not isinstance(open_exit, Mapping):
                raise ValueError(f"raw source {expected_index} lacks exit provenance")
            if set(open_exit) != {
                "callback_invocations",
                "successful_samples_per_invocation",
                "failed_samples_per_invocation",
                "failure_reasons",
                "unresolved_stage_edges",
                "returned_pieces",
                "explicit_empty_callback",
                "mapgraph_empty_image",
                "active_target_cells",
                "missing_in_domain_target_cells",
                "missing_in_domain_witnesses",
                "target_pieces",
                "ambient_boundary_pieces",
                "wholly_outside_active_family_pieces",
                "has_open_exit",
            }:
                raise ValueError(
                    f"raw source {expected_index} exit provenance is not canonical"
                )
            cells.append(cell)
            dyadic_cells.append(dyadic)
            relation[expected_index] = frozenset(image)
            streamed_relation_edges += len(image)
            if (
                max_relation_edges is not None
                and streamed_relation_edges > max_relation_edges
            ):
                raise GarciaRelationSizeLimitExceeded(
                    kind="raw_relation_edge",
                    limit=max_relation_edges,
                    observed=streamed_relation_edges,
                )
            estimated_relation_bytes = _estimated_raw_relation_storage_bytes(
                len(cells), streamed_relation_edges
            )
            if (
                max_relation_storage_bytes is not None
                and estimated_relation_bytes > max_relation_storage_bytes
            ):
                raise GarciaRelationSizeLimitExceeded(
                    kind="raw_relation_estimated_storage_byte",
                    limit=max_relation_storage_bytes,
                    observed=estimated_relation_bytes,
                )
            if (
                max_adjacency_storage_bytes is not None
                and _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_CELL * len(cells)
                > max_adjacency_storage_bytes
            ):
                raise GarciaRelationSizeLimitExceeded(
                    kind="sparse_adjacency_storage_byte",
                    limit=max_adjacency_storage_bytes,
                    observed=(
                        _SPARSE_ADJACENCY_ESTIMATED_BYTES_PER_CELL * len(cells)
                    ),
                )
            provenance[expected_index] = _source_provenance_from_dict(
                expected_index, open_exit
            )
            raw_open_exit_flags[expected_index] = bool(open_exit["has_open_exit"])
    if trailer is None:
        raise ValueError("raw checkpoint lacks a complete trailer")
    expected_trailer_keys = {
        "record",
        "schema",
        "checkpoint_complete",
        "audit_complete",
        "scientific_result_accepted",
        "cell_count",
        "relation_edges",
        "quotient_neighbor_pairs",
        "morse_graph",
        "morse_snapshot_sha256",
        "raw_relation_fingerprint",
    }
    if (
        set(trailer) != expected_trailer_keys
        or trailer.get("schema") != _GARCIA_RAW_CHECKPOINT_SCHEMA
        or trailer.get("checkpoint_complete") is not True
        or trailer.get("audit_complete") is not False
        or trailer.get("scientific_result_accepted") is not False
    ):
        raise ValueError("raw checkpoint trailer is malformed or claims acceptance")
    if type(trailer["cell_count"]) is not int or trailer["cell_count"] != len(cells):
        raise ValueError("raw checkpoint trailer has the wrong cell count")
    if any(
        target < 0 or target >= len(cells)
        for targets in relation.values()
        for target in targets
    ):
        raise ValueError("raw checkpoint relation references an unknown target")
    raw_pairs = trailer["quotient_neighbor_pairs"]
    if not isinstance(raw_pairs, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(value) is not int or value < 0 for value in pair)
        for pair in raw_pairs
    ):
        raise ValueError("raw checkpoint quotient pairs are malformed")
    pairs = tuple((pair[0], pair[1]) for pair in raw_pairs)
    if pairs != tuple(sorted(set(pairs))):
        raise ValueError("raw checkpoint quotient pairs are not canonical")
    if any(
        first >= len(cells) or second >= len(cells) or first >= second
        for first, second in pairs
    ):
        raise ValueError("raw checkpoint quotient pair is invalid")
    pair_payload = [[first, second] for first, second in pairs]
    _update_raw_content_hash(digest, "quotient_neighbor_pairs", pair_payload)
    relation_edges = sum(len(targets) for targets in relation.values())
    if (
        type(trailer["relation_edges"]) is not int
        or trailer["relation_edges"] != relation_edges
    ):
        raise ValueError("raw checkpoint trailer has the wrong edge count")
    expected_fingerprint = _raw_fingerprint(
        configuration=configuration,
        content_sha256=digest.hexdigest(),
        cell_count=len(cells),
        relation_edges=relation_edges,
    )
    if (
        not isinstance(trailer["raw_relation_fingerprint"], Mapping)
        or dict(trailer["raw_relation_fingerprint"]) != expected_fingerprint
    ):
        raise ValueError("raw checkpoint fingerprint does not match its relation")
    morse_snapshot = trailer["morse_graph"]
    if not isinstance(morse_snapshot, Mapping):
        raise ValueError("raw checkpoint Morse snapshot is malformed")
    if trailer["morse_snapshot_sha256"] != _sha256_json(dict(morse_snapshot)):
        raise ValueError("raw checkpoint Morse snapshot hash does not match")

    family = GarciaMixedDyadicFamily(
        tuple(sorted(dyadic_cells)),
        role=str(configuration["family_role"]),
    )
    if int(configuration["active_cells"]) != len(family.cells):
        raise ValueError("raw checkpoint active-cell summary is inconsistent")
    if int(configuration["min_axis_depth"]) != family.min_axis_depth:
        raise ValueError("raw checkpoint minimum-depth summary is inconsistent")
    if int(configuration["max_axis_depth"]) != family.max_axis_depth:
        raise ValueError("raw checkpoint maximum-depth summary is inconsistent")
    if expected_family is not None and family != expected_family:
        raise ValueError("raw checkpoint active family differs from the requested family")
    # Enforce spatial and memory caps before nonlinear quotient reconstruction
    # or global SCC/Morse validation.  The exact stored quotient pairs are
    # hash-bound here and are independently recomputed below before acceptance.
    prevalidated_connectivity = audit_garcia_sparse_relation_connectivity(
        relation,
        tuple(dyadic_cells),
        pairs,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )
    scalar_expectations = (
        ("t_star", expected_t_star),
        ("padding_cells", expected_padding_cells),
        ("gamma", expected_gamma),
        ("guard_delta", expected_guard_delta),
        ("transversality_eta", expected_transversality_eta),
        ("max_step", expected_max_step),
    )
    for key, expected in scalar_expectations:
        if expected is not None and abs(float(configuration[key]) - float(expected)) > 1e-13:
            raise ValueError(f"raw checkpoint {key} differs from the requested run")
    integer_expectations = (
        ("samples_per_axis", expected_samples_per_axis),
        ("max_jumps", expected_max_jumps),
    )
    for key, expected in integer_expectations:
        if expected is not None and int(configuration[key]) != int(expected):
            raise ValueError(f"raw checkpoint {key} differs from the requested run")

    walker = GarciaPassiveWalker(
        gamma=float(configuration["gamma"]),
        guard_delta=float(configuration["guard_delta"]),
        transversality_eta=float(configuration["transversality_eta"]),
        domain_bounds=[
            (float(interval[0]), float(interval[1]))
            for interval in configuration["base_bounds"]
        ],
        max_jumps=int(configuration["max_jumps"]),
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    if not np.allclose(
        np.asarray(configuration["handle_bounds"], dtype=np.float64),
        np.asarray(charts.handle_bounds, dtype=np.float64),
        atol=1e-13,
        rtol=0.0,
    ):
        raise ValueError("raw checkpoint handle chart disagrees with physical parameters")
    box_map = _AuditedGarciaSuspensionBoxMap(
        walker.system,
        charts,
        float(configuration["t_star"]),
        samples_per_axis=int(configuration["samples_per_axis"]),
        padding_cells=float(configuration["padding_cells"]),
        max_jumps=int(configuration["max_jumps"]),
        max_step=float(configuration["max_step"]),
        require_domain_path=bool(configuration["require_domain_path"]),
        diagnostics_limit=0,
    )

    class _CheckpointCallback:
        def __init__(self, chart_data: SuspensionAtlasCharts) -> None:
            self.charts = chart_data

        def __call__(self, *_args: object) -> list[object]:  # pragma: no cover
            raise AssertionError("raw checkpoint validation must not evaluate dynamics")

    model = build_cmgdb_atlas_model(
        _CheckpointCallback(charts),  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    )
    atlas = model.phaseSpace()
    if int(atlas.size()) != len(cells):
        raise ValueError("raw checkpoint cell count disagrees with its Atlas family")
    for index, (cell, dyadic) in enumerate(zip(cells, dyadic_cells, strict=True)):
        atlas_cell = atlas.cell(index)
        atlas_bounds = tuple(float(value) for value in atlas_cell.bounds)
        if int(atlas_cell.chart_id) != cell.chart_id or not np.allclose(
            atlas_bounds, cell.bounds, atol=1e-13, rtol=0.0
        ):
            raise ValueError(f"raw checkpoint cell {index} disagrees with native Atlas order")
        if not np.allclose(
            dyadic.bounds(charts), cell.bounds, atol=1e-13, rtol=0.0
        ):
            raise ValueError(f"raw checkpoint cell {index} has inconsistent dyadic geometry")

    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )
    recomputed_pairs = _quotient_cross_chart_pairs(cells, incidence, charts, atlas)
    if recomputed_pairs != pairs:
        raise ValueError("raw checkpoint quotient pairs disagree with cell geometry")
    quotient_neighbors_by_handle: dict[int, set[int]] = {}
    for first, second in pairs:
        handle = first if cells[first].chart_id == charts.handle_chart_id else second
        base = second if handle == first else first
        quotient_neighbors_by_handle.setdefault(handle, set()).add(base)
    frozen_quotient_neighbors = {
        handle: frozenset(neighbors)
        for handle, neighbors in quotient_neighbors_by_handle.items()
    }
    expected_samples = int(configuration["samples_per_axis"]) ** 4
    for source_index, source_provenance in provenance.items():
        if source_provenance.callback_invocations < 1:
            raise ValueError(f"raw source {source_index} was never evaluated")
        if (
            source_provenance.successful_samples + source_provenance.failed_samples
            != expected_samples
        ):
            raise ValueError(f"raw source {source_index} has inconsistent sample total")
        if sum(count for _reason, count in source_provenance.failure_reasons) != (
            source_provenance.failed_samples
        ):
            raise ValueError(f"raw source {source_index} has inconsistent failures")
        if (
            source_provenance.unresolved_stage_edges
            != tuple(sorted(set(source_provenance.unresolved_stage_edges)))
            or any(
                first > 2 * int(configuration["max_jumps"])
                or second > 2 * int(configuration["max_jumps"])
                or first >= second
                for first, second in source_provenance.unresolved_stage_edges
            )
        ):
            raise ValueError(f"raw source {source_index} has invalid stage-edge index")
        recovered = frozenset(
            int(target)
            for chart_id, bounds in source_provenance.target_pieces
            for target in atlas.cover(chart_id, bounds)
        )
        if recovered != relation[source_index]:
            raise ValueError(
                f"raw source {source_index} target pieces disagree with MapGraph image"
            )
        if source_provenance.mapgraph_empty_image != (not recovered):
            raise ValueError(f"raw source {source_index} has inconsistent empty image")
        if source_provenance.returned_pieces != len(source_provenance.target_pieces):
            raise ValueError(f"raw source {source_index} has inconsistent piece count")
        if source_provenance.explicit_empty_callback != (
            not source_provenance.target_pieces
        ):
            raise ValueError(f"raw source {source_index} has inconsistent callback empty")
        if source_provenance.active_target_cells != len(recovered):
            raise ValueError(f"raw source {source_index} has inconsistent active cover")
        recomputed_missing: set[GarciaDyadicCell] = set()
        recomputed_ambient = 0
        recomputed_outside = 0
        for target_chart, target_bounds in source_provenance.target_pieces:
            covered = tuple(int(value) for value in atlas.cover(target_chart, target_bounds))
            recomputed_outside += int(not covered)
            fine_cover, crossings = _dyadic_rectangle_cover(
                charts,
                target_chart,
                target_bounds,
                family.max_axis_depth,
            )
            recomputed_missing.update(
                cell for cell in fine_cover if family.covering_cell(cell) is None
            )
            recomputed_ambient += int(
                _piece_has_unmatched_ambient_exit(
                    charts,
                    incidence,
                    atlas,
                    target_chart,
                    target_bounds,
                    crossings,
                    source_provenance.target_pieces,
                    quotient_neighbors_by_handle=frozen_quotient_neighbors,
                )
            )
        if frozenset(recomputed_missing) != frozenset(
            source_provenance.missing_in_domain_witnesses
        ):
            raise ValueError(f"raw source {source_index} has inconsistent missing support")
        if source_provenance.missing_in_domain_target_cells != len(recomputed_missing):
            raise ValueError(f"raw source {source_index} has inconsistent missing count")
        if source_provenance.ambient_boundary_pieces != recomputed_ambient:
            raise ValueError(f"raw source {source_index} has inconsistent ambient exits")
        if source_provenance.wholly_outside_active_family_pieces != recomputed_outside:
            raise ValueError(f"raw source {source_index} has inconsistent outside pieces")
        if raw_open_exit_flags[source_index] != source_provenance.has_open_exit:
            raise ValueError(f"raw source {source_index} has inconsistent open-exit status")

    required_metric_keys = (
        "box_map_unique_source_boxes_evaluated",
        "new_whole_source_values_evaluated",
        "reused_whole_source_values",
        "box_map_sampled_points_evaluated",
        "unique_physical_endpoint_evaluations",
        "reused_lattice_endpoint_evaluations",
        "box_map_failed_samples",
        "box_map_empty_images",
        "box_map_unresolved_stage_edges",
    )
    for key in required_metric_keys:
        value = compute_metadata_header[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"raw checkpoint compute metric {key} is invalid")
    logical_samples = sum(
        item.successful_samples + item.failed_samples for item in provenance.values()
    )
    if int(compute_metadata_header["box_map_unique_source_boxes_evaluated"]) != len(
        cells
    ):
        raise ValueError("raw checkpoint source-box total is inconsistent")
    if (
        int(compute_metadata_header["new_whole_source_values_evaluated"])
        + int(compute_metadata_header["reused_whole_source_values"])
        != len(cells)
    ):
        raise ValueError("raw checkpoint whole-source cache totals are inconsistent")
    if int(compute_metadata_header["box_map_sampled_points_evaluated"]) != logical_samples:
        raise ValueError("raw checkpoint sampled-point total is inconsistent")
    if (
        int(compute_metadata_header["unique_physical_endpoint_evaluations"])
        + int(compute_metadata_header["reused_lattice_endpoint_evaluations"])
        != logical_samples
    ):
        raise ValueError("raw checkpoint endpoint-cache totals are inconsistent")
    if int(compute_metadata_header["box_map_failed_samples"]) != sum(
        item.failed_samples for item in provenance.values()
    ):
        raise ValueError("raw checkpoint failed-sample total is inconsistent")
    if int(compute_metadata_header["box_map_empty_images"]) != sum(
        item.explicit_empty_callback for item in provenance.values()
    ):
        raise ValueError("raw checkpoint empty-callback total is inconsistent")
    if int(compute_metadata_header["box_map_unresolved_stage_edges"]) != sum(
        len(item.unresolved_stage_edges) for item in provenance.values()
    ):
        raise ValueError("raw checkpoint unresolved-stage total is inconsistent")
    _validate_endpoint_precompute_metadata(
        compute_metadata_header.get("endpoint_precompute"),
        logical_samples=logical_samples,
        unique_endpoint_evaluations=int(
            compute_metadata_header["unique_physical_endpoint_evaluations"]
        ),
        reused_endpoint_evaluations=int(
            compute_metadata_header["reused_lattice_endpoint_evaluations"]
        ),
        new_source_values=int(
            compute_metadata_header["new_whole_source_values_evaluated"]
        ),
        reused_source_values=int(
            compute_metadata_header["reused_whole_source_values"]
        ),
    )

    _validate_raw_morse_snapshot(morse_snapshot, relation)
    resumed_metadata = dict(compute_metadata_header)
    resumed_metadata["raw_checkpoint_resumed"] = True
    return _GarciaRawRelationStage(
        family=family,
        walker=walker,
        charts=charts,
        box_map=box_map,
        model=model,
        morse_graph=_StoredMorseGraph(morse_snapshot),
        map_graph=_StoredMapGraph(relation),
        cells=tuple(cells),
        dyadic_cells=tuple(dyadic_cells),
        relation=relation,
        quotient_neighbor_pairs=pairs,
        source_provenance=provenance,
        raw_elapsed_seconds=float(raw_elapsed_header),
        compute_metadata=resumed_metadata,
        checkpoint_path=str(source),
        connectivity_audit=prevalidated_connectivity,
    )


def _finalize_garcia_raw_relation(
    stage: _GarciaRawRelationStage,
    *,
    reference_base_samples_per_stride: int,
    reference_handle_samples: int,
    reference_max_step: float,
    max_relation_edges: int | None,
    max_relation_storage_bytes: int | None,
    max_undirected_adjacencies: int | None,
    max_adjacency_storage_bytes: int | None,
) -> GarciaLocalRelationRun:
    """Run every derived audit after the atomic raw checkpoint boundary."""

    postprocess_started = time.perf_counter()
    cells = stage.cells
    relation = stage.relation
    connectivity = stage.connectivity_audit
    if connectivity is None:
        connectivity = audit_garcia_sparse_relation_connectivity(
            relation,
            stage.dyadic_cells,
            stage.quotient_neighbor_pairs,
            max_relation_edges=max_relation_edges,
            max_relation_storage_bytes=max_relation_storage_bytes,
            max_undirected_adjacencies=max_undirected_adjacencies,
            max_adjacency_storage_bytes=max_adjacency_storage_bytes,
        )
    disconnected = connectivity.disconnected_sources
    disconnected_components = connectivity.disconnected_image_components

    setup = GarciaWalkerAtlasSetup(
        walker=stage.walker,
        charts=stage.charts,
        box_map=stage.box_map,
        model=stage.model,
    )
    probes, reference_failures, gait_sources = _stored_gait_probes(
        setup,
        cells,
        base_samples_per_stride=reference_base_samples_per_stride,
        handle_samples=reference_handle_samples,
        reference_max_step=reference_max_step,
    )
    relation_view = _CellRelationView(cells, relation)
    reference_audit = _audit_atlas_endpoint_probes(
        relation_view,
        probes,
        cells,
        stage.box_map,
    )
    node_by_cell = _morse_node_by_phase_cell(stage.morse_graph)
    cell_index = relation_view.index_by_cell
    gait_source_indices = frozenset(cell_index[cell] for cell in gait_sources)
    reference_hits = Counter(
        node_by_cell[index]
        for index in gait_source_indices
        if index in node_by_cell
    )
    candidate_node = None
    if reference_hits:
        ordered_hits = sorted(reference_hits.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered_hits) == 1 or ordered_hits[0][1] > ordered_hits[1][1]:
            candidate_node = int(ordered_hits[0][0])

    s_cells = (
        frozenset(int(value) for value in stage.morse_graph.morse_set(candidate_node))
        if candidate_node is not None
        else frozenset()
    )
    f_s = frozenset(target for source in s_cells for target in relation[source])
    x_cells = s_cells | f_s
    a_cells = x_cells - s_cells

    recurrent = list(_relation_recurrent_components(relation, x_cells))
    boundary, margins = _candidate_boundary_audit(s_cells, cells, stage.charts)
    pair_second = tuple(
        source
        for source in sorted(a_cells)
        if frozenset(
            target for target in relation[source] if target in x_cells
        ) - a_cells
    )
    source_indices_by_label: dict[str, set[int]] = {}
    for probe in probes:
        source_indices_by_label.setdefault(probe.label, set()).add(
            cell_index[probe.source]
        )
    missing_reference_labels = tuple(
        sorted(
            label
            for label, source_indices in source_indices_by_label.items()
            if not source_indices.intersection(s_cells)
        )
    )
    refinement_locator = _safe_refinement_locator(
        relation,
        stage.source_provenance,
        disconnected,
        gait_source_indices,
    )
    candidate = GarciaLocalCandidateAudit(
        morse_node=candidate_node,
        s_cells=s_cells,
        x_cells=x_cells,
        a_cells=a_cells,
        reference_source_cells=gait_source_indices,
        reference_source_cells_in_candidate=frozenset(gait_source_indices & s_cells),
        reference_labels_total=len(source_indices_by_label),
        reference_labels_recovered=(
            len(source_indices_by_label) - len(missing_reference_labels)
        ),
        reference_missing_labels=missing_reference_labels,
        reference_endpoint_misses=len(reference_audit.missed),
        reference_evaluation_failures=len(reference_failures),
        failed_sources_in_s=tuple(
            source
            for source in sorted(s_cells)
            if stage.source_provenance[source].has_sample_failure
        ),
        unresolved_stage_sources_in_s=tuple(
            source
            for source in sorted(s_cells)
            if stage.source_provenance[source].unresolved_stage_edges
        ),
        open_exit_sources_in_s=tuple(
            source
            for source in sorted(s_cells)
            if stage.source_provenance[source].has_open_exit
        ),
        failed_sources_in_a=tuple(
            source
            for source in sorted(a_cells)
            if stage.source_provenance[source].has_sample_failure
        ),
        empty_sources_in_s=tuple(
            source for source in sorted(s_cells) if not relation[source]
        ),
        disconnected_sources_in_s=tuple(sorted(s_cells & disconnected)),
        disconnected_sources_in_a=tuple(sorted(a_cells & disconnected)),
        disconnected_image_components=disconnected_components,
        missing_in_domain_sources_in_s=tuple(
            source
            for source in sorted(s_cells)
            if stage.source_provenance[source].missing_in_domain_target_cells
        ),
        missing_in_domain_sources_in_x=tuple(
            source
            for source in sorted(x_cells)
            if stage.source_provenance[source].missing_in_domain_target_cells
        ),
        a_exit_sources=tuple(
            source
            for source in sorted(a_cells)
            if stage.source_provenance[source].has_open_exit
            or any(target not in x_cells for target in relation[source])
        ),
        pair_second_condition_violations=pair_second,
        recurrent_components_in_x=tuple(recurrent),
        touches_nonglued_ambient_boundary=boundary,
        minimum_boundary_margin_by_chart_axis=margins,
    )
    return GarciaLocalRelationRun(
        family=stage.family,
        walker=stage.walker,
        charts=stage.charts,
        box_map=stage.box_map,
        morse_graph=stage.morse_graph,
        map_graph=stage.map_graph,
        cells=cells,
        dyadic_cells=stage.dyadic_cells,
        relation=relation,
        node_by_cell=node_by_cell,
        quotient_neighbor_pairs=stage.quotient_neighbor_pairs,
        source_provenance=stage.source_provenance,
        reference_node_hits=dict(sorted(reference_hits.items())),
        reference_probe_count=len(reference_audit.witnesses),
        reference_miss_count=len(reference_audit.missed),
        reference_evaluation_failures=reference_failures,
        candidate=candidate,
        refinement_locator=refinement_locator,
        elapsed_seconds=(
            stage.raw_elapsed_seconds + time.perf_counter() - postprocess_started
        ),
        t_star=float(stage.box_map.t_star),
        samples_per_axis=int(stage.box_map.samples_per_axis),
        padding_cells=float(stage.box_map.padding_cells),
        max_step=float(stage.box_map.max_step),
        max_jumps=int(stage.box_map.max_jumps),
        connectivity_audit=connectivity,
        raw_checkpoint_path=stage.checkpoint_path,
        raw_compute_metadata=stage.compute_metadata,
        relation_reference=stage.relation_reference,
        family_provenance=stage.family_provenance,
    )


def resume_garcia_local_relation_from_raw_checkpoint(
    path: str | Path,
    *,
    expected_family: GarciaMixedDyadicFamily | None = None,
    t_star: float | None = None,
    samples_per_axis: int | None = None,
    padding_cells: float | None = None,
    gamma: float | None = None,
    guard_delta: float | None = None,
    transversality_eta: float | None = None,
    max_step: float | None = None,
    max_jumps: int | None = None,
    reference_base_samples_per_stride: int = 17,
    reference_handle_samples: int = 9,
    reference_max_step: float = 0.005,
    max_relation_edges: int | None = None,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
) -> GarciaLocalRelationRun:
    """Resume only the deterministic audits after a completed raw checkpoint."""

    stage = read_garcia_raw_relation_checkpoint(
        path,
        expected_family=expected_family,
        expected_t_star=t_star,
        expected_samples_per_axis=samples_per_axis,
        expected_padding_cells=padding_cells,
        expected_gamma=gamma,
        expected_guard_delta=guard_delta,
        expected_transversality_eta=transversality_eta,
        expected_max_step=max_step,
        expected_max_jumps=max_jumps,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )
    return _finalize_garcia_raw_relation(
        stage,
        reference_base_samples_per_stride=reference_base_samples_per_stride,
        reference_handle_samples=reference_handle_samples,
        reference_max_step=reference_max_step,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )


def migrate_garcia_relation_to_raw_checkpoint(
    payload_or_path: Mapping[str, object] | str | Path,
    output_path: str | Path,
) -> Path:
    """Create a v1 raw checkpoint from a fully provenanced v3/v4 relation.

    This performs no ODE evaluation.  It is intentionally unavailable for the
    older depth-8 fixture, which lacks the target-piece and stage provenance
    needed for strict reconstruction.
    """

    payload = (
        read_garcia_local_relation(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    validate_garcia_resume_fingerprint(payload)
    metadata = payload["metadata"]
    charts_payload = payload["charts"]
    cell_payloads = payload["cells"]
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(charts_payload, Mapping)
        or not isinstance(cell_payloads, list)
    ):
        raise ValueError("legacy relation payload is malformed")
    walker = GarciaPassiveWalker(
        gamma=float(metadata["gamma"]),
        guard_delta=float(metadata["guard_delta"]),
        transversality_eta=float(metadata["transversality_eta"]),
        domain_bounds=[
            (float(interval[0]), float(interval[1]))
            for interval in charts_payload["base"]["bounds"]
        ],
        max_jumps=int(metadata["max_jumps"]),
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    dyadic_cells: list[GarciaDyadicCell] = []
    cells: list[AtlasWalkerCell] = []
    relation: dict[int, frozenset[int]] = {}
    provenance: dict[int, GarciaSourceExitProvenance] = {}
    for expected_index, raw in enumerate(cell_payloads):
        if not isinstance(raw, Mapping) or int(raw["index"]) != expected_index:
            raise ValueError("legacy relation cells are incomplete or unordered")
        open_exit = raw.get("open_exit")
        if not isinstance(open_exit, Mapping) or "target_pieces" not in open_exit:
            raise ValueError(
                "legacy relation lacks raw target pieces and cannot be migrated"
            )
        if "unresolved_stage_edges" not in open_exit:
            raise ValueError(
                "legacy relation lacks stage provenance and cannot be migrated"
            )
        dyadic = GarciaDyadicCell(
            int(raw["chart_id"]),
            int(raw["axis_depth"]),
            tuple(int(value) for value in raw["coordinates"]),
        )
        dyadic_cells.append(dyadic)
        cells.append(
            AtlasWalkerCell(
                expected_index,
                dyadic.chart_id,
                tuple(float(value) for value in raw["bounds"]),
            )
        )
        relation[expected_index] = frozenset(int(value) for value in raw["image"])
        provenance[expected_index] = _source_provenance_from_dict(
            expected_index, open_exit
        )
    family = GarciaMixedDyadicFamily(
        tuple(sorted(dyadic_cells)),
        role=str(metadata["family_role"]),
    )
    # This no-ODE validation recomputes all derivable geometry, relation-cover,
    # quotient, SCC, candidate, and locator fields before extracting raw data.
    _validate_persisted_locator_and_relation(payload, charts, family)

    box_map = _AuditedGarciaSuspensionBoxMap(
        walker.system,
        charts,
        float(metadata["t_star"]),
        samples_per_axis=int(metadata["samples_per_axis"]),
        padding_cells=float(metadata["padding_cells"]),
        max_jumps=int(metadata["max_jumps"]),
        max_step=float(metadata["max_step"]),
        require_domain_path=True,
        diagnostics_limit=0,
    )

    class _MigrationCallback:
        def __init__(self, chart_data: SuspensionAtlasCharts) -> None:
            self.charts = chart_data

        def __call__(self, *_args: object) -> list[object]:  # pragma: no cover
            raise AssertionError("raw migration must not evaluate dynamics")

    model = build_cmgdb_atlas_model(
        _MigrationCallback(charts),  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    )
    morse_payload = payload["morse_graph"]
    if not isinstance(morse_payload, Mapping):
        raise ValueError("legacy relation Morse graph is malformed")
    canonical_morse = _canonicalize_morse_snapshot(morse_payload)
    _validate_raw_morse_snapshot(canonical_morse, relation)
    compute_metadata = {
        "box_map_unique_source_boxes_evaluated": int(
            metadata["box_map_unique_source_boxes_evaluated"]
        ),
        "new_whole_source_values_evaluated": int(
            metadata.get("new_whole_source_values_evaluated", len(cells))
        ),
        "reused_whole_source_values": int(metadata.get("reused_whole_source_values", 0)),
        "box_map_sampled_points_evaluated": int(
            metadata["box_map_sampled_points_evaluated"]
        ),
        "unique_physical_endpoint_evaluations": int(
            metadata["unique_physical_endpoint_evaluations"]
        ),
        "reused_lattice_endpoint_evaluations": int(
            metadata["reused_lattice_endpoint_evaluations"]
        ),
        "box_map_failed_samples": int(metadata["box_map_failed_samples"]),
        "box_map_empty_images": int(metadata["box_map_empty_images"]),
        "box_map_unresolved_stage_edges": int(
            metadata["box_map_unresolved_stage_edges"]
        ),
        "endpoint_precompute": metadata.get("endpoint_precompute"),
        "migrated_without_ode_recomputation": True,
        "migration_source_derived_revision": metadata["resume_fingerprint"][
            "algorithm_revision"
        ],
    }
    stage = _GarciaRawRelationStage(
        family=family,
        walker=walker,
        charts=charts,
        box_map=box_map,
        model=model,
        morse_graph=_StoredMorseGraph(canonical_morse),
        map_graph=_StoredMapGraph(relation),
        cells=tuple(cells),
        dyadic_cells=tuple(dyadic_cells),
        relation=relation,
        quotient_neighbor_pairs=tuple(
            (int(pair[0]), int(pair[1]))
            for pair in payload["quotient_neighbor_pairs"]
        ),
        source_provenance=provenance,
        raw_elapsed_seconds=float(metadata["elapsed_seconds"]),
        compute_metadata=compute_metadata,
        checkpoint_path=str(Path(output_path)),
    )
    target = write_garcia_raw_relation_checkpoint(stage, output_path)
    # Read it back through the strict v1 validator before claiming migration.
    read_garcia_raw_relation_checkpoint(target, expected_family=family)
    return target


def _collect_garcia_source_provenance(
    box_map: _AuditedGarciaSuspensionBoxMap,
    tracked: _OpenExitTrackingBoxMap,
    cells: Sequence[AtlasWalkerCell],
    relation: Mapping[int, Collection[int]],
) -> tuple[
    dict[int, GarciaSourceExitProvenance],
    Mapping[tuple[object, ...], object],
    Mapping[tuple[object, ...], object],
]:
    """Collect the complete per-source records after CMGDB has evaluated them."""

    diagnostic_records, diagnostic_invocations = box_map.source_records()
    coverage_records = tracked.records()
    provenance: dict[int, GarciaSourceExitProvenance] = {}
    for index, cell in enumerate(cells):
        key = _source_key(cell.chart_id, cell.bounds)
        diagnostic = diagnostic_records.get(key)
        coverage = coverage_records.get(key)
        if diagnostic is None or coverage is None:
            raise RuntimeError(f"missing callback provenance for Atlas source {index}")
        reasons = Counter(failure.reason for failure in diagnostic.failures)
        provenance[index] = GarciaSourceExitProvenance(
            source_index=index,
            callback_invocations=max(
                diagnostic_invocations.get(key, 0), coverage.callback_invocations
            ),
            successful_samples=diagnostic.successful_samples,
            failed_samples=len(diagnostic.failures),
            failure_reasons=tuple(sorted(reasons.items())),
            unresolved_stage_edges=diagnostic.unresolved_stage_edges,
            returned_pieces=coverage.returned_pieces,
            explicit_empty_callback=coverage.explicit_empty_callback,
            mapgraph_empty_image=not relation[index],
            active_target_cells=coverage.active_target_cells,
            missing_in_domain_target_cells=len(coverage.missing_in_domain_cells),
            missing_in_domain_witnesses=tuple(sorted(coverage.missing_in_domain_cells)),
            target_pieces=coverage.target_pieces,
            ambient_boundary_pieces=coverage.ambient_boundary_pieces,
            wholly_outside_active_family_pieces=(
                coverage.wholly_outside_active_family_pieces
            ),
        )
    return provenance, diagnostic_records, coverage_records


def compute_garcia_local_relation(
    family: GarciaMixedDyadicFamily,
    *,
    coordinate_system: str = "physical",
    t_star: float = 0.5,
    samples_per_axis: int = 3,
    padding_cells: float = 1.0,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    max_step: float = 0.02,
    max_jumps: int = 20,
    reference_base_samples_per_stride: int = 17,
    reference_handle_samples: int = 9,
    reference_max_step: float = 0.005,
    precompute_workers: int = 0,
    reuse_evaluations_from: GarciaLocalRelationRun | None = None,
    raw_checkpoint_path: str | Path | None = None,
    csr_bundle_path: str | Path | None = None,
    family_provenance: Mapping[str, object] | None = None,
    max_csr_payload_bytes: int | None = None,
    max_native_cache_bytes: int | None = None,
    max_relation_edges: int | None = None,
    max_relation_storage_bytes: int | None = None,
    max_undirected_adjacencies: int | None = None,
    max_adjacency_storage_bytes: int | None = None,
) -> GarciaLocalRelationRun:
    """Evaluate one complete mixed-depth local relation and its gait audit."""

    if precompute_workers < 0:
        raise ValueError("precompute_workers must be non-negative")
    if coordinate_system not in {"physical", "guard_aligned"}:
        raise ValueError("coordinate_system must be 'physical' or 'guard_aligned'")
    for name, cap in (
        ("max_relation_edges", max_relation_edges),
        ("max_relation_storage_bytes", max_relation_storage_bytes),
        ("max_undirected_adjacencies", max_undirected_adjacencies),
        ("max_adjacency_storage_bytes", max_adjacency_storage_bytes),
        ("max_csr_payload_bytes", max_csr_payload_bytes),
        ("max_native_cache_bytes", max_native_cache_bytes),
    ):
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError(f"{name} must be a nonnegative exact integer or None")
    if reuse_evaluations_from is not None and raw_checkpoint_path is not None:
        raise ValueError(
            "acceptance raw checkpoints require a fresh-family evaluation; "
            "whole-source replay has different per-run endpoint accounting"
        )
    if csr_bundle_path is not None and raw_checkpoint_path is not None:
        raise ValueError("CSR bundles replace the legacy standalone raw checkpoint")
    if csr_bundle_path is None and family_provenance is not None:
        raise ValueError("family provenance is meaningful only for a CSR-backed run")
    if csr_bundle_path is not None:
        csr_target = Path(csr_bundle_path)
        if csr_target.exists():
            raise FileExistsError(f"refusing to overwrite CSR bundle {csr_target}")
        csr_target.parent.mkdir(parents=True, exist_ok=True)
        if (
            max_relation_edges is None
            or max_csr_payload_bytes is None
            or max_native_cache_bytes is None
        ):
            raise ValueError(
                "CSR persistence requires explicit edge, disk-payload, and "
                "native-cache byte caps"
            )
    started = time.perf_counter()
    if coordinate_system == "guard_aligned":
        walker = GuardAlignedGarciaPassiveWalker(
            gamma=gamma,
            guard_delta=guard_delta,
            transversality_eta=transversality_eta,
            domain_bounds=list(GUARD_ALIGNED_DOMAIN_BOUNDS),
            max_jumps=max_jumps,
        )
        charts = garcia_guard_aligned_atlas_charts(
            base_bounds=walker.domain_bounds,
            guard_delta=guard_delta,
            transversality_eta=transversality_eta,
        )
        incidence = GuardAlignedGarciaWalkerQuotientIncidence(
            charts,
            transversality_eta=walker.transversality_eta,
        )
    else:
        walker = GarciaPassiveWalker(
            gamma=gamma,
            guard_delta=guard_delta,
            transversality_eta=transversality_eta,
            domain_bounds=list(DEFAULT_BASE_BOUNDS),
            max_jumps=max_jumps,
        )
        charts = garcia_passive_walker_atlas_charts(
            base_bounds=walker.domain_bounds,
            guard_delta=guard_delta,
            transversality_eta=transversality_eta,
        )
        incidence = GarciaWalkerQuotientIncidence(
            charts,
            phi_dot_min=charts.base_bounds[3][0],
            phi_dot_max=charts.base_bounds[3][1],
            transversality_eta=walker.transversality_eta,
        )
    if reuse_evaluations_from is None:
        box_map = _AuditedGarciaSuspensionBoxMap(
            walker.system,
            charts,
            t_star,
            samples_per_axis=samples_per_axis,
            padding_cells=padding_cells,
            max_jumps=max_jumps,
            max_step=max_step,
            require_domain_path=True,
            diagnostics_limit=0,
        )
    else:
        previous = reuse_evaluations_from
        compatible = bool(
            _garcia_coordinate_system(previous.walker) == coordinate_system
            and
            previous.family.cells == family.cells
            and abs(previous.t_star - float(t_star)) <= 1.0e-13
            and previous.samples_per_axis == int(samples_per_axis)
            and abs(previous.padding_cells - float(padding_cells)) <= 1.0e-13
            and abs(previous.walker.gamma - float(gamma)) <= 1.0e-13
            and abs(previous.walker.guard_delta - float(guard_delta)) <= 1.0e-13
            and abs(
                previous.walker.transversality_eta - float(transversality_eta)
            )
            <= 1.0e-13
            and previous.max_step == float(max_step)
            and previous.max_jumps == int(max_jumps)
            and np.allclose(
                previous.charts.base_bounds,
                charts.base_bounds,
                atol=1.0e-13,
                rtol=0.0,
            )
            and np.allclose(
                previous.charts.handle_bounds,
                charts.handle_bounds,
                atol=1.0e-13,
                rtol=0.0,
            )
        )
        if not compatible:
            raise ValueError(
                "Garcia evaluation reuse requires the identical active family "
                "and physical box-map configuration"
            )
        box_map = previous.box_map
    tracked = _OpenExitTrackingBoxMap(box_map, family, incidence)
    model = build_cmgdb_atlas_model(
        tracked,  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    )
    tracked.bind_atlas(model.phaseSpace())

    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB fork is not installed") from error

    box_map.reset_diagnostics()
    box_map.reset_source_records(
        preserve_evaluation_cache=reuse_evaluations_from is not None
    )
    tracked.reset_records()
    if precompute_workers:
        atlas = model.phaseSpace()
        cached_source_keys = box_map.source_value_cache_keys()
        new_source_boxes = tuple(
            (
                int(atlas.cell(index).chart_id),
                tuple(float(value) for value in atlas.cell(index).bounds),
            )
            for index in range(int(atlas.size()))
            if _source_key(
                int(atlas.cell(index).chart_id),
                atlas.cell(index).bounds,
            )
            not in cached_source_keys
        )
        precompute = precompute_garcia_endpoint_cache(
            new_source_boxes,
            GarciaEndpointPrecomputeConfig(
                t_star=float(t_star),
                samples_per_axis=int(samples_per_axis),
                padding_cells=float(padding_cells),
                gamma=float(gamma),
                guard_delta=float(guard_delta),
                transversality_eta=float(transversality_eta),
                domain_bounds=tuple(
                    (float(lower), float(upper))
                    for lower, upper in walker.domain_bounds
                ),
                max_step=float(max_step),
                max_jumps=int(max_jumps),
                atol=float(box_map.atol),
                require_domain_path=bool(box_map.require_domain_path),
                coordinate_system=coordinate_system,
            ),
            workers=int(precompute_workers),
        )
        if reuse_evaluations_from is None:
            box_map.seed_point_cache(
                precompute.entries,
                precompute_result=precompute,
            )
        else:
            box_map.extend_point_cache(
                precompute.entries,
                precompute_result=precompute,
            )
    native_cap_values = {
        "CMGDB_MAPGRAPH_HARD_MAX_VERTICES": len(family.cells),
        "CMGDB_MAPGRAPH_HARD_MAX_EDGES": max_relation_edges,
        "CMGDB_MAPGRAPH_HARD_MAX_CACHE_BYTES": (
            max_native_cache_bytes
        ),
    }
    previous_native_caps: dict[str, str | None] = {}
    if csr_bundle_path is not None:
        for key, value in native_cap_values.items():
            if value is None:  # guarded above for all practical CSR calls
                continue
            previous_native_caps[key] = os.environ.get(key)
            os.environ[key] = str(int(value))
    try:
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    finally:
        for key, previous in previous_native_caps.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    native_map_graph_ref = (
        weakref.ref(map_graph) if csr_bundle_path is not None else None
    )
    cells = _atlas_cells(morse_graph, int(map_graph.num_vertices()))
    geometry = _family_geometry_map(family, charts)
    dyadic_cells = tuple(
        geometry[_source_key(cell.chart_id, cell.bounds)] for cell in cells
    )
    relation_reference: Mapping[str, object] | None = None
    temporary_bundle: Path | None = None
    if csr_bundle_path is not None:
        from .garcia_passive_walker_csr import write_and_load_garcia_map_graph_csr

        bundle_target = Path(csr_bundle_path)
        temporary_bundle = Path(
            tempfile.mkdtemp(
                prefix=f".{bundle_target.name}.",
                suffix=".tmp",
                dir=bundle_target.parent,
            )
        )
        provisional_stage = _GarciaRawRelationStage(
            family=family,
            walker=walker,
            charts=charts,
            box_map=box_map,
            model=model,
            morse_graph=morse_graph,
            map_graph=map_graph,
            cells=cells,
            dyadic_cells=dyadic_cells,
            relation={},
            quotient_neighbor_pairs=tracked.quotient_neighbor_pairs,
            source_provenance={},
            raw_elapsed_seconds=0.0,
            compute_metadata={},
            family_provenance=family_provenance,
        )
        try:
            relation, relation_reference = write_and_load_garcia_map_graph_csr(
                provisional_stage,
                temporary_bundle / "relation.csr",
                family_provenance=family_provenance,
                max_vertices=len(family.cells),
                max_edges=int(max_relation_edges),
                max_payload_bytes=int(max_csr_payload_bytes),
            )
        except BaseException:
            shutil.rmtree(temporary_bundle, ignore_errors=True)
            raise
        del provisional_stage
    else:
        mutable_relation: dict[int, frozenset[int]] = {}
        streamed_relation_edges = 0
        for index in range(len(cells)):
            targets = frozenset(int(target) for target in map_graph.adjacencies(index))
            mutable_relation[index] = targets
            streamed_relation_edges += len(targets)
            if (
                max_relation_edges is not None
                and streamed_relation_edges > max_relation_edges
            ):
                raise GarciaRelationSizeLimitExceeded(
                    kind="raw_relation_edge",
                    limit=max_relation_edges,
                    observed=streamed_relation_edges,
                )
            estimated_relation_bytes = _estimated_raw_relation_storage_bytes(
                index + 1, streamed_relation_edges
            )
            if (
                max_relation_storage_bytes is not None
                and estimated_relation_bytes > max_relation_storage_bytes
            ):
                raise GarciaRelationSizeLimitExceeded(
                    kind="raw_relation_estimated_storage_byte",
                    limit=max_relation_storage_bytes,
                    observed=estimated_relation_bytes,
                )
        relation = mutable_relation

    try:
        provenance, diagnostic_records, coverage_records = (
            _collect_garcia_source_provenance(box_map, tracked, cells, relation)
        )
        stored_morse_snapshot = _canonicalize_morse_snapshot(
            _morse_snapshot(morse_graph)
        )
        raw_stage = _GarciaRawRelationStage(
            family=family,
            walker=walker,
            charts=charts,
            box_map=box_map,
            model=model,
            morse_graph=_StoredMorseGraph(stored_morse_snapshot),
            map_graph=relation if relation_reference is not None else map_graph,
            cells=cells,
            dyadic_cells=dyadic_cells,
            relation=relation,
            quotient_neighbor_pairs=tracked.quotient_neighbor_pairs,
            source_provenance=provenance,
            raw_elapsed_seconds=time.perf_counter() - started,
            compute_metadata=_raw_compute_metadata(box_map),
            checkpoint_path=(
                str(Path(csr_bundle_path))
                if csr_bundle_path is not None
                else None
                if raw_checkpoint_path is None
                else str(Path(raw_checkpoint_path))
            ),
            relation_reference=relation_reference,
            family_provenance=family_provenance,
        )
    except BaseException:
        if temporary_bundle is not None:
            shutil.rmtree(temporary_bundle, ignore_errors=True)
        raise
    if raw_checkpoint_path is not None or csr_bundle_path is not None:
        if relation_reference is None:
            assert raw_checkpoint_path is not None
            write_garcia_raw_relation_checkpoint(raw_stage, raw_checkpoint_path)
        else:
            from .garcia_passive_walker_csr import (
                complete_garcia_csr_bundle,
            )

            assert temporary_bundle is not None and csr_bundle_path is not None
            # The mmap CSR and canonical Morse snapshot are now complete.
            # Release all native graph owners before serializing provenance or
            # traversing global audits, then verify the lifetime boundary.
            del map_graph, morse_graph, diagnostic_records, coverage_records
            if native_map_graph_ref is not None and native_map_graph_ref() is not None:
                shutil.rmtree(temporary_bundle, ignore_errors=True)
                raise RuntimeError(
                    "native MapGraph remained alive after authoritative CSR write"
                )
            try:
                complete_garcia_csr_bundle(
                    raw_stage,
                    temporary_bundle,
                    csr_bundle_path,
                )
            except BaseException:
                shutil.rmtree(temporary_bundle, ignore_errors=True)
                raise
            _validate_raw_morse_snapshot(stored_morse_snapshot, relation)
        tracked.reset_records()
        box_map.release_evaluation_caches()
        raw_stage = replace(raw_stage, map_graph=relation)
        # Drop the native MapGraph and copied callback-record dictionaries before
        # sparse incidence/SCC postprocessing.  The immutable Python relation and
        # per-source provenance above are the complete audited inputs from here.
        if relation_reference is None:
            del map_graph, morse_graph, diagnostic_records, coverage_records
    return _finalize_garcia_raw_relation(
        raw_stage,
        reference_base_samples_per_stride=reference_base_samples_per_stride,
        reference_handle_samples=reference_handle_samples,
        reference_max_step=reference_max_step,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )


def _expanded_fine_cells(
    charts: SuspensionAtlasCharts,
    cell: AtlasWalkerCell,
    target_axis_depth: int,
    collar_fraction: float,
) -> frozenset[GarciaDyadicCell]:
    chart_bounds = np.asarray(charts.bounds_for(cell.chart_id), dtype=np.float64)
    span = chart_bounds[:, 1] - chart_bounds[:, 0]
    lower = cell.lower - collar_fraction * span
    upper = cell.upper + collar_fraction * span
    return _dyadic_rectangle_cover(
        charts,
        cell.chart_id,
        tuple(float(value) for value in (*lower, *upper)),
        target_axis_depth,
    )[0]


def _relation_saturated_family(
    *,
    family: GarciaMixedDyadicFamily,
    charts: SuspensionAtlasCharts,
    cells: Sequence[AtlasWalkerCell],
    dyadic_cells: Sequence[GarciaDyadicCell],
    x_cells: Collection[int],
    missing_by_source: Mapping[int, Collection[GarciaDyadicCell]],
    target_pieces_by_source: Mapping[
        int, Collection[tuple[int, tuple[float, ...]]]
    ],
    quotient_neighbor_pairs: Collection[tuple[int, int]],
    target_axis_depth: int,
    collar_fraction: float,
    include_target_support: bool = True,
    retain_unselected: bool = True,
    max_output_cells: int | None = None,
) -> GarciaMixedDyadicFamily:
    if not x_cells:
        raise ValueError("cannot refine without a reference-carrying candidate")
    if target_axis_depth < family.max_axis_depth:
        raise ValueError("target_axis_depth cannot coarsen the active family")
    if not np.isfinite(collar_fraction) or collar_fraction <= 0.0:
        raise ValueError("collar_fraction must be finite and positive")
    if max_output_cells is not None and max_output_cells < 1:
        raise ValueError("max_output_cells must be positive")

    selected: set[GarciaDyadicCell] = set()
    additions: set[GarciaDyadicCell] = set()
    output_cell_lower_bound = len(family.cells) if retain_unselected else 0

    def check_cap() -> None:
        if (
            max_output_cells is not None
            and output_cell_lower_bound > max_output_cells
        ):
            raise GarciaRefinementSizeLimitExceeded(
                limit=max_output_cells,
                lower_bound=output_cell_lower_bound,
            )

    def select_source_cell(cell: GarciaDyadicCell) -> None:
        nonlocal output_cell_lower_bound
        if cell in selected:
            return
        selected.add(cell)
        descendants = 2 ** (4 * (target_axis_depth - cell.axis_depth))
        output_cell_lower_bound += descendants - int(retain_unselected)
        check_cap()

    def add_missing_cell(cell: GarciaDyadicCell) -> None:
        nonlocal output_cell_lower_bound
        if cell in additions:
            return
        additions.add(cell)
        output_cell_lower_bound += 1
        check_cap()

    check_cap()

    def select_fine(fine: GarciaDyadicCell) -> None:
        covering = family.covering_cell(fine)
        if covering is None:
            add_missing_cell(fine)
        elif covering.axis_depth < target_axis_depth:
            select_source_cell(covering)
        elif not retain_unselected:
            select_source_cell(covering)

    for source in x_cells:
        token = dyadic_cells[source]
        select_source_cell(token)
        for fine in _expanded_fine_cells(
            charts,
            cells[source],
            target_axis_depth,
            collar_fraction,
        ):
            select_fine(fine)
        # Refine the complete in-domain geometric F(source) support, including
        # target cells outside the collar.  Merely retaining a coarse outer
        # carrier would hide a support-expansion step at the next resolution.
        if include_target_support:
            for target_chart, target_bounds in target_pieces_by_source.get(source, ()):
                fine_targets, _crosses_ambient = _dyadic_rectangle_cover(
                    charts,
                    target_chart,
                    target_bounds,
                    target_axis_depth,
                )
                for fine in fine_targets:
                    select_fine(fine)
    for source in x_cells:
        for missing in missing_by_source.get(source, ()):
            if missing.axis_depth == target_axis_depth:
                add_missing_cell(missing)

    # A refined support must be closed under the two quotient seam
    # identifications.  The stored pairs are exact incidences between current
    # Atlas leaves; selecting either side refines the other side conservatively.
    changed = True
    while changed:
        changed = False
        for first, second in quotient_neighbor_pairs:
            first_token = dyadic_cells[first]
            second_token = dyadic_cells[second]
            if first_token in selected and second_token not in selected:
                select_source_cell(second_token)
                changed = True
            if second_token in selected and first_token not in selected:
                select_source_cell(first_token)
                changed = True

    result: set[GarciaDyadicCell] = set(additions)
    for cell in family.cells:
        if cell in selected and cell.axis_depth < target_axis_depth:
            result.update(cell.descendants(target_axis_depth))
        elif retain_unselected or cell in selected:
            result.add(cell)
    # Additions can be covered by an existing coarse cell when a serialized
    # witness came from an earlier maximum depth.  Keep only the antichain.
    ordered = sorted(
        result,
        key=lambda cell: (cell.chart_id, cell.axis_depth, cell.coordinates),
    )
    antichain: list[GarciaDyadicCell] = []
    accepted: set[GarciaDyadicCell] = set()
    for cell in ordered:
        if any(
            GarciaDyadicCell(
                cell.chart_id,
                depth,
                tuple(
                    value >> (cell.axis_depth - depth)
                    for value in cell.coordinates
                ),
            )
            in accepted
            for depth in range(cell.axis_depth + 1)
        ):
            continue
        antichain.append(cell)
        accepted.add(cell)
    return GarciaMixedDyadicFamily(
        tuple(sorted(antichain)),
        role=(
            f"{'mixed ambient' if retain_unselected else 'local open window'}; "
            f"relation support with physical collar fraction {collar_fraction:g}, "
            f"refined through axis depth {target_axis_depth}"
        ),
    )


def relation_saturated_refinement(
    run: GarciaLocalRelationRun,
    *,
    target_axis_depth: int,
    collar_fraction: float = 0.125,
    retain_unselected: bool = True,
    max_output_cells: int | None = None,
) -> GarciaMixedDyadicFamily:
    """Refine the candidate, its in-domain images, and a physical collar.

    Missing fine target cells are inserted explicitly.  Existing coarse outer
    cells are retained unless selected for refinement, yielding an antichain and
    a complete in-domain cover.  Re-running this function after each computed
    relation gives the intended saturation iteration.
    """

    if run.candidate.morse_node is None:
        raise ValueError("cannot refine without a reference-carrying candidate")
    return _relation_saturated_family(
        family=run.family,
        charts=run.charts,
        cells=run.cells,
        dyadic_cells=run.dyadic_cells,
        x_cells=run.candidate.x_cells,
        missing_by_source={
            source: run.source_provenance[source].missing_in_domain_witnesses
            for source in run.candidate.x_cells
        },
        target_pieces_by_source={
            source: run.source_provenance[source].target_pieces
            for source in run.candidate.x_cells
        },
        quotient_neighbor_pairs=run.quotient_neighbor_pairs,
        target_axis_depth=target_axis_depth,
        collar_fraction=collar_fraction,
        retain_unselected=retain_unselected,
        max_output_cells=max_output_cells,
    )


def exit_aware_index_pair_refinement(
    run: GarciaLocalRelationRun,
    *,
    target_axis_depth: int,
    max_output_cells: int | None = None,
) -> tuple[GarciaMixedDyadicFamily, GarciaExitAwareSupportAudit]:
    """Close only missing in-domain support of the local pair interior.

    This is the same-depth saturation step used after the physical-collar
    window has been evaluated.  It intentionally does *not* add images of the
    exit set ``L=A``.  Those values are part of the open-exit certificate.
    """

    if run.candidate.morse_node is None or not run.candidate.x_cells:
        raise ValueError("cannot audit support without a reference-carrying pair")
    if target_axis_depth != run.family.max_axis_depth:
        raise ValueError(
            "exit-aware support closure is same-depth; refine the locator first"
        )
    n_cells = run.candidate.x_cells
    l_cells = run.candidate.a_cells
    n_minus_l = n_cells - l_cells
    expandable_sources = frozenset(
        source
        for source in n_minus_l
        if run.source_provenance[source].missing_in_domain_witnesses
    )
    added: set[GarciaDyadicCell] = set()
    for source in expandable_sources:
        for witness in run.source_provenance[source].missing_in_domain_witnesses:
            if run.family.covering_cell(witness) is not None:
                continue
            added.add(witness)
            if (
                max_output_cells is not None
                and len(run.family.cells) + len(added) > max_output_cells
            ):
                raise GarciaRefinementSizeLimitExceeded(
                    limit=max_output_cells,
                    lower_bound=len(run.family.cells) + len(added),
                )
    added_cells = frozenset(added)
    nonexpandable_open = frozenset(
        source
        for source in n_minus_l
        if run.source_provenance[source].has_open_exit
        and not run.source_provenance[source].missing_in_domain_witnesses
    )
    preserved_l_exits = frozenset(
        source
        for source in l_cells
        if run.source_provenance[source].has_open_exit
        or any(target not in n_cells for target in run.relation[source])
    )
    terminal_blockers: list[str] = []
    if not run.candidate.reference_recovered:
        terminal_blockers.append("reference_gait_not_recovered")
    if nonexpandable_open:
        terminal_blockers.append("nonexpandable_open_exit_in_N_minus_L")
    if run.candidate.pair_second_condition_violations:
        terminal_blockers.append("finite_pair_second_condition_violation")
    if run.candidate.touches_nonglued_ambient_boundary:
        terminal_blockers.append("S_touches_nonglued_ambient_boundary")
    disconnected_without_missing = frozenset(
        source
        for source in run.candidate.disconnected_sources_in_s
        if not run.source_provenance[source].missing_in_domain_witnesses
    )
    if disconnected_without_missing:
        terminal_blockers.append("disconnected_S_image_without_missing_support")
    audit = GarciaExitAwareSupportAudit(
        n_cells=frozenset(n_cells),
        l_cells=frozenset(l_cells),
        n_minus_l_cells=frozenset(n_minus_l),
        expandable_sources=expandable_sources,
        added_cells=added_cells,
        nonexpandable_open_exit_sources=nonexpandable_open,
        preserved_l_exit_sources=preserved_l_exits,
        pair_second_condition_violations=frozenset(
            run.candidate.pair_second_condition_violations
        ),
        terminal_blockers=tuple(terminal_blockers),
    )
    if not added_cells:
        return run.family, audit

    result = set(run.family.cells)
    result.update(added_cells)
    # Missing witnesses are emitted at the active family's maximum depth.  A
    # covering-cell check above prevents ancestor/descendant overlap, but keep
    # this assertion explicit because an antichain is scientific provenance.
    expanded = GarciaMixedDyadicFamily(
        tuple(sorted(result)),
        role=(
            f"local open window; exit-aware N\\L image support through axis "
            f"depth {target_axis_depth}"
        ),
    )
    return expanded, audit


def safe_locator_refinement(
    run: GarciaLocalRelationRun,
    *,
    target_axis_depth: int,
    collar_fraction: float = 0.0625,
    include_target_support: bool = True,
    retain_unselected: bool = False,
    max_output_cells: int | None = None,
) -> GarciaMixedDyadicFamily:
    """Refine around the failure-pruned locator without pruning the relation."""

    if not run.refinement_locator.selected_component:
        raise ValueError("safe refinement locator has no recurrent component")
    return _relation_saturated_family(
        family=run.family,
        charts=run.charts,
        cells=run.cells,
        dyadic_cells=run.dyadic_cells,
        x_cells=run.refinement_locator.seed_cells,
        missing_by_source={
            source: run.source_provenance[source].missing_in_domain_witnesses
            for source in run.refinement_locator.seed_cells
        },
        target_pieces_by_source={
            source: run.source_provenance[source].target_pieces
            for source in run.refinement_locator.seed_cells
        },
        quotient_neighbor_pairs=run.quotient_neighbor_pairs,
        target_axis_depth=target_axis_depth,
        collar_fraction=collar_fraction,
        include_target_support=include_target_support,
        retain_unselected=retain_unselected,
        max_output_cells=max_output_cells,
    )


def read_garcia_local_relation(
    path: str | Path,
    *,
    require_resume_fingerprint: bool = True,
) -> dict[str, object]:
    """Read and minimally validate a persisted local-relation payload."""

    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "garcia-walker-open-exit-atlas-relation-v1":
        raise ValueError("resume file is not a Garcia open-exit relation")
    if require_resume_fingerprint:
        validate_garcia_resume_fingerprint(payload)
    return payload


def persisted_refinement_size_bounds(
    payload_or_path: Mapping[str, object] | str | Path,
    *,
    target_axis_depth: int,
    selector: str = "candidate_X",
    retain_unselected: bool = True,
) -> dict[str, int | float | bool]:
    """Return cheap lower/upper cell bounds before materializing refinement.

    The lower bound refines the currently stored candidate ``X`` and nothing
    else.  The actual relation-saturated physical collar can only add cells.
    When the mixed family covers both complete chart rectangles, the upper
    bound is the complete uniform Atlas at the target depth.
    """

    payload = (
        read_garcia_local_relation(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    validate_garcia_resume_fingerprint(payload)
    cell_payloads = payload["cells"]
    candidate = payload["candidate"]
    metadata = payload["metadata"]
    if not isinstance(cell_payloads, list) or not isinstance(candidate, Mapping):
        raise ValueError("refinement payload has malformed cells or candidate")
    relation = {
        int(raw["index"]): frozenset(int(value) for value in raw["image"])
        for raw in cell_payloads
    }
    recomputed_s, recomputed_x, recomputed_a = _recomputed_candidate_sets(
        payload,
        relation,
    )
    if (
        recomputed_s != frozenset(int(value) for value in candidate["S"])
        or recomputed_x != frozenset(int(value) for value in candidate["X"])
        or recomputed_a != frozenset(int(value) for value in candidate["A"])
    ):
        raise ValueError("persisted candidate S/X/A disagrees with raw relation")
    if selector == "candidate_X":
        selected_payload = recomputed_x
    elif selector == "safe_locator":
        locator = payload.get("refinement_locator")
        if not isinstance(locator, Mapping):
            raise ValueError("payload has no persisted safe refinement locator")
        selected_payload = locator["seed_cells"]
    else:
        raise ValueError("selector must be 'candidate_X' or 'safe_locator'")
    x_cells = frozenset(int(value) for value in selected_payload)
    if target_axis_depth < max(int(raw["axis_depth"]) for raw in cell_payloads):
        raise ValueError("target_axis_depth cannot coarsen the persisted family")
    descendant_counts = [
        2 ** (4 * (target_axis_depth - int(raw["axis_depth"])))
        for raw in cell_payloads
    ]
    candidate_only_cells = sum(
        descendant_counts[index]
        if index in x_cells
        else int(retain_unselected)
        for index in range(len(cell_payloads))
    )
    represented_fine_cells = sum(descendant_counts)
    complete_uniform_cells = 2 * 2 ** (4 * target_axis_depth)
    complete_ambient_cover = represented_fine_cells == complete_uniform_cells
    upper_cells = (
        complete_uniform_cells if complete_ambient_cover else represented_fine_cells
    )
    samples_per_axis = int(metadata["samples_per_axis"])
    return {
        "source_cells": len(cell_payloads),
        "candidate_X_cells": len(x_cells),
        "candidate_fraction": len(x_cells) / len(cell_payloads),
        "selector": selector,
        "retain_unselected": retain_unselected,
        "target_axis_depth": target_axis_depth,
        "target_total_depth": 4 * target_axis_depth,
        "candidate_only_cell_lower_bound": candidate_only_cells,
        "complete_ambient_cell_upper_bound": upper_cells,
        "candidate_only_logical_sample_lower_bound": (
            candidate_only_cells * samples_per_axis**4
        ),
        "complete_ambient_logical_sample_upper_bound": (
            upper_cells * samples_per_axis**4
        ),
        "complete_ambient_cover": complete_ambient_cover,
    }


def _recomputed_candidate_sets(
    payload: Mapping[str, object],
    relation: Mapping[int, frozenset[int]],
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """Derive the standard finite pair ``S, X=S union F(S), A=X-S``."""

    morse = payload.get("morse_graph")
    candidate = payload.get("candidate")
    reference = payload.get("reference_audit")
    if not isinstance(morse, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("resume payload has malformed Morse/candidate data")
    if not isinstance(reference, Mapping):
        raise ValueError("resume payload has malformed reference audit")
    raw_sets = morse.get("morse_sets")
    if not isinstance(raw_sets, Mapping):
        raise ValueError("resume payload has malformed Morse sets")

    recurrent_components = {
        frozenset(component) for component in _relation_recurrent_components(relation)
    }
    normalized_sets = {
        int(node): frozenset(int(value) for value in component)
        for node, component in raw_sets.items()
    }
    stored_components = set(normalized_sets.values())
    if recurrent_components != stored_components:
        raise ValueError("persisted Morse sets disagree with raw relation SCCs")
    stored_nodes = frozenset(int(value) for value in morse.get("nodes", ()))
    if stored_nodes != frozenset(normalized_sets):
        raise ValueError("persisted Morse node list disagrees with its Morse sets")
    node_by_cell: dict[int, int] = {}
    for node, component in normalized_sets.items():
        for index in component:
            if index in node_by_cell:
                raise ValueError("persisted Morse sets overlap")
            node_by_cell[index] = node
    cell_payloads = payload.get("cells")
    if not isinstance(cell_payloads, list):
        raise ValueError("resume payload cells are malformed")
    for raw in cell_payloads:
        index = int(raw["index"])
        stored_cell_node = raw.get("morse_node")
        if stored_cell_node is not None:
            stored_cell_node = int(stored_cell_node)
        if stored_cell_node != node_by_cell.get(index):
            raise ValueError("persisted cell Morse membership disagrees with Morse sets")

    raw_hits = reference.get("morse_node_hits")
    if not isinstance(raw_hits, Mapping):
        raise ValueError("resume payload has malformed reference-node hits")
    hits = {int(node): int(count) for node, count in raw_hits.items()}
    selected_node: int | None = None
    if hits:
        ordered = sorted(hits.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
            selected_node = ordered[0][0]
    stored_node = candidate.get("morse_node")
    if stored_node is not None:
        stored_node = int(stored_node)
    if selected_node != stored_node:
        raise ValueError(
            "persisted reference-selected Morse node disagrees with raw hits"
        )
    if selected_node is not None and selected_node not in normalized_sets:
        raise ValueError("reference audit selects an unknown Morse node")
    s_cells = (
        normalized_sets[selected_node]
        if selected_node is not None
        else frozenset()
    )
    f_s = frozenset(
        target for source in s_cells for target in relation[source]
    )
    x_cells = s_cells | f_s
    return s_cells, x_cells, x_cells - s_cells


def _validate_persisted_locator_and_relation(
    payload: Mapping[str, object],
    charts: SuspensionAtlasCharts,
    family: GarciaMixedDyadicFamily,
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """Cross-check raw pieces, MapGraph edges, exits, and locator before resume."""

    class _ChartsOnlyCallback:
        def __init__(self, value: SuspensionAtlasCharts) -> None:
            self.charts = value

        def __call__(self, *_args: object) -> list[object]:  # pragma: no cover
            raise AssertionError("resume validation must not evaluate dynamics")

    atlas = build_cmgdb_atlas_model(
        _ChartsOnlyCallback(charts),  # type: ignore[arg-type]
        depth=0,
        active_dyadic_cells=family.tagged_cells(),
    ).phaseSpace()
    cell_payloads = payload["cells"]
    if not isinstance(cell_payloads, list):
        raise ValueError("resume payload cells are malformed")
    cells = tuple(
        AtlasWalkerCell(
            int(raw["index"]),
            int(raw["chart_id"]),
            tuple(float(value) for value in raw["bounds"]),
        )
        for raw in cell_payloads
    )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("resume payload metadata is malformed")
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=float(metadata["transversality_eta"]),
    )
    recomputed_pairs = _quotient_cross_chart_pairs(
        cells,
        incidence,
        charts,
        atlas,
    )
    stored_pairs = tuple(
        sorted(
            (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
            for pair in payload.get("quotient_neighbor_pairs", ())
        )
    )
    if recomputed_pairs != stored_pairs:
        raise ValueError("persisted quotient-neighbor pairs disagree with cell geometry")
    quotient_neighbors_by_handle: dict[int, set[int]] = {}
    for first, second in recomputed_pairs:
        handle = first if cells[first].chart_id == charts.handle_chart_id else second
        base = second if handle == first else first
        quotient_neighbors_by_handle.setdefault(handle, set()).add(base)
    frozen_quotient_neighbors = {
        handle: frozenset(neighbors)
        for handle, neighbors in quotient_neighbors_by_handle.items()
    }

    relation: dict[int, frozenset[int]] = {}
    for raw in cell_payloads:
        source = int(raw["index"])
        if type(raw.get("relation_evaluated")) is not bool:
            raise ValueError(
                f"resume relation-evaluated flag is nonboolean at source {source}"
            )
        if not bool(raw["relation_evaluated"]):
            raise ValueError(f"resume relation was not evaluated at source {source}")
        stored = frozenset(int(value) for value in raw["image"])
        open_exit = raw.get("open_exit")
        if not isinstance(open_exit, Mapping):
            raise ValueError(f"resume source {source} lacks open-exit provenance")
        pieces = open_exit.get("target_pieces")
        if not isinstance(pieces, list):
            raise ValueError(f"resume source {source} lacks raw target pieces")
        recovered = frozenset(
            int(target)
            for piece in pieces
            for target in atlas.cover(int(piece["chart_id"]), piece["bounds"])
        )
        if recovered != stored:
            raise ValueError(
                f"resume source {source} raw target cover disagrees with its relation"
            )
        if bool(open_exit["mapgraph_empty_image"]) != (not stored):
            raise ValueError(f"resume source {source} has inconsistent empty-image flag")
        if int(open_exit["returned_pieces"]) != len(pieces):
            raise ValueError(f"resume source {source} has inconsistent piece count")
        if bool(open_exit["explicit_empty_callback"]) != (not pieces):
            raise ValueError(f"resume source {source} has inconsistent callback-empty flag")
        if int(open_exit["active_target_cells"]) != len(recovered):
            raise ValueError(f"resume source {source} has inconsistent active cover count")

        recomputed_missing: set[GarciaDyadicCell] = set()
        recomputed_ambient = 0
        recomputed_wholly_outside = 0
        encoded_pieces = tuple(
            (
                int(piece["chart_id"]),
                tuple(float(value) for value in piece["bounds"]),
            )
            for piece in pieces
        )
        for target_chart, target_bounds in encoded_pieces:
            covered = tuple(int(value) for value in atlas.cover(target_chart, target_bounds))
            recomputed_wholly_outside += int(not covered)
            fine_cover, crossings = _dyadic_rectangle_cover(
                charts,
                target_chart,
                target_bounds,
                family.max_axis_depth,
            )
            recomputed_missing.update(
                cell for cell in fine_cover if family.covering_cell(cell) is None
            )
            recomputed_ambient += int(
                _piece_has_unmatched_ambient_exit(
                    charts,
                    incidence,
                    atlas,
                    target_chart,
                    target_bounds,
                    crossings,
                    encoded_pieces,
                    quotient_neighbors_by_handle=frozen_quotient_neighbors,
                )
            )
        stored_missing = frozenset(
            GarciaDyadicCell(
                int(item["chart_id"]),
                int(item["axis_depth"]),
                tuple(int(value) for value in item["coordinates"]),
            )
            for item in open_exit["missing_in_domain_witnesses"]
        )
        if frozenset(recomputed_missing) != stored_missing:
            raise ValueError(
                f"resume source {source} missing-target witnesses disagree with raw pieces"
            )
        if int(open_exit["missing_in_domain_target_cells"]) != len(stored_missing):
            raise ValueError(f"resume source {source} has inconsistent missing-target count")
        if int(open_exit["ambient_boundary_pieces"]) != recomputed_ambient:
            raise ValueError(f"resume source {source} has inconsistent ambient-exit count")
        if (
            int(open_exit["wholly_outside_active_family_pieces"])
            != recomputed_wholly_outside
        ):
            raise ValueError(f"resume source {source} has inconsistent outside-piece count")
        failure_reasons = open_exit.get("failure_reasons")
        if not isinstance(failure_reasons, Mapping):
            raise ValueError(f"resume source {source} has malformed failure reasons")
        failed_samples = int(open_exit["failed_samples_per_invocation"])
        successful_samples = int(open_exit["successful_samples_per_invocation"])
        expected_samples = int(metadata["samples_per_axis"]) ** 4
        if successful_samples + failed_samples != expected_samples:
            raise ValueError(f"resume source {source} has inconsistent sample total")
        if int(open_exit["callback_invocations"]) < 1:
            raise ValueError(f"resume source {source} was never evaluated")
        if sum(int(value) for value in failure_reasons.values()) != failed_samples:
            raise ValueError(f"resume source {source} has inconsistent failure counts")
        recomputed_has_open_exit = bool(
            failed_samples
            or open_exit["unresolved_stage_edges"]
            or not pieces
            or not stored
            or stored_missing
            or recomputed_ambient
            or recomputed_wholly_outside
        )
        if type(open_exit.get("has_open_exit")) is not bool:
            raise ValueError(f"resume source {source} has nonboolean open-exit flag")
        if bool(open_exit["has_open_exit"]) != recomputed_has_open_exit:
            raise ValueError(f"resume source {source} has inconsistent open-exit flag")
        relation[source] = stored

    recomputed_s, recomputed_x, recomputed_a = _recomputed_candidate_sets(
        payload,
        relation,
    )
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("resume payload candidate is malformed")
    if (
        recomputed_s != frozenset(int(value) for value in candidate["S"])
        or recomputed_x != frozenset(int(value) for value in candidate["X"])
        or recomputed_a != frozenset(int(value) for value in candidate["A"])
    ):
        raise ValueError("persisted candidate S/X/A disagrees with raw relation")

    locator = payload.get("refinement_locator")
    if not isinstance(locator, Mapping):
        raise ValueError("resume payload lacks a safe refinement locator")
    dyadic_cells = tuple(
        GarciaDyadicCell(
            int(raw["chart_id"]),
            int(raw["axis_depth"]),
            tuple(int(value) for value in raw["coordinates"]),
        )
        for raw in cell_payloads
    )
    connectivity = audit_garcia_sparse_relation_connectivity(
        relation,
        dyadic_cells,
        recomputed_pairs,
    )
    disconnected = connectivity.disconnected_sources
    if disconnected != frozenset(
        int(value) for value in locator["disconnected_sources_excluded"]
    ):
        raise ValueError(
            "persisted disconnected-source exclusions disagree with raw relation"
        )
    eligible = frozenset(
        source
        for source, targets in relation.items()
        if targets
        and not bool(cell_payloads[source]["open_exit"]["has_open_exit"])
        and source not in disconnected
    )
    if eligible != frozenset(int(value) for value in locator["eligible_cells"]):
        raise ValueError("persisted safe-locator eligibility disagrees with raw relation")
    recurrent = _relation_recurrent_components(relation, eligible)
    if recurrent != tuple(
        tuple(int(value) for value in component)
        for component in locator["recurrent_components"]
    ):
        raise ValueError("persisted safe-locator SCCs disagree with raw relation")
    references = frozenset(
        int(value) for value in payload["candidate"]["reference_source_cells"]
    )
    selected = frozenset()
    if recurrent:
        selected = frozenset(
            max(
                recurrent,
                key=lambda component: (
                    len(references.intersection(component)),
                    len(component),
                    -component[0],
                ),
            )
        )
    if selected != frozenset(int(value) for value in locator["selected_component"]):
        raise ValueError("persisted safe-locator selection disagrees with raw relation")
    if selected | references != frozenset(
        int(value) for value in locator["seed_cells"]
    ):
        raise ValueError("persisted safe-locator seed omits selected/reference cells")
    return recomputed_s, recomputed_x, recomputed_a


def relation_saturated_refinement_from_payload(
    payload_or_path: Mapping[str, object] | str | Path,
    *,
    target_axis_depth: int,
    collar_fraction: float = 0.125,
    selector: str = "candidate_X",
    include_target_support: bool = True,
    retain_unselected: bool = True,
    max_output_cells: int | None = None,
) -> GarciaMixedDyadicFamily:
    """Build the next mixed family from a complete persisted relation.

    This makes the expensive depth-12 discovery pass resumable.  The function
    consumes only disclosed cell geometry, candidate membership, and the
    stored per-source missing-target witnesses; it does not infer new edges.
    """

    payload = (
        read_garcia_local_relation(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    validate_garcia_resume_fingerprint(payload)
    metadata = payload["metadata"]
    charts_payload = payload["charts"]
    candidate_payload = payload["candidate"]
    cell_payloads = payload["cells"]
    if not isinstance(metadata, Mapping) or not isinstance(charts_payload, Mapping):
        raise ValueError("resume payload has malformed metadata or chart data")
    if not isinstance(candidate_payload, Mapping) or not isinstance(cell_payloads, list):
        raise ValueError("resume payload has malformed candidate or cell data")

    base_payload = charts_payload["base"]
    handle_payload = charts_payload["handle"]
    if not isinstance(base_payload, Mapping) or not isinstance(handle_payload, Mapping):
        raise ValueError("resume payload has malformed chart records")
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=base_payload["bounds"],
        guard_delta=float(metadata["guard_delta"]),
        transversality_eta=float(metadata["transversality_eta"]),
    )
    stored_handle_bounds = np.asarray(handle_payload["bounds"], dtype=np.float64)
    if not np.allclose(stored_handle_bounds, charts.handle_bounds, atol=1.0e-13):
        raise ValueError("resume payload handle chart is inconsistent with its model")

    cells: list[AtlasWalkerCell] = []
    dyadic_cells: list[GarciaDyadicCell] = []
    missing_by_source: dict[int, tuple[GarciaDyadicCell, ...]] = {}
    target_pieces_by_source: dict[
        int, tuple[tuple[int, tuple[float, ...]], ...]
    ] = {}
    for expected_index, raw in enumerate(cell_payloads):
        if not isinstance(raw, Mapping) or int(raw["index"]) != expected_index:
            raise ValueError("resume payload cells are not a complete indexed relation")
        token = GarciaDyadicCell(
            int(raw["chart_id"]),
            int(raw["axis_depth"]),
            tuple(int(value) for value in raw["coordinates"]),
        )
        bounds = tuple(float(value) for value in raw["bounds"])
        if not np.allclose(bounds, token.bounds(charts), atol=1.0e-13):
            raise ValueError(f"resume cell {expected_index} has inconsistent geometry")
        cells.append(AtlasWalkerCell(expected_index, token.chart_id, bounds))
        dyadic_cells.append(token)
        open_exit = raw["open_exit"]
        if not isinstance(open_exit, Mapping):
            raise ValueError(f"resume cell {expected_index} lacks exit provenance")
        missing_by_source[expected_index] = tuple(
            GarciaDyadicCell(
                int(item["chart_id"]),
                int(item["axis_depth"]),
                tuple(int(value) for value in item["coordinates"]),
            )
            for item in open_exit["missing_in_domain_witnesses"]
        )
        target_pieces_by_source[expected_index] = tuple(
            (
                int(item["chart_id"]),
                tuple(float(value) for value in item["bounds"]),
            )
            for item in open_exit.get("target_pieces", ())
        )
    family = GarciaMixedDyadicFamily(
        tuple(sorted(dyadic_cells)),
        role=str(metadata["family_role"]),
    )
    _recomputed_s, recomputed_x, _recomputed_a = (
        _validate_persisted_locator_and_relation(payload, charts, family)
    )
    if selector == "candidate_X":
        selected_payload = recomputed_x
    elif selector == "safe_locator":
        locator_payload = payload.get("refinement_locator")
        if not isinstance(locator_payload, Mapping):
            raise ValueError("resume payload has no safe refinement locator")
        selected_payload = locator_payload["seed_cells"]
    else:
        raise ValueError("selector must be 'candidate_X' or 'safe_locator'")
    x_cells = frozenset(int(value) for value in selected_payload)
    if any(index < 0 or index >= len(cells) for index in x_cells):
        raise ValueError("resume candidate X references an unknown cell")
    return _relation_saturated_family(
        family=family,
        charts=charts,
        cells=tuple(cells),
        dyadic_cells=tuple(dyadic_cells),
        x_cells=x_cells,
        missing_by_source=missing_by_source,
        target_pieces_by_source=target_pieces_by_source,
        quotient_neighbor_pairs=tuple(
            (int(pair[0]), int(pair[1]))
            for pair in payload.get("quotient_neighbor_pairs", ())
        ),
        target_axis_depth=target_axis_depth,
        collar_fraction=collar_fraction,
        include_target_support=include_target_support,
        retain_unselected=retain_unselected,
        max_output_cells=max_output_cells,
    )


def safe_locator_refinement_from_payload(
    payload_or_path: Mapping[str, object] | str | Path,
    *,
    target_axis_depth: int,
    collar_fraction: float = 0.0625,
    include_target_support: bool = True,
    retain_unselected: bool = False,
    max_output_cells: int | None = None,
) -> GarciaMixedDyadicFamily:
    return relation_saturated_refinement_from_payload(
        payload_or_path,
        target_axis_depth=target_axis_depth,
        collar_fraction=collar_fraction,
        selector="safe_locator",
        include_target_support=include_target_support,
        retain_unselected=retain_unselected,
        max_output_cells=max_output_cells,
    )


def estimated_callback_point_evaluations(
    family: GarciaMixedDyadicFamily,
    samples_per_axis: int,
    *,
    cmgdb_passes: int = 1,
) -> int:
    """Return the conservative logical tensor-point count.

    The Garcia adapter caches deterministic source values across CMGDB's two
    graph traversals, hence the default of one physical source pass.  Shared
    lattice nodes of adjacent dyadic cells are cached as well, so the actual
    number of ODE integrations is lower than this disclosed cap.
    """

    return len(family.cells) * int(samples_per_axis) ** 4 * int(cmgdb_passes)


__all__ = [
    "GarciaDyadicCell",
    "GarciaMixedDyadicFamily",
    "GarciaRefinementSizeLimitExceeded",
    "GarciaRelationSizeLimitExceeded",
    "garcia_failure_reason_is_explicit_domain_exit",
    "garcia_non_domain_failure_sources",
    "GarciaSparseConnectivityAudit",
    "GarciaCoverageRecord",
    "GarciaSourceExitProvenance",
    "GarciaLocalCandidateAudit",
    "GarciaSafeRefinementLocator",
    "GarciaExitAwareSupportAudit",
    "GarciaLocalRelationRun",
    "full_garcia_dyadic_family",
    "compute_garcia_local_relation",
    "audit_garcia_sparse_relation_connectivity",
    "write_garcia_raw_relation_checkpoint",
    "read_garcia_raw_relation_checkpoint",
    "resume_garcia_local_relation_from_raw_checkpoint",
    "migrate_garcia_relation_to_raw_checkpoint",
    "relation_saturated_refinement",
    "exit_aware_index_pair_refinement",
    "safe_locator_refinement",
    "read_garcia_local_relation",
    "persisted_refinement_size_bounds",
    "relation_saturated_refinement_from_payload",
    "safe_locator_refinement_from_payload",
    "estimated_callback_point_evaluations",
    "attach_garcia_resume_fingerprint",
    "validate_garcia_resume_fingerprint",
]
