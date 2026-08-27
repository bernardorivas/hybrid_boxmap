"""Diagnostic strict-source restriction of a persisted Garcia relation.

This module never evaluates the walker dynamics.  It consumes the immutable
CSR relation and its already recorded per-source provenance.  A source row is
replaced by the empty set when *any* of the following fixed conditions holds:

* at least one callback/sample failure;
* an unresolved suspension-stage edge;
* any recorded open/ambient exit;
* a target missing from the active Atlas family; or
* a quotient-disconnected represented image.

The original relation and provenance remain authoritative and unchanged.  The
restricted graph is deliberately diagnostic: deleting uncertain source rows
is not an outer enclosure of the continuous map and cannot certify a Conley
index or a physical gait.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .garcia_passive_walker_local import (
    GarciaSourceExitProvenance,
    _GarciaRawRelationStage,
    _relation_recurrent_components,
    _relation_strongly_connected_components,
    audit_garcia_sparse_relation_connectivity,
)
from .garcia_passive_walker_atlas import GarciaWalkerQuotientIncidence
from .garcia_passive_walker_tube import _handle_attachment_carrier


STRICT_SOURCE_RESTRICTION_SCHEMA: Final = (
    "garcia-walker-strict-source-restriction-audit-v1"
)
STRICT_SOURCE_RESTRICTION_RULE: Final = (
    "empty_every_source_row_with_any_sample_failure_unresolved_stage_"
    "open_or_ambient_exit_missing_active_target_or_quotient_disconnected_image"
)

# These are the 66 distinct source cells obtained by the predeclared reference
# audit used throughout the Garcia Atlas pipeline: 17 base samples per stride,
# 9 handle samples per jump, two strides/jumps, and reference max step 0.005.
# They were independently recomputed once for the pinned depth-20 family.  The
# strict restriction audit consumes this fixed cell list and does not rerun the
# reference ODE.  The report authenticates the physical/family configuration
# through its sibling CSR bundle reference.
PINNED_D20_REFERENCE_SOURCE_CELLS: Final = (
    1078,
    2207,
    2751,
    4419,
    5475,
    5782,
    6998,
    7380,
    7595,
    7781,
    8585,
    8820,
    8878,
    9369,
    9854,
    10888,
    12792,
    12805,
    13676,
    14220,
    14553,
    14891,
    16134,
    16990,
    17030,
    17045,
    17212,
    18397,
    18437,
    18449,
    19886,
    20043,
    20080,
    20252,
    22278,
    22295,
    22422,
    22439,
    22878,
    22895,
    23022,
    23039,
    24382,
    24399,
    24574,
    24591,
    24862,
    24879,
    25054,
    25071,
    26278,
    26295,
    26422,
    26439,
    26878,
    26895,
    27022,
    27039,
    28382,
    28399,
    28574,
    28591,
    28862,
    28879,
    29054,
    29071,
)

_REASON_ORDER: Final = (
    "callback_or_sample_failure",
    "unresolved_stage",
    "composite_has_open_exit",
    "missing_active_target",
    "quotient_disconnected_image",
)


class GarciaStrictSourceRelation(Mapping[int, Collection[int]]):
    """Lazy row-masked view which leaves the authoritative CSR untouched."""

    def __init__(
        self,
        relation: Mapping[int, Collection[int]],
        restricted_sources: Collection[int],
    ) -> None:
        count = len(relation)
        if set(relation) != set(range(count)):
            raise ValueError("source relation must use consecutive integer rows")
        restricted = frozenset(restricted_sources)
        if any(
            type(source) is not int or source < 0 or source >= count
            for source in restricted
        ):
            raise ValueError("strict restriction references an unknown source")
        self._relation = relation
        self.restricted_sources = restricted

    @property
    def source_relation(self) -> Mapping[int, Collection[int]]:
        return self._relation

    @property
    def metadata(self) -> object:
        """Forward mmap payload metadata for conservative storage accounting."""

        return getattr(self._relation, "metadata", None)

    def __len__(self) -> int:
        return len(self._relation)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, source: int) -> Collection[int]:
        if type(source) is not int or source < 0 or source >= len(self):
            raise KeyError(source)
        if source in self.restricted_sources:
            return ()
        return self._relation[source]


def garcia_strict_source_exclusion_reasons(
    provenance: Mapping[int, GarciaSourceExitProvenance],
    disconnected_sources: Collection[int],
) -> dict[int, tuple[str, ...]]:
    """Return every fixed restriction reason, preserving category overlap."""

    disconnected = frozenset(disconnected_sources)
    if set(provenance) != set(range(len(provenance))):
        raise ValueError("source provenance must use consecutive integer rows")
    if any(source < 0 or source >= len(provenance) for source in disconnected):
        raise ValueError("disconnected-image audit references an unknown source")
    result: dict[int, tuple[str, ...]] = {}
    for source in range(len(provenance)):
        item = provenance[source]
        reasons: list[str] = []
        if item.has_sample_failure:
            reasons.append("callback_or_sample_failure")
        if item.unresolved_stage_edges:
            reasons.append("unresolved_stage")
        if item.has_open_exit:
            reasons.append("composite_has_open_exit")
        if (
            item.missing_in_domain_target_cells
            or item.wholly_outside_active_family_pieces
        ):
            reasons.append("missing_active_target")
        if source in disconnected:
            reasons.append("quotient_disconnected_image")
        ordered = tuple(reason for reason in _REASON_ORDER if reason in reasons)
        if ordered:
            result[source] = ordered
    return result


def _morse_order(
    relation: Mapping[int, Collection[int]],
    recurrent_components: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return exact recurrent reachability closure and its Hasse reduction."""

    node_by_cell = [-1] * len(relation)
    for node, component in enumerate(recurrent_components):
        for source in component:
            node_by_cell[source] = node
    closure: set[tuple[int, int]] = set()
    for node, component in enumerate(recurrent_components):
        visited = bytearray(len(relation))
        stack = list(component)
        for source in component:
            visited[source] = 1
        while stack:
            source = stack.pop()
            for target in relation[source]:
                target_node = node_by_cell[target]
                if target_node >= 0 and target_node != node:
                    closure.add((node, target_node))
                if not visited[target]:
                    visited[target] = 1
                    stack.append(target)
    for source, target in closure:
        if (target, source) in closure:
            raise ValueError("distinct recurrent SCCs have cyclic reachability")
    hasse = tuple(
        sorted(
            (source, target)
            for source, target in closure
            if not any(
                middle not in (source, target)
                and (source, middle) in closure
                and (middle, target) in closure
                for middle in range(len(recurrent_components))
            )
        )
    )
    return tuple(sorted(closure)), hasse


def _source_list_sha256(sources: Collection[int]) -> str:
    encoded = json.dumps(
        sorted(sources),
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_open_boundary_indices(
    stage: _GarciaRawRelationStage,
) -> tuple[frozenset[int], frozenset[int]]:
    """Reconstruct tube boundaries from bundle geometry and quotient data."""

    if stage.family.min_axis_depth != stage.family.max_axis_depth:
        raise ValueError("strict tube boundary audit requires a uniform family")
    depth = stage.family.max_axis_depth
    subdivisions = 2**depth
    active = frozenset(stage.dyadic_cells)
    index_by_cell = {cell: index for index, cell in enumerate(stage.dyadic_cells)}
    glued_handle_indices: set[int] = set()
    for first, second in stage.quotient_neighbor_pairs:
        if stage.dyadic_cells[first].chart_id == stage.charts.handle_chart_id:
            glued_handle_indices.add(first)
        if stage.dyadic_cells[second].chart_id == stage.charts.handle_chart_id:
            glued_handle_indices.add(second)
    glued_handles = {stage.dyadic_cells[index] for index in glued_handle_indices}
    offsets = tuple(
        offset
        for offset in itertools.product((-1, 0, 1), repeat=4)
        if any(offset)
    )
    same_chart: set[int] = set()
    for index, cell in enumerate(stage.dyadic_cells):
        for offset in offsets:
            coordinates = tuple(
                value + delta
                for value, delta in zip(cell.coordinates, offset, strict=True)
            )
            outside_axes = {
                axis
                for axis, value in enumerate(coordinates)
                if value < 0 or value >= subdivisions
            }
            if outside_axes:
                if cell in glued_handles and outside_axes == {3}:
                    continue
                same_chart.add(index)
                break
            neighbor = type(cell)(cell.chart_id, depth, coordinates)
            if neighbor not in active:
                same_chart.add(index)
                break

    incidence = GarciaWalkerQuotientIncidence(
        stage.charts,
        phi_dot_min=stage.charts.base_bounds[3][0],
        phi_dot_max=stage.charts.base_bounds[3][1],
        transversality_eta=stage.walker.transversality_eta,
    )
    reverse_base: set[int] = set()
    for first_three in itertools.product(range(subdivisions), repeat=3):
        for phase in (0, subdivisions - 1):
            handle = type(stage.dyadic_cells[0])(
                stage.charts.handle_chart_id,
                depth,
                (*first_three, phase),
            )
            if handle in active:
                continue
            for base in _handle_attachment_carrier(
                handle,
                charts=stage.charts,
                incidence=incidence,
            ):
                index = index_by_cell.get(base)
                if index is not None:
                    reverse_base.add(index)
    return frozenset(same_chart), frozenset(reverse_base)


@dataclass(frozen=True)
class GarciaStrictSourceRestrictionAudit:
    relation: GarciaStrictSourceRelation
    restriction_reasons: Mapping[int, tuple[str, ...]]
    original_connectivity: Any
    restricted_connectivity: Any
    recurrent_components: tuple[tuple[int, ...], ...]
    full_scc_count: int
    full_scc_size_histogram: Mapping[int, int]
    induced_safe_scc_count: int
    induced_safe_scc_size_histogram: Mapping[int, int]
    induced_safe_recurrent_components: tuple[tuple[int, ...], ...]
    retained_to_retained_edges: int
    retained_to_restricted_edges: int
    morse_reachability_edges: tuple[tuple[int, int], ...]
    morse_hasse_edges: tuple[tuple[int, int], ...]
    reference_source_cells: frozenset[int]
    reference_probe_configuration: Mapping[str, object]
    reference_raw_provenance_sources: Mapping[str, tuple[int, ...]]
    reference_failure_reasons: Mapping[str, int]
    raw_provenance_cause_counts: Mapping[str, int]
    reference_node_hits: Mapping[int, int]
    selected_morse_node: int | None
    s_cells: frozenset[int]
    x_cells: frozenset[int]
    a_cells: frozenset[int]
    recurrent_components_in_x: tuple[tuple[int, ...], ...]
    pair_second_condition_violations: tuple[int, ...]
    same_chart_open_boundary_cells: frozenset[int]
    reverse_open_boundary_cells: frozenset[int]

    @property
    def removed_reference_sources(self) -> frozenset[int]:
        return self.reference_source_cells & self.relation.restricted_sources

    @property
    def pair_defined(self) -> bool:
        return self.selected_morse_node is not None and bool(self.s_cells)

    @property
    def diagnostic_pair_conditions_passed(self) -> bool:
        return bool(
            self.pair_defined
            and not self.pair_second_condition_violations
            and self.recurrent_components_in_x == (tuple(sorted(self.s_cells)),)
            and not self.removed_reference_sources
        )

    def to_dict(self) -> dict[str, object]:
        category_counts = Counter(
            reason
            for reasons in self.restriction_reasons.values()
            for reason in reasons
        )
        reference_reason_sources = {
            reason: sorted(
                source
                for source in self.reference_source_cells
                if reason in self.restriction_reasons.get(source, ())
            )
            for reason in _REASON_ORDER
        }
        blockers: list[str] = []
        if not self.recurrent_components:
            blockers.append("no_recurrent_SCC_after_strict_source_restriction")
        if self.selected_morse_node is None:
            blockers.append("no_unique_reference_carrier_Morse_node")
        if self.removed_reference_sources:
            blockers.append("strict_restriction_removed_reference_sources")
        if self.pair_second_condition_violations:
            blockers.append("finite_pair_second_condition_violation")
        if self.pair_defined and self.recurrent_components_in_x != (
            tuple(sorted(self.s_cells)),
        ):
            blockers.append("additional_recurrence_in_candidate_X")
        return {
            "restriction": {
                "rule": STRICT_SOURCE_RESTRICTION_RULE,
                "source_rows": len(self.relation),
                "restricted_source_rows": len(self.relation.restricted_sources),
                "retained_source_rows": len(self.relation)
                - len(self.relation.restricted_sources),
                "reason_counts_with_overlap": {
                    reason: category_counts[reason] for reason in _REASON_ORDER
                },
                "raw_provenance_cause_counts": dict(
                    sorted(self.raw_provenance_cause_counts.items())
                ),
                "sources": [
                    {"source": source, "reasons": list(reasons)}
                    for source, reasons in sorted(self.restriction_reasons.items())
                ],
                "original_provenance_retained_unchanged": True,
                "implementation": (
                    "lazy row mask over the immutable mmap CSR; restricted rows "
                    "return the empty collection"
                ),
            },
            "relation_counts": {
                "original_edges": self.original_connectivity.relation_edges,
                "restricted_edges": self.restricted_connectivity.relation_edges,
                "removed_edges": self.original_connectivity.relation_edges
                - self.restricted_connectivity.relation_edges,
                "restricted_nonempty_images": (
                    self.restricted_connectivity.relation_nonempty_images
                ),
                "original_quotient_disconnected_images": len(
                    self.original_connectivity.disconnected_sources
                ),
                "restricted_quotient_disconnected_images": len(
                    self.restricted_connectivity.disconnected_sources
                ),
                "retained_to_retained_edges": self.retained_to_retained_edges,
                "retained_to_restricted_edges": self.retained_to_restricted_edges,
            },
            "morse_graph": {
                "nodes": list(range(len(self.recurrent_components))),
                "morse_sets": {
                    str(node): list(component)
                    for node, component in enumerate(self.recurrent_components)
                },
                "reachability_closure_edges": [
                    list(edge) for edge in self.morse_reachability_edges
                ],
                "hasse_edges": [list(edge) for edge in self.morse_hasse_edges],
                "counts": {
                    "nodes": len(self.recurrent_components),
                    "reachability_closure_edges": len(
                        self.morse_reachability_edges
                    ),
                    "hasse_edges": len(self.morse_hasse_edges),
                },
                "all_SCC_audit": {
                    "empty_row_relation_SCCs": self.full_scc_count,
                    "empty_row_relation_size_histogram": {
                        str(size): count
                        for size, count in sorted(
                            self.full_scc_size_histogram.items()
                        )
                    },
                    "induced_safe_relation_SCCs": self.induced_safe_scc_count,
                    "induced_safe_relation_size_histogram": {
                        str(size): count
                        for size, count in sorted(
                            self.induced_safe_scc_size_histogram.items()
                        )
                    },
                    "empty_row_recurrent_components": [
                        list(component) for component in self.recurrent_components
                    ],
                    "induced_safe_recurrent_components": [
                        list(component)
                        for component in self.induced_safe_recurrent_components
                    ],
                    "recurrence_parity": self.recurrent_components
                    == self.induced_safe_recurrent_components,
                    "interpretation": (
                        "restricted vertices remain as transient empty-row cells; "
                        "the induced retained-source graph has identical recurrence"
                    ),
                },
            },
            "reference_candidate": {
                "rule": (
                    "same unique maximum recurrent-node hit count over the fixed "
                    "66 distinct source cells from the standard 17/9 two-stride "
                    "stored-gait probe; this diagnostic does not rerun that ODE"
                ),
                "reference_source_provenance": (
                    "externally supplied one-time standard stored-gait probe IDs; "
                    "fixed cell IDs consumed without ODE in this audit"
                ),
                "reference_probe_configuration": dict(
                    self.reference_probe_configuration
                ),
                "source_cells": sorted(self.reference_source_cells),
                "safe_source_cells": sorted(
                    self.reference_source_cells - self.removed_reference_sources
                ),
                "removed_source_cells": sorted(self.removed_reference_sources),
                "removed_source_reasons": reference_reason_sources,
                "raw_provenance_audit": {
                    "source_lists": {
                        reason: list(sources)
                        for reason, sources in sorted(
                            self.reference_raw_provenance_sources.items()
                        )
                    },
                    "counts": {
                        reason: len(sources)
                        for reason, sources in sorted(
                            self.reference_raw_provenance_sources.items()
                        )
                    },
                    "failure_reason_sample_counts": dict(
                        sorted(self.reference_failure_reasons.items())
                    ),
                },
                "morse_node_hits": {
                    str(node): count
                    for node, count in sorted(self.reference_node_hits.items())
                },
                "selected_morse_node": self.selected_morse_node,
                "source_cells_in_selected_morse_set": sorted(
                    self.reference_source_cells & self.s_cells
                ),
            },
            "candidate_pair": {
                "definition": "S=M; X=S union F_restricted(S); A=X minus S",
                "S": sorted(self.s_cells),
                "X": sorted(self.x_cells),
                "A": sorted(self.a_cells),
                "counts": {
                    "S": len(self.s_cells),
                    "X": len(self.x_cells),
                    "A": len(self.a_cells),
                },
                "F_S_subset_X": all(
                    target in self.x_cells
                    for source in self.s_cells
                    for target in self.relation[source]
                ),
                "F_A_intersection_X_subset_A": not (
                    self.pair_second_condition_violations
                ),
                "pair_second_condition_violations": list(
                    self.pair_second_condition_violations
                ),
                "recurrent_components_in_X": [
                    list(component) for component in self.recurrent_components_in_x
                ],
                "pair_defined": self.pair_defined,
                "conditions_vacuous_because_pair_undefined": not self.pair_defined,
                "diagnostic_pair_conditions_passed": (
                    self.diagnostic_pair_conditions_passed
                ),
            },
            "local_open_boundary_audit": {
                "same_chart_closed_one_ring_boundary_cells": len(
                    self.same_chart_open_boundary_cells
                ),
                "reverse_open_quotient_boundary_cells": len(
                    self.reverse_open_boundary_cells
                ),
                "retained_sources_on_same_chart_boundary": sorted(
                    (set(range(len(self.relation))) - self.relation.restricted_sources)
                    & self.same_chart_open_boundary_cells
                ),
                "retained_sources_on_reverse_open_boundary": sorted(
                    (set(range(len(self.relation))) - self.relation.restricted_sources)
                    & self.reverse_open_boundary_cells
                ),
                "safe_reference_sources_on_same_chart_boundary": sorted(
                    (self.reference_source_cells - self.removed_reference_sources)
                    & self.same_chart_open_boundary_cells
                ),
                "safe_reference_sources_on_reverse_open_boundary": sorted(
                    (self.reference_source_cells - self.removed_reference_sources)
                    & self.reverse_open_boundary_cells
                ),
                "S_on_same_chart_boundary": sorted(
                    self.s_cells & self.same_chart_open_boundary_cells
                ),
                "S_on_reverse_open_boundary": sorted(
                    self.s_cells & self.reverse_open_boundary_cells
                ),
                "S_boundary_condition_vacuous_because_S_empty": not self.s_cells,
            },
            "diagnostic_blockers": sorted(set(blockers)),
            "interpretation": (
                "Source invalidation does not recover the gait. Emptying uncertain "
                "rows destroys all recurrence in the pinned depth-20 relation."
                if not self.recurrent_components
                else "This is a diagnostic finite row-restricted relation only."
            ),
        }


def audit_garcia_strict_source_restriction(
    stage: _GarciaRawRelationStage,
    *,
    reference_source_cells: Collection[int] = PINNED_D20_REFERENCE_SOURCE_CELLS,
    max_relation_edges: int,
    max_relation_storage_bytes: int,
    max_undirected_adjacencies: int,
    max_adjacency_storage_bytes: int,
) -> GarciaStrictSourceRestrictionAudit:
    """Recompute the restricted graph audits without evaluating any ODE."""

    original_connectivity = stage.connectivity_audit
    if original_connectivity is None:
        original_connectivity = audit_garcia_sparse_relation_connectivity(
            stage.relation,
            stage.dyadic_cells,
            stage.quotient_neighbor_pairs,
            max_relation_edges=max_relation_edges,
            max_relation_storage_bytes=max_relation_storage_bytes,
            max_undirected_adjacencies=max_undirected_adjacencies,
            max_adjacency_storage_bytes=max_adjacency_storage_bytes,
        )
    reasons = garcia_strict_source_exclusion_reasons(
        stage.source_provenance,
        original_connectivity.disconnected_sources,
    )
    restricted_relation = GarciaStrictSourceRelation(stage.relation, reasons)
    restricted_connectivity = audit_garcia_sparse_relation_connectivity(
        restricted_relation,
        stage.dyadic_cells,
        stage.quotient_neighbor_pairs,
        max_relation_edges=max_relation_edges,
        max_relation_storage_bytes=max_relation_storage_bytes,
        max_undirected_adjacencies=max_undirected_adjacencies,
        max_adjacency_storage_bytes=max_adjacency_storage_bytes,
    )
    full_sccs = _relation_strongly_connected_components(restricted_relation)
    recurrent = tuple(
        component
        for component in full_sccs
        if len(component) > 1 or component[0] in restricted_relation[component[0]]
    )
    retained_sources = frozenset(range(len(restricted_relation))) - frozenset(reasons)
    induced_safe_sccs = _relation_strongly_connected_components(
        restricted_relation,
        retained_sources,
    )
    induced_safe_recurrent = tuple(
        component
        for component in induced_safe_sccs
        if len(component) > 1 or component[0] in restricted_relation[component[0]]
    )
    full_scc_histogram = Counter(len(component) for component in full_sccs)
    induced_scc_histogram = Counter(len(component) for component in induced_safe_sccs)
    retained_to_retained_edges = 0
    retained_to_restricted_edges = 0
    for source in retained_sources:
        for target in restricted_relation[source]:
            if target in retained_sources:
                retained_to_retained_edges += 1
            else:
                retained_to_restricted_edges += 1
    if (
        retained_to_retained_edges + retained_to_restricted_edges
        != restricted_connectivity.relation_edges
    ):
        raise ValueError("restricted edge partition is inconsistent")
    closure, hasse = _morse_order(restricted_relation, recurrent)
    reference_sources = frozenset(reference_source_cells)
    if any(
        type(source) is not int or source < 0 or source >= len(restricted_relation)
        for source in reference_sources
    ):
        raise ValueError("reference-source audit names a cell outside the relation")
    pinned_reference = reference_sources == frozenset(
        PINNED_D20_REFERENCE_SOURCE_CELLS
    )
    if pinned_reference:
        configuration = stage.relation_reference.get("configuration", {})
        family_provenance = stage.family_provenance or {}
        expected_binding = {
            "physical_model_revision": "garcia-physical-event-reset-v2",
            "boxmap_revision": "garcia-fixed-time-suspension-sample-bloat-v2",
            "family_cells_sha256": (
                "dfdf00cf2045301922367a3a63ef74226b512f81bc823463e9cf26882461bec1"
            ),
        }
        for key, expected in expected_binding.items():
            if configuration.get(key) != expected:
                raise ValueError(f"pinned reference binding disagrees at {key}")
        tube_fingerprint = (
            "8b762570020296fb80aaa4160c2a72bfd8645972c94e7e7d5e5bedb43a85f16a"
        )
        if family_provenance.get("fingerprint") != tube_fingerprint:
            raise ValueError("pinned reference binding disagrees with tube geometry")
        reference_probe_configuration: Mapping[str, object] = {
            "base_samples_per_stride": 17,
            "handle_samples_per_jump": 9,
            "completed_strides": 2,
            "completed_jumps": 2,
            "reference_max_step": 0.005,
            **expected_binding,
            "tube_geometry_fingerprint": tube_fingerprint,
            "source_cells_sha256": _source_list_sha256(reference_sources),
        }
    else:
        reference_probe_configuration = {
            "scope": "caller_supplied_nonpinned_reference_source_cells",
            "source_cells_sha256": _source_list_sha256(reference_sources),
        }
    reference_raw_sources = {
        "sample_failure": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].has_sample_failure
            )
        ),
        "unresolved_stage": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].unresolved_stage_edges
            )
        ),
        "missing_in_domain_target": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].missing_in_domain_target_cells
            )
        ),
        "ambient_boundary_piece": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].ambient_boundary_pieces
            )
        ),
        "wholly_outside_active_family_piece": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].wholly_outside_active_family_pieces
            )
        ),
        "explicit_empty_callback": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].explicit_empty_callback
            )
        ),
        "mapgraph_empty_image": tuple(
            sorted(
                source
                for source in reference_sources
                if stage.source_provenance[source].mapgraph_empty_image
            )
        ),
        "quotient_disconnected_image": tuple(
            sorted(reference_sources & original_connectivity.disconnected_sources)
        ),
    }
    reference_failure_reasons = Counter(
        reason
        for source in reference_sources
        for reason, count in stage.source_provenance[source].failure_reasons
        for _ in range(count)
    )
    raw_provenance_cause_counts = {
        "sample_failure_sources": sum(
            item.has_sample_failure for item in stage.source_provenance.values()
        ),
        "failed_sample_evaluations": sum(
            item.failed_samples for item in stage.source_provenance.values()
        ),
        "unresolved_stage_sources": sum(
            bool(item.unresolved_stage_edges)
            for item in stage.source_provenance.values()
        ),
        "composite_has_open_exit_sources": sum(
            item.has_open_exit for item in stage.source_provenance.values()
        ),
        "missing_in_domain_target_sources": sum(
            bool(item.missing_in_domain_target_cells)
            for item in stage.source_provenance.values()
        ),
        "ambient_boundary_piece_sources": sum(
            bool(item.ambient_boundary_pieces)
            for item in stage.source_provenance.values()
        ),
        "wholly_outside_active_family_piece_sources": sum(
            bool(item.wholly_outside_active_family_pieces)
            for item in stage.source_provenance.values()
        ),
        "explicit_empty_callback_sources": sum(
            item.explicit_empty_callback for item in stage.source_provenance.values()
        ),
        "mapgraph_empty_image_sources": sum(
            item.mapgraph_empty_image for item in stage.source_provenance.values()
        ),
        "quotient_disconnected_image_sources": len(
            original_connectivity.disconnected_sources
        ),
    }
    same_chart_boundary, reverse_open_boundary = _local_open_boundary_indices(stage)
    node_by_cell: dict[int, int] = {
        source: node
        for node, component in enumerate(recurrent)
        for source in component
    }
    reference_hits = Counter(
        node_by_cell[source]
        for source in reference_sources
        if source in node_by_cell
    )
    selected_node: int | None = None
    if reference_hits:
        ordered = sorted(reference_hits.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
            selected_node = int(ordered[0][0])
    s_cells = (
        frozenset(recurrent[selected_node])
        if selected_node is not None
        else frozenset()
    )
    f_s = frozenset(
        target for source in s_cells for target in restricted_relation[source]
    )
    x_cells = s_cells | f_s
    a_cells = x_cells - s_cells
    pair_second = tuple(
        source
        for source in sorted(a_cells)
        if frozenset(
            target
            for target in restricted_relation[source]
            if target in x_cells
        )
        - a_cells
    )
    recurrent_in_x = (
        _relation_recurrent_components(restricted_relation, x_cells)
        if x_cells
        else ()
    )
    return GarciaStrictSourceRestrictionAudit(
        relation=restricted_relation,
        restriction_reasons=reasons,
        original_connectivity=original_connectivity,
        restricted_connectivity=restricted_connectivity,
        recurrent_components=recurrent,
        full_scc_count=len(full_sccs),
        full_scc_size_histogram=dict(full_scc_histogram),
        induced_safe_scc_count=len(induced_safe_sccs),
        induced_safe_scc_size_histogram=dict(induced_scc_histogram),
        induced_safe_recurrent_components=induced_safe_recurrent,
        retained_to_retained_edges=retained_to_retained_edges,
        retained_to_restricted_edges=retained_to_restricted_edges,
        morse_reachability_edges=closure,
        morse_hasse_edges=hasse,
        reference_source_cells=reference_sources,
        reference_probe_configuration=reference_probe_configuration,
        reference_raw_provenance_sources=reference_raw_sources,
        reference_failure_reasons=dict(reference_failure_reasons),
        raw_provenance_cause_counts=raw_provenance_cause_counts,
        reference_node_hits=dict(reference_hits),
        selected_morse_node=selected_node,
        s_cells=s_cells,
        x_cells=x_cells,
        a_cells=a_cells,
        recurrent_components_in_x=recurrent_in_x,
        pair_second_condition_violations=pair_second,
        same_chart_open_boundary_cells=same_chart_boundary,
        reverse_open_boundary_cells=reverse_open_boundary,
    )


__all__ = [
    "PINNED_D20_REFERENCE_SOURCE_CELLS",
    "STRICT_SOURCE_RESTRICTION_RULE",
    "STRICT_SOURCE_RESTRICTION_SCHEMA",
    "GarciaStrictSourceRelation",
    "GarciaStrictSourceRestrictionAudit",
    "audit_garcia_strict_source_restriction",
    "garcia_strict_source_exclusion_reasons",
]
