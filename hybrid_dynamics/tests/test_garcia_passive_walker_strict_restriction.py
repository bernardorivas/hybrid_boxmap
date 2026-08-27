"""Tests for the provenance-only Garcia strict-source diagnostic."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    read_garcia_csr_provenance_checkpoint,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaSourceExitProvenance,
    _relation_recurrent_components,
    compute_garcia_local_relation,
    full_garcia_dyadic_family,
)
from hybrid_dynamics.examples.garcia_passive_walker_strict_restriction import (
    PINNED_D20_REFERENCE_SOURCE_CELLS,
    GarciaStrictSourceRelation,
    _morse_order,
    audit_garcia_strict_source_restriction,
    garcia_strict_source_exclusion_reasons,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    _GarciaHybridSystem,
)


def _provenance(
    source: int,
    *,
    failed: int = 0,
    unresolved: tuple[tuple[int, int], ...] = (),
    missing: int = 0,
    ambient: int = 0,
    outside: int = 0,
    empty: bool = False,
) -> GarciaSourceExitProvenance:
    return GarciaSourceExitProvenance(
        source_index=source,
        callback_invocations=1,
        successful_samples=81 - failed,
        failed_samples=failed,
        failure_reasons=(("sample failure", failed),) if failed else (),
        unresolved_stage_edges=unresolved,
        returned_pieces=0 if empty else 1,
        explicit_empty_callback=empty,
        mapgraph_empty_image=empty,
        active_target_cells=0 if empty else 1,
        missing_in_domain_target_cells=missing,
        missing_in_domain_witnesses=(),
        target_pieces=(),
        ambient_boundary_pieces=ambient,
        wholly_outside_active_family_pieces=outside,
    )


def test_fixed_restriction_reasons_preserve_all_overlapping_causes():
    provenance = {
        0: _provenance(0),
        1: _provenance(1, failed=2, missing=3, ambient=1),
        2: _provenance(2, unresolved=((0, 1),)),
        3: _provenance(3, outside=1),
    }
    reasons = garcia_strict_source_exclusion_reasons(provenance, (1, 3))
    assert 0 not in reasons
    assert reasons[1] == (
        "callback_or_sample_failure",
        "composite_has_open_exit",
        "missing_active_target",
        "quotient_disconnected_image",
    )
    assert reasons[2] == ("unresolved_stage", "composite_has_open_exit")
    assert reasons[3] == (
        "composite_has_open_exit",
        "missing_active_target",
        "quotient_disconnected_image",
    )


def test_lazy_source_mask_and_morse_order_do_not_modify_original_relation():
    original = {0: (0, 1), 1: (2,), 2: (2,), 3: ()}
    masked = GarciaStrictSourceRelation(original, (1,))
    assert tuple(masked[0]) == (0, 1)
    assert tuple(masked[1]) == ()
    assert original[1] == (2,)

    unmasked = GarciaStrictSourceRelation(original, ())
    recurrent = _relation_recurrent_components(unmasked)
    assert recurrent == ((0,), (2,))
    closure, hasse = _morse_order(unmasked, recurrent)
    assert closure == ((0, 1),)
    assert hasse == ((0, 1),)


def test_bundle_read_and_strict_audit_execute_with_ODE_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = tmp_path / "bundle"
    family = full_garcia_dyadic_family(1)
    family_provenance = {"schema": "strict-source-test-family-v1"}
    compute_garcia_local_relation(
        family,
        precompute_workers=0,
        csr_bundle_path=bundle,
        family_provenance=family_provenance,
        max_relation_edges=100_000,
        max_csr_payload_bytes=10_000_000,
        max_native_cache_bytes=10_000_000,
        max_relation_storage_bytes=100_000_000,
        max_undirected_adjacencies=100_000,
        max_adjacency_storage_bytes=100_000_000,
    )

    def forbid_ode(*_args, **_kwargs):
        raise AssertionError("strict-source audit must not run an ODE")

    monkeypatch.setattr(_GarciaHybridSystem, "simulate", forbid_ode)
    stage = read_garcia_csr_provenance_checkpoint(
        bundle,
        expected_family=family,
        expected_family_provenance=family_provenance,
        expected_run_configuration={
            "t_star": 0.5,
            "gamma": 0.0172,
            "guard_delta": 0.05,
            "transversality_eta": 0.1,
            "samples_per_axis": 3,
            "padding_cells": 1.0,
            "max_step": 0.02,
            "max_jumps": 20,
            "require_domain_path": True,
        },
        max_vertices=100,
        max_relation_edges=100_000,
        max_csr_payload_bytes=10_000_000,
        max_relation_storage_bytes=100_000_000,
        max_undirected_adjacencies=100_000,
        max_adjacency_storage_bytes=100_000_000,
    )
    audit = audit_garcia_strict_source_restriction(
        stage,
        reference_source_cells=(0,),
        max_relation_edges=100_000,
        max_relation_storage_bytes=100_000_000,
        max_undirected_adjacencies=100_000,
        max_adjacency_storage_bytes=100_000_000,
    )
    payload = audit.to_dict()
    assert payload["restriction"]["original_provenance_retained_unchanged"] is True
    assert audit.relation.source_relation is stage.relation
    assert payload["relation_counts"]["restricted_edges"] <= payload[
        "relation_counts"
    ]["original_edges"]
    if not audit.pair_defined:
        assert payload["candidate_pair"][
            "conditions_vacuous_because_pair_undefined"
        ] is True


def test_pinned_reference_source_list_is_canonical_and_distinct():
    assert len(PINNED_D20_REFERENCE_SOURCE_CELLS) == 66
    assert PINNED_D20_REFERENCE_SOURCE_CELLS == tuple(
        sorted(set(PINNED_D20_REFERENCE_SOURCE_CELLS))
    )
