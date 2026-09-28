from __future__ import annotations

import hashlib
import gzip
import json
import warnings

import numpy as np
import pytest

from hybrid_dynamics.examples.garcia_passive_walker_csr import (
    resume_garcia_local_relation_from_csr_checkpoint,
)
from hybrid_dynamics.examples.garcia_passive_walker_guard_aligned import (
    GUARD_ALIGNED_FAMILY_ALGORITHM,
    PINNED_UNSAFE_REFERENCE_SOURCE_IDS,
    PINNED_UNSAFE_SOURCE_IDS_SHA256,
    _transform_intrinsic_handle_rectangle,
    _transform_physical_rectangle,
    build_guard_aligned_garcia_tube,
    default_legacy_tube_bundle,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    compute_garcia_local_relation,
    full_garcia_dyadic_family,
)


def test_rectangle_transforms_contain_every_physical_and_intrinsic_corner():
    physical = (-0.3, -0.2, -0.5, -0.4, -0.1, 0.02, 0.6, 0.1)
    transformed = _transform_physical_rectangle(physical)
    lower, upper = np.asarray(transformed[:4]), np.asarray(transformed[4:])
    for theta in (physical[0], physical[4]):
        for omega in (physical[1], physical[5]):
            for phi in (physical[2], physical[6]):
                for phi_dot in (physical[3], physical[7]):
                    point = np.asarray(
                        [theta, omega, phi - 2 * theta, phi_dot - 2 * omega]
                    )
                    assert np.all(point >= lower) and np.all(point <= upper)

    handle = (-0.3, -0.34, 0.1, 0.2, -0.1, 0.02, 0.9, 0.8)
    transformed_handle = _transform_intrinsic_handle_rectangle(
        handle,
        phi_dot_min=-0.55,
        phi_dot_max=0.15,
        transversality_eta=0.1,
    )
    lower, upper = (
        np.asarray(transformed_handle[:4]),
        np.asarray(transformed_handle[4:]),
    )
    for theta in (handle[0], handle[4]):
        for omega in (handle[1], handle[5], -0.325):
            for rho in (handle[2], handle[6]):
                minimum = max(-0.55, 2 * omega + 0.1)
                nu = minimum + rho * (0.15 - minimum) - 2 * omega
                point = np.asarray([theta, omega, nu, handle[3]])
                assert np.all(point >= lower - 1e-15)
                point[-1] = handle[7]
                assert np.all(point <= upper + 1e-15)


def test_guard_aligned_tube_is_pinned_and_uses_complete_internal_attachments(
    tmp_path,
):
    if not default_legacy_tube_bundle().is_dir():
        pytest.skip("the local tube relation bundle is not in the repository")
    construction = build_guard_aligned_garcia_tube(axis_depth=1)
    payload = construction.to_dict()
    identifiers = json.dumps(
        list(PINNED_UNSAFE_REFERENCE_SOURCE_IDS),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(identifiers).hexdigest() == PINNED_UNSAFE_SOURCE_IDS_SHA256
    assert payload["algorithm"] == GUARD_ALIGNED_FAMILY_ALGORITHM
    assert payload["fixed_prior_diagnostic_enrichment"][
        "selection_uses_new_relation_or_SCC_counts"
    ] is False
    assert payload["legacy_negative_diagnostic"]["unsafe_source_ids"] == list(
        PINNED_UNSAFE_REFERENCE_SOURCE_IDS
    )
    assert construction.attachment_carriers
    for handle, bases in construction.attachment_carriers:
        assert handle.coordinates[-1] in (0, 1)
        assert bases
        for base in bases:
            q_lower = base.bounds(construction.charts)[2]
            q_upper = base.bounds(construction.charts)[6]
            assert q_lower <= 0.0 <= q_upper

    output = construction.write(tmp_path / "geometry.json.gz")
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert json.load(stream) == json.loads(json.dumps(payload))
    assert not tuple(tmp_path.glob(".geometry.json.gz.*.tmp"))


def test_guard_aligned_flagship_rejects_post_hoc_parameter_changes():
    for keyword, value in (
        ("normalized_radius", 0.125),
        ("gamma", 0.02),
        ("guard_delta", 0.04),
        ("transversality_eta", 0.09),
        ("reference_max_step", 0.01),
        ("maximum_base_samples_per_stride", 8193),
    ):
        try:
            build_guard_aligned_garcia_tube(axis_depth=1, **{keyword: value})
        except ValueError as error:
            assert "fixed predeclared settings" in str(error)
        else:
            raise AssertionError(f"flagship builder accepted changed {keyword}")


def test_guard_aligned_csr_bundle_resume_preserves_relation_and_candidate(tmp_path):
    family = full_garcia_dyadic_family(0)
    bundle = tmp_path / "guard-aligned-bundle"
    provenance = {"schema": "guard-aligned-test-family-v1"}
    caps = {
        "max_relation_edges": 1000,
        "max_csr_payload_bytes": 100_000,
        "max_native_cache_bytes": 100_000,
        "max_relation_storage_bytes": 100_000,
        "max_undirected_adjacencies": 1000,
        "max_adjacency_storage_bytes": 100_000,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fresh = compute_garcia_local_relation(
            family,
            coordinate_system="guard_aligned",
            reference_base_samples_per_stride=5,
            reference_handle_samples=3,
            csr_bundle_path=bundle,
            family_provenance=provenance,
            **caps,
        )
        resumed = resume_garcia_local_relation_from_csr_checkpoint(
            bundle,
            expected_family=family,
            expected_family_provenance=provenance,
            expected_run_configuration={
                "coordinate_system": "guard_aligned",
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
            max_vertices=2,
            max_relation_edges=caps["max_relation_edges"],
            max_csr_payload_bytes=caps["max_csr_payload_bytes"],
            max_relation_storage_bytes=caps["max_relation_storage_bytes"],
            max_undirected_adjacencies=caps["max_undirected_adjacencies"],
            max_adjacency_storage_bytes=caps["max_adjacency_storage_bytes"],
            reference_base_samples_per_stride=5,
            reference_handle_samples=3,
        )
    assert fresh.to_dict()["metadata"]["coordinate_system"] == "guard_aligned"
    assert tuple(tuple(fresh.relation[i]) for i in fresh.relation) == tuple(
        tuple(resumed.relation[i]) for i in resumed.relation
    )
    assert fresh.candidate == resumed.candidate
    assert fresh.quotient_neighbor_pairs == resumed.quotient_neighbor_pairs
