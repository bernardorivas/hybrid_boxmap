from __future__ import annotations

import warnings
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from hybrid_dynamics.examples.garcia_endpoint_precompute import (
    GarciaEndpointPrecomputeConfig,
    endpoint_cache_key,
    precompute_garcia_endpoint_cache,
)
from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    DEFAULT_BASE_BOUNDS,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    _AuditedGarciaSuspensionBoxMap,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
    GUARD_ALIGNED_DOMAIN_BOUNDS,
    PERIOD_TWO_POINT_A,
    guard_aligned_post_impact_state,
    post_impact_state,
)


def _small_source_box() -> tuple[int, tuple[float, ...]]:
    center = post_impact_state(*PERIOD_TWO_POINT_A)
    radius = np.asarray((2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4))
    return 0, tuple(float(value) for value in (*(center - radius), *(center + radius)))


def _config() -> GarciaEndpointPrecomputeConfig:
    return GarciaEndpointPrecomputeConfig(t_star=0.02, max_step=0.01)


def _failing_degenerate_source_box() -> tuple[int, tuple[float, ...]]:
    point = (-0.1, -0.1, 0.0, 0.0)
    return 0, (*point, *point)


def _normalized(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, type) and issubclass(value, BaseException):
        return ("exception_type", value.__module__, value.__qualname__)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple((field.name, _normalized(getattr(value, field.name))) for field in fields(value)),
        )
    if isinstance(value, tuple):
        return tuple(_normalized(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalized(item) for item in value)
    return value


def test_endpoint_key_matches_closed_face_rounding() -> None:
    assert endpoint_cache_key(1, (0.1 + 0.2, -0.0, 1.0, 0.5)) == (
        1,
        0.3,
        -0.0,
        1.0,
        0.5,
    )


def test_duplicate_source_boxes_keep_first_seen_tensor_order() -> None:
    source = _small_source_box()
    result = precompute_garcia_endpoint_cache(
        (source, source),
        _config(),
        workers=1,
    )
    assert result.logical_tensor_points == 162
    assert result.unique_points == 81
    assert result.duplicate_tensor_points == 81
    assert result.entries[0][0] == endpoint_cache_key(0, source[1][:4])
    assert len({entry[0] for entry in result.entries}) == 81


def test_process_pool_is_bitwise_equal_to_serial_precompute() -> None:
    sources = (_small_source_box(), _failing_degenerate_source_box())
    serial = precompute_garcia_endpoint_cache(sources, _config(), workers=1)
    parallel = precompute_garcia_endpoint_cache(
        sources,
        _config(),
        workers=2,
        chunksize=7,
        submission_buffer=3,
    )
    assert serial.mode == "serial"
    assert parallel.mode == "process_pool"
    assert tuple(key for key, _value in parallel.entries) == tuple(
        key for key, _value in serial.entries
    )
    assert _normalized(parallel.entries) == _normalized(serial.entries)
    assert serial.entries[-1][1][0] is False


def test_pool_setup_failure_restarts_from_scratch_serially() -> None:
    source = _small_source_box()
    expected = precompute_garcia_endpoint_cache((source,), _config(), workers=1)
    fallback = precompute_garcia_endpoint_cache(
        (source,),
        _config(),
        workers=2,
        start_method="not-a-real-start-method",
    )
    assert fallback.mode == "serial_fallback"
    assert fallback.used_workers == 1
    assert fallback.fallback_reason is not None
    assert _normalized(fallback.entries) == _normalized(expected.entries)


def test_seeded_callback_has_exact_serial_pieces_and_diagnostics() -> None:
    source = _small_source_box()
    config = _config()
    walker = GarciaPassiveWalker(
        domain_bounds=list(DEFAULT_BASE_BOUNDS),
        max_jumps=config.max_jumps,
    )
    charts = garcia_passive_walker_atlas_charts(base_bounds=walker.domain_bounds)

    def box_map() -> _AuditedGarciaSuspensionBoxMap:
        return _AuditedGarciaSuspensionBoxMap(
            walker.system,
            charts,
            config.t_star,
            samples_per_axis=config.samples_per_axis,
            padding_cells=config.padding_cells,
            max_jumps=config.max_jumps,
            max_step=config.max_step,
            atol=config.atol,
            require_domain_path=config.require_domain_path,
            diagnostics_limit=0,
        )

    ordinary = box_map()
    ordinary_pieces = ordinary(*source)
    ordinary_records = ordinary.source_records()
    ordinary_counts = ordinary.point_cache_counts()

    precomputed = precompute_garcia_endpoint_cache((source,), config, workers=1)
    seeded = box_map()
    seeded.seed_point_cache(precomputed.entries, precompute_result=precomputed)
    seeded_pieces = seeded(*source)

    assert seeded_pieces == ordinary_pieces
    assert seeded.source_records() == ordinary_records
    assert seeded.diagnostics() == ordinary.diagnostics()
    assert seeded.point_cache_counts() == ordinary_counts == (81, 0)
    assert seeded.endpoint_precompute_result() is precomputed


def test_guard_aligned_precompute_matches_the_guard_aligned_callback_exactly() -> None:
    center = guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A)
    radius = np.full(4, 2.0e-4)
    source = (
        0,
        tuple(float(value) for value in (*(center - radius), *(center + radius))),
    )
    config = GarciaEndpointPrecomputeConfig(
        t_star=0.02,
        max_step=0.01,
        domain_bounds=GUARD_ALIGNED_DOMAIN_BOUNDS,
        coordinate_system="guard_aligned",
    )
    walker = GuardAlignedGarciaPassiveWalker(
        domain_bounds=list(GUARD_ALIGNED_DOMAIN_BOUNDS),
        max_jumps=config.max_jumps,
    )
    charts = garcia_guard_aligned_atlas_charts(base_bounds=walker.domain_bounds)
    ordinary = _AuditedGarciaSuspensionBoxMap(
        walker.system,
        charts,
        config.t_star,
        samples_per_axis=config.samples_per_axis,
        padding_cells=config.padding_cells,
        max_jumps=config.max_jumps,
        max_step=config.max_step,
        atol=config.atol,
        require_domain_path=config.require_domain_path,
        diagnostics_limit=0,
    )
    expected_pieces = ordinary(*source)
    precomputed = precompute_garcia_endpoint_cache((source,), config, workers=1)
    seeded = _AuditedGarciaSuspensionBoxMap(
        walker.system,
        charts,
        config.t_star,
        samples_per_axis=config.samples_per_axis,
        padding_cells=config.padding_cells,
        max_jumps=config.max_jumps,
        max_step=config.max_step,
        atol=config.atol,
        require_domain_path=config.require_domain_path,
        diagnostics_limit=0,
    )
    seeded.seed_point_cache(precomputed.entries, precompute_result=precomputed)
    assert seeded(*source) == expected_pieces
    assert seeded.source_records() == ordinary.source_records()


def test_precompute_worker_validation_is_early() -> None:
    from hybrid_dynamics.examples.garcia_passive_walker_local import (
        compute_garcia_local_relation,
        full_garcia_dyadic_family,
    )

    with pytest.raises(ValueError, match="precompute_workers"):
        compute_garcia_local_relation(
            full_garcia_dyadic_family(0),
            precompute_workers=-1,
        )


def test_complete_depth_zero_relation_matches_serial_payload_exactly() -> None:
    from hybrid_dynamics.examples.garcia_passive_walker_local import (
        compute_garcia_local_relation,
        full_garcia_dyadic_family,
    )

    family = full_garcia_dyadic_family(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        serial = compute_garcia_local_relation(family, precompute_workers=0)
        parallel = compute_garcia_local_relation(family, precompute_workers=2)

    serial_payload = serial.to_dict()
    parallel_payload = parallel.to_dict()
    execution = parallel_payload["metadata"]["endpoint_precompute"]
    assert execution["requested_workers"] == 2
    assert execution["used_workers"] == 2
    assert execution["process_count"] == 2
    assert execution["parity_provenance"]["source_box_order"] == (
        "native Atlas cell-index order"
    )
    serial_payload["metadata"].pop("elapsed_seconds")
    parallel_payload["metadata"].pop("elapsed_seconds")
    serial_payload["metadata"].pop("endpoint_precompute")
    parallel_payload["metadata"].pop("endpoint_precompute")
    assert parallel_payload == serial_payload
