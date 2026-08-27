"""Physical integration tests for the native CMGDB Atlas wheel model."""

from __future__ import annotations

import pytest

from hybrid_dynamics.examples.rimless_wheel_atlas import (
    AtlasWheelCell,
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    RimlessWheelQuotientIncidence,
    build_rimless_wheel_atlas_model,
    compute_rimless_wheel_atlas_acceptance,
    rimless_wheel_atlas_charts,
)


def test_wheel_charts_use_the_outgoing_guard_and_physical_reset_seams():
    charts = rimless_wheel_atlas_charts()

    assert charts.base_bounds == ((-0.2, 0.6), (-0.5, 1.0))
    assert charts.handle_bounds == ((0.0, 1.0), (0.0, 1.0))
    assert charts.encode_handle((0.6, 0.75), 0.25) == (0.75, 0.25)

    incidence = RimlessWheelQuotientIncidence(charts, alpha=0.4, gamma=0.2)
    guard_base = AtlasWheelCell(0, BASE_CHART_ID, (0.55, 0.7, 0.6, 0.8))
    reset_base = AtlasWheelCell(1, BASE_CHART_ID, (-0.2, 0.45, -0.15, 0.55))
    bottom_handle = AtlasWheelCell(2, HANDLE_CHART_ID, (0.7, 0.0, 0.8, 0.1))
    top_handle = AtlasWheelCell(3, HANDLE_CHART_ID, (0.7, 0.9, 0.8, 1.0))
    middle_handle = AtlasWheelCell(4, HANDLE_CHART_ID, (0.7, 0.4, 0.8, 0.6))

    assert incidence.intersects(guard_base, bottom_handle)
    assert incidence.intersects(reset_base, top_handle)
    assert not incidence.intersects(guard_base, middle_handle)
    assert not incidence.intersects(reset_base, middle_handle)


def test_wheel_builder_installs_the_tagged_callback_in_atlas_model():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    setup = build_rimless_wheel_atlas_model(depth=4)

    assert setup.model.chart_ids() == [BASE_CHART_ID, HANDLE_CHART_ID]
    assert setup.box_map.require_domain_path is True
    base_value = setup.box_map(BASE_CHART_ID, (-0.2, 0.5, 0.0, 0.7))
    assert base_value
    assert {piece[0] for piece in base_value} <= {BASE_CHART_ID, HANDLE_CHART_ID}


def test_coarse_native_wheel_run_passes_topological_falsification_gates():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    result = compute_rimless_wheel_atlas_acceptance(
        depth=4,
        reference_base_samples=31,
        reference_handle_samples=17,
    )

    assert result.reference_endpoint_audit.passed
    assert result.image_connectivity_audit.passed
    assert result.gait_base_connected
    assert result.box_map_diagnostics.unresolved_stage_edges == 0
    summary = result.summary()
    assert summary["relation_scope"] == "nonempty local-window relation"
    assert summary["empty_sources_excluded_from_connectivity_audit"] is True
    assert summary["global_attractor_lattice_interpretation"] is False
    assert summary["conley_index_computed"] is False
    # This deliberately coarse run is a plumbing regression, not the claimed
    # two-node scientific result: it still merges the saddle and gait.
    assert result.gait_node == result.saddle_node


def test_wheel_depth_must_give_square_dyadic_charts():
    with pytest.raises(ValueError, match="even depth"):
        build_rimless_wheel_atlas_model(depth=5)
