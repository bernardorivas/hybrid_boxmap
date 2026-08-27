"""Physical plumbing tests for the native CMGDB Garcia walker Atlas."""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    AtlasWalkerCell,
    GarciaWalkerQuotientIncidence,
    GuardAlignedGarciaWalkerQuotientIncidence,
    build_garcia_gait_active_family,
    build_garcia_passive_walker_atlas_model,
    compute_garcia_passive_walker_atlas_acceptance,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    PERIOD_TWO_POINT_A,
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
    guard_aligned_post_impact_state,
    heelstrike_transversality,
    post_impact_state,
)


def _cell(index, chart_id, lower, upper):
    return AtlasWalkerCell(
        index,
        chart_id,
        tuple(float(value) for value in (*lower, *upper)),
    )


def test_walker_handle_chart_is_intrinsic_transverse_and_contains_both_impacts():
    walker = GarciaPassiveWalker(max_jumps=4)
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )

    assert charts.base_bounds == tuple(walker.domain_bounds)
    assert charts.handle_dimension == 4
    assert charts.handle_bounds[0] == pytest.approx((-0.32, -0.05))
    assert charts.handle_bounds[1] == pytest.approx((-0.35, 0.024))
    assert charts.handle_bounds[2:] == ((0.0, 1.0), (0.0, 1.0))

    trajectory = walker.system.simulate(
        post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=0.01,
    )
    for guard, _reset in trajectory.jump_states[:2]:
        encoded = charts.encode_handle(guard, 0.4)
        decoded, phase = charts.decode_handle(encoded)
        np.testing.assert_allclose(decoded, guard, atol=2e-12)
        assert phase == pytest.approx(0.4)

    # Every corner of the intrinsic guard patch lies on phi=2*theta and is
    # transverse; the chart does not fill its handle with inadmissible points.
    for intrinsic in itertools.product(
        *[(interval[0], interval[1]) for interval in charts.guard_bounds]
    ):
        guard = np.asarray(charts.guard_embedding(intrinsic), dtype=float)
        assert guard[2] == pytest.approx(2.0 * guard[0])
        assert guard[0] <= -walker.guard_delta + 1e-12
        assert heelstrike_transversality(guard) >= walker.transversality_eta - 1e-12
        assert all(
            lower - 1e-12 <= value <= upper + 1e-12
            for value, (lower, upper) in zip(guard, charts.base_bounds)
        )
        reset = walker.system.apply_reset_map(guard)
        assert all(
            lower - 1e-12 <= value <= upper + 1e-12
            for value, (lower, upper) in zip(reset, charts.base_bounds)
        )


def test_walker_quotient_incidence_uses_guard_and_nonlinear_reset_faces():
    walker = GarciaPassiveWalker(max_jumps=2)
    charts = garcia_passive_walker_atlas_charts(base_bounds=walker.domain_bounds)
    trajectory = walker.system.simulate(
        post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 5.0),
        max_jumps=1,
        dense_output=True,
        max_step=0.01,
    )
    guard, reset = trajectory.jump_states[0]
    intrinsic = np.asarray(charts.encode_handle(guard, 0.0)[:-1])
    epsilon = 2e-3
    guard_base = _cell(0, BASE_CHART_ID, guard - epsilon, guard + epsilon)
    reset_base = _cell(1, BASE_CHART_ID, reset - epsilon, reset + epsilon)
    bottom = _cell(
        2,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.0],
        np.r_[intrinsic + epsilon, 0.1],
    )
    top = _cell(
        3,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.9],
        np.r_[intrinsic + epsilon, 1.0],
    )
    middle = _cell(
        4,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.4],
        np.r_[intrinsic + epsilon, 0.6],
    )
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )

    assert incidence.intersects(guard_base, bottom)
    assert incidence.intersects(reset_base, top)
    assert not incidence.intersects(guard_base, middle)
    assert not incidence.intersects(reset_base, middle)


def test_guard_aligned_chart_has_internal_q_face_and_exact_two_attachments():
    walker = GuardAlignedGarciaPassiveWalker(max_jumps=2)
    charts = garcia_guard_aligned_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    assert charts.base_bounds[2][0] == pytest.approx(-charts.base_bounds[2][1])
    assert charts.base_bounds[2][0] < 0.0 < charts.base_bounds[2][1]
    assert charts.guard_bounds == (
        (-0.32, -0.05),
        (-0.35, 0.05),
        (0.1, 0.85),
    )

    trajectory = walker.system.simulate(
        guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 5.0),
        max_jumps=1,
        dense_output=True,
        max_step=0.005,
    )
    guard, reset = trajectory.jump_states[0]
    intrinsic = np.asarray(charts.encode_handle(guard, 0.0)[:-1])
    decoded, phase = charts.decode_handle((*intrinsic, 0.37))
    np.testing.assert_allclose(decoded, guard, rtol=0.0, atol=2e-10)
    assert phase == pytest.approx(0.37)
    assert guard[0] <= -walker.guard_delta
    assert reset[0] >= walker.guard_delta
    assert abs(guard[2]) < 1e-10
    assert reset[2] == 0.0

    epsilon = 2e-3
    guard_base = _cell(20, BASE_CHART_ID, guard - epsilon, guard + epsilon)
    reset_base = _cell(21, BASE_CHART_ID, reset - epsilon, reset + epsilon)
    bottom = _cell(
        22,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.0],
        np.r_[intrinsic + epsilon, 0.1],
    )
    top = _cell(
        23,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.9],
        np.r_[intrinsic + epsilon, 1.0],
    )
    middle = _cell(
        24,
        HANDLE_CHART_ID,
        np.r_[intrinsic - epsilon, 0.4],
        np.r_[intrinsic + epsilon, 0.6],
    )
    incidence = GuardAlignedGarciaWalkerQuotientIncidence(
        charts,
        transversality_eta=walker.transversality_eta,
    )
    assert incidence.intersects(guard_base, bottom)
    assert incidence.intersects(reset_base, top)
    assert not incidence.intersects(reset_base, bottom)
    assert not incidence.intersects(guard_base, top)
    assert not incidence.intersects(guard_base, middle)


def test_q_zero_is_an_exact_dyadic_base_face_at_every_positive_depth():
    charts = garcia_guard_aligned_atlas_charts()
    lower, upper = charts.base_bounds[2]
    for axis_depth in range(1, 7):
        subdivisions = 2**axis_depth
        midpoint = subdivisions // 2
        left_upper = lower + (upper - lower) * midpoint / subdivisions
        right_lower = lower + (upper - lower) * midpoint / subdivisions
        assert left_upper == 0.0
        assert right_lower == 0.0


def test_walker_builder_installs_four_dimensional_tagged_callback():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    setup = build_garcia_passive_walker_atlas_model(depth=4)
    initial = post_impact_state(*PERIOD_TWO_POINT_A)
    value = setup.box_map(BASE_CHART_ID, tuple(initial) + tuple(initial))

    assert setup.model.chart_ids() == [BASE_CHART_ID, HANDLE_CHART_ID]
    assert setup.charts.base_dimension == 4
    assert setup.charts.handle_dimension == 4
    assert setup.box_map.require_domain_path is True
    assert value
    assert {piece[0] for piece in value} <= {BASE_CHART_ID, HANDLE_CHART_ID}


def test_coarse_native_walker_run_passes_only_the_declared_plumbing_gates():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = compute_garcia_passive_walker_atlas_acceptance(
            depth=4,
            reference_base_samples_per_stride=7,
            reference_handle_samples=5,
        )

    assert result.reference_covered
    assert result.image_values_connected
    assert result.box_map_diagnostics.unresolved_stage_edges == 0
    assert result.plumbing_gates_passed
    assert not result.sampled_self_map_complete

    summary = result.summary()
    assert summary["map_cells"] == 32
    assert summary["morse_nodes"] == 1
    assert summary["box_map_failed_samples"] > 0
    assert (
        summary["domain_exit_sample_failures"]
        + summary["inadmissible_guard_sample_failures"]
        + summary["incomplete_trajectory_sample_failures"]
        == summary["box_map_failed_samples"]
    )
    assert summary["inadmissible_guard_sample_failures"] > 0
    assert summary["target_chart_sample_failures"] == 0
    assert summary["active_subgrid_used"] is False
    assert summary["active_subgrid_api_available"] is True
    assert summary["scientific_morse_recovery_claimed"] is False
    assert summary["global_attractor_lattice_interpretation"] is False
    assert summary["conley_index_computed"] is False


def test_walker_depth_must_refine_all_four_axes_equally():
    with pytest.raises(ValueError, match="multiple of four"):
        build_garcia_passive_walker_atlas_model(depth=6)


def test_gait_local_family_is_a_thick_full_dimensional_dyadic_neighborhood():
    walker = GarciaPassiveWalker(max_jumps=20)
    charts = garcia_passive_walker_atlas_charts(base_bounds=walker.domain_bounds)
    family = build_garcia_gait_active_family(
        walker,
        charts,
        axis_depth=2,
        stencil_radius=1,
    )

    counts = family.chart_counts()
    assert family.subdivisions_per_axis == 4
    assert len(family.base_seed_cells) < counts[BASE_CHART_ID] < 4**4
    assert len(family.handle_seed_cells) < counts[HANDLE_CHART_ID] < 4**4
    assert len(family.tagged_cells) < 2 * 4**4
    assert all(depth == 2 and len(coordinates) == 4 for _, depth, coordinates in family.tagged_cells)


def test_walker_active_builder_uses_native_selected_cells_without_map_evaluation():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb.AtlasModel, "set_active_subgrid"):
        pytest.skip("installed CMGDB predates the active-subgrid extension")

    setup = build_garcia_passive_walker_atlas_model(
        depth=8,
        active_stencil_radius=1,
    )
    assert setup.active_family is not None
    assert setup.active_tracker is not None
    assert setup.model.active_subgrid_configured()
    assert setup.model.initial_cell_count() == len(setup.active_family.tagged_cells)
    assert setup.model.initial_cell_count() < 2 * 4**4
    assert setup.model.phaseSpace().num_charts() == 2
    assert setup.active_tracker.diagnostics().records == ()


def test_walker_active_builder_refuses_an_orbit_only_seed():
    with pytest.raises(ValueError, match="orbit-only"):
        build_garcia_passive_walker_atlas_model(
            depth=8,
            active_stencil_radius=0,
        )


def test_recorded_active_subgrid_screen_rejects_every_morse_result():
    record_path = (
        Path(__file__).parents[2]
        / "data"
        / "garcia_passive_walker_atlas"
        / "active_subgrid_screen_tau050.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["family_is_orbit_only"] is False
    assert record["whole_cell_outer_enclosure_certified"] is False
    assert record["scientific_morse_recovery_claimed"] is False
    assert len(record["cases"]) == 5
    assert all(case["reference_misses"] == 0 for case in record["cases"])
    assert all(case["unresolved_stage_edges"] == 0 for case in record["cases"])
    assert all(
        case["disconnected_images"] > 0
        or case["empty_graph_images"] > 0
        or case["active_boundary_exit_sources"] > 0
        for case in record["cases"]
    )
