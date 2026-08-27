"""Physical integration tests for the native CMGDB Atlas bouncing ball."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_dynamics.examples.bouncing_ball_atlas import (
    AtlasBallCell,
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    BouncingBallQuotientIncidence,
    analytic_bouncing_ball_suspension_endpoint,
    bouncing_ball_atlas_charts,
    build_bouncing_ball_atlas_model,
    compute_bouncing_ball_atlas_acceptance,
    _is_domain_exit_failure_reason,
)
from hybrid_dynamics.src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
)


def test_ball_charts_use_only_the_outgoing_guard_and_physical_reset_seams():
    charts = bouncing_ball_atlas_charts()

    assert charts.base_bounds == ((0.0, 2.0), (-5.0, 5.0))
    assert charts.handle_bounds == ((-5.0, 0.0), (0.0, 1.0))
    assert charts.encode_handle((0.0, -2.0), 0.25) == (-2.0, 0.25)

    incidence = BouncingBallQuotientIncidence(charts, restitution=0.8)
    guard_base = AtlasBallCell(0, BASE_CHART_ID, (0.0, -2.1, 0.1, -1.9))
    reset_base = AtlasBallCell(1, BASE_CHART_ID, (0.0, 1.5, 0.1, 1.7))
    bottom_handle = AtlasBallCell(2, HANDLE_CHART_ID, (-2.1, 0.0, -1.9, 0.1))
    top_handle = AtlasBallCell(3, HANDLE_CHART_ID, (-2.1, 0.9, -1.9, 1.0))
    middle_handle = AtlasBallCell(4, HANDLE_CHART_ID, (-2.1, 0.4, -1.9, 0.6))

    assert incidence.intersects(guard_base, bottom_handle)
    assert incidence.intersects(reset_base, top_handle)
    assert not incidence.intersects(guard_base, middle_handle)
    assert not incidence.intersects(reset_base, middle_handle)


def test_analytic_reference_handles_ballistic_impact_and_zeno_rest_fiber():
    falling = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (1.0, 0.0),
        0.5,
    )
    assert isinstance(falling, HandleSuspensionSample)
    impact_time = np.sqrt(2.0 / 9.81)
    assert falling.continuous_time == pytest.approx(impact_time)
    assert falling.phase == pytest.approx(0.5 - impact_time)

    rest_half_turn = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (0.0, 0.0),
        1.5,
    )
    assert isinstance(rest_half_turn, HandleSuspensionSample)
    assert rest_half_turn.jump_index == 1
    assert rest_half_turn.phase == pytest.approx(0.5)

    rest_two_turns = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (0.0, 0.0),
        2.0,
    )
    assert isinstance(rest_two_turns, BaseSuspensionSample)
    assert rest_two_turns.jumps_completed == 2
    assert np.allclose(rest_two_turns.state, (0.0, 0.0))


def test_analytic_reference_respects_both_source_quotient_identifications():
    tau = 0.2
    from_base_guard = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (0.0, -2.0),
        tau,
    )
    from_handle_zero = analytic_bouncing_ball_suspension_endpoint(
        HANDLE_CHART_ID,
        (-2.0, 0.0),
        tau,
    )
    assert isinstance(from_base_guard, HandleSuspensionSample)
    assert isinstance(from_handle_zero, HandleSuspensionSample)
    assert from_base_guard.phase == pytest.approx(from_handle_zero.phase)
    assert np.allclose(from_base_guard.guard_state, from_handle_zero.guard_state)

    from_base_reset = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (0.0, 1.6),
        tau,
    )
    from_handle_one = analytic_bouncing_ball_suspension_endpoint(
        HANDLE_CHART_ID,
        (-2.0, 1.0),
        tau,
    )
    assert isinstance(from_base_reset, BaseSuspensionSample)
    assert isinstance(from_handle_one, BaseSuspensionSample)
    assert np.allclose(from_base_reset.state, from_handle_one.state)


def test_ball_builder_installs_the_tagged_callback_in_atlas_model():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    setup = build_bouncing_ball_atlas_model(depth=4, t_star=2.0)

    assert setup.model.chart_ids() == [BASE_CHART_ID, HANDLE_CHART_ID]
    assert setup.box_map.samples_per_axis == 3
    assert setup.box_map.padding_cells == 1.0
    assert setup.box_map.require_domain_path is True
    value = setup.box_map(BASE_CHART_ID, (0.0, -1.0, 0.5, 1.0))
    assert value
    assert {piece[0] for piece in value} <= {BASE_CHART_ID, HANDLE_CHART_ID}

    analytic = analytic_bouncing_ball_suspension_endpoint(
        BASE_CHART_ID,
        (0.4, 1.0),
        2.0,
    )
    sampled = setup.box_map.evaluate_point(BASE_CHART_ID, (0.4, 1.0))
    assert type(analytic) is type(sampled)
    if isinstance(analytic, BaseSuspensionSample):
        assert np.allclose(analytic.state, sampled.state, atol=1e-9)
        assert analytic.jumps_completed == sampled.jumps_completed
    else:
        assert isinstance(sampled, HandleSuspensionSample)
        assert np.allclose(analytic.guard_state, sampled.guard_state, atol=1e-9)
        assert analytic.phase == pytest.approx(sampled.phase, abs=1e-9)
        assert analytic.jump_index == sampled.jump_index


def test_coarse_native_ball_run_passes_local_tau_two_falsification_gates():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates the AtlasModel extension")

    result = compute_bouncing_ball_atlas_acceptance(
        depth=4,
        t_star=2.0,
        reference_speed_samples=7,
        reference_base_samples=5,
        reference_handle_samples=5,
    )

    assert result.reference_endpoint_audit.passed
    assert result.image_connectivity_audit.passed
    assert result.zeno_support_connected
    assert result.zeno_base_connected
    assert result.box_map_diagnostics.unresolved_stage_edges == 0
    assert result.one_morse_node
    summary = result.summary()
    assert summary["reference_includes_exact_rest_fiber"] is True
    assert summary["relation_scope"] == "nonempty local-window relation"
    assert summary["empty_sources_excluded_from_connectivity_audit"] is True
    assert summary["box_map_failed_samples"] > 0
    assert summary["domain_exit_sample_failures"] == summary["box_map_failed_samples"]
    assert summary["other_sample_failures"] == 0
    assert summary["global_attractor_lattice_interpretation"] is False
    assert summary["conley_index_computed"] is False


@pytest.mark.parametrize(
    "reason",
    (
        "ValueError: trajectory leaves the declared state space",
        "ValueError: reset leaves the declared state space",
        "ValueError: guard face exits the declared state space",
        "ValueError: reset face exits the declared nonrectangular state space",
        "ValueError: trajectory leaves the declared state-space bounds",
    ),
)
def test_ball_failure_summary_recognizes_domain_exit_reasons(reason):
    assert _is_domain_exit_failure_reason(reason)


def test_ball_failure_summary_keeps_non_domain_failures_separate():
    assert not _is_domain_exit_failure_reason(
        "RuntimeError: recorded trajectory does not cover suspension time"
    )


def test_ball_depth_must_give_square_dyadic_charts():
    with pytest.raises(ValueError, match="even depth"):
        build_bouncing_ball_atlas_model(depth=5)


def test_positive_guard_velocities_are_rejected():
    with pytest.raises(ValueError, match="v_guard <= 0"):
        bouncing_ball_atlas_charts(guard_velocity_bounds=(-5.0, 0.1))
