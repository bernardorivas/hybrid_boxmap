"""Regression tests for the Garcia passive-walker suspension pipeline."""

import numpy as np

from hybrid_dynamics.examples._fixed_time_suspension import CemeteryCell
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    GUARD_ALIGNED_DOMAIN_BOUNDS,
    PERIOD_TWO_POINT_A,
    PERIOD_TWO_POINT_B,
    RAW_FAILURE_LABELS,
    GarciaPassiveWalker,
    GuardAlignedGarciaPassiveWalker,
    build_garcia_passive_walker_suspension_pipeline,
    decode_guard_aligned_state,
    encode_guard_aligned_state,
    encode_guard_aligned_velocity,
    guard_aligned_ode_fun,
    guard_aligned_post_impact_state,
    guard_aligned_reset_map,
    heelstrike_gap,
    heelstrike_transversality,
    ode_fun,
    post_impact_state,
    reset_map,
)
from hybrid_dynamics.src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
    simulate_suspension_endpoint,
)


def _assert_coordinate_conjugate_endpoint(physical, aligned, *, atol=2e-9):
    assert type(physical) is type(aligned)
    assert physical.total_time == aligned.total_time
    assert physical.jumps_completed == aligned.jumps_completed if isinstance(
        physical, BaseSuspensionSample
    ) else physical.jump_index == aligned.jump_index
    assert abs(physical.continuous_time - aligned.continuous_time) <= atol
    if isinstance(physical, BaseSuspensionSample):
        np.testing.assert_allclose(
            aligned.state,
            encode_guard_aligned_state(physical.state),
            rtol=0.0,
            atol=atol,
        )
    else:
        assert abs(physical.phase - aligned.phase) <= atol
        np.testing.assert_allclose(
            aligned.guard_state,
            encode_guard_aligned_state(physical.guard_state),
            rtol=0.0,
            atol=atol,
        )
        np.testing.assert_allclose(
            aligned.reset_state,
            encode_guard_aligned_state(physical.reset_state),
            rtol=0.0,
            atol=atol,
        )


def test_guard_aligned_coordinate_transform_and_vector_field_are_conjugate():
    physical = GarciaPassiveWalker()
    aligned = GuardAlignedGarciaPassiveWalker()
    rng = np.random.default_rng(60218501)
    bounds = np.asarray(physical.domain_bounds, dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(GUARD_ALIGNED_DOMAIN_BOUNDS),
        np.asarray([[-0.32, 0.32], [-0.35, 0.05], [-1.29, 1.29], [-0.65, 0.85]]),
        rtol=0.0,
        atol=1e-15,
    )
    for state in rng.uniform(bounds[:, 0], bounds[:, 1], size=(128, 4)):
        encoded = encode_guard_aligned_state(state)
        np.testing.assert_allclose(
            decode_guard_aligned_state(encoded),
            state,
            rtol=0.0,
            atol=2e-16,
        )
        np.testing.assert_allclose(
            guard_aligned_ode_fun(0.37, encoded, aligned),
            encode_guard_aligned_velocity(ode_fun(0.37, state, physical)),
            rtol=0.0,
            atol=5e-16,
        )
        assert aligned.event_function(0.0, encoded) == physical.event_function(
            0.0,
            state,
        )


def test_guard_aligned_reset_formula_matches_physical_reset_on_full_guard_patch():
    physical = GarciaPassiveWalker()
    aligned = GuardAlignedGarciaPassiveWalker()
    for theta in np.linspace(-0.32, -physical.guard_delta, 12):
        for omega in np.linspace(-0.35, 0.05, 9):
            for nu in np.linspace(physical.transversality_eta, 0.85, 7):
                guard = np.asarray([theta, omega, 0.0, nu], dtype=np.float64)
                assert aligned.is_admissible_heelstrike(guard)
                expected = encode_guard_aligned_state(
                    reset_map(decode_guard_aligned_state(guard))
                )
                actual = guard_aligned_reset_map(guard)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
                assert actual[2] == 0.0
                assert actual[0] >= physical.guard_delta


def test_guard_aligned_period_two_strides_and_fixed_time_endpoints_match_physical():
    physical = GarciaPassiveWalker(max_jumps=6)
    aligned = GuardAlignedGarciaPassiveWalker(max_jumps=6)
    initial = post_impact_state(*PERIOD_TWO_POINT_A)
    aligned_initial = guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A)

    physical_trajectory = physical.system.simulate(
        initial,
        (0.0, 10.5),
        max_jumps=3,
        dense_output=True,
        max_step=0.005,
    )
    aligned_trajectory = aligned.system.simulate(
        aligned_initial,
        (0.0, 10.5),
        max_jumps=3,
        dense_output=True,
        max_step=0.005,
    )
    assert len(physical_trajectory.jump_states) >= 2
    assert len(aligned_trajectory.jump_states) >= 2
    np.testing.assert_allclose(
        aligned_trajectory.jump_times[:2],
        physical_trajectory.jump_times[:2],
        rtol=0.0,
        atol=2e-10,
    )
    for physical_jump, aligned_jump in zip(
        physical_trajectory.jump_states[:2],
        aligned_trajectory.jump_states[:2],
    ):
        for physical_state, aligned_state in zip(physical_jump, aligned_jump):
            np.testing.assert_allclose(
                aligned_state,
                encode_guard_aligned_state(physical_state),
                rtol=0.0,
                atol=2e-9,
            )

    first_flow_time = float(physical_trajectory.jump_times[0])
    for total_time in (0.5, first_flow_time + 0.5, first_flow_time + 1.25):
        physical_endpoint = simulate_suspension_endpoint(
            physical.system,
            initial,
            total_time,
            max_jumps=3,
            max_step=0.005,
        )
        aligned_endpoint = simulate_suspension_endpoint(
            aligned.system,
            aligned_initial,
            total_time,
            max_jumps=3,
            max_step=0.005,
        )
        _assert_coordinate_conjugate_endpoint(physical_endpoint, aligned_endpoint)


def test_garcia_reset_and_closed_transverse_guard_conventions():
    walker = GarciaPassiveWalker()
    post_a = post_impact_state(*PERIOD_TWO_POINT_A)
    assert heelstrike_gap(post_a) == 0.0
    assert post_a[0] > 0.0  # outgoing/scuffing root is excluded by theta margin

    trajectory = walker.system.simulate(
        post_a,
        (0.0, 5.0),
        max_jumps=1,
        dense_output=True,
        max_step=0.01,
    )
    guard, reset = trajectory.jump_states[0]
    assert walker.is_admissible_heelstrike(guard)
    assert abs(heelstrike_gap(guard)) < 1e-8
    assert heelstrike_transversality(guard) > 0.4
    np.testing.assert_allclose(reset, reset_map(guard), atol=1e-12)
    np.testing.assert_allclose(reset[:2], PERIOD_TWO_POINT_B, atol=1e-7)


def test_period_two_gait_and_failure_compactification_are_separately_labeled():
    result = build_garcia_passive_walker_suspension_pipeline()

    assert result.diagnostics["first_postimpact_residual_to_b"] < 1e-7
    assert result.diagnostics["second_postimpact_residual_to_a"] < 1e-7
    assert result.diagnostics["first_guard_transversality"] > 0.4
    assert result.diagnostics["second_guard_transversality"] > 0.4
    assert tuple(result.diagnostics["raw_failure_labels"]) == RAW_FAILURE_LABELS
    assert isinstance(result.witnesses[1].endpoint, HandleSuspensionSample)

    cemetery = CemeteryCell("garcia_failure_cemetery")
    assert result.relation[cemetery] == frozenset({cemetery})
    assert frozenset({cemetery}) in result.recurrent_sccs
    gait_components = [
        component
        for component in result.recurrent_sccs
        if cemetery not in component
    ]
    assert len(gait_components) == 1
    assert len(result.suspension_complex_ingredients.handles) == 2
    assert len(result.phase_descriptors) == 1
    assert result.descriptor_expansion_matches_relation
    assert len(result.scc_reconstruction.components) == 8
    assert result.suspension_complex.betti_numbers(modulus=5) == (1, 1)
    assert result.cmgdb_payload.cell_counts == (18, 18)
    assert "not generated by ODE" in result.diagnostics["failure_cell_status"]
