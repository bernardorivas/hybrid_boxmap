"""Regression tests: an initial state jumps only if it lies on the guard.

The legacy simulator reset every initial state with a nonnegative event value
(for ``direction = +1``).  For the rimless wheel this reset the states
``(alpha + gamma, omega)`` with ``omega < 0``, which are not on the guard
``G = {theta = alpha + gamma, omega >= 0}``.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_dynamics import HybridSystem
from hybrid_dynamics.examples.bouncing_ball import BouncingBall
from hybrid_dynamics.examples.paper_examples import rimless_wheel_problem
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.examples.spiking_neuron import SpikingNeuron, V_PEAK, V_RESET
from hybrid_dynamics.examples.thermostat import Thermostat
from hybrid_dynamics.src.sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
    simulate_suspension_endpoint,
)

WIDE = [(-1.0e6, 1.0e6), (-1.0e6, 1.0e6)]


def _starts_with_jump(system: HybridSystem, state) -> bool:
    """Simulate briefly; report whether the first jump happens at time zero."""

    trajectory = system.simulate(np.asarray(state, dtype=np.float64), (0.0, 0.05))
    return bool(trajectory.jump_times) and trajectory.jump_times[0] == 0.0


@pytest.mark.parametrize(
    "wheel",
    [RimlessWheel(), RimlessWheel(domain_bounds=WIDE)],
    ids=["default-domain", "wide-domain"],
)
def test_wheel_event_surface_points_jump_only_with_nonnegative_speed(wheel):
    # The window edge 0.6 and the exact guard angle alpha + gamma differ by
    # one unit in the last place; the default domain contains only the first.
    angles = [0.6] if wheel.domain_bounds[0][1] == 0.6 else [0.6, wheel.alpha + wheel.gamma]
    for theta in angles:
        assert not wheel.system.jumps_at_start(0.0, np.array([theta, -0.3]))
        assert not _starts_with_jump(wheel.system, [theta, -0.3])
        trajectory = wheel.system.simulate(np.array([theta, -0.3]), (0.0, 0.1))
        assert trajectory.num_jumps == 0
        assert trajectory.interpolate(0.1)[0] < 0.6
        for omega in (0.3, 0.0):
            assert wheel.system.jumps_at_start(0.0, np.array([theta, omega]))
            trajectory = wheel.system.simulate(np.array([theta, omega]), (0.0, 0.05))
            assert trajectory.jump_times[0] == 0.0
            guard_state, reset_state = trajectory.jump_states[0]
            assert np.allclose(guard_state, [theta, omega])
            assert np.allclose(
                reset_state,
                [wheel.gamma - wheel.alpha, np.cos(2.0 * wheel.alpha) * omega],
            )


def test_wheel_on_guard_matches_the_guard_of_the_model():
    system = RimlessWheel(domain_bounds=WIDE).system
    guard_angle = 0.4 + 0.2
    assert system.on_guard(0.0, np.array([guard_angle, 0.5]))
    assert system.on_guard(0.0, np.array([guard_angle, 0.0]))
    assert not system.on_guard(0.0, np.array([guard_angle, -1.0e-3]))
    assert not system.on_guard(0.0, np.array([guard_angle - 1.0e-3, 0.5]))


def test_wheel_sampled_suspension_does_not_reset_off_guard_points():
    # The legacy relation of the old Atlas inherited the reset of these points.
    problem = rimless_wheel_problem()
    guard_angle = 0.4 + 0.2
    sample = simulate_suspension_endpoint(
        problem.system, [guard_angle, -0.3], 0.1, max_step=0.02
    )
    assert isinstance(sample, BaseSuspensionSample)
    assert sample.jumps_completed == 0 and sample.state[0] < guard_angle
    sample = simulate_suspension_endpoint(
        problem.system, [guard_angle, 0.3], 0.1, max_step=0.02
    )
    assert isinstance(sample, HandleSuspensionSample)
    assert np.isclose(sample.phase, 0.1)


def test_ball_post_reset_points_flow_and_guard_points_jump():
    ball = BouncingBall()
    # (0, v) with v > 0 is a post-reset state, not a guard point.
    for v in (2.0, 1.0e-9):
        assert not ball.system.jumps_at_start(0.0, np.array([0.0, v]))
        assert not _starts_with_jump(ball.system, [0.0, v])
    trajectory = ball.system.simulate(np.array([0.0, 2.0]), (0.0, 0.1))
    assert trajectory.num_jumps == 0 and trajectory.interpolate(0.1)[0] > 0.0
    # (0, v) with v <= 0 lies on the guard G = {h = 0, v <= 0}.
    for v in (-2.0, 0.0):
        assert ball.system.on_guard(0.0, np.array([0.0, v]))
        assert _starts_with_jump(ball.system, [0.0, v])
        trajectory = ball.system.simulate(np.array([0.0, v]), (0.0, 0.05))
        assert np.allclose(trajectory.jump_states[0][1], [0.0, -ball.c * v])


def test_neuron_guard_points_jump_and_subthreshold_points_flow():
    neuron = SpikingNeuron()
    for u in (-250.0, 0.0, 150.0):
        assert neuron.system.on_guard(0.0, np.array([V_PEAK, u]))
        trajectory = neuron.system.simulate(np.array([V_PEAK, u]), (0.0, 0.05))
        assert trajectory.jump_times[0] == 0.0
        assert np.allclose(trajectory.jump_states[0][1], [V_RESET, u + 100.0])
        assert not neuron.system.jumps_at_start(0.0, np.array([V_PEAK - 1.0, u]))
        assert not _starts_with_jump(neuron.system, [V_PEAK - 1.0, u])


def test_thermostat_jump_set_includes_its_boundary():
    thermostat = Thermostat()
    # q = 0 switches on in {z <= z_min}; q = 1 switches off in {z >= z_max}.
    for state in ([70.0, 0.0], [70.0 - 1.0e-9, 0.0], [65.0, 0.0], [80.0, 1.0], [85.0, 1.0]):
        assert thermostat.system.jumps_at_start(0.0, np.array(state)), state
    for state in ([75.0, 0.0], [75.0, 1.0]):
        assert not thermostat.system.jumps_at_start(0.0, np.array(state)), state
    # Before the fix a state just inside the boundary never switched on.
    trajectory = thermostat.system.simulate(np.array([70.0 - 1.0e-9, 0.0]), (0.0, 0.05))
    assert trajectory.jump_times[0] == 0.0 and trajectory.jump_states[0][1][1] == 1.0


def test_explicit_guard_predicate_decides_initial_jumps():
    # The flow moves away from the event surface x = 1 against the event
    # direction, so the event rule alone never jumps there; the predicate
    # declares the half y >= 0 of the surface a guard.
    def ode(_t, state):
        return np.array([-1.0, 0.0])

    def event(_t, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1

    def reset(state):
        return np.array([0.0, float(state[1])])

    def guard(state):
        return bool(abs(state[0] - 1.0) <= 1.0e-12 and state[1] >= 0.0)

    plain = HybridSystem(ode, event, reset, domain_bounds=None, event_direction=1)
    explicit = HybridSystem(
        ode, event, reset, domain_bounds=None, event_direction=1, guard_predicate=guard
    )
    assert not plain.jumps_at_start(0.0, np.array([1.0, 0.5]))
    assert explicit.jumps_at_start(0.0, np.array([1.0, 0.5]))
    assert explicit.on_guard(0.0, np.array([1.0, 0.5]))
    assert not explicit.jumps_at_start(0.0, np.array([1.0, -0.5]))
    trajectory = explicit.simulate(np.array([1.0, 0.5]), (0.0, 0.1))
    assert trajectory.jump_times[0] == 0.0
    assert np.allclose(trajectory.interpolate(0.1), [-0.1, 0.5])
    trajectory = explicit.simulate(np.array([1.0, -0.5]), (0.0, 0.1))
    assert trajectory.num_jumps == 0
    assert np.allclose(trajectory.interpolate(0.1), [0.9, -0.5])


def test_failed_initial_reset_stops_the_trajectory():
    # A failed reset at time zero ends the trajectory, as a failed reset during
    # integration does, instead of recording a flow segment without its jump.
    def ode(_t, state):
        return np.array([1.0, 0.0])

    def event(_t, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1

    def reset(state):
        raise ValueError("reset rejected")

    system = HybridSystem(ode, event, reset, domain_bounds=None, event_direction=1)
    with pytest.warns(RuntimeWarning, match="Initial reset map failed"):
        trajectory = system.simulate(np.array([1.0, 0.0]), (0.0, 0.1))
    assert trajectory.num_jumps == 0
    assert len(trajectory.segments) == 1
    assert trajectory.total_duration == 0.0
