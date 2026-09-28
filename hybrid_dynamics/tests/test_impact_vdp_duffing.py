"""Tests for the impacting van der Pol-Duffing oscillator.

The reference values (equilibria, impact cycles, Lienard identity) are the
numerically verified values of the default example with ``eps = 1``,
``beta = 0.8``, ``w = 0.8``, ``c = 0.7``; the last test checks the variant
``impact-vdp-duffing-beta076`` against the values verified at ``beta = 0.76``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hybrid_dynamics import SuspensionFlow, build_suspension_grid, check_suspension_grid
from hybrid_dynamics.examples import ImpactVanDerPolDuffing
from hybrid_dynamics.examples.impact_vdp_duffing import energy, potential
from hybrid_dynamics.examples.paper_examples import (
    PAPER_PROBLEMS,
    PAPER_REFERENCE_SETS,
    impact_vdp_duffing_problem,
    impact_vdp_duffing_reference_sets,
    paper_problem,
    paper_problem_factory,
)
from hybrid_dynamics.src.suspension_grid_relation import ENDPOINT_BASE, ENDPOINT_HANDLE


@pytest.fixture(scope="module")
def oscillator():
    return ImpactVanDerPolDuffing()


def test_vector_field_and_equilibria(oscillator):
    rng = np.random.default_rng(0)
    for x, v in rng.uniform(-2.0, 0.8, size=(10, 2)):
        expected = [v, x - x**3 + 1.0 * (0.8 - x * x) * v]
        assert np.allclose(oscillator.system.ode(0.0, np.array([x, v])), expected)
    equilibria = oscillator.equilibria()
    focus, focus_eigenvalues = equilibria["F"]
    saddle, saddle_eigenvalues = equilibria["S"]
    assert np.allclose(oscillator.system.ode(0.0, focus), 0.0)
    assert np.allclose(oscillator.system.ode(0.0, saddle), 0.0)
    assert np.allclose(np.sort_complex(focus_eigenvalues), [-0.1 - 1.4106736j, -0.1 + 1.4106736j])
    assert np.allclose(np.sort(saddle_eigenvalues.real), [-0.6770330, 1.4770330])
    # The force at the stop points into the stop.
    assert np.isclose(oscillator.system.ode(0.0, np.array([0.8, 0.0]))[1], 0.288)


def test_guard_is_the_stop_with_nonnegative_velocity(oscillator):
    system = oscillator.system
    assert system.event_crossing_direction == 1
    for v in (0.0, 0.3, 1.9):
        assert system.on_guard(0.0, np.array([0.8, v]))
        assert system.jumps_at_start(0.0, np.array([0.8, v]))
    # Post-impact states (w, -c v) and interior points flow.
    for state in ([0.8, -0.3], [0.8, -1e-3], [0.0, 0.0], [0.79, 0.5]):
        assert not system.jumps_at_start(0.0, np.array(state))
    # A trajectory from the interior jumps exactly at the stop.
    unbounded = ImpactVanDerPolDuffing(domain_bounds=[(-1e6, 1e6), (-1e6, 1e6)])
    trajectory = unbounded.system.simulate(np.array([0.5, 0.5]), (0.0, 1.0), max_jumps=1)
    assert trajectory.num_jumps == 1
    jump_time = trajectory.jump_times[0]
    before = trajectory.segments[0].state_values[-1]
    after = trajectory.segments[1].state_values[0]
    assert 0.0 < jump_time < 1.0
    assert np.isclose(before[0], 0.8, atol=1e-9) and before[1] > 0.0
    assert np.allclose(after, [0.8, -0.7 * before[1]])


def test_reset_map(oscillator):
    for v in (0.0, 0.5, 1.95):
        assert np.allclose(oscillator.system.reset_map(np.array([0.8, v])), [0.8, -0.7 * v])
    problem = impact_vdp_duffing_problem()
    u = np.linspace(*problem.guard.u_bounds, 9)
    assert np.allclose(problem.guard.gamma(u), np.stack((np.full(9, 0.8), u), axis=1))
    assert np.allclose(problem.guard.reset(u), np.stack((np.full(9, 0.8), -0.7 * u), axis=1))
    # r(G cap R) lies in R.
    (x_low, x_high), (v_low, v_high) = problem.window.ambient_bounds
    reset = problem.guard.reset(u)
    assert np.all((reset[:, 0] <= x_high) & (reset[:, 1] >= v_low) & (reset[:, 1] <= v_high))


def test_lienard_identity(oscillator):
    rng = np.random.default_rng(1)
    states = np.column_stack((rng.uniform(-1.9, 0.8, 50), rng.uniform(-2.3, 1.9, 50)))
    # dL/dt = grad L . f = -eps Gf(x) V'(x), pointwise.
    step = 1e-6
    for state in states:
        field = oscillator.system.ode(0.0, state)
        derivative = (
            oscillator.lienard_function(state + step * field)
            - oscillator.lienard_function(state - step * field)
        ) / (2 * step)
        assert np.isclose(derivative, oscillator.lienard_rate(state[0]), atol=1e-6)
    # Along an orbit without impacts, L(t) - L(0) = int_0^t dL/dt.
    solution = solve_ivp(
        lambda _t, y: np.append(oscillator.system.ode(0.0, y[:2]), oscillator.lienard_rate(y[0])),
        (0.0, 3.0),
        [-1.2, 0.4, 0.0],
        rtol=1e-11,
        atol=1e-12,
    )
    final = solution.y[:, -1]
    assert np.max(solution.y[0]) < 0.8
    assert np.isclose(
        oscillator.lienard_function(final[:2]) - oscillator.lienard_function([-1.2, 0.4]),
        final[2],
        atol=1e-8,
    )
    # At the stop L = E, and an impact with speed v lowers it by (1 - c^2) v^2 / 2.
    for v in (0.2, 1.0, 1.7):
        guard_point = np.array([0.8, v])
        assert np.isclose(oscillator.lienard_function(guard_point), energy(guard_point))
        reset = oscillator.system.reset_map(guard_point)
        loss = oscillator.lienard_function(guard_point) - oscillator.lienard_function(reset)
        assert np.isclose(loss, oscillator.impact_loss(v))
    assert np.isclose(potential(-1.0), -0.25)


def test_impact_cycles(oscillator):
    attracting = oscillator.impact_cycle((1.5, 1.9))
    repelling = oscillator.impact_cycle((0.55, 0.66))
    assert abs(attracting - 1.6905562) < 1e-6
    assert abs(repelling - 0.6208138) < 1e-6
    _speed, flight, _solution = oscillator.impact_return(attracting)
    assert abs(flight + 1.0 - 5.496083) < 1e-5
    _speed, flight, _solution = oscillator.impact_return(repelling)
    assert abs(flight + 1.0 - 3.916957) < 1e-5
    # Multipliers by central differences of the impact map.
    for speed, multiplier in ((attracting, 0.244774), (repelling, 2.024450)):
        h = 1e-5
        slope = (
            oscillator.impact_return(speed + h)[0] - oscillator.impact_return(speed - h)[0]
        ) / (2 * h)
        assert abs(slope - multiplier) < 1e-4


def test_zeno_point_circulates_through_the_handle():
    problem = impact_vdp_duffing_problem()
    flow = SuspensionFlow(problem.system, max_step=0.02)
    path = flow.path(np.array([0.8, 0.0]), 2.5, start_on_handle=True)
    kind, state, phase = path.evaluate(np.array([0.25, 1.5, 2.0]))
    assert kind.tolist() == [ENDPOINT_HANDLE, ENDPOINT_HANDLE, ENDPOINT_BASE]
    assert np.allclose(phase[:2], [0.25, 0.5])
    assert np.allclose(state, [0.8, 0.0])
    # A post-impact state flows back to the stop and enters the handle again.
    # The flight from (w, -0.1) lasts about 2 * 0.1 / 0.288 = 0.69.
    kind, state, phase = flow.path(np.array([0.8, -0.1]), 1.0).evaluate(1.0)
    assert kind[0] == ENDPOINT_HANDLE and np.isclose(state[0, 0], 0.8)
    assert 0.0 < state[0, 1] < 0.11 and 0.0 < phase[0] < 0.5


def test_paper_problem_and_reference_sets():
    problem = impact_vdp_duffing_problem()
    assert problem.window.ambient_bounds == ((-1.95, 0.8), (-2.35, 1.95))
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    assert check_suspension_grid(grid).passed
    sets = impact_vdp_duffing_reference_sets(problem, samples=2000)
    assert set(sets) == {"F", "S", "Z", "C", "U_Z"}
    cycle = sets["C"]["base"]
    assert np.isclose(cycle[:, 0].min(), -1.742871, atol=1e-3)
    assert np.isclose(cycle[:, 0].max(), 0.8)
    inner = sets["U_Z"]["base"]
    assert np.isclose(inner[:, 0].min(), 0.423591, atol=1e-3)


def test_beta076_variant():
    name = "impact-vdp-duffing-beta076"
    # A named variant, run only when named; the example keeps beta = 0.8.
    assert name not in PAPER_PROBLEMS
    assert impact_vdp_duffing_problem().parameters["beta"] == 0.8
    example = paper_problem("impact-vdp-duffing", tau=1.0, level_offset=4)
    variant = paper_problem(name, tau=1.0, level_offset=4)
    assert (example.name, variant.name) == ("impact-vdp-duffing", name)
    assert (example.parameters["beta"], variant.parameters["beta"]) == (0.8, 0.76)
    assert {key: value for key, value in variant.parameters.items() if key != "beta"} == {
        key: value for key, value in example.parameters.items() if key != "beta"
    }
    # The same window R and guard, so the same grid.
    assert variant.window.ambient_bounds == ((-1.95, 0.8), (-2.35, 1.95))
    assert variant.window.cells_per_axis(6) == 1024
    u = np.linspace(*variant.guard.u_bounds, 5)
    assert np.allclose(variant.guard.reset(u), example.guard.reset(u))
    state = np.array([0.3, 0.5])
    difference = variant.system.ode(0.0, state) - example.system.ode(0.0, state)
    assert np.allclose(difference, [0.0, (0.76 - 0.8) * 0.5])
    with pytest.raises(TypeError, match="fixes beta"):
        paper_problem(name, beta=0.8)
    # Worker processes rebuild the variant from a pickled factory.
    rebuilt = pickle.loads(pickle.dumps(paper_problem_factory(name, tau=1.0)))()
    assert (rebuilt.name, rebuilt.parameters["beta"], rebuilt.tau) == (name, 0.76, 1.0)
    # The impact cycles at beta = 0.76: pre-impact speeds, extents, periods.
    sets = PAPER_REFERENCE_SETS[name](variant, samples=2000)
    assert set(sets) == {"F", "S", "Z", "C", "U_Z"}
    assert abs(sets["C"]["handle"][0, 0] - 1.5098383) < 1e-6
    assert abs(sets["U_Z"]["handle"][0, 0] - 0.6569620) < 1e-6
    assert np.isclose(sets["C"]["base"][:, 0].min(), -1.685907, atol=1e-4)
    assert np.isclose(sets["C"]["base"][:, 1].min(), -1.876882, atol=1e-4)
    assert np.isclose(sets["U_Z"]["base"][:, 0].min(), 0.376129, atol=1e-4)
    oscillator = ImpactVanDerPolDuffing(beta=0.76)
    for speed, period in ((1.5098383, 5.955741), (0.6569620, 4.167507)):
        assert abs(oscillator.impact_return(speed)[1] + 1.0 - period) < 1e-5
