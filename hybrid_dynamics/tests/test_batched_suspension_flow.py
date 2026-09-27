"""The batched suspension flow against the scalar vector fields and SuspensionFlow paths."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from hybrid_dynamics.examples.paper_grid_examples import paper_grid_problem
from hybrid_dynamics.src import suspension_grid_relation as relation_module
from hybrid_dynamics.src.batched_suspension_flow import batched_suspension_endpoints
from hybrid_dynamics.src.suspension_grid import build_suspension_grid
from hybrid_dynamics.src.suspension_grid_relation import (
    _evaluate_base_endpoints_by_path,
    _make_flow,
    compute_suspension_grid_relation,
    evaluate_base_endpoints,
)

#: The examples with the ``tau`` of the manuscript runs.
EXAMPLES = {
    "bouncing-ball": 0.5,
    "rimless-wheel": 0.5,
    "spiking-neuron": 5.0,
    "impact-vdp-duffing": 1.0,
    "impact-vdp-duffing-beta076": 1.0,
}

#: Points near which many resets happen in a short time: the Zeno points of
#: the ball and the oscillator, and a guard point of the wheel and the neuron.
BUSY_POINTS = {
    "bouncing-ball": (0.0, 0.0),
    "rimless-wheel": (0.6, 0.3),
    "spiking-neuron": (35.0, 0.0),
    "impact-vdp-duffing": (0.8, 0.0),
    "impact-vdp-duffing-beta076": (0.8, 0.0),
}


def _samples(name: str, count: int, seed: int) -> tuple[object, np.ndarray, np.ndarray]:
    """Lattice corners of the window, guard points, and points near ``BUSY_POINTS``."""

    problem = paper_grid_problem(name, tau=EXAMPLES[name], level_offset=4)
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = grid.window.ambient_bounds
    cells = 2 ** (grid.level + grid.window.level_offset)
    corners = np.stack(
        (
            x0 + rng.integers(0, cells + 1, count) * (x1 - x0) / cells,
            y0 + rng.integers(0, cells + 1, count) * (y1 - y0) / cells,
        ),
        axis=1,
    )
    corners = corners[grid.window.contains(corners, grid.level)]
    u = rng.uniform(*problem.guard.u_bounds, count // 4)
    on_guard = problem.guard.gamma(u)
    busy = np.asarray(BUSY_POINTS[name]) + rng.normal(scale=0.02, size=(count // 4, 2)) * [
        x1 - x0,
        y1 - y0,
    ]
    busy = busy[grid.window.contains(busy, grid.level)]
    points = np.concatenate((corners, on_guard, busy))
    return problem, grid, points


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_batch_dynamics_equal_the_scalar_functions(name):
    problem = paper_grid_problem(name)
    system = problem.system
    dynamics = problem.batch_dynamics
    rng = np.random.default_rng(1)
    (x0, x1), (y0, y1) = problem.window.ambient_bounds
    states = np.stack((rng.uniform(x0, x1, 500), rng.uniform(y0, y1, 500)), axis=1)
    states[:50, 1] = 0.0
    field = np.array([system.ode(0.0, state) for state in states])
    event = np.array([system.event_function(0.0, state) for state in states])
    reset = np.array([system.reset_map(state) for state in states])
    assert np.array_equal(dynamics.vector_field(states), field)
    assert np.array_equal(dynamics.event(states), event)
    assert np.array_equal(dynamics.reset(states), reset)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_batched_endpoints_equal_the_paths(name):
    problem, grid, points = _samples(name, 240, seed=2)
    flow = _make_flow(problem, grid.level)
    assert flow.batch is not None
    guard_u = grid.guard_membership(points)
    assert np.isfinite(guard_u).sum() >= 40
    batched = evaluate_base_endpoints(flow, points, guard_u, problem.tau)
    reference = _evaluate_base_endpoints_by_path(flow, points, guard_u, problem.tau)
    assert np.array_equal(batched.kind, reference.kind)
    assert np.array_equal(batched.left_window, reference.left_window)
    assert np.allclose(batched.state, reference.state, rtol=0.0, atol=1e-9, equal_nan=True)
    assert np.allclose(batched.phase, reference.phase, rtol=0.0, atol=1e-9, equal_nan=True)
    assert batched.failures == reference.failures
    # Every endpoint kind occurs, and some paths leave the window.
    assert set(np.unique(reference.kind).tolist()) >= {0, 1}


def test_several_handles_within_tau():
    # tau = 3.5 lets a ball near the Zeno point pass through several handles.
    problem = paper_grid_problem("bouncing-ball", tau=3.5)
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    rng = np.random.default_rng(3)
    points = np.stack((rng.uniform(0.0, 0.3, 200), rng.uniform(-2.0, 2.0, 200)), axis=1)
    points[:40, 0] = 0.0
    points[:40, 1] = -np.abs(points[:40, 1])
    flow = _make_flow(problem, grid.level)
    guard_u = grid.guard_membership(points)
    batched = evaluate_base_endpoints(flow, points, guard_u, problem.tau)
    reference = _evaluate_base_endpoints_by_path(flow, points, guard_u, problem.tau)
    assert np.array_equal(batched.kind, reference.kind)
    assert np.array_equal(batched.left_window, reference.left_window)
    assert np.allclose(batched.state, reference.state, rtol=0.0, atol=1e-9, equal_nan=True)
    assert np.allclose(batched.phase, reference.phase, rtol=0.0, atol=1e-9, equal_nan=True)


def test_unresolved_points_are_evaluated_by_the_path(monkeypatch):
    problem, grid, points = _samples("impact-vdp-duffing", 60, seed=4)
    flow = _make_flow(problem, grid.level)
    guard_u = grid.guard_membership(points)
    capped = batched_suspension_endpoints(
        flow.batch,
        points,
        np.isfinite(guard_u),
        problem.tau,
        direction=1.0,
        rtol=problem.system.rtol,
        atol=problem.system.atol,
        max_step=problem.max_step,
        in_window=flow.in_window,
        guard_in_window=flow.guard_in_window,
        max_iterations=3,
    )
    assert capped.unresolved.sum() > 0
    monkeypatch.setattr(
        relation_module,
        "batched_suspension_endpoints",
        functools.partial(batched_suspension_endpoints, max_iterations=3),
    )
    batched = evaluate_base_endpoints(flow, points, guard_u, problem.tau)
    reference = _evaluate_base_endpoints_by_path(flow, points, guard_u, problem.tau)
    assert np.array_equal(batched.kind, reference.kind)
    assert np.array_equal(batched.left_window, reference.left_window)
    assert np.allclose(batched.state, reference.state, rtol=0.0, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("name", ["rimless-wheel", "impact-vdp-duffing-beta076"])
def test_batched_relation_equals_the_path_relation(name):
    problem = paper_grid_problem(name, tau=EXAMPLES[name], level_offset=2)
    grid = build_suspension_grid(problem.window, problem.guard, 3)
    batched = compute_suspension_grid_relation(grid, problem, gap_refinement_depth=3)
    reference = compute_suspension_grid_relation(
        grid, problem, gap_refinement_depth=3, batched_flow=False
    )
    assert (batched.matrix != reference.matrix).nnz == 0
    assert (batched.sampled != reference.sampled).nnz == 0
    statistics = {k: v for k, v in batched.statistics.items() if not k.startswith("seconds")}
    expected = {k: v for k, v in reference.statistics.items() if not k.startswith("seconds")}
    assert statistics == expected
