"""Focused tests for unit-handle suspension sampling and graph collapse."""

import unittest

import networkx as nx
import numpy as np

from hybrid_dynamics import (
    BaseSuspensionSample,
    BaseCell,
    Grid,
    HandleSuspensionSample,
    HybridSystem,
    PhaseOnlyRecurrentComponentError,
    PhaseCell,
    assert_recurrent_morse_equivalence,
    build_augmented_outer_graph,
    collapse_phase_paths,
    crossing_completed_state,
    diagnose_recurrent_morse_collapse,
    locate_augmented_cells,
    simulate_suspension_endpoint,
)


def unit_speed_reset_system() -> HybridSystem:
    """Flow from zero to one, then reset to zero and repeat."""

    def ode(_time, state):
        return np.ones_like(state)

    def event(_time, state):
        return float(state[0] - 1.0)

    def reset(_state):
        return np.array([0.0])

    return HybridSystem(
        ode=ode,
        event_function=event,
        reset_map=reset,
        domain_bounds=[(-0.1, 1.1)],
        max_jumps=20,
        event_direction=1,
    )


class SuspensionClockTests(unittest.TestCase):
    def setUp(self):
        self.system = unit_speed_reset_system()
        self.initial = np.array([0.0])

    def assert_state(self, sample, expected):
        np.testing.assert_allclose(sample.state, [expected], atol=1e-8)

    def test_base_flow_before_first_handle(self):
        sample = simulate_suspension_endpoint(self.system, self.initial, 0.5)
        self.assertIsInstance(sample, BaseSuspensionSample)
        self.assert_state(sample, 0.5)
        self.assertEqual(sample.jumps_completed, 0)

    def test_guard_and_reset_are_base_quotient_endpoints(self):
        guard = simulate_suspension_endpoint(self.system, self.initial, 1.0)
        reset = simulate_suspension_endpoint(self.system, self.initial, 2.0)

        self.assertIsInstance(guard, BaseSuspensionSample)
        self.assertIsInstance(reset, BaseSuspensionSample)
        self.assert_state(guard, 1.0)
        self.assert_state(reset, 0.0)
        self.assertEqual(guard.jumps_completed, 0)
        self.assertEqual(reset.jumps_completed, 1)

    def test_handle_has_unit_duration_and_crossing_completion(self):
        sample = simulate_suspension_endpoint(self.system, self.initial, 1.4)

        self.assertIsInstance(sample, HandleSuspensionSample)
        np.testing.assert_allclose(sample.guard_state, [1.0], atol=1e-8)
        np.testing.assert_allclose(sample.reset_state, [0.0], atol=1e-8)
        self.assertAlmostEqual(sample.phase, 0.4, places=8)
        np.testing.assert_allclose(crossing_completed_state(sample), [0.0])

    def test_second_flow_and_second_handle_use_total_clock(self):
        base = simulate_suspension_endpoint(self.system, self.initial, 2.25)
        handle = simulate_suspension_endpoint(self.system, self.initial, 3.4)
        second_reset = simulate_suspension_endpoint(self.system, self.initial, 4.0)

        self.assertIsInstance(base, BaseSuspensionSample)
        self.assert_state(base, 0.25)
        self.assertEqual(base.jumps_completed, 1)
        self.assertIsInstance(handle, HandleSuspensionSample)
        self.assertEqual(handle.jump_index, 1)
        self.assertAlmostEqual(handle.phase, 0.4, places=8)
        self.assert_state(second_reset, 0.0)
        self.assertEqual(second_reset.jumps_completed, 2)

    def test_initial_guard_starts_a_handle_at_zero(self):
        sample = simulate_suspension_endpoint(
            self.system,
            np.array([1.0]),
            0.5,
        )
        self.assertIsInstance(sample, HandleSuspensionSample)
        self.assertAlmostEqual(sample.phase, 0.5, places=8)
        np.testing.assert_allclose(sample.guard_state, [1.0])
        np.testing.assert_allclose(sample.reset_state, [0.0])

    def test_subunit_sampling_stops_after_the_first_zero_time_reset(self):
        reset_calls = 0

        def ode(_time, state):
            return np.zeros_like(state)

        def event(_time, _state):
            return 0.0

        def reset(state):
            nonlocal reset_calls
            reset_calls += 1
            return state.copy()

        zeno = HybridSystem(
            ode=ode,
            event_function=event,
            reset_map=reset,
            domain_bounds=[(-1.0, 1.0)],
            max_jumps=100,
            event_direction=-1,
        )
        sample = simulate_suspension_endpoint(zeno, np.array([0.0]), 0.5)

        self.assertIsInstance(sample, HandleSuspensionSample)
        self.assertEqual(reset_calls, 1)

    def test_unit_handle_budget_caps_instantaneous_reset_integration(self):
        recorded_limits = []
        original_simulate = self.system.simulate

        def recording_simulate(initial_state, time_span, **kwargs):
            recorded_limits.append(kwargs["max_jumps"])
            return original_simulate(initial_state, time_span, **kwargs)

        self.system.simulate = recording_simulate
        horizons = (0.5, 1.0, 1.0 + 0.5e-9, 1.0 + 2.0e-9, 2.0, 3.4)
        for horizon in horizons:
            simulate_suspension_endpoint(self.system, self.initial, horizon)

        self.assertEqual(recorded_limits, [0, 0, 0, 1, 1, 3])

    def test_exact_integer_horizon_handles_zero_flow_reset_chain(self):
        reset_calls = 0

        def ode(_time, state):
            return np.zeros_like(state)

        def event(_time, _state):
            return 0.0

        def reset(state):
            nonlocal reset_calls
            reset_calls += 1
            return state.copy()

        zero_flow = HybridSystem(
            ode=ode,
            event_function=event,
            reset_map=reset,
            domain_bounds=[(-1.0, 1.0)],
            max_jumps=100,
            event_direction=-1,
        )
        sample = simulate_suspension_endpoint(
            zero_flow,
            np.array([0.0]),
            3.0,
        )

        self.assertIsInstance(sample, BaseSuspensionSample)
        self.assertEqual(sample.jumps_completed, 3)
        self.assertEqual(reset_calls, 3)

    def test_truncated_simulation_is_not_replaced_by_last_state(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            simulate_suspension_endpoint(
                self.system,
                self.initial,
                2.25,
                max_jumps=0,
            )

    def test_locator_keeps_base_and_handle_cells_distinct(self):
        grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[2])
        base = BaseSuspensionSample(
            state=np.array([0.5]),
            total_time=0.0,
            continuous_time=0.0,
            jumps_completed=0,
        )
        handle = HandleSuspensionSample(
            guard_state=np.array([0.25]),
            reset_state=np.array([0.75]),
            phase=0.5,
            total_time=1.5,
            continuous_time=1.0,
            jump_index=0,
        )

        self.assertEqual(
            locate_augmented_cells(base, grid, handle_slabs=4),
            frozenset({BaseCell(0), BaseCell(1)}),
        )
        self.assertEqual(
            locate_augmented_cells(handle, grid, handle_slabs=4),
            frozenset({PhaseCell(0, 1), PhaseCell(0, 2)}),
        )


class PhasePathCollapseTests(unittest.TestCase):
    def test_outer_graph_keeps_isolated_cells_and_rejects_unknown_outputs(self):
        a = BaseCell(0)
        b = BaseCell(1)
        phase = PhaseCell(0, 0)
        cells = [a, b, phase]

        graph = build_augmented_outer_graph(
            cells,
            lambda cell: {phase} if cell == a else ({b} if cell == phase else set()),
        )
        self.assertEqual(set(graph.nodes), set(cells))
        self.assertEqual(set(graph.edges), {(a, phase), (phase, b)})
        self.assertEqual(graph.out_degree(b), 0)

        with self.assertRaisesRegex(ValueError, "undeclared"):
            build_augmented_outer_graph(cells, lambda _cell: {BaseCell(99)})

    def test_collapse_bypasses_only_phase_nodes(self):
        graph = nx.DiGraph(
            [
                ("a", "h1"),
                ("h1", "h2"),
                ("h2", "b"),
                ("b", "h3"),
                ("h3", "c"),
            ],
        )

        collapsed = collapse_phase_paths(graph, {"h1", "h2", "h3"})

        self.assertEqual(set(collapsed.nodes), {"a", "b", "c"})
        self.assertEqual(set(collapsed.edges), {("a", "b"), ("b", "c")})
        self.assertNotIn(("a", "c"), collapsed.edges)

    def test_mixed_base_phase_recurrence_is_preserved(self):
        graph = nx.DiGraph([("a", "h"), ("h", "a")])
        diagnostic = assert_recurrent_morse_equivalence(graph, {"h"})

        self.assertTrue(diagnostic.equivalent)
        self.assertEqual(
            set(diagnostic.collapsed_recurrent_components),
            {frozenset({"a"})},
        )

    def test_phase_only_recurrence_fails_diagnostic(self):
        graph = nx.DiGraph(
            [("a", "h"), ("h", "h"), ("h", "b"), ("b", "b")],
        )
        diagnostic = diagnose_recurrent_morse_collapse(graph, {"h"})

        self.assertFalse(diagnostic.equivalent)
        self.assertEqual(
            diagnostic.phase_only_recurrent_components,
            (frozenset({"h"}),),
        )
        with self.assertRaises(PhaseOnlyRecurrentComponentError):
            assert_recurrent_morse_equivalence(graph, {"h"})

    def test_diagnostic_compares_recurrent_reachability_order(self):
        graph = nx.DiGraph(
            [("a", "a"), ("a", "h"), ("h", "b"), ("b", "b")],
        )
        diagnostic = assert_recurrent_morse_equivalence(graph, {"h"})
        expected_order = frozenset(
            {(frozenset({"a"}), frozenset({"b"}))},
        )

        self.assertTrue(diagnostic.reachability_order_equivalent)
        self.assertEqual(diagnostic.full_recurrent_order, expected_order)
        self.assertEqual(diagnostic.collapsed_recurrent_order, expected_order)


if __name__ == "__main__":
    unittest.main()
