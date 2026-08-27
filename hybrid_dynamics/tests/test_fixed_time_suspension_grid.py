"""Tests for the cellwise fixed-time suspension relation."""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

from hybrid_dynamics.src.fixed_time_suspension_grid import (
    SuspensionCemeteryCell,
    _descriptor_reconstruction,
    _grid_cells_in_rectangle,
    _handle_cover,
    _safe_endpoint,
    compute_fixed_time_suspension_grid,
)
from hybrid_dynamics.src.grid import Grid
from hybrid_dynamics.src.hybrid_system import HybridSystem
from hybrid_dynamics.src.implicit_phase_scc import (
    VirtualPhaseNode,
    reconstruct_from_base_descriptors,
)
from hybrid_dynamics.src.sampled_suspension import (
    BaseCell,
    HandleSuspensionSample,
    PhaseCell,
)
from hybrid_dynamics.examples.physical_suspension_grid import (
    build_rimless_wheel_suspension_grid,
)


def _unit_reset_system() -> HybridSystem:
    def ode(_time, _state):
        return np.asarray([1.0])

    def event(_time, state):
        return float(state[0] - 1.0)

    event.terminal = True
    event.direction = 1
    return HybridSystem(
        ode=ode,
        event_function=event,
        reset_map=lambda _state: np.asarray([0.0]),
        domain_bounds=[(0.0, 1.0)],
        event_direction=1,
    )


def _handle_sample(
    guard_state: float,
    *,
    phase: float,
    jump_index: int,
) -> HandleSuspensionSample:
    state = np.asarray([guard_state])
    return HandleSuspensionSample(
        guard_state=state,
        reset_state=state,
        phase=phase,
        total_time=0.1,
        continuous_time=0.0,
        jump_index=jump_index,
    )


def test_handle_endpoint_outside_declared_state_space_is_rejected():
    def ode(_time, _state):
        return np.asarray([1.0])

    def event(_time, state):
        return float(state[0] - 1.2)

    event.terminal = True
    event.direction = 1
    system = HybridSystem(
        ode=ode,
        event_function=event,
        reset_map=lambda _state: np.asarray([0.0]),
        domain_bounds=[(0.0, 1.0)],
        event_direction=1,
    )

    assert _safe_endpoint(
        system,
        [0.9],
        0.5,
        max_jumps=2,
        max_step=0.01,
    ) is None


def test_closed_rectangle_includes_all_cells_incident_to_grid_faces():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])

    assert set(
        _grid_cells_in_rectangle(
            grid,
            [0.5],
            [0.5],
            padding_cells=0.0,
        )
    ) == {1, 2}
    assert set(
        _grid_cells_in_rectangle(
            grid,
            [0.25],
            [0.5],
            padding_cells=0.0,
        )
    ) == {0, 1, 2}


def test_handle_cover_hulls_each_jump_branch_but_not_distinct_branches():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[8])
    guard_cells = frozenset(range(8))
    same_branch = _handle_cover(
        (
            _handle_sample(0.125, phase=0.2, jump_index=0),
            _handle_sample(0.875, phase=0.2, jump_index=0),
        ),
        grid,
        handle_slabs=10,
        padding_cells=0.0,
        guard_cells=guard_cells,
    )
    distinct_branches = _handle_cover(
        (
            _handle_sample(0.125, phase=0.2, jump_index=0),
            _handle_sample(0.875, phase=0.2, jump_index=1),
        ),
        grid,
        handle_slabs=10,
        padding_cells=0.0,
        guard_cells=guard_cells,
    )

    assert PhaseCell(4, 1) in same_branch
    assert PhaseCell(4, 2) in same_branch
    assert all(target.guard_cell != 4 for target in distinct_branches)


def test_phase_crossing_samples_reset_and_records_closed_quotient_incidence():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[10])

    def guard_sampler(lower, upper):
        return (np.asarray([1.0]),) if lower[0] <= 1.0 <= upper[0] else ()

    result = compute_fixed_time_suspension_grid(
        model_name="unit_reset_crossing",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=0.3,
        handle_slabs=2,
        padding_cells=0.0,
        max_step=0.02,
    )
    targets = result.relation[PhaseCell(9, 1)]

    # s=0.7 is inside this source slab.  Its image is the reset x=0; the
    # residual-flow endpoint x=0.3 lies on the same post-reset branch, so the
    # ordinary rectangular image hull also contains the intervening cell.
    assert BaseCell(0) in targets
    assert BaseCell(1) in targets
    # The reset point is the quotient image of the terminal handle face.
    assert PhaseCell(9, 1) in targets


def test_full_grid_relation_declares_base_phase_and_cemetery_cells():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])

    def guard_sampler(lower, upper):
        if lower[0] <= 1.0 <= upper[0]:
            return (np.asarray([1.0]),)
        return ()

    result = compute_fixed_time_suspension_grid(
        model_name="unit_reset",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=0.5,
        handle_slabs=4,
        padding_cells=0.0,
        max_step=0.02,
    )

    base_nodes = {node for node in result.relation if isinstance(node, BaseCell)}
    phase_nodes = {node for node in result.relation if isinstance(node, PhaseCell)}
    cemetery_nodes = {
        node for node in result.relation if isinstance(node, SuspensionCemeteryCell)
    }
    assert base_nodes == {BaseCell(index) for index in range(4)}
    assert phase_nodes == {PhaseCell(3, slab) for slab in range(4)}
    assert len(cemetery_nodes) == 1
    assert result.descriptor_expansion_matches_relation
    assert result.diagnostics["graph_parity_certificate"]["exact"] is True
    assert result.relation_graph.graph["whole_cell_outer_enclosure"] is False
    assert result.relation_graph.graph["base_source_event_strata_completed"] is False
    assert result.diagnostics["whole_cell_outer_enclosure"] is False
    assert result.diagnostics["base_source_event_strata_completed"] is False


def test_active_base_family_maps_inactive_targets_to_cemetery_without_leaks():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])
    guard_calls = 0

    def guard_sampler(lower, upper):
        nonlocal guard_calls
        guard_calls += 1
        return (np.asarray([1.0]),) if lower[0] <= 1.0 <= upper[0] else ()

    result = compute_fixed_time_suspension_grid(
        model_name="active_unit_reset",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        active_base_cells={0, 1},
        t_star=0.3,
        handle_slabs=2,
        padding_cells=0.0,
    )
    cemetery = next(
        node
        for node in result.relation
        if isinstance(node, SuspensionCemeteryCell)
    )

    assert guard_calls == 2
    assert set(result.relation) == {BaseCell(0), BaseCell(1), cemetery}
    assert BaseCell(1) in result.relation[BaseCell(0)]
    assert cemetery in result.relation[BaseCell(0)]
    assert not any(
        isinstance(node, BaseCell) and node.index in {2, 3}
        for node in result.relation_graph
    )
    assert not result.phase_cells
    assert result.diagnostics["ambient_grid_cells"] == 4
    assert result.diagnostics["active_base_cells"] == 2
    assert result.diagnostics["unique_corner_samples"] == 3
    assert result.diagnostics["ambient_unique_corner_samples"] == 5


def test_phase_cell_translation_uses_the_unit_handle_clock():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])

    def guard_sampler(lower, upper):
        return (np.asarray([1.0]),) if upper[0] == 1.0 else ()

    result = compute_fixed_time_suspension_grid(
        model_name="unit_reset",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=0.5,
        handle_slabs=4,
        padding_cells=0.0,
    )

    # The lower half of the handle advances by exactly two phase slabs.
    assert PhaseCell(3, 2) in result.relation[PhaseCell(3, 0)]
    # The upper half crosses the quotient endpoint and returns to base space.
    assert any(
        isinstance(target, BaseCell)
        for target in result.relation[PhaseCell(3, 2)]
    )
    # The crossing phase s=1-t*=1/2 is a shared face of source slabs 1 and 2.
    # Both closed source cells therefore see both quotient-chart incidences.
    for source in (PhaseCell(3, 1), PhaseCell(3, 2)):
        assert BaseCell(0) in result.relation[source]
        assert PhaseCell(3, 3) in result.relation[source]


def test_subunit_phase_relation_is_preserved_exactly():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])

    def guard_sampler(lower, upper):
        return (np.asarray([1.0]),) if lower[0] <= 1.0 <= upper[0] else ()

    result = compute_fixed_time_suspension_grid(
        model_name="subunit_regression",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=0.5,
        handle_slabs=4,
        padding_cells=0.0,
        max_step=0.02,
    )

    assert {
        source: result.relation[source]
        for source in sorted(result.phase_cells, key=lambda cell: cell.slab)
    } == {
        PhaseCell(3, 0): frozenset(
            {PhaseCell(3, 1), PhaseCell(3, 2), PhaseCell(3, 3)}
        ),
        PhaseCell(3, 1): frozenset(
            {BaseCell(0), PhaseCell(3, 2), PhaseCell(3, 3)}
        ),
        PhaseCell(3, 2): frozenset(
            {BaseCell(0), BaseCell(1), PhaseCell(3, 3)}
        ),
        PhaseCell(3, 3): frozenset(
            {BaseCell(0), BaseCell(1), BaseCell(2)}
        ),
    }
    assert result.diagnostics["phase_sampling_mode"] == (
        "legacy-subunit-single-crossing"
    )
    assert result.diagnostics["reset_trajectory_simulations"] == 0


def test_long_horizon_samples_chart_change_after_multiple_resets():
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[4])

    def guard_sampler(lower, upper):
        return (np.asarray([1.0]),) if lower[0] <= 1.0 <= upper[0] else ()

    result = compute_fixed_time_suspension_grid(
        model_name="multi_reset_event_driven",
        system=_unit_reset_system(),
        grid=grid,
        guard_sampler=guard_sampler,
        t_star=3.5,
        handle_slabs=4,
        padding_cells=0.0,
        max_jumps=20,
        max_step=0.02,
    )

    # From source phase s=1/2 the residual clock after the current handle is
    # three.  The unit-reset system has already traversed another complete
    # handle, and the endpoint is the entry face of the following handle.
    # Both source slabs incident to s=1/2 therefore contain its phase-zero cell.
    for source in (PhaseCell(3, 1), PhaseCell(3, 2)):
        assert BaseCell(3) in result.relation[source]
        assert PhaseCell(3, 0) in result.relation[source]

    assert result.diagnostics["phase_sampling_mode"] == (
        "event-driven-all-crossings"
    )
    assert result.diagnostics["reset_trajectory_simulations"] == 1
    assert result.diagnostics["phase_chart_boundaries"] == 2
    assert result.descriptor_expansion_matches_relation


def _component_edge_labels(reconstruction):
    return {
        (
            reconstruction.components[source],
            reconstruction.components[target],
        )
        for source, target in reconstruction.condensation.edges
    }


def test_linear_reconstruction_matches_reference_expansion_exactly():
    a, b, c = BaseCell(0), BaseCell(1), BaseCell(2)
    p0, p1 = PhaseCell(0, 0), PhaseCell(0, 1)
    transient_phase = PhaseCell(2, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from((a, b, c, p0, p1, transient_phase))
    graph.add_edges_from(
        (
            (a, p0),
            (p0, p1),
            (p1, p0),
            (p1, b),
            (b, a),
            (c, transient_phase),
            (transient_phase, transient_phase),
            (c, c),
        )
    )

    descriptors, direct, result, matches = _descriptor_reconstruction(
        "parity", graph
    )
    reference = reconstruct_from_base_descriptors(
        (a, b, c), descriptors, direct_base_edges=direct
    )

    assert matches
    assert set(result.virtual_graph.nodes) == set(reference.virtual_graph.nodes)
    assert set(result.virtual_graph.edges) == set(reference.virtual_graph.edges)
    assert set(result.components) == set(reference.components)
    assert _component_edge_labels(result) == _component_edge_labels(reference)
    assert (
        result.macro_edges_realized_by_phase
        == reference.macro_edges_realized_by_phase
    )
    assert result.retained_direct_base_edges == reference.retained_direct_base_edges

    certificate = result.virtual_graph.graph["graph_parity_certificate"]
    assert certificate == {
        "exact": True,
        "node_bijection": True,
        "node_set_equal": True,
        "edge_set_equal": True,
        "source_node_count": graph.number_of_nodes(),
        "expanded_node_count": graph.number_of_nodes(),
        "source_edge_count": graph.number_of_edges(),
        "expanded_edge_count": graph.number_of_edges(),
    }
    assert frozenset(
        {
            VirtualPhaseNode(descriptors[0].descriptor_id, transient_phase),
        }
    ) in result.phase_only_components


def test_reconstruction_does_not_issue_per_component_reachability_queries(
    monkeypatch,
):
    def forbidden_descendants(*_args, **_kwargs):
        raise AssertionError("linear reconstruction must not call nx.descendants")

    monkeypatch.setattr(nx, "descendants", forbidden_descendants)
    start, finish = BaseCell(0), BaseCell(1)
    phase_nodes = [PhaseCell(0, slab) for slab in range(4000)]
    graph = nx.DiGraph()
    graph.add_nodes_from((start, finish, *phase_nodes))
    graph.add_edge(start, phase_nodes[0])
    graph.add_edges_from(zip(phase_nodes, phase_nodes[1:]))
    graph.add_edge(phase_nodes[-1], finish)

    descriptors, direct, result, matches = _descriptor_reconstruction(
        "large_chain", graph
    )

    assert matches
    assert direct == frozenset()
    assert result.macro_edges_realized_by_phase == frozenset({(start, finish)})
    assert len(result.components) == graph.number_of_nodes()
    assert result.condensation.number_of_edges() == graph.number_of_edges()
    assert len(descriptors[0].phase_nodes) == len(phase_nodes)


def test_rimless_guard_on_domain_boundary_generates_phase_cells():
    result = build_rimless_wheel_suspension_grid(
        subdivisions=(8, 8),
        handle_slabs=2,
        padding_cells=0.0,
    )

    assert result.phase_cells
    assert result.suspension_complex_ingredients.handles


def test_rimless_guard_outside_domain_is_not_clamped_to_boundary():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = build_rimless_wheel_suspension_grid(
            subdivisions=(8, 8),
            handle_slabs=2,
            padding_cells=0.0,
            alpha=0.7,
            gamma=0.2,
        )

    assert not result.phase_cells
    assert not result.suspension_complex_ingredients.handles
