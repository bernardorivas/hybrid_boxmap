import networkx as nx
import pytest

from hybrid_dynamics.src.implicit_phase_scc import (
    PhaseGadgetDescriptor,
    PhasePathDescriptor,
    VirtualPhaseNode,
    build_base_macro_graph,
    reconstruct_from_base_descriptors,
    reconstruct_full_scc_condensation,
)


def component_edges(result):
    return set(result.condensation.edges)


def test_one_way_phase_path_reconstructs_three_ordered_sccs():
    # Base macro edge a => b suppresses a -> h -> b.
    macro = nx.DiGraph([("a", "b")])
    path = PhasePathDescriptor("jump", "a", "b", ("h",))

    result = reconstruct_full_scc_condensation(macro, [path])
    h = VirtualPhaseNode("jump", "h")

    assert {component for component in result.components} == {
        frozenset({"a"}),
        frozenset({h}),
        frozenset({"b"}),
    }
    assert component_edges(result) == {
        (result.component_of["a"], result.component_of[h]),
        (result.component_of[h], result.component_of["b"]),
    }
    assert ("a", "b") not in result.virtual_graph.edges
    assert result.phase_only_components == (frozenset({h}),)


def test_base_data_builder_derives_macro_edge_and_full_sccs():
    path = PhasePathDescriptor("jump", "a", "b", ("h",))

    macro = build_base_macro_graph(("a", "b"), (path,))
    result = reconstruct_from_base_descriptors(("a", "b"), (path,))
    h = VirtualPhaseNode("jump", "h")

    assert set(macro.edges) == {("a", "b")}
    assert set(result.components) == {
        frozenset({"a"}),
        frozenset({h}),
        frozenset({"b"}),
    }


def test_base_data_builder_rejects_unknown_attachment():
    path = PhasePathDescriptor("jump", "missing", "b", ("h",))

    with pytest.raises(ValueError, match="unknown base node"):
        build_base_macro_graph(("a", "b"), (path,))


def test_return_through_base_absorbs_entire_phase_path():
    # a => b is the phase path and b -> a is a direct base edge.
    macro = nx.DiGraph([("a", "b"), ("b", "a")])
    path = PhasePathDescriptor("jump", "a", "b", ("h0", "h1"))

    result = reconstruct_full_scc_condensation(macro, [path])
    h0 = VirtualPhaseNode("jump", "h0")
    h1 = VirtualPhaseNode("jump", "h1")

    assert result.components == (frozenset({"a", "b", h0, h1}),)
    assert result.phase_only_components == ()
    assert result.condensation.number_of_nodes() == 1
    assert result.condensation.number_of_edges() == 0


def test_phase_cycle_remains_one_phase_only_scc():
    macro = nx.DiGraph([("a", "b")])
    gadget = PhaseGadgetDescriptor(
        descriptor_id="guard-reset",
        phase_nodes=("p", "q"),
        phase_edges=(("p", "q"), ("q", "p")),
        entries=(("a", "p"),),
        exits=(("q", "b"),),
    )

    result = reconstruct_full_scc_condensation(macro, [gadget])
    p = VirtualPhaseNode("guard-reset", "p")
    q = VirtualPhaseNode("guard-reset", "q")
    phase_component = frozenset({p, q})

    assert phase_component in result.phase_only_components
    assert component_edges(result) == {
        (result.component_of["a"], result.component_of[p]),
        (result.component_of[p], result.component_of["b"]),
    }


def test_parallel_direct_edge_is_retained_only_when_declared():
    macro = nx.DiGraph([("a", "b")])
    path = PhasePathDescriptor("jump", "a", "b", ("h",))

    suppressed = reconstruct_full_scc_condensation(macro, [path])
    parallel = reconstruct_full_scc_condensation(
        macro, [path], direct_base_edges=(("a", "b"),)
    )

    assert ("a", "b") not in suppressed.virtual_graph.edges
    assert ("a", "b") in parallel.virtual_graph.edges
    assert parallel.retained_direct_base_edges == frozenset({("a", "b")})


def test_gadget_must_be_reflected_in_macro_graph():
    macro = nx.DiGraph()
    macro.add_nodes_from(("a", "b"))
    path = PhasePathDescriptor("jump", "a", "b", ("h",))

    with pytest.raises(ValueError, match="missing edges realized"):
        reconstruct_full_scc_condensation(macro, [path])


def test_reconstruction_matches_sccs_of_descriptor_realization():
    macro = nx.DiGraph()
    macro.add_nodes_from(("a", "b", "c", "isolated"))
    macro.add_edges_from((("a", "b"), ("b", "a"), ("b", "c")))
    descriptors = [
        PhasePathDescriptor("ab", "a", "b", (0, 1)),
        PhasePathDescriptor("bc", "b", "c", (0,)),
    ]

    result = reconstruct_full_scc_condensation(macro, descriptors)
    explicit_components = {
        frozenset(component)
        for component in nx.strongly_connected_components(result.virtual_graph)
    }

    assert set(result.components) == explicit_components
    assert nx.is_directed_acyclic_graph(result.condensation)
    assert frozenset({"isolated"}) in result.components
