"""Recover full suspension SCCs from a base macro graph.

The input contains no cylinder geometry.  It consists of

* a directed macro graph on base cells, and
* finite descriptors for the phase vertices suppressed by selected macro
  edges.

For a descriptor, ``entries`` point from base vertices into its local phase
graph and ``exits`` point back to base vertices.  A macro edge ``b -> c`` is
realized by that descriptor exactly when an entry from ``b`` can reach an
exit to ``c`` through the local phase graph.  Macro edges not realized by a
descriptor are interpreted as direct base edges.  If both realizations are
present, the direct edge must also be listed in ``direct_base_edges``.

The reconstruction follows the base-trace theorem rather than computing SCCs
of an augmented geometric grid: base-containing SCCs start with SCCs of the
macro graph and absorb precisely the virtual phase vertices mutually
reachable with them.  SCCs of the remaining phase-induced graph are the
phase-only SCCs.  Relabeling descriptor endpoints then gives the full
condensation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx


Edge = Tuple[Hashable, Hashable]


def _check_hashable(value: object, description: str) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{description} must be hashable: {value!r}") from error


def _normalize_nodes(
    values: Iterable[Hashable], description: str
) -> Tuple[Hashable, ...]:
    result = tuple(values)
    for value in result:
        _check_hashable(value, description)
    if len(set(result)) != len(result):
        raise ValueError(f"{description} must not contain duplicates")
    return result


def _normalize_edges(values: Iterable[Edge], description: str) -> Tuple[Edge, ...]:
    result = []
    for value in values:
        try:
            source, target = value
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"each {description} item must be a pair: {value!r}"
            ) from error
        _check_hashable(source, f"{description} source")
        _check_hashable(target, f"{description} target")
        result.append((source, target))
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class VirtualPhaseNode:
    """A collision-free name for one descriptor-local phase vertex."""

    descriptor_id: Hashable
    local_node: Hashable

    def __post_init__(self) -> None:
        _check_hashable(self.descriptor_id, "descriptor_id")
        _check_hashable(self.local_node, "local phase node")


@dataclass(frozen=True)
class PhaseGadgetDescriptor:
    """A finite phase graph and its directed attachments to base vertices.

    ``phase_nodes`` and ``phase_edges`` use local names.  Entries are pairs
    ``(base, local_phase)`` and exits are pairs ``(local_phase, base)``.
    Different gadgets may reuse local names because reconstructed vertices are
    namespaced by ``descriptor_id``.
    """

    descriptor_id: Hashable
    phase_nodes: Tuple[Hashable, ...]
    phase_edges: Tuple[Edge, ...] = ()
    entries: Tuple[Edge, ...] = ()
    exits: Tuple[Edge, ...] = ()

    def __post_init__(self) -> None:
        _check_hashable(self.descriptor_id, "descriptor_id")
        nodes = _normalize_nodes(self.phase_nodes, "phase_nodes")
        phase_edges = _normalize_edges(self.phase_edges, "phase_edges")
        entries = _normalize_edges(self.entries, "entries")
        exits = _normalize_edges(self.exits, "exits")
        node_set = set(nodes)

        for source, target in phase_edges:
            if source not in node_set or target not in node_set:
                raise ValueError(
                    "phase_edges may reference only declared phase_nodes; "
                    f"got {(source, target)!r}"
                )
        for _, target in entries:
            if target not in node_set:
                raise ValueError(
                    f"entry target {target!r} is not a declared phase node"
                )
        for source, _ in exits:
            if source not in node_set:
                raise ValueError(
                    f"exit source {source!r} is not a declared phase node"
                )

        object.__setattr__(self, "phase_nodes", nodes)
        object.__setattr__(self, "phase_edges", phase_edges)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "exits", exits)


@dataclass(frozen=True)
class PhasePathDescriptor:
    """Convenience descriptor for ``source -> phase_nodes... -> target``."""

    descriptor_id: Hashable
    source: Hashable
    target: Hashable
    phase_nodes: Tuple[Hashable, ...]

    def __post_init__(self) -> None:
        _check_hashable(self.descriptor_id, "descriptor_id")
        _check_hashable(self.source, "path source")
        _check_hashable(self.target, "path target")
        nodes = _normalize_nodes(self.phase_nodes, "phase_nodes")
        if not nodes:
            raise ValueError("a phase path must contain at least one phase node")
        object.__setattr__(self, "phase_nodes", nodes)

    def as_gadget(self) -> PhaseGadgetDescriptor:
        """Return the equivalent general gadget descriptor."""

        return PhaseGadgetDescriptor(
            descriptor_id=self.descriptor_id,
            phase_nodes=self.phase_nodes,
            phase_edges=tuple(zip(self.phase_nodes, self.phase_nodes[1:])),
            entries=((self.source, self.phase_nodes[0]),),
            exits=((self.phase_nodes[-1], self.target),),
        )


PhaseDescriptor = Union[PhasePathDescriptor, PhaseGadgetDescriptor]


@dataclass
class FullSCCReconstruction:
    """The exact combinatorial SCC partition and its condensation."""

    components: Tuple[FrozenSet[Hashable], ...]
    component_of: Mapping[Hashable, int]
    condensation: nx.DiGraph
    virtual_graph: nx.DiGraph
    macro_edges_realized_by_phase: FrozenSet[Edge]
    retained_direct_base_edges: FrozenSet[Edge]

    @property
    def base_containing_components(self) -> Tuple[FrozenSet[Hashable], ...]:
        """SCCs containing at least one base vertex."""

        return tuple(
            component
            for component in self.components
            if any(not isinstance(node, VirtualPhaseNode) for node in component)
        )

    @property
    def phase_only_components(self) -> Tuple[FrozenSet[Hashable], ...]:
        """SCCs containing only virtual phase vertices, including transients."""

        return tuple(
            component
            for component in self.components
            if all(isinstance(node, VirtualPhaseNode) for node in component)
        )


def _normalize_descriptors(
    descriptors: Iterable[PhaseDescriptor],
) -> Tuple[PhaseGadgetDescriptor, ...]:
    gadgets = []
    identifiers = set()
    for descriptor in descriptors:
        if isinstance(descriptor, PhasePathDescriptor):
            gadget = descriptor.as_gadget()
        elif isinstance(descriptor, PhaseGadgetDescriptor):
            gadget = descriptor
        else:
            raise TypeError(
                "descriptors must contain PhasePathDescriptor or "
                "PhaseGadgetDescriptor values"
            )
        if gadget.descriptor_id in identifiers:
            raise ValueError(
                f"duplicate phase descriptor id: {gadget.descriptor_id!r}"
            )
        identifiers.add(gadget.descriptor_id)
        gadgets.append(gadget)
    return tuple(gadgets)


def _phase_realized_macro_edges(
    gadget: PhaseGadgetDescriptor,
) -> FrozenSet[Edge]:
    local_graph = nx.DiGraph()
    local_graph.add_nodes_from(gadget.phase_nodes)
    local_graph.add_edges_from(gadget.phase_edges)

    result = set()
    for base_source, entry in gadget.entries:
        reachable = {entry}
        reachable.update(nx.descendants(local_graph, entry))
        for exit_node, base_target in gadget.exits:
            if exit_node in reachable:
                result.add((base_source, base_target))
    return frozenset(result)


def build_base_macro_graph(
    base_nodes: Iterable[Hashable],
    descriptors: Iterable[PhaseDescriptor],
    *,
    direct_base_edges: Collection[Edge] = (),
) -> nx.DiGraph:
    """Build the base macro graph directly from base and phase data.

    The result contains every declared direct base edge and, for each phase
    gadget, every base pair joined by a path through that gadget.  Phase
    vertices remain stored in the descriptors; they are not graph vertices in
    this macro graph.
    """

    bases = _normalize_nodes(base_nodes, "base_nodes")
    base_set = set(bases)
    if any(isinstance(node, VirtualPhaseNode) for node in bases):
        raise TypeError(
            "base nodes may not be VirtualPhaseNode values; that type is "
            "reserved for reconstructed phase vertices"
        )

    gadgets = _normalize_descriptors(descriptors)
    direct_edges = _normalize_edges(direct_base_edges, "direct_base_edges")

    for source, target in direct_edges:
        if source not in base_set or target not in base_set:
            raise ValueError(
                "direct_base_edges may reference only declared base nodes; "
                f"got {(source, target)!r}"
            )

    graph = nx.DiGraph()
    graph.graph["semantics"] = "base macro graph with implicit phase gadgets"
    graph.add_nodes_from(bases)
    graph.add_edges_from(direct_edges, kind="direct")

    for gadget in gadgets:
        for base_source, _ in gadget.entries:
            if base_source not in base_set:
                raise ValueError(
                    f"gadget entry references unknown base node {base_source!r}"
                )
        for _, base_target in gadget.exits:
            if base_target not in base_set:
                raise ValueError(
                    f"gadget exit references unknown base node {base_target!r}"
                )
        for source, target in _phase_realized_macro_edges(gadget):
            if graph.has_edge(source, target):
                graph[source][target]["phase_realized"] = True
            else:
                graph.add_edge(source, target, kind="phase", phase_realized=True)

    return graph


def _copy_base_edge_data(
    graph: nx.DiGraph, source: Hashable, target: Hashable
) -> Dict[str, object]:
    if graph.is_multigraph():
        return {}
    data = graph.get_edge_data(source, target, default={})
    return dict(data)


def _build_virtual_graph(
    base_macro_graph: nx.DiGraph,
    gadgets: Sequence[PhaseGadgetDescriptor],
    direct_base_edges: Collection[Edge],
) -> Tuple[nx.DiGraph, FrozenSet[Edge], FrozenSet[Edge]]:
    if not base_macro_graph.is_directed():
        raise TypeError("base_macro_graph must be directed")

    base_nodes = set(base_macro_graph.nodes)
    if any(isinstance(node, VirtualPhaseNode) for node in base_nodes):
        raise TypeError(
            "base nodes may not be VirtualPhaseNode values; that type is "
            "reserved for reconstructed phase vertices"
        )

    macro_edges = set(base_macro_graph.edges())
    direct_edges = set(_normalize_edges(direct_base_edges, "direct_base_edges"))
    unknown_direct = direct_edges - macro_edges
    if unknown_direct:
        raise ValueError(
            "direct_base_edges must be edges of the base macro graph; "
            f"unknown edges: {sorted(unknown_direct, key=repr)!r}"
        )

    realized_edges = set()
    for gadget in gadgets:
        for base_source, _ in gadget.entries:
            if base_source not in base_nodes:
                raise ValueError(
                    f"gadget entry references unknown base node {base_source!r}"
                )
        for _, base_target in gadget.exits:
            if base_target not in base_nodes:
                raise ValueError(
                    f"gadget exit references unknown base node {base_target!r}"
                )
        realized_edges.update(_phase_realized_macro_edges(gadget))

    missing_macro_edges = realized_edges - macro_edges
    if missing_macro_edges:
        raise ValueError(
            "the base macro graph is missing edges realized by phase "
            "descriptors: "
            f"{sorted(missing_macro_edges, key=repr)!r}"
        )

    # A macro edge supported by a gadget is a suppressed path.  Every other
    # macro edge is direct.  The explicit list resolves the case in which both
    # a direct edge and a suppressed path have the same endpoints.
    retained_base_edges = (macro_edges - realized_edges) | direct_edges

    virtual_graph = nx.DiGraph()
    virtual_graph.graph.update(base_macro_graph.graph)
    for base_node, attributes in base_macro_graph.nodes(data=True):
        copied = dict(attributes)
        copied.setdefault("kind", "base")
        virtual_graph.add_node(base_node, **copied)

    for source, target in retained_base_edges:
        copied = _copy_base_edge_data(base_macro_graph, source, target)
        copied.setdefault("kind", "base")
        virtual_graph.add_edge(source, target, **copied)

    for gadget in gadgets:
        def phase(local_node: Hashable) -> VirtualPhaseNode:
            return VirtualPhaseNode(gadget.descriptor_id, local_node)

        for local_node in gadget.phase_nodes:
            virtual_graph.add_node(
                phase(local_node),
                kind="phase",
                descriptor_id=gadget.descriptor_id,
                local_node=local_node,
            )
        for source, target in gadget.phase_edges:
            virtual_graph.add_edge(phase(source), phase(target), kind="phase")
        for base_source, phase_target in gadget.entries:
            virtual_graph.add_edge(
                base_source, phase(phase_target), kind="phase_entry"
            )
        for phase_source, base_target in gadget.exits:
            virtual_graph.add_edge(
                phase(phase_source), base_target, kind="phase_exit"
            )

    return (
        virtual_graph,
        frozenset(realized_edges),
        frozenset(retained_base_edges),
    )


def _base_trace_components(
    base_macro_graph: nx.DiGraph, virtual_graph: nx.DiGraph
) -> Tuple[Tuple[FrozenSet[Hashable], ...], FrozenSet[Hashable]]:
    """Lift macro SCCs by mutual reachability and return absorbed phases."""

    base_components = tuple(
        frozenset(component)
        for component in nx.strongly_connected_components(base_macro_graph)
    )
    lifted = []
    absorbed_phase_nodes = set()

    for base_component in base_components:
        representative = next(iter(base_component))
        forward = {representative}
        forward.update(nx.descendants(virtual_graph, representative))
        backward = {representative}
        backward.update(nx.ancestors(virtual_graph, representative))
        mutually_reachable = forward & backward
        phases = {
            node
            for node in mutually_reachable
            if isinstance(node, VirtualPhaseNode)
        }
        absorbed_phase_nodes.update(phases)
        lifted.append(frozenset(set(base_component) | phases))

    if sum(len(component) for component in lifted) != (
        len(base_macro_graph.nodes) + len(absorbed_phase_nodes)
    ):
        raise RuntimeError(
            "a virtual phase node was assigned to more than one base component; "
            "the macro graph and descriptors violate the base-trace contract"
        )

    return tuple(lifted), frozenset(absorbed_phase_nodes)


def _assemble_condensation(
    virtual_graph: nx.DiGraph,
    components: Tuple[FrozenSet[Hashable], ...],
) -> Tuple[Mapping[Hashable, int], nx.DiGraph]:
    component_of: Dict[Hashable, int] = {}
    condensation = nx.DiGraph()

    for index, component in enumerate(components):
        for node in component:
            if node in component_of:
                raise RuntimeError(f"node assigned to two SCCs: {node!r}")
            component_of[node] = index
        base_members = frozenset(
            node for node in component if not isinstance(node, VirtualPhaseNode)
        )
        phase_members = frozenset(
            node for node in component if isinstance(node, VirtualPhaseNode)
        )
        condensation.add_node(
            index,
            members=component,
            base_members=base_members,
            phase_members=phase_members,
            kind="base-containing" if base_members else "phase-only",
        )

    missing = set(virtual_graph.nodes) - set(component_of)
    if missing:
        raise RuntimeError(f"nodes missing from reconstructed SCCs: {missing!r}")

    for source, target in virtual_graph.edges:
        source_component = component_of[source]
        target_component = component_of[target]
        if source_component != target_component:
            condensation.add_edge(source_component, target_component)

    if not nx.is_directed_acyclic_graph(condensation):
        raise RuntimeError(
            "reconstructed condensation is cyclic; the macro graph and "
            "descriptors violate the reconstruction contract"
        )
    return component_of, condensation


def reconstruct_full_scc_condensation(
    base_macro_graph: nx.DiGraph,
    descriptors: Iterable[PhaseDescriptor],
    *,
    direct_base_edges: Collection[Edge] = (),
) -> FullSCCReconstruction:
    """Reconstruct every full SCC and the full condensation graph.

    The base graph must contain exactly the macro relation intended by the
    caller: an edge for every direct base edge and for every positive path
    whose internal vertices lie in one supplied phase gadget.  The function
    checks the descriptor-to-macro direction.  Macro edges not supported by a
    descriptor are taken to be direct.

    Ordered path descriptors require no special SCC case: when their endpoints
    are in one base SCC, mutual reachability absorbs the whole path; otherwise
    the path's phase vertices appear as ordered singleton SCCs.
    """

    gadgets = _normalize_descriptors(descriptors)
    virtual_graph, realized_edges, retained_edges = _build_virtual_graph(
        base_macro_graph, gadgets, direct_base_edges
    )

    base_components, absorbed = _base_trace_components(
        base_macro_graph, virtual_graph
    )
    remaining_phase_nodes = {
        node
        for node in virtual_graph.nodes
        if isinstance(node, VirtualPhaseNode) and node not in absorbed
    }
    phase_graph = virtual_graph.subgraph(remaining_phase_nodes)
    phase_only_components = tuple(
        frozenset(component)
        for component in nx.strongly_connected_components(phase_graph)
    )
    components = base_components + phase_only_components
    component_of, condensation = _assemble_condensation(virtual_graph, components)

    return FullSCCReconstruction(
        components=components,
        component_of=component_of,
        condensation=condensation,
        virtual_graph=virtual_graph,
        macro_edges_realized_by_phase=realized_edges,
        retained_direct_base_edges=retained_edges,
    )


def reconstruct_from_base_descriptors(
    base_nodes: Iterable[Hashable],
    descriptors: Iterable[PhaseDescriptor],
    *,
    direct_base_edges: Collection[Edge] = (),
) -> FullSCCReconstruction:
    """Build the macro graph and reconstruct its full implicit-phase SCCs."""

    descriptor_data = tuple(descriptors)
    direct_edges = tuple(direct_base_edges)
    macro_graph = build_base_macro_graph(
        base_nodes,
        descriptor_data,
        direct_base_edges=direct_edges,
    )
    return reconstruct_full_scc_condensation(
        macro_graph,
        descriptor_data,
        direct_base_edges=direct_edges,
    )


__all__ = [
    "FullSCCReconstruction",
    "PhaseDescriptor",
    "PhaseGadgetDescriptor",
    "PhasePathDescriptor",
    "VirtualPhaseNode",
    "build_base_macro_graph",
    "reconstruct_from_base_descriptors",
    "reconstruct_full_scc_condensation",
]
