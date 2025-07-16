"""
Morse graph computation for hybrid dynamical systems.

This module implements the computation of Hasse diagrams for non-trivial
strongly connected components, which represent the Morse decomposition
of the dynamics.
"""

from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def create_morse_graph(G: nx.DiGraph) -> Tuple[nx.DiGraph, List[Set]]:
    """
    Create a Hasse diagram of non-trivial strongly connected components (Morse sets).

    This is computed by taking the transitive reduction of the condensation graph
    restricted to the non-trivial SCCs. A non-trivial SCC is one that has more
    than one node, or a single node with a self-loop.

    Args:
        G: The input directed graph.

    Returns:
        Tuple of (hasse_diagram, nontrivial_sccs) where:
        - hasse_diagram: DiGraph representing the partial order of non-trivial SCCs.
        - nontrivial_sccs: List of non-trivial SCCs, where the index in the list
                           corresponds to the node ID in the Hasse diagram.
    """

    # Step 1: Find all strongly connected components
    sccs = list(nx.strongly_connected_components(G))

    # Step 2: Identify which SCCs are non-trivial (recurrent)
    nontrivial_sccs = []
    for scc in sccs:
        if len(scc) > 1 or (len(scc) == 1 and G.has_edge(list(scc)[0], list(scc)[0])):
            nontrivial_sccs.append(scc)

    # Step 3: Get the condensation graph
    condensation = nx.condensation(G, sccs)

    # Create a mapping from the original SCC frozenset to its condensation node ID
    scc_to_cond_node = {frozenset(scc): i for i, scc in enumerate(sccs)}

    # Step 4: Build a reachability graph between non-trivial SCCs
    morse_reachability = nx.DiGraph()
    for i, scc1 in enumerate(nontrivial_sccs):
        morse_reachability.add_node(i)
        for j, scc2 in enumerate(nontrivial_sccs):
            if i == j:
                continue

            # Find condensation nodes
            cond1 = scc_to_cond_node[frozenset(scc1)]
            cond2 = scc_to_cond_node[frozenset(scc2)]

            # If there's a path in the condensation graph, add an edge
            if nx.has_path(condensation, cond1, cond2):
                morse_reachability.add_edge(i, j)

    # Step 5: The Hasse diagram is the transitive reduction of this reachability graph
    hasse_diagram = nx.transitive_reduction(morse_reachability)

    return hasse_diagram, nontrivial_sccs


def visualize_morse_graph_comparison(
    G: nx.DiGraph, output_path: str = None,
) -> Tuple[nx.DiGraph, List[Set]]:
    """
    Visualize both the original graph and its Morse graph (Hasse diagram of non-trivial SCCs).

    Args:
        G: The input directed graph
        output_path: Optional path to save the visualization

    Returns:
        Tuple of (hasse_diagram, nontrivial_sccs)
    """
    # Create Hasse diagram
    hasse, nontrivial_sccs = create_morse_graph(G)

    # Visualize both graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original graph
    pos1 = nx.spring_layout(G)
    nx.draw(
        G,
        pos=pos1,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=16,
        font_weight="bold",
        ax=ax1,
    )
    nx.draw_networkx_edge_labels(
        G, pos1, edge_labels={(u, v): f"{u}→{v}" for u, v in G.edges()}, ax=ax1,
    )
    ax1.set_title("Original Graph")

    # Hasse diagram
    if hasse.nodes():
        pos2 = nx.spring_layout(hasse)
        node_labels = {
            i: f"M{i}\\n({len(scc)} nodes)" for i, scc in enumerate(nontrivial_sccs)
        }

        # Use a colormap
        colors = plt.cm.get_cmap("turbo", max(len(nontrivial_sccs), 1))
        node_colors = [colors(i) for i in range(len(nontrivial_sccs))]

        nx.draw(
            hasse,
            pos=pos2,
            labels=node_labels,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
            font_weight="bold",
            ax=ax2,
        )
        ax2.set_title("Morse Graph (Hasse Diagram)")
    else:
        ax2.text(
            0.5,
            0.5,
            "No non-trivial SCCs with ordering",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Hasse Diagram (Empty)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        # Successfully saved
        plt.close()
    else:
        plt.show()

    return hasse, nontrivial_sccs


def main_example():
    """
    Example usage with the test graph G=(V,E) with V={a,b,c} and E={(a,a),(a,b),(b,c),(c,c)}.
    """
    # Create the example graph G=(V,E) with V={a,b,c} and E={(a,a),(a,b),(b,c),(c,c)}
    G = nx.DiGraph()
    G.add_edges_from([("a", "a"), ("a", "b"), ("b", "c"), ("c", "c")])

    # Create test graph

    # Create and visualize Hasse diagram
    output_file = "morse_graph_example.png"
    hasse, nontrivial_sccs = visualize_morse_graph_comparison(
        G, output_path=output_file,
    )

    return G, hasse, nontrivial_sccs


if __name__ == "__main__":
    G, hasse, nontrivial_sccs = main_example()
