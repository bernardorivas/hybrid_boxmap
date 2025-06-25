#!/usr/bin/env python3
"""
Test script for analyzing the HybridBoxMap as a directed graph.
"""
from pathlib import Path
import networkx as nx

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel

def analyze_boxmap_graph():
    """
    Computes a HybridBoxMap, converts it to a NetworkX graph,
    and performs a simple analysis.
    """
    print("--- Running HybridBoxMap Graph Analysis Test ---")

    # 1. Compute the Box Map
    print("Computing box map...")
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=10)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[30, 30])
    tau = 0.2
    
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
    )
    print(f"Box map computed with {len(box_map)} source nodes.")

    # 2. Convert to NetworkX graph
    print("\nConverting box map to NetworkX DiGraph...")
    graph = box_map.to_networkx()
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # 3. Perform graph analysis
    # Example: Find all "terminal" nodes (nodes with an out-degree of 0)
    print("\nAnalyzing graph structure...")
    terminal_nodes = [node for node, out_degree in graph.out_degree() if out_degree == 0]
    
    # We also need to account for nodes that are in the grid but are not sources in the map
    all_grid_nodes = set(range(len(grid)))
    source_nodes = set(box_map.keys())
    unmapped_nodes = all_grid_nodes - source_nodes
    
    # Terminal nodes are those in the map with no outgoing edges, plus all unmapped nodes
    all_terminal_nodes = sorted(list(set(terminal_nodes).union(unmapped_nodes)))

    print(f"Found {len(all_terminal_nodes)} terminal nodes (boxes that map to themselves or nowhere).")
    print("A few examples:", all_terminal_nodes[:10], "..." if len(all_terminal_nodes) > 10 else "")

    # Example: Find the number of weakly connected components
    num_components = nx.number_weakly_connected_components(graph)
    print(f"\nThe graph has {num_components} weakly connected components.")

    print("\n✓ Graph analysis script completed successfully!")

if __name__ == "__main__":
    analyze_boxmap_graph() 