"""
Region of Attraction (ROA) analysis for hybrid dynamical systems.

This module provides utilities to compute and analyze the basins of attraction
(regions of attraction) for Morse sets in hybrid box maps.
"""

from typing import Dict, List, Set, Tuple

import networkx as nx

from .config import config


def compute_regions_of_attraction(
    graph: nx.DiGraph, morse_sets: List[Set[int]],
) -> Dict[int, Set[int]]:
    """
    Compute the regions of attraction (basins of attraction) for each Morse set.

    A region of attraction for a Morse set M is the set of all boxes (nodes)
    from which the system eventually flows into M. This is computed using
    backward reachability analysis on the box map graph.

    Args:
        graph: NetworkX directed graph representing the box map transitions
        morse_sets: List of Morse sets, where each Morse set is a set of box indices

    Returns:
        Dictionary mapping morse_set_index -> set of boxes in its region of attraction.
        The region of attraction includes the Morse set itself.
    """
    logger = config.get_logger(__name__)

    if config.logging.verbose:
        logger.info(
            f"Computing regions of attraction for {len(morse_sets)} Morse sets...",
        )

    # Dictionary to store results: morse_set_index -> ROA boxes
    roa_dict = {}

    # For each Morse set, compute its region of attraction
    for morse_idx, morse_set in enumerate(morse_sets):
        roa_boxes = set()

        # Find all nodes that can reach any node in the Morse set
        # We need to use backward reachability (ancestors in graph theory)
        for morse_box in morse_set:
            if morse_box in graph:
                # Get all ancestors (nodes that can reach this morse_box)
                ancestors = nx.ancestors(graph, morse_box)
                roa_boxes.update(ancestors)

        # Include the Morse set itself in its region of attraction
        roa_boxes.update(morse_set)

        roa_dict[morse_idx] = roa_boxes

        if config.logging.verbose:
            logger.info(
                f"  Morse set {morse_idx}: {len(morse_set)} boxes, ROA: {len(roa_boxes)} boxes",
            )

    return roa_dict


def compute_regions_of_attraction_efficient(
    graph: nx.DiGraph, morse_sets: List[Set[int]],
) -> Dict[int, Set[int]]:
    """
    Efficient computation of regions of attraction using reverse graph traversal.

    This version creates the transpose graph once and performs forward reachability
    from each Morse set, which can be more efficient for large graphs.

    Args:
        graph: NetworkX directed graph representing the box map transitions
        morse_sets: List of Morse sets, where each Morse set is a set of box indices

    Returns:
        Dictionary mapping morse_set_index -> set of boxes in its region of attraction.
    """
    logger = config.get_logger(__name__)

    if config.logging.verbose:
        logger.info(f"Computing ROA (efficient) for {len(morse_sets)} Morse sets...")

    # Create the transpose (reverse) graph
    reverse_graph = graph.reverse()

    roa_dict = {}

    for morse_idx, morse_set in enumerate(morse_sets):
        roa_boxes = set(morse_set)  # Start with the Morse set itself

        # In the reverse graph, perform forward reachability from Morse set nodes
        # This gives us all nodes that can reach the Morse set in the original graph
        nodes_to_visit = list(morse_set)
        visited = set(morse_set)

        while nodes_to_visit:
            current_node = nodes_to_visit.pop()

            # Get all neighbors in the reverse graph (predecessors in original graph)
            for neighbor in reverse_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_to_visit.append(neighbor)
                    roa_boxes.add(neighbor)

        roa_dict[morse_idx] = roa_boxes

        if config.logging.verbose:
            logger.info(
                f"  Morse set {morse_idx}: {len(morse_set)} boxes, ROA: {len(roa_boxes)} boxes",
            )

    return roa_dict


def analyze_roa_coverage(
    roa_dict: Dict[int, Set[int]], total_grid_boxes: int,
) -> Tuple[Dict[str, float], Set[int]]:
    """
    Analyze the coverage and overlap of regions of attraction.

    Args:
        roa_dict: Dictionary mapping morse_set_index -> ROA boxes
        total_grid_boxes: Total number of boxes in the grid

    Returns:
        Tuple of (statistics_dict, uncovered_boxes):
        - statistics_dict: Contains coverage percentages and overlap information
        - uncovered_boxes: Set of boxes not in any ROA
    """
    logger = config.get_logger(__name__)

    all_roa_boxes = set()
    overlapping_boxes = set()

    # Find all boxes covered by ROAs and identify overlaps
    for morse_idx, roa_boxes in roa_dict.items():
        overlaps_with_existing = all_roa_boxes.intersection(roa_boxes)
        if overlaps_with_existing:
            overlapping_boxes.update(overlaps_with_existing)

        all_roa_boxes.update(roa_boxes)

    # Find uncovered boxes
    all_grid_boxes_set = set(range(total_grid_boxes))
    uncovered_boxes = all_grid_boxes_set - all_roa_boxes

    # Compute statistics
    coverage_percentage = (len(all_roa_boxes) / total_grid_boxes) * 100
    overlap_percentage = (
        (len(overlapping_boxes) / len(all_roa_boxes)) * 100 if all_roa_boxes else 0
    )

    statistics = {
        "total_boxes": total_grid_boxes,
        "covered_boxes": len(all_roa_boxes),
        "uncovered_boxes": len(uncovered_boxes),
        "overlapping_boxes": len(overlapping_boxes),
        "coverage_percentage": coverage_percentage,
        "overlap_percentage": overlap_percentage,
        "num_morse_sets": len(roa_dict),
    }

    if config.logging.verbose:
        logger.info("ROA Coverage Analysis:")
        logger.info(f"  Total grid boxes: {total_grid_boxes}")
        logger.info(
            f"  Covered by ROAs: {len(all_roa_boxes)} ({coverage_percentage:.1f}%)",
        )
        logger.info(
            f"  Uncovered boxes: {len(uncovered_boxes)} ({100-coverage_percentage:.1f}%)",
        )
        logger.info(
            f"  Overlapping boxes: {len(overlapping_boxes)} ({overlap_percentage:.1f}% of covered)",
        )

    return statistics, uncovered_boxes


def find_transient_boxes(
    graph: nx.DiGraph, morse_sets: List[Set[int]], roa_dict: Dict[int, Set[int]],
) -> Set[int]:
    """
    Find boxes that are transient (not in any Morse set but in some ROA).

    These are boxes that eventually flow into Morse sets but are not themselves
    part of any recurrent component.

    Args:
        graph: NetworkX directed graph representing the box map transitions
        morse_sets: List of Morse sets
        roa_dict: Dictionary of regions of attraction

    Returns:
        Set of transient box indices
    """
    # Get all boxes that are in Morse sets
    all_morse_boxes = set()
    for morse_set in morse_sets:
        all_morse_boxes.update(morse_set)

    # Get all boxes in ROAs
    all_roa_boxes = set()
    for roa_boxes in roa_dict.values():
        all_roa_boxes.update(roa_boxes)

    # Transient boxes are in ROAs but not in Morse sets
    transient_boxes = all_roa_boxes - all_morse_boxes

    return transient_boxes


# Convenience function that uses the more efficient algorithm by default
def compute_roa(graph: nx.DiGraph, morse_sets: List[Set[int]]) -> Dict[int, Set[int]]:
    """
    Convenience function to compute regions of attraction.

    Uses the efficient algorithm by default.

    Args:
        graph: NetworkX directed graph representing the box map transitions
        morse_sets: List of Morse sets

    Returns:
        Dictionary mapping morse_set_index -> set of boxes in its region of attraction
    """
    return compute_regions_of_attraction_efficient(graph, morse_sets)
