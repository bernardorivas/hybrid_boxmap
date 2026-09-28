#!/usr/bin/env python3
"""
Thermostat MultiGrid Demo

Demonstrates hybrid system analysis using the MultiGrid framework for
systems with discrete modes. The thermostat has two modes (heater off/on)
with temperature-threshold-driven transitions between modes.

This demo showcases:
- MultiGrid with 1000 temperature subdivisions for high resolution
- Discrete mode structure: union of two 1D grids
- Inter-mode and intra-mode transition analysis
- Morse graph analysis of multi-modal dynamics
"""

from pathlib import Path
import numpy as np
import warnings
import time

from hybrid_dynamics.examples.thermostat import Thermostat
from hybrid_dynamics.src.multigrid import MultiGrid, MultiGridBoxMap
from hybrid_dynamics.src.grid import Grid
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    setup_demo_directories,
    create_progress_callback,
)


# Factory function for parallel processing
def create_thermostat_system(T_c=0.5, T_h=1.0, T_off=30.0, T_on=70.0, max_jumps=50):
    """Factory function to create Thermostat system for parallel processing."""
    thermostat = Thermostat(T_c=T_c, T_h=T_h, T_off=T_off, T_on=T_on, max_jumps=max_jumps)
    return thermostat.system


# ========== CONFIGURATION PARAMETERS ==========
TAU = 0.5                    # Integration time horizon
TEMP_SUBDIVISIONS = 100      # Temperature subdivisions
USE_INTERVAL_METHOD = True   # Use interval-based computation (more accurate for 1D systems)
# ===============================================


def create_thermostat_embedding_grid(multigrid: MultiGrid, y_subdivisions: int = 50) -> Grid:
    """
    Create a 2D embedding grid specifically for thermostat visualization.
    
    This creates a grid of size (n, y_subdivisions) where n is the temperature
    subdivision from the multigrid, and the y-bounds are [-eps, 1+eps] such that
    the bottom and top layers of boxes are centered at y=0 and y=1.
    
    Args:
        multigrid: MultiGrid with 1D mode grids for thermostat
        y_subdivisions: Number of y-subdivisions for visualization
        
    Returns:
        2D Grid with bottom/top box layers centered at y=0 and y=1
    """
    # Get temperature bounds and subdivisions from mode grids
    first_grid = next(iter(multigrid.mode_grids.values()))
    temp_bounds = first_grid.bounds[0]  # [0.0, 100.0]
    temp_subdivisions = first_grid.subdivisions[0]  # e.g., 1000
    
    # Calculate eps such that:
    # -eps + box_height/2 = 0  (bottom box centered at 0)
    # 1+eps - box_height/2 = 1  (top box centered at 1)
    # 
    # Total y-span = (1 + eps) - (-eps) = 1 + 2*eps
    # box_height = (1 + 2*eps) / y_subdivisions
    # 
    # From -eps + box_height/2 = 0:
    # eps = box_height/2 = (1 + 2*eps) / (2 * y_subdivisions)
    # eps * 2 * y_subdivisions = 1 + 2*eps
    # eps * (2 * y_subdivisions - 2) = 1
    # eps = 1 / (2 * y_subdivisions - 2)
    
    eps = 1.0 / (2 * y_subdivisions - 2)
    y_min = -eps
    y_max = 1.0 + eps
    
    return Grid(
        bounds=[temp_bounds, [y_min, y_max]], 
        subdivisions=[temp_subdivisions, y_subdivisions]
    )


def map_mode_boxes_to_embedding(multigrid: MultiGrid, embedding_grid: Grid, 
                               morse_sets: list, mode_prefix: str = "") -> dict:
    """
    Map mode-specific Morse sets to embedding grid boxes.
    
    Args:
        multigrid: MultiGrid with 1D mode grids
        embedding_grid: 2D embedding grid with centered boxes
        morse_sets: List of Morse sets (each is a set of NetworkX node IDs)
        mode_prefix: Prefix for node IDs (empty string for direct mapping)
        
    Returns:
        Dictionary mapping morse_set_id -> list of embedding box indices
    """
    morse_sets_for_grid = {}
    
    for morse_set_id, node_set in enumerate(morse_sets):
        grid_boxes = []
        
        for node in node_set:
            # Parse node ID to get mode and box index
            parts = node.split('_')
            if len(parts) != 2:
                continue
                
            try:
                mode, box_idx = int(parts[0]), int(parts[1])
            except ValueError:
                continue
                
            # Get temperature center from 1D mode grid
            temp_grid = multigrid.mode_grids[mode]
            temp_bounds = temp_grid.get_box_bounds(box_idx)
            temp_center = (temp_bounds[0] + temp_bounds[1]) / 2
            
            # Map to embedding grid: find box whose center is closest to (temp_center, mode)
            target_point = [temp_center[0], float(mode)]
            
            try:
                embedding_box = embedding_grid.get_box_from_point(target_point)
                grid_boxes.append(embedding_box)
            except (ValueError, IndexError):
                continue
                
        if grid_boxes:
            morse_sets_for_grid[morse_set_id] = grid_boxes
            
    return morse_sets_for_grid


def run_thermostat_demo():
    """Thermostat MultiGrid with discrete mode analysis."""
    
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    print("=== Thermostat MultiGrid Demo ===")
    print(f"Configuration:")
    print(f"  Time: {TAU}")
    print(f"  Temperature subdivisions: {TEMP_SUBDIVISIONS}")
    print(f"  Resolution: {100/TEMP_SUBDIVISIONS:.2f}°F per box")
    print()
    
    # Setup paths
    data_dir, figures_base_dir = setup_demo_directories("thermostat")
    
    # Create thermostat system
    thermostat = Thermostat(
        z0=60.0,       # Base temperature
        zdelta=30.0,   # Temperature increment when heater is on
        zmin=70.0,     # Lower threshold (heater turns on)
        zmax=80.0,     # Upper threshold (heater turns off)
        max_jumps=50
    )
    
    print(f"Thermostat parameters:")
    print(f"  Base temperature: {thermostat.z0}°F")
    print(f"  Heater increment: {thermostat.zdelta}°F")
    print(f"  Thresholds: {thermostat.zmin}°F - {thermostat.zmax}°F")
    print()
    
    # Create MultiGrid
    multigrid = thermostat.create_multigrid(temp_subdivisions=TEMP_SUBDIVISIONS)
    
    
    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, TAU, [TEMP_SUBDIVISIONS, 2])
    
    # Create progress callback
    progress_callback = create_progress_callback(update_interval=50)
    
    # Compute MultiGrid BoxMap
    compute_start_time = time.time()
    if USE_INTERVAL_METHOD:
        print("Computing MultiGrid BoxMap using interval method...")
        multi_boxmap = multigrid.compute_multi_boxmap_interval(
            system=thermostat.system,
            tau=TAU,
            bloat_factor=0.2,  # Increased bloat factor to catch all transitions
            progress_callback=progress_callback
        )
    else:
        print("Computing MultiGrid BoxMap using corner sampling...")
        multi_boxmap = multigrid.compute_multi_boxmap(
            system=thermostat.system,
            tau=TAU,
            bloat_factor=0.2,  # Increased bloat factor to catch all transitions
            progress_callback=progress_callback
        )
    compute_time = time.time() - compute_start_time
    print(f"  MultiGrid BoxMap computation time: {compute_time:.2f} seconds")
    
    print(f"\\nBoxMap computation results:")
    print(f"  Source boxes with transitions: {len(multi_boxmap)}")
    print(f"  Total transitions: {sum(len(destinations) for destinations in multi_boxmap.values())}")
    
    # Analyze transition types
    intra_mode_count = 0
    inter_mode_count = 0
    
    for (src_mode, src_box), destinations in multi_boxmap.items():
        for dest_mode, dest_box in destinations:
            if src_mode == dest_mode:
                intra_mode_count += 1
            else:
                inter_mode_count += 1
    
    print(f"  Intra-mode transitions: {intra_mode_count}")
    print(f"  Inter-mode transitions: {inter_mode_count}")
    print()
    
    # Analyze mode-specific behavior
    for mode in multigrid.modes:
        mode_transitions = multi_boxmap.get_transitions_from_mode(mode)
        intra_transitions = multi_boxmap.get_intra_mode_transitions(mode)
        
        mode_name = "Heater Off" if mode == 0 else "Heater On"
        print(f"Mode {mode} ({mode_name}):")
        print(f"  Boxes with outgoing transitions: {len(intra_transitions)}")
        print(f"  Transitions to other modes: {sum(len(dests) for m, dests in mode_transitions.items() if m != mode)}")
    
    print()
    
    # Create thermostat-specific embedding grid for centered visualization
    y_subdivisions = 50  # Fine resolution with boxes centered at y=0 and y=1
    embedding_grid = create_thermostat_embedding_grid(multigrid, y_subdivisions=y_subdivisions)
    print(f"Created thermostat-specific embedding grid: {embedding_grid}")
    print(f"  Grid resolution: {embedding_grid.subdivisions[0]} × {embedding_grid.subdivisions[1]} (temperature × visualization)")
    print(f"  Y-bounds: {embedding_grid.bounds[1]} (designed for boxes centered at y=0 and y=1)")
    print(f"  Box height in y-direction: {(embedding_grid.bounds[1][1] - embedding_grid.bounds[1][0]) / embedding_grid.subdivisions[1]:.3f}")
    
    # Initial conditions for phase portrait
    initial_conditions_2d = [
        [65.0, 0.0],  # Start below threshold, heater off
        [85.0, 1.0],  # Start above threshold, heater on
        [75.0, 0.0],  # Start between thresholds, heater off
        [75.0, 1.0],  # Start between thresholds, heater on
        [20.0, 0.0],  # Start far below, heater off
        [95.0, 1.0],  # Start far above, heater on
    ]
    
    # Generate visualizations
    print("\\nGenerating visualizations...")
    
    # Phase portrait showing both discrete levels
    from hybrid_dynamics.src.plot_utils import HybridPlotter
    
    plotter = HybridPlotter()
    # Create enlarged domain bounds for better visualization
    enlarged_bounds = [
        (thermostat.domain_bounds[0][0] - 5, thermostat.domain_bounds[0][1] + 5),  # Temperature: [-5, 105]
        (thermostat.domain_bounds[1][0] - 0.1, thermostat.domain_bounds[1][1] + 0.1)  # Controller: [-0.1, 1.1]
    ]
    
    plotter.create_phase_portrait_with_trajectories(
        system=thermostat.system,
        initial_conditions=initial_conditions_2d,
        time_span=(0.0, 2.0),
        output_path=str(run_dir / "phase_portrait.png"),
        max_jumps=20,
        max_step=0.05,
        title="Phase Portrait",
        xlabel="Temperature (°F)",
        ylabel="Controller State (0=Off, 1=On)",
        domain_bounds=enlarged_bounds,
        figsize=(12, 8),
        dpi=150,
        show_legend=True
    )
    
    print(f"✓ Phase portrait saved to {run_dir / 'phase_portrait.png'}")
    
    # Convert MultiGridBoxMap to standard NetworkX graph for morse analysis
    print("\\nConverting to NetworkX for Morse analysis...")
    
    import networkx as nx
    graph = nx.DiGraph()
    
    # Add nodes for each (mode, box) pair
    for mode, grid in multigrid.mode_grids.items():
        for box_idx in grid.box_indices:
            node_id = f"{mode}_{box_idx}"
            graph.add_node(node_id, mode=mode, box=box_idx)
    
    # Add edges from MultiGridBoxMap
    for (src_mode, src_box), destinations in multi_boxmap.items():
        src_node = f"{src_mode}_{src_box}"
        for dest_mode, dest_box in destinations:
            dest_node = f"{dest_mode}_{dest_box}"
            graph.add_edge(src_node, dest_node, 
                          transition_type='intra' if src_mode == dest_mode else 'inter')
    
    print(f"Created NetworkX graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Morse graph analysis
    morse_start_time = time.time()
    from hybrid_dynamics.src import create_morse_graph, plot_morse_graph_viz
    
    morse_graph, morse_sets = create_morse_graph(graph)
    morse_time = time.time() - morse_start_time
    
    if morse_graph.number_of_nodes() > 0:
        print(f"✓ Morse graph computed: {morse_graph.number_of_nodes()} Morse sets")
        print(f"  Morse graph analysis time: {morse_time:.3f} seconds")
        
        # Plot Morse graph
        plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
        print(f"✓ Morse graph saved to {run_dir / 'morse_graph.png'}")
        
        # Plot Morse sets on embedded grid for visualization
        from hybrid_dynamics import plot_morse_sets_on_grid_fast, plot_morse_sets_with_roa_fast, compute_roa
        
        # Use custom mapping function for thermostat-specific embedding
        morse_sets_for_grid = map_mode_boxes_to_embedding(multigrid, embedding_grid, morse_sets)
        
        if morse_sets_for_grid:
            # Convert morse_sets_for_grid dict to list of sets for plot_morse_sets_on_grid
            sccs_list = [set(boxes) for boxes in morse_sets_for_grid.values()]
            
            # Calculate which boxes to exclude from background plotting
            # For thermostat, we only want to show bottom and top layers
            exclude_boxes = set()
            y_subdivisions = embedding_grid.subdivisions[1]
            temp_subdivisions = embedding_grid.subdivisions[0]
            
            # We want to keep only the bottom layer (y index 0) and top layer (y index y_subdivisions-1)
            # All other y-layers should be excluded
            for y_idx in range(1, y_subdivisions - 1):  # Skip first and last y-layers
                for x_idx in range(temp_subdivisions):
                    # Convert grid coordinates to linear box index
                    box_idx = int(np.ravel_multi_index([x_idx, y_idx], embedding_grid.subdivisions))
                    exclude_boxes.add(box_idx)
            
            # Use fast plotting function without aspect ratio constraint
            import matplotlib.pyplot as plt
            from matplotlib import patches
            from matplotlib.collections import PatchCollection
            
            # First plot morse sets using custom aspect ratio
            fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure for temperature range
            
            # Define colors
            colors = plt.cm.get_cmap("turbo", len(sccs_list))
            
            # Get grid parameters
            x_dim, y_dim = 0, 1
            box_width_x = embedding_grid.box_widths[x_dim]
            box_width_y = embedding_grid.box_widths[y_dim]
            margin = 0.02 * max(embedding_grid.bounds[0][1] - embedding_grid.bounds[0][0],
                               embedding_grid.bounds[1][1] - embedding_grid.bounds[1][0])
            
            # Set limits
            x_min, x_max = embedding_grid.bounds[x_dim]
            y_min, y_max = embedding_grid.bounds[y_dim]
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(-0.1, 1.1)  # Fixed y-limits for better visualization
            
            # Plot each Morse set
            for i, component in enumerate(sccs_list):
                if not component:
                    continue
                
                color = colors(i)
                
                # Filter out excluded boxes
                component_filtered = component - exclude_boxes
                
                # Create rectangles
                rectangles = []
                for box_idx in component_filtered:
                    center = embedding_grid.get_sample_points(box_idx, mode="center")[0]
                    rect = patches.Rectangle(
                        (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                        box_width_x,
                        box_width_y,
                    )
                    rectangles.append(rect)
                
                # Create PatchCollection
                pc = PatchCollection(rectangles, facecolor=color, edgecolor="none", alpha=0.7)
                ax.add_collection(pc)
                
                # Add label
                ax.plot([], [], "s", color=color, label=f"M({i})", markersize=10)
            
            ax.set_xlabel("Temperature (°F)", fontsize=14)
            ax.set_ylabel("Controller State (0=Off, 1=On)", fontsize=14)
            ax.set_title("Morse sets", fontsize=16, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            
            # Add horizontal lines at y=0 and y=1 for clarity
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            
            # Set y-ticks to only show 0 and 1
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Off', 'On'])
            
            if len(sccs_list) <= 10:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(str(run_dir / "morse_sets.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Morse sets visualization saved to {run_dir / 'morse_sets.png'}")
            
            # Compute regions of attraction
            roa_dict = compute_roa(graph, morse_sets)
            
            # Map ROA to embedding grid using same logic as morse sets
            roa_nodes_as_morse_format = []
            for morse_id, roa_nodes in roa_dict.items():
                roa_nodes_as_morse_format.append(roa_nodes)
            
            roa_for_grid_dict = map_mode_boxes_to_embedding(multigrid, embedding_grid, roa_nodes_as_morse_format)
            roa_for_grid = {}
            for i, (morse_id, boxes) in enumerate(roa_for_grid_dict.items()):
                roa_for_grid[i] = set(boxes)
            
            # Plot Morse sets with regions of attraction using same style as morse sets
            fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure for temperature range
            
            # Get grid parameters
            x_dim, y_dim = 0, 1
            box_width_x = embedding_grid.box_widths[x_dim]
            box_width_y = embedding_grid.box_widths[y_dim]
            margin = 0.02 * max(embedding_grid.bounds[0][1] - embedding_grid.bounds[0][0],
                               embedding_grid.bounds[1][1] - embedding_grid.bounds[1][0])
            
            # Set limits
            x_min, x_max = embedding_grid.bounds[x_dim]
            y_min, y_max = embedding_grid.bounds[y_dim]
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(-0.1, 1.1)  # Fixed y-limits for better visualization
            
            # Use same colors as morse sets plot
            colors = plt.cm.get_cmap("turbo", len(sccs_list))
            
            # Plot ROA boxes (excluding middle layers to match morse_sets plot)
            for morse_idx, roa_boxes in roa_for_grid.items():
                if morse_idx >= len(sccs_list):
                    continue
                
                color = colors(morse_idx)
                morse_set = sccs_list[morse_idx]
                roa_only_boxes = roa_boxes - morse_set - exclude_boxes  # Also exclude middle layers
                
                if roa_only_boxes:
                    roa_rectangles = []
                    for box_idx in roa_only_boxes:
                        center = embedding_grid.get_sample_points(box_idx, mode="center")[0]
                        rect = patches.Rectangle(
                            (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                            box_width_x, box_width_y
                        )
                        roa_rectangles.append(rect)
                    
                    pc = PatchCollection(roa_rectangles, facecolor=color, edgecolor="none", alpha=0.15)
                    ax.add_collection(pc)
            
            # Plot Morse sets (excluding middle layers to match morse_sets plot)
            for morse_idx, morse_set in enumerate(sccs_list):
                color = colors(morse_idx)
                morse_rectangles = []
                morse_set_filtered = morse_set - exclude_boxes  # Exclude middle layers
                for box_idx in morse_set_filtered:
                    center = embedding_grid.get_sample_points(box_idx, mode="center")[0]
                    rect = patches.Rectangle(
                        (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                        box_width_x, box_width_y
                    )
                    morse_rectangles.append(rect)
                
                if morse_rectangles:
                    pc = PatchCollection(morse_rectangles, facecolor=color, edgecolor="none", 
                                       alpha=0.7)
                    ax.add_collection(pc)
                    
                # Add label
                ax.plot([], [], "s", color=color, label=f"M({morse_idx}) + ROA", markersize=10)
            
            # Labels and formatting - matching morse_sets plot
            ax.set_xlabel("Temperature (°F)", fontsize=14)
            ax.set_ylabel("Controller State (0=Off, 1=On)", fontsize=14)
            ax.set_title("Morse sets and their regions of attraction", fontsize=16, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            
            # Add horizontal lines at y=0 and y=1 for clarity
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            
            # Set y-ticks to only show 0 and 1
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Off', 'On'])
            
            if len(sccs_list) <= 10:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(str(run_dir / "morse_sets_with_roa.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Morse sets with ROA saved to {run_dir / 'morse_sets_with_roa.png'}")
        
        # Analyze Morse sets by mode
        mode_morse_sets = {0: [], 1: []}
        for morse_set_id, node_set in enumerate(morse_sets):
            for node in node_set:
                mode = int(node.split('_')[0])
                mode_morse_sets[mode].append(morse_set_id)
        
        print(f"\\nMorse sets by mode:")
        for mode in [0, 1]:
            mode_name = "Heater Off" if mode == 0 else "Heater On"
            unique_sets = set(mode_morse_sets[mode])
            print(f"  Mode {mode} ({mode_name}): {len(unique_sets)} Morse sets")
        
    else:
        print("⚠ Warning: No Morse sets found")
    
    # Summary of key findings
    print(f"\\n=== Analysis Summary ===")
    print(f"Temperature range analyzed: [0, 100]°F with {TEMP_SUBDIVISIONS} subdivisions")
    print(f"Switching thresholds: {thermostat.zmin}°F (on) and {thermostat.zmax}°F (off)")
    print(f"Time horizon: {TAU} time units")
    print(f"Total state space boxes: {multigrid.total_boxes}")
    print(f"Active transitions found: {len(multi_boxmap)}")
    print(f"Mode switching behavior: {inter_mode_count} inter-mode transitions")
    print(f"Results saved to: {run_dir}")
    print()
    
    print("✓ Thermostat MultiGrid analysis complete!")


if __name__ == "__main__":
    run_thermostat_demo()