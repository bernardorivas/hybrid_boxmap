#!/usr/bin/env python3
"""
Unstable Periodic Demo: SCC Analysis and Visualization

Generates visualizations of strongly connected components and condensation graph
for the unstable periodic hybrid dynamical system.
"""
from pathlib import Path
import networkx as nx
import warnings

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.unstableperiodic import UnstablePeriodicSystem
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
)
from hybrid_dynamics import plot_morse_sets_with_roa_fast, compute_roa
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# Factory function for parallel processing
def create_unstableperiodic_system(max_jumps=15):
    """Factory function to create UnstablePeriodicSystem for parallel processing."""
    system = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-2.0, 2.0)],
        max_jumps=max_jumps
    )
    return system.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]  # Integration time horizons to test
SUBDIVISIONS = [100, 201] # Grid subdivisions [x_subdivisions, y_subdivisions]
SAMPLING_MODE = 'corners' # Sampling mode for box map computation
ENCLOSURE = True          # Use enclosure mode for corners sampling
# ===============================================


def run_unstableperiodic_demo():
    """Unstable periodic Morse set analysis and visualization."""
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("unstable_periodic")
    
    # 1. Define configuration (same for all tau values)
    system_obj = UnstablePeriodicSystem(
        domain_bounds=[(0.0, 1.0), (-2.0, 2.0)],
        max_jumps=15
    )
    grid = Grid(bounds=system_obj.domain_bounds, subdivisions=SUBDIVISIONS)
    
    print(f"Running unstable periodic demo with tau values: {TAU_VALUES}")
    print(f"Grid subdivisions: {SUBDIVISIONS}")
    print(f"Domain bounds: {system_obj.domain_bounds}")
    print()
    
    # Process each tau value
    for tau in TAU_VALUES:
        print(f"\n{'='*60}")
        print(f"Processing tau = {tau}")
        print(f"{'='*60}")
        
        box_map_file = data_dir / f"unstable_periodic_box_map_tau_{tau}.pkl"
        
        # Initial conditions for phase portrait
        initial_conditions = [
            [0.1, 1.2],    # Start near left boundary
            [0.3, -0.8],   # Start with negative y
            [0.8, 0.4],    # Start near right boundary 
            [0.5, 0.0],    # Start in middle
            [0.2, -1.5],   # Start in lower region
            [0.7, 1.8]     # Start in upper region
        ]
        
        # Create run directory
        run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())
        
        # Create configuration hash for validation
        system_params = {
            "max_jumps": system_obj.max_jumps,
            "sampling_mode": SAMPLING_MODE,
            "enclosure": ENCLOSURE
        }
        current_config_hash = create_config_hash(
            system_params, 
            system_obj.domain_bounds, 
            grid.subdivisions.tolist(), 
            tau
        )
        
        # 2. Try to load from cache
        box_map = load_box_map_from_cache(grid, system_obj.system, tau, box_map_file, current_config_hash)
        
        # 3. Compute new if cache miss
        if box_map is None:
            progress_callback = create_progress_callback()
            box_map = HybridBoxMap.compute(
                grid=grid,
                system=system_obj.system,
                tau=tau,
                sampling_mode=SAMPLING_MODE,
                enclosure=ENCLOSURE,
                discard_out_of_bounds_destinations=True,
                progress_callback=progress_callback,
                parallel=True,
                system_factory=create_unstableperiodic_system,
                system_args=(system_obj.max_jumps,)
            )
            
            # Save to cache
            config_details = {
                'system_params': system_params,
                'domain_bounds': system_obj.domain_bounds,
                'grid_subdivisions': grid.subdivisions.tolist(),
                'tau': tau,
                'sampling_mode': SAMPLING_MODE,
                'enclosure': ENCLOSURE
            }
            save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)

        # Convert to NetworkX and analyze
        graph = box_map.to_networkx()
        morse_graph, morse_sets = create_morse_graph(graph)
        
        # Check and report Morse graph computation results
        if morse_graph.number_of_nodes() > 0:
            print(f"Morse graph has {morse_graph.number_of_nodes()} nodes and {morse_graph.number_of_edges()} edges")
            print(f"Found {len(morse_sets)} Morse sets")
        else:
            print("Warning: Morse graph computation resulted in empty graph")

        # Generate visualizations
        
        # Phase portrait
        plotter = HybridPlotter()
        plotter.create_phase_portrait_with_trajectories(
            system=system_obj.system,
            initial_conditions=initial_conditions,
            time_span=(0.0, 4.0),
            output_path=str(run_dir / "phase_portrait.png"),
            max_jumps=8,
            title=f"Phase Portrait (tau={tau})",
            xlabel="x",
            ylabel="y",
            domain_bounds=system_obj.domain_bounds,
            figsize=(10, 8),
            dpi=150,
            show_legend=False
        )
        
        plot_morse_sets_on_grid(grid, morse_sets, str(run_dir / "morse_sets.png"), xlabel="x", ylabel="y")
        plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
        
        # Compute regions of attraction
        print("\nComputing regions of attraction...")
        roa_dict = compute_roa(graph, morse_sets)
        
        # Plot Morse sets with regions of attraction with custom aspect ratio
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.collections import PatchCollection
        
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted figure size
        
        # Define colors
        colors = plt.cm.get_cmap("turbo", len(morse_sets))
        
        # Get grid parameters
        x_dim, y_dim = 0, 1
        box_width_x = grid.box_widths[x_dim]
        box_width_y = grid.box_widths[y_dim]
        margin = 0.02 * max(grid.bounds[0][1] - grid.bounds[0][0],
                           grid.bounds[1][1] - grid.bounds[1][0])
        
        # Set limits without equal aspect
        x_min, x_max = grid.bounds[x_dim]
        y_min, y_max = grid.bounds[y_dim]
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        # Note: NOT setting aspect ratio to equal
        
        # First pass: Plot ROA boxes with dimmer colors
        for morse_idx, roa_boxes in roa_dict.items():
            if morse_idx >= len(morse_sets):
                continue
            
            color = colors(morse_idx)
            morse_set = morse_sets[morse_idx]
            roa_only_boxes = roa_boxes - morse_set
            
            if roa_only_boxes:
                roa_rectangles = []
                for box_idx in roa_only_boxes:
                    center = grid.get_sample_points(box_idx, mode="center")[0]
                    rect = patches.Rectangle(
                        (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                        box_width_x, box_width_y
                    )
                    roa_rectangles.append(rect)
                
                pc = PatchCollection(roa_rectangles, facecolor=color, edgecolor="none", alpha=0.15)
                ax.add_collection(pc)
        
        # Second pass: Plot Morse sets with vibrant colors
        for morse_idx, morse_set in enumerate(morse_sets):
            if not morse_set:
                continue
            
            color = colors(morse_idx)
            morse_rectangles = []
            for box_idx in morse_set:
                center = grid.get_sample_points(box_idx, mode="center")[0]
                rect = patches.Rectangle(
                    (center[x_dim] - box_width_x / 2, center[y_dim] - box_width_y / 2),
                    box_width_x, box_width_y
                )
                morse_rectangles.append(rect)
            
            if morse_rectangles:
                pc = PatchCollection(morse_rectangles, facecolor=color, edgecolor="none", alpha=0.7)
                ax.add_collection(pc)
                
            # Add label
            ax.plot([], [], "s", color=color, label=f"M({morse_idx}) + ROA", markersize=10)
        
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(f"Morse sets and their regions of attraction (tau={tau})", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        if len(morse_sets) <= 10:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(str(run_dir / "morse_sets_with_roa.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Morse sets with ROA saved to {run_dir / 'morse_sets_with_roa.png'}")
        
        # Final summary for this tau
        print(f"\nResults saved to: {run_dir}")
        print(f"Box map computed with tau={tau}, grid subdivisions={SUBDIVISIONS}")
        print(f"Enclosure mode: {ENCLOSURE}")
    
    print(f"\n{'='*60}")
    print("All tau values processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_unstableperiodic_demo()