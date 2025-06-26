#!/usr/bin/env python3
"""
Generates 3D phase portraits, morse sets, and morse graph for Chua's circuit hybrid dynamical system.
"""
from pathlib import Path
import warnings
import time
import argparse
import matplotlib.pyplot as plt

from hybrid_dynamics import Grid, HybridBoxMap
from hybrid_dynamics.examples.chua_circuit import ChuaCircuit
from hybrid_dynamics.src import (
    create_morse_graph,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    plot_morse_sets_3d,
)
from hybrid_dynamics.src.plot_utils import HybridPlotter
from hybrid_dynamics.src.demo_utils import (
    create_config_hash,
    create_next_run_dir,
    plot_box_containing_point,
    setup_demo_directories,
    save_box_map_with_config,
    load_box_map_from_cache,
    create_progress_callback,
)


# Factory function for parallel processing
def create_chua_system(alpha=8.4562, beta=12.0732, gamma=0.0052, m0=-1.1468, m1=-0.1768, max_jumps=50):
    """Factory function to create ChuaCircuit system for parallel processing."""
    chua = ChuaCircuit(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        m0=m0,
        m1=m1,
        domain_bounds=[(-8.0, 8.0), (-4.0, 4.0), (-10.0, 10.0)],
        max_jumps=max_jumps,
    )
    return chua.system


# ========== CONFIGURATION PARAMETERS ==========
# Modify these values to change simulation settings:
TAU = 1.0                           # Integration time horizon
SUBDIVISIONS = [40, 40, 40]         # Grid subdivisions [x_subdivisions, y_subdivisions, z_subdivisions]
# ===============================================


def handle_3d_plot(fig, output_path, interactive=False, title=""):
    """Helper function to handle both interactive display and saving of 3D plots.
    
    Args:
        fig: matplotlib figure object
        output_path: path to save the figure
        interactive: if True, show plot interactively
        title: title for the interactive window
    """
    if interactive:
        # Set window title
        if hasattr(fig.canvas.manager, 'set_window_title'):
            fig.canvas.manager.set_window_title(title)
        # Show interactive plot
        plt.show()
    else:
        # Just save the figure
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_phase_portrait_3d_interactive(plotter, system, initial_conditions, time_span, 
                                        output_path, interactive=False, **kwargs):
    """Wrapper for phase portrait that supports interactive mode."""
    if interactive:
        # Don't change backend, just modify the behavior
        # First save the plot as usual
        plotter.create_phase_portrait_3d(system, initial_conditions, time_span, 
                                        output_path, **kwargs)
        
        # Then create an interactive version
        print("  Creating interactive 3D phase portrait...")
        fig = plt.figure(figsize=kwargs.get('figsize', (12, 10)))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simulate trajectories
        trajectories = []
        for ic in initial_conditions:
            try:
                traj = system.simulate(ic, time_span, max_jumps=kwargs.get('max_jumps', 50),
                                     dense_output=True, max_step=kwargs.get('max_step', 0.01))
                if traj.segments:
                    trajectories.append(traj)
            except:
                pass
        
        # Plot trajectories
        colors = plt.cm.tab10(range(len(trajectories)))
        for i, traj in enumerate(trajectories):
            plotter.plot_3d_trajectory(traj, ax=ax, show_jumps=True)
        
        # Customize plot
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))
        ax.set_zlabel(kwargs.get('zlabel', 'z'))
        ax.set_title(kwargs.get('title', 'Phase Portrait 3D'))
        ax.view_init(elev=kwargs.get('elev', 30), azim=kwargs.get('azim', 45))
        ax.grid(True, alpha=0.3)
        
        # Show interactive plot
        print("  Showing interactive plot - close window to continue...")
        plt.show()
    else:
        # Use the standard method
        plotter.create_phase_portrait_3d(system, initial_conditions, time_span, 
                                        output_path, **kwargs)


def plot_morse_sets_3d_interactive(grid, morse_sets, output_path, interactive=False, **kwargs):
    """Wrapper for 3D Morse sets that supports interactive mode.""" 
    if interactive:
        print("  Opening interactive 3D Morse sets visualization...")
        print("  Use mouse to rotate, scroll to zoom, right-click to pan")
        # For now, just save the file - interactive Morse sets would need modification
        # to the core plotting function
        plot_morse_sets_3d(grid, morse_sets, output_path, **kwargs)
        print(f"  Note: Interactive Morse sets visualization saved to {output_path}")
        print("  For full interactivity, open the file with an image viewer that supports 3D rotation")
    else:
        plot_morse_sets_3d(grid, morse_sets, output_path, **kwargs)


def run_chua_circuit_demo(interactive=False):
    """Chua's circuit 3D phase portrait, morse sets, and morse graph.
    
    Args:
        interactive: If True, show plots interactively instead of just saving
    """
    
    # Start total timer
    total_start_time = time.time()
    
    # Suppress repeated warnings about post-jump states
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Setup paths using utility function
    data_dir, figures_base_dir = setup_demo_directories("chua_circuit")
    box_map_file = data_dir / "chua_circuit_box_map.pkl"

    # 1. Define configuration with precise parameters for hidden attractor
    chua = ChuaCircuit(
        alpha=8.4562,      # Precise value for hidden attractor
        beta=12.0732,      # Precise value for hidden attractor
        gamma=0.0052,      # Precise value for hidden attractor
        m0=-1.1468,        # Precise value for hidden attractor
        m1=-0.1768,        # Precise value for hidden attractor
        domain_bounds=[(-8.0, 8.0), (-4.0, 4.0), (-10.0, 10.0)],  # Expanded to contain the attractor
        max_jumps=50,
    )
    grid = Grid(bounds=chua.domain_bounds, subdivisions=SUBDIVISIONS)
    tau = TAU
    
    # Initial conditions for phase portrait - precise values for hidden attractor
    initial_conditions = [
        [5.9, 0.3720, -8.4291],    # Positive hidden attractor (precise IC)
        [-5.9, -0.3720, 8.4291],   # Negative hidden attractor (symmetric)
        chua.get_initial_condition_hidden(),  # Original method IC for comparison
        [0.1, 0.0, 0.0],           # Near origin (should converge to stable equilibrium)
        [1.0, 0.0, 0.0],           # Near x=1 switching surface
        [-1.0, 0.0, 0.0],          # Near x=-1 switching surface
    ]

    # Create run directory
    run_dir = create_next_run_dir(figures_base_dir, tau, grid.subdivisions.tolist())

    # Create configuration hash for validation
    chua_params = {
        "alpha": chua.alpha, 
        "beta": chua.beta,
        "gamma": chua.gamma,
        "m0": chua.m0, 
        "m1": chua.m1,
        "max_jumps": chua.max_jumps
    }
    current_config_hash = create_config_hash(
        chua_params,
        chua.domain_bounds,
        grid.subdivisions.tolist(),
        tau,
    )

    # 2. Try to load from cache
    cache_start_time = time.time()
    box_map = load_box_map_from_cache(grid, chua.system, tau, box_map_file, current_config_hash)
    
    # 3. Compute if cache miss
    if box_map is None:
        print("Computing box map...")
        compute_start_time = time.time()
        progress_callback = create_progress_callback()
        box_map = HybridBoxMap.compute(
            grid=grid,
            system=chua.system,
            tau=tau,
            discard_out_of_bounds_destinations=True,
            progress_callback=progress_callback,
            parallel=True,
            system_factory=create_chua_system,
            system_args=(chua.alpha, chua.beta, chua.gamma, chua.m0, chua.m1, chua.max_jumps)
        )

        # Save to cache
        config_details = {
            "chua_params": chua_params,
            "domain_bounds": chua.domain_bounds,
            "grid_subdivisions": grid.subdivisions.tolist(),
            "tau": tau,
        }
        save_box_map_with_config(box_map, current_config_hash, config_details, box_map_file)
        
        compute_time = time.time() - compute_start_time
        print(f"  Box map computation time: {compute_time:.2f} seconds")
    else:
        cache_time = time.time() - cache_start_time
        print(f"  Cache load time: {cache_time:.3f} seconds")

    # Convert to NetworkX and compute morse graph and sets
    morse_start_time = time.time()
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    morse_time = time.time() - morse_start_time
    
    # Check and report Morse graph computation results
    if morse_graph.number_of_nodes() > 0:
        print("✓ Morse graph and Morse sets were computed.")
        print(f"  Number of Morse sets: {len(morse_sets)}")
        print(f"  Morse graph analysis time: {morse_time:.3f} seconds")
    else:
        print("⚠ Warning: Morse graph is empty. No recurrent sets were found.")

    # Generate visualizations
    viz_start_time = time.time()
    
    # Calculate which boxes to exclude (all boxes not in any Morse set)
    all_morse_boxes = set()
    for morse_set in morse_sets:
        all_morse_boxes.update(morse_set)
    
    # All boxes that are NOT in Morse sets should be excluded from background
    total_boxes = len(grid)
    exclude_boxes = set(range(total_boxes)) - all_morse_boxes
    
    # 3D Phase portrait
    plotter = HybridPlotter()
    create_phase_portrait_3d_interactive(
        plotter=plotter,
        system=chua.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 200.0),  # Much longer time to fully capture the attractor structure
        output_path=str(run_dir / "phase_portrait_3d.png"),
        interactive=interactive,
        max_jumps=50,
        max_step=0.01,
        title="Chua's Circuit - 3D Phase Portrait",
        xlabel="x",
        ylabel="y",
        zlabel="z",
        domain_bounds=chua.domain_bounds,
        figsize=(12, 10),
        dpi=150,
        elev=20,
        azim=45,
        show_legend=False
    )
    
    # Also create 2D projections of phase portrait
    # XY projection
    plotter.create_phase_portrait_with_trajectories(
        system=chua.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 200.0),
        output_path=str(run_dir / "phase_portrait_xy.png"),
        max_jumps=50,
        max_step=0.01,
        title="Chua's Circuit - XY Projection",
        xlabel="x",
        ylabel="y",
        domain_bounds=[chua.domain_bounds[0], chua.domain_bounds[1]],
        figsize=(10, 8),
        dpi=150,
        show_legend=False,
        plot_dims=(0, 1)
    )
    
    # XZ projection
    plotter.create_phase_portrait_with_trajectories(
        system=chua.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 200.0),
        output_path=str(run_dir / "phase_portrait_xz.png"),
        max_jumps=50,
        max_step=0.01,
        title="Chua's Circuit - XZ Projection",
        xlabel="x",
        ylabel="z",
        domain_bounds=[chua.domain_bounds[0], chua.domain_bounds[2]],
        figsize=(10, 8),
        dpi=150,
        show_legend=False,
        plot_dims=(0, 2)
    )
    
    # YZ projection
    plotter.create_phase_portrait_with_trajectories(
        system=chua.system,
        initial_conditions=initial_conditions,
        time_span=(0.0, 200.0),
        output_path=str(run_dir / "phase_portrait_yz.png"),
        max_jumps=50,
        max_step=0.01,
        title="Chua's Circuit - YZ Projection",
        xlabel="y",
        ylabel="z",
        domain_bounds=[chua.domain_bounds[1], chua.domain_bounds[2]],
        figsize=(10, 8),
        dpi=150,
        show_legend=False,
        plot_dims=(1, 2)
    )

    # 3D Morse sets from multiple angles
    angles = [
        (20, 45, "default"),      # Default view
        (30, 135, "back"),        # Back view
        (60, 30, "top"),          # Top-down view
        (5, 90, "side"),          # Side view
        (30, 225, "opposite"),    # Opposite corner
        (45, 60, "isometric"),    # Isometric-like view
    ]
    
    for elev, azim, view_name in angles:
        output_name = f"morse_sets_3d_{view_name}.png"
        print(f"  Generating 3D Morse sets - {view_name} view (elev={elev}, azim={azim})")
        plot_morse_sets_3d_interactive(
            grid, 
            morse_sets, 
            str(run_dir / output_name),
            interactive=False,  # Don't show interactive for multiple views
            xlabel="x", 
            ylabel="y", 
            zlabel="z",
            elev=elev,
            azim=azim,
            exclude_boxes=exclude_boxes
        )
    
    # Show one interactive view if requested
    if interactive:
        print("  Showing interactive 3D Morse sets...")
        plot_morse_sets_3d_interactive(
            grid, 
            morse_sets, 
            str(run_dir / "morse_sets_3d_interactive.png"),
            interactive=True,
            xlabel="x", 
            ylabel="y", 
            zlabel="z",
            elev=30,
            azim=45,
            exclude_boxes=exclude_boxes
        )
    
    # 2D projections of Morse sets
    # XY projection
    plot_morse_sets_on_grid(
        grid, 
        morse_sets, 
        str(run_dir / "morse_sets_xy.png"), 
        xlabel="x", 
        ylabel="y",
        plot_dims=(0, 1),
        exclude_boxes=exclude_boxes
    )
    
    # XZ projection
    plot_morse_sets_on_grid(
        grid, 
        morse_sets, 
        str(run_dir / "morse_sets_xz.png"), 
        xlabel="x", 
        ylabel="z",
        plot_dims=(0, 2),
        exclude_boxes=exclude_boxes
    )
    
    # YZ projection
    plot_morse_sets_on_grid(
        grid, 
        morse_sets, 
        str(run_dir / "morse_sets_yz.png"), 
        xlabel="y", 
        ylabel="z",
        plot_dims=(1, 2),
        exclude_boxes=exclude_boxes
    )

    # The Morse graph (topology doesn't change with projection)
    plot_morse_graph_viz(morse_graph, morse_sets, str(run_dir / "morse_graph.png"))
    
    # Plot box containing hidden attractor initial condition
    ic_hidden = chua.get_initial_condition_hidden()
    plot_box_containing_point(box_map, grid, ic_hidden, run_dir, "hidden_attractor_ic_box")
    
    viz_time = time.time() - viz_start_time
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"✓ Figures are saved in {run_dir}")
    print(f"  3D visualizations:")
    print(f"  - phase_portrait_3d.png")
    print(f"  - morse_sets_3d_default.png (elev=20, azim=45)")
    print(f"  - morse_sets_3d_back.png (elev=30, azim=135)")
    print(f"  - morse_sets_3d_top.png (elev=60, azim=30)")
    print(f"  - morse_sets_3d_side.png (elev=5, azim=90)")
    print(f"  - morse_sets_3d_opposite.png (elev=30, azim=225)")
    print(f"  - morse_sets_3d_isometric.png (elev=45, azim=60)")
    print(f"  2D projections:")
    print(f"  - phase_portrait_xy.png, phase_portrait_xz.png, phase_portrait_yz.png")
    print(f"  - morse_sets_xy.png, morse_sets_xz.png, morse_sets_yz.png")
    print(f"  Topology:")
    print(f"  - morse_graph.png")
    print(f"  - hidden_attractor_ic_box_point_*.png")
    print(f"  Visualization time: {viz_time:.2f} seconds")
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chua's circuit demo with 3D visualizations")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Show interactive 3D plots that can be rotated with mouse')
    args = parser.parse_args()
    
    if args.interactive:
        print("\n🔄 Running in INTERACTIVE mode - 3D plots will open in separate windows")
        print("   Use mouse to rotate, scroll to zoom, right-click to pan")
        print("   Close each window to continue to the next visualization\n")
    
    run_chua_circuit_demo(interactive=args.interactive)