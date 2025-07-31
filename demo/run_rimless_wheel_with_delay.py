#!/usr/bin/env python3
"""
Rimless Wheel with Jump Time Delay

This demo computes the box map for the rimless wheel with jump time penalties,
using a high-resolution grid to show the effects of discrete transition delays.

Parameters:
- Grid: 151×151 subdivisions  
- Time horizon: τ = 2.0
- Jump penalty: Adaptive (based on grid resolution)
- Enclosure: Standard rectangular enclosure
"""
import time
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from hybrid_dynamics import (
    Grid, 
    HybridBoxMap, 
    create_morse_graph,
    compute_roa,
    analyze_roa_coverage,
    config
)
from hybrid_dynamics.examples.rimless_wheel import RimlessWheel
from hybrid_dynamics.src.plot_utils import (
    plot_morse_sets_on_grid,
    plot_morse_sets_with_roa
)
from hybrid_dynamics.src.demo_utils import (
    create_progress_callback,
    create_next_run_dir
)


def run_rimless_wheel_with_delay():
    """Run rimless wheel analysis with jump time delay."""
    
    # Suppress post-jump warnings
    warnings.filterwarnings('ignore', message='Post-jump state outside domain bounds')
    
    # Enable verbose logging to see adaptive epsilon
    config.logging.verbose = True
    
    print("=" * 80)
    print("RIMLESS WHEEL WITH JUMP TIME DELAY")
    print("=" * 80)
    
    # Setup parameters
    wheel = RimlessWheel(alpha=0.4, gamma=0.2, max_jumps=50)
    grid = Grid(bounds=wheel.domain_bounds, subdivisions=[151, 151])
    tau = 2.0
    bloat_factor = 0.12
    
    print(f"Parameters:")
    print(f"  - Grid: {grid.subdivisions.tolist()}")
    print(f"  - Time horizon: τ = {tau}")
    print(f"  - Domain: θ ∈ [{wheel.domain_bounds[0][0]:.2f}, {wheel.domain_bounds[0][1]:.2f}], "
          f"ω ∈ [{wheel.domain_bounds[1][0]:.2f}, {wheel.domain_bounds[1][1]:.2f}]")
    print(f"  - Bloat factor: {bloat_factor}")
    print(f"  - Jump penalty: Adaptive (will be calculated)")
    print()
    
    # Create run directory with standard naming
    base_dir = Path("figures/rimless_wheel")
    run_dir = create_next_run_dir(base_dir, tau, grid.subdivisions.tolist(), bloat_factor)
    
    # Rename to include penalty indicator
    penalty_dir = run_dir.parent / (run_dir.name.replace("_001", "_penalty_001"))
    if penalty_dir.exists():
        # Find next available number
        i = 2
        while True:
            penalty_dir = run_dir.parent / (run_dir.name.replace("_001", f"_penalty_{i:03d}"))
            if not penalty_dir.exists():
                break
            i += 1
    run_dir.rename(penalty_dir)
    run_dir = penalty_dir
    
    print(f"Created run directory: {run_dir.name}")
    
    # Progress callback
    progress_callback = create_progress_callback(update_interval=2000)
    
    # Compute box map with jump delay
    print("\nComputing box map with jump time delay...")
    start_time = time.time()
    
    box_map = HybridBoxMap.compute(
        grid=grid,
        system=wheel.system,
        tau=tau,
        sampling_mode='corners',
        bloat_factor=bloat_factor,
        enclosure=True,
        discard_out_of_bounds_destinations=True,
        progress_callback=progress_callback,
        parallel=False,
        jump_time_penalty=True,
        jump_time_penalty_epsilon=None  # Use adaptive epsilon
    )
    
    computation_time = time.time() - start_time
    
    # Calculate what epsilon was used
    box_diagonal = np.sqrt(np.sum(grid.box_widths ** 2))
    adaptive_epsilon = min(tau / 20.0, box_diagonal / 10.0)  # max_velocity = 5.0
    adaptive_epsilon = max(adaptive_epsilon, 1e-6)
    
    print(f"\nComputation completed in {computation_time:.1f} seconds")
    print(f"Adaptive epsilon used: {adaptive_epsilon:.6f}")
    print(f"Active boxes: {len(box_map)}")
    print(f"Total transitions: {sum(len(dests) for dests in box_map.values())}")
    
    # Compute Morse decomposition
    print("\nComputing Morse decomposition...")
    graph = box_map.to_networkx()
    morse_graph, morse_sets = create_morse_graph(graph)
    
    print(f"Number of Morse sets: {len(morse_sets)}")
    for i, morse_set in enumerate(morse_sets):
        print(f"  Morse set {i}: {len(morse_set)} boxes")
    
    # Compute regions of attraction
    print("\nComputing regions of attraction...")
    roa_dict = compute_roa(graph, morse_sets)
    stats, _ = analyze_roa_coverage(roa_dict, len(grid))
    
    print(f"ROA coverage: {stats['coverage_percentage']:.1f}%")
    print(f"Uncovered boxes: {stats['uncovered_boxes']}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Morse sets visualization
    plot_morse_sets_on_grid(
        grid, morse_sets, 
        str(run_dir / "morse_sets.png"),
        xlabel="Angle θ (rad)",
        ylabel="Angular Velocity ω (rad/s)"
    )
    print("  - Generated morse_sets.png")
    
    # 2. Regions of attraction (renamed to match convention)
    plot_morse_sets_with_roa(
        grid, morse_sets, roa_dict,
        str(run_dir / "morse_set_roa.png"),
        xlabel="Angle θ (rad)",
        ylabel="Angular Velocity ω (rad/s)"
    )
    print("  - Generated morse_set_roa.png")
    
    # 3. Phase portrait (renamed from sample_trajectories)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot some sample trajectories
    initial_conditions = [
        np.array([0.0, 0.4]),
        np.array([0.1, 0.3]),
        np.array([-0.1, 0.5]),
        np.array([0.15, 0.2])
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))
    
    for ic, color in zip(initial_conditions, colors):
        traj = wheel.system.simulate(
            ic, (0, tau), 
            jump_time_penalty=True,
            jump_time_penalty_epsilon=adaptive_epsilon,
            max_jumps=10
        )
        
        if traj.segments:
            states = traj.get_all_states()
            if states.size > 0:
                ax.plot(states[:, 0], states[:, 1], '-', 
                       color=color, linewidth=2, alpha=0.8,
                       label=f"IC: ({ic[0]:.1f}, {ic[1]:.1f})")
                ax.plot(ic[0], ic[1], 'o', color=color, markersize=8)
                
                # Mark jump points
                for jump_time, (pre_state, post_state) in traj.jump_states:
                    if hasattr(pre_state, '__len__'):  # Check if it's an array
                        ax.plot(pre_state[0], pre_state[1], 'x', 
                               color=color, markersize=10, markeredgewidth=2)
    
    ax.set_xlim(wheel.domain_bounds[0])
    ax.set_ylim(wheel.domain_bounds[1])
    ax.set_xlabel("Angle θ (rad)")
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.set_title(f"Phase Portrait with Jump Delay (ε={adaptive_epsilon:.4f})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "phase_portrait.png", dpi=200)
    plt.close()
    print("  - Generated phase_portrait.png")
    
    # 4. Try to generate Morse graph visualization
    try:
        from hybrid_dynamics import plot_morse_graph_viz
        plot_morse_graph_viz(
            morse_graph, morse_sets,
            str(run_dir / "morse_graph.png")
        )
        print("  - Generated morse_graph.png")
    except Exception as e:
        print(f"  - Could not generate Morse graph visualization: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_rimless_wheel_with_delay()