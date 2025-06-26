# 3D Visualization Guide

This guide explains the new 3D visualization capabilities added to the hybrid_dynamics library.

## Overview

The library now supports visualization of 3D hybrid dynamical systems with:
- Full 3D Morse set visualization
- 2D projections (XY, XZ, YZ planes)
- 3D phase portraits with trajectory plotting
- Automatic dimension detection and appropriate visualization selection

## New Functions

### 1. `plot_morse_sets_3d()`
Creates a 3D visualization of Morse sets using matplotlib's 3D capabilities.

```python
from hybrid_dynamics.src import plot_morse_sets_3d

plot_morse_sets_3d(
    grid,                      # 3D Grid object
    morse_sets,               # List of Morse sets
    "output.png",             # Output path
    xlabel="x", ylabel="y", zlabel="z",
    elev=30, azim=45         # View angles
)
```

### 2. `plot_morse_sets_projections()`
Automatically creates XY, XZ, and YZ projections for 3D+ systems.

```python
from hybrid_dynamics.src import plot_morse_sets_projections

plot_morse_sets_projections(
    grid,                     # Grid object (any dimension)
    morse_sets,              # List of Morse sets  
    "output_dir/"            # Directory for projection files
)
```

This creates:
- `morse_sets_xy.png` - XY plane projection
- `morse_sets_xz.png` - XZ plane projection  
- `morse_sets_yz.png` - YZ plane projection

### 3. `HybridPlotter` 3D Methods

The `HybridPlotter` class now includes:

```python
plotter = HybridPlotter()

# 3D phase portrait
plotter.create_phase_portrait_3d(
    system, initial_conditions, time_span,
    "phase_3d.png",
    elev=30, azim=45
)

# Phase portrait projections
plotter.create_phase_portrait_projections(
    system, initial_conditions, time_span,
    "output_dir/"
)
```

## Usage Patterns

### Automatic Dimension Detection

Demo scripts can now automatically detect system dimensionality and generate appropriate visualizations:

```python
if grid.dimension == 2:
    # Standard 2D visualization
    plot_morse_sets_on_grid(grid, morse_sets, "morse_sets.png")
    
elif grid.dimension == 3:
    # 3D visualization + projections
    plot_morse_sets_3d(grid, morse_sets, "morse_sets_3d.png")
    plot_morse_sets_projections(grid, morse_sets, output_dir)
    
else:
    # Higher dimensions: projections only
    plot_morse_sets_on_grid(
        grid, morse_sets, "morse_sets.png",
        plot_dims=(0, 1)  # Choose which dims to project
    )
```

### Projection Support

The existing 2D functions now support a `plot_dims` parameter for projecting higher-dimensional data:

```python
# Project 4D system to XZ plane
plot_morse_sets_on_grid(
    grid, morse_sets, "morse_xz.png",
    plot_dims=(0, 2),  # X and Z dimensions
    xlabel="x", ylabel="z"
)
```

## Backward Compatibility

All existing 2D functionality remains unchanged:
- 2D systems work exactly as before
- No changes needed to existing scripts
- New features are opt-in through new functions

## Example Output Structure

For a 3D system, the demo scripts now generate:

```
figures/system_name/run_xxx/
├── morse_graph.png          # Unchanged
├── morse_sets_3d.png        # New: 3D visualization
├── morse_sets_xy.png        # New: XY projection
├── morse_sets_xz.png        # New: XZ projection  
├── morse_sets_yz.png        # New: YZ projection
├── phase_portrait_3d.png    # New: 3D trajectories (if generated)
├── phase_portrait_xy.png    # New: XY projection
├── phase_portrait_xz.png    # New: XZ projection
└── phase_portrait_yz.png    # New: YZ projection
```

## Testing

Run the test script to see the 3D features in action:

```bash
python demo/test_3d_visualization.py
```

This creates sample visualizations in `figures/test_3d/`.