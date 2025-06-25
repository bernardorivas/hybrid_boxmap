"""
Unified plotting interface for hybrid dynamical systems.

This module provides comprehensive visualization tools for hybrid trajectories,
phase portraits, and cubification analysis.
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import warnings
import networkx as nx

from .hybrid_system import HybridSystem
from .hybrid_trajectory import HybridTrajectory
from .cubifier import DatasetCubifier
from .box import Box, SquareBox
from .grid import Grid
from .hybrid_boxmap import HybridBoxMap


class HybridPlotter:
    """Unified plotting interface for hybrid systems."""
    
    def __init__(self, style_config: Optional[Dict[str, Any]] = None):
        """Initialize with custom style configuration.
        
        Args:
            style_config: Custom style overrides
        """
        self.config = self._default_config()
        if style_config:
            self.config.update(style_config)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default style configuration for small-scale plots."""
        return {
            # Figure settings
            'figure_size': (6, 5),
            'dpi': 150,
            
            # Trajectory settings
            'trajectory_color': 'blue',
            'trajectory_alpha': 0.8,
            'trajectory_width': 1.5,
            'trajectory_marker': None,
            'trajectory_markersize': 3,
            
            # Jump/event settings
            'event_marker_size': 50,
            'event_marker_color': 'red',
            'event_marker': 'o',
            'jump_arrow_color': 'red',
            'jump_arrow_alpha': 0.7,
            'jump_arrow_width': 1.5,
            'jump_arrow_style': '->',
            
            # Vector field settings
            'vector_field_color': 'gray',
            'vector_field_alpha': 0.6,
            'vector_field_width': 0.5,
            'vector_field_density': 10,
            
            # Grid and axes
            'grid_color': 'gray',
            'grid_alpha': 0.3,
            'grid_style': '-',
            'grid_width': 0.5,
            
            # Box/cubification settings
            'box_edge_color': 'black',
            'box_face_color': 'lightblue',
            'box_face_alpha': 0.2,
            'box_edge_width': 1,
            
            # Text and labels
            'font_size': 10,
            'title_size': 12,
            'label_size': 10,
            'tick_size': 8,
            
            # Colors for multiple trajectories
            'trajectory_colors': ['blue', 'red', 'green', 'orange', 'purple', 'brown'],
            
            # Time series settings
            'time_color': 'black',
            'time_alpha': 0.9,
            'time_width': 1.2,
        }
    
    def _setup_axes(self, ax: plt.Axes) -> plt.Axes:
        """Apply common axes styling."""
        ax.grid(True, 
                color=self.config['grid_color'], 
                alpha=self.config['grid_alpha'],
                linestyle=self.config['grid_style'],
                linewidth=self.config['grid_width'])
        
        ax.tick_params(labelsize=self.config['tick_size'])
        ax.set_xlabel('x₁', fontsize=self.config['label_size'])
        ax.set_ylabel('x₂', fontsize=self.config['label_size'])
        
        return ax
    
    def plot_hybrid_trajectory(self, 
                              trajectory: HybridTrajectory, 
                              ax: Optional[plt.Axes] = None,
                              show_jumps: bool = True,
                              show_time_labels: bool = False,
                              color: Optional[str] = None,
                              label: Optional[str] = None) -> plt.Axes:
        """Plot trajectory with optional jump visualization.
        
        Args:
            trajectory: HybridTrajectory to plot
            ax: Matplotlib axes to plot on
            show_jumps: Whether to show jump points and arrows
            show_time_labels: Whether to show hybrid time labels
            color: Override trajectory color
            label: Legend label for trajectory
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], 
                                 dpi=self.config['dpi'])
        
        if not trajectory.segments:
            warnings.warn("Empty trajectory - nothing to plot")
            return ax
        
        # Determine trajectory color
        traj_color = color if color is not None else self.config['trajectory_color']
        
        # Plot each segment
        for i, segment in enumerate(trajectory.segments):
            if segment.state_values.shape[1] < 2:
                warnings.warn(f"Segment {i} has dimension < 2, cannot plot")
                continue
            
            # Plot continuous trajectory
            line_label = label if i == 0 else None  # Only label first segment
            ax.plot(segment.state_values[:, 0], segment.state_values[:, 1],
                   color=traj_color,
                   alpha=self.config['trajectory_alpha'],
                   linewidth=self.config['trajectory_width'],
                   marker=self.config['trajectory_marker'],
                   markersize=self.config['trajectory_markersize'],
                   label=line_label)
            
            # Add hybrid time labels if requested
            if show_time_labels and len(segment.time_values) > 0:
                mid_idx = len(segment.time_values) // 2
                mid_state = segment.state_values[mid_idx]
                mid_time = segment.time_values[mid_idx]
                ax.annotate(f'({mid_time:.2f}, {segment.jump_index})',
                           xy=(mid_state[0], mid_state[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.config['font_size'] - 1,
                           alpha=0.7)
        
        # Plot jumps if requested
        if show_jumps and trajectory.jump_times:
            for i, (jump_time, (state_before, state_after)) in enumerate(
                zip(trajectory.jump_times, trajectory.jump_states)):
                
                if len(state_before) < 2 or len(state_after) < 2:
                    continue
                
                # Draw reset map as dashed line segment (not arrow)
                # Use same color as trajectory
                jump_color = color if color is not None else self.config['trajectory_color']
                ax.plot([state_before[0], state_after[0]], 
                       [state_before[1], state_after[1]],
                       color=jump_color,
                       alpha=self.config['jump_arrow_alpha'],
                       linewidth=self.config['jump_arrow_width'],
                       linestyle='--')
        
        self._setup_axes(ax)
        return ax
    
    def plot_phase_portrait(self, 
                           system: HybridSystem,
                           trajectories: List[HybridTrajectory],
                           ax: Optional[plt.Axes] = None,
                           show_vector_field: bool = True,
                           vector_field_bounds: Optional[List[Tuple[float, float]]] = None,
                           colors: Optional[List[str]] = None,
                           show_equilibria: bool = False,
                           equilibria_search_bounds: Optional[List[Tuple[float, float]]] = None,
                           show_legend: bool = True) -> plt.Axes:
        """Plot multiple trajectories in phase space.
        
        Args:
            system: HybridSystem for vector field computation
            trajectories: List of trajectories to plot
            ax: Matplotlib axes to plot on
            show_vector_field: Whether to show background vector field
            vector_field_bounds: Bounds for vector field [(x_min, x_max), (y_min, y_max)]
            colors: Custom colors for trajectories
            show_equilibria: Whether to find and show equilibrium points
            equilibria_search_bounds: Bounds for equilibrium search
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], 
                                 dpi=self.config['dpi'])
        
        # Plot vector field if requested
        if show_vector_field:
            self._plot_vector_field(system, ax, bounds=vector_field_bounds)
        
        # Determine colors for trajectories
        if colors is None:
            colors = self.config['trajectory_colors']
        
        # Plot trajectories
        for i, trajectory in enumerate(trajectories):
            color = colors[i % len(colors)]
            label = f"Trajectory {i+1}"
            self.plot_hybrid_trajectory(trajectory, ax=ax, color=color, label=label)
        
        # Find and plot equilibria if requested
        if show_equilibria and equilibria_search_bounds:
            try:
                equilibria = system.find_equilibria(equilibria_search_bounds)
                for eq in equilibria:
                    if len(eq) >= 2:
                        ax.scatter(eq[0], eq[1], 
                                 s=self.config['event_marker_size'] * 1.5,
                                 c='black', marker='x', 
                                 alpha=0.8, zorder=15,
                                 label='Equilibrium' if eq is equilibria[0] else None)
            except Exception as e:
                warnings.warn(f"Could not find equilibria: {e}")
        
        # Add legend if requested and if multiple trajectories or equilibria shown
        if show_legend and (len(trajectories) > 1 or show_equilibria):
            ax.legend(fontsize=self.config['font_size'])
        
        self._setup_axes(ax)
        return ax
    
    def _plot_vector_field(self, 
                          system: HybridSystem, 
                          ax: plt.Axes,
                          bounds: Optional[List[Tuple[float, float]]] = None):
        """Plot vector field for continuous dynamics."""
        # Determine bounds
        if bounds is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bounds = [(xlim[0], xlim[1]), (ylim[0], ylim[1])]
        
        # Create grid
        density = self.config['vector_field_density']
        x = np.linspace(bounds[0][0], bounds[0][1], density)
        y = np.linspace(bounds[1][0], bounds[1][1], density)
        X, Y = np.meshgrid(x, y)
        
        # Compute vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]])
                try:
                    if system.is_valid_state(state):
                        derivative = system.evaluate_ode(0.0, state)
                        if len(derivative) >= 2:
                            U[i, j] = derivative[0]
                            V[i, j] = derivative[1]
                except Exception:
                    U[i, j] = 0
                    V[i, j] = 0
        
        # Plot vector field
        ax.quiver(X, Y, U, V,
                 color=self.config['vector_field_color'],
                 alpha=self.config['vector_field_alpha'],
                 width=self.config['vector_field_width'] * 0.002,
                 scale_units='width',
                 scale=np.max([np.max(np.abs(U)), np.max(np.abs(V))]) * 20)
    
    def plot_time_series(self, 
                        trajectory: HybridTrajectory,
                        state_indices: Optional[List[int]] = None,
                        ax: Optional[plt.Axes] = None,
                        show_jumps: bool = True) -> plt.Axes:
        """Plot trajectory components vs time.
        
        Args:
            trajectory: HybridTrajectory to plot
            state_indices: Which state components to plot (default: all)
            ax: Matplotlib axes to plot on
            show_jumps: Whether to mark jump times
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], 
                                 dpi=self.config['dpi'])
        
        if not trajectory.segments:
            warnings.warn("Empty trajectory - nothing to plot")
            return ax
        
        # Determine which components to plot
        if state_indices is None:
            state_dim = trajectory.segments[0].state_values.shape[1]
            state_indices = list(range(min(state_dim, 4)))  # Limit to 4 components
        
        # Plot each component
        colors = self.config['trajectory_colors']
        
        for i, state_idx in enumerate(state_indices):
            color = colors[i % len(colors)]
            
            # Plot each segment
            for segment in trajectory.segments:
                if segment.state_values.shape[1] <= state_idx:
                    continue
                
                ax.plot(segment.time_values, segment.state_values[:, state_idx],
                       color=color,
                       alpha=self.config['time_alpha'],
                       linewidth=self.config['time_width'],
                       label=f'x_{state_idx+1}' if segment.jump_index == 0 else None)
        
        # Mark jump times
        if show_jumps and trajectory.jump_times:
            for jump_time in trajectory.jump_times:
                ax.axvline(x=jump_time, 
                          color=self.config['event_marker_color'],
                          alpha=self.config['jump_arrow_alpha'],
                          linestyle='--',
                          linewidth=self.config['jump_arrow_width'])
        
        ax.set_xlabel('Time', fontsize=self.config['label_size'])
        ax.set_ylabel('State', fontsize=self.config['label_size'])
        ax.grid(True, alpha=self.config['grid_alpha'])
        ax.legend(fontsize=self.config['font_size'])
        
        return ax
    
    def plot_cubification(self, 
                         cubifier: DatasetCubifier,
                         trajectory: HybridTrajectory,
                         ax: Optional[plt.Axes] = None,
                         show_boxes: bool = True,
                         show_trajectory: bool = True) -> plt.Axes:
        """Plot trajectory with its cubification.
        
        Args:
            cubifier: DatasetCubifier instance
            trajectory: HybridTrajectory to cubify and plot
            ax: Matplotlib axes to plot on
            show_boxes: Whether to show cubification boxes
            show_trajectory: Whether to show trajectory
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], 
                                 dpi=self.config['dpi'])
        
        # Compute cubification
        if show_boxes:
            boxes = cubifier.cubify_trajectory(trajectory)
            
            # Plot boxes
            for box in boxes:
                if hasattr(box, 'plot_2d'):
                    box.plot_2d(ax=ax,
                               facecolor=self.config['box_face_color'],
                               edgecolor=self.config['box_edge_color'],
                               alpha=self.config['box_face_alpha'],
                               linewidth=self.config['box_edge_width'])
        
        # Plot trajectory
        if show_trajectory:
            self.plot_hybrid_trajectory(trajectory, ax=ax, show_jumps=True)
        
        self._setup_axes(ax)
        return ax
    
    def create_example_figure(self, 
                             example_name: str,
                             system: HybridSystem,
                             trajectory: HybridTrajectory) -> plt.Figure:
        """Generate standard figure for given example.
        
        Args:
            example_name: Name of the example
            system: HybridSystem instance
            trajectory: HybridTrajectory to visualize
            
        Returns:
            Complete matplotlib Figure
        """
        fig = plt.figure(figsize=(12, 5), dpi=self.config['dpi'])
        
        # Phase portrait
        ax1 = fig.add_subplot(121)
        self.plot_phase_portrait(system, [trajectory], ax=ax1)
        ax1.set_title(f"{example_name} - Phase Portrait", 
                     fontsize=self.config['title_size'])
        
        # Time series
        ax2 = fig.add_subplot(122)
        self.plot_time_series(trajectory, ax=ax2)
        ax2.set_title(f"{example_name} - Time Series", 
                     fontsize=self.config['title_size'])
        
        # Add hybrid time domain info
        domain_str = trajectory.to_hybrid_time_notation()
        fig.suptitle(f"Domain: {domain_str}", fontsize=self.config['font_size'])
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectory(self, 
                          trajectory: HybridTrajectory,
                          ax: Optional[Axes3D] = None,
                          show_jumps: bool = True) -> Axes3D:
        """Plot 3D hybrid trajectory.
        
        Args:
            trajectory: HybridTrajectory to plot
            ax: 3D matplotlib axes
            show_jumps: Whether to show jump points
            
        Returns:
            3D matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=self.config['figure_size'], 
                           dpi=self.config['dpi'])
            ax = fig.add_subplot(111, projection='3d')
        
        if not trajectory.segments:
            warnings.warn("Empty trajectory - nothing to plot")
            return ax
        
        # Plot segments
        for segment in trajectory.segments:
            if segment.state_values.shape[1] < 3:
                warnings.warn(f"Segment has dimension < 3, cannot plot in 3D")
                continue
            
            ax.plot(segment.state_values[:, 0], 
                   segment.state_values[:, 1], 
                   segment.state_values[:, 2],
                   color=self.config['trajectory_color'],
                   alpha=self.config['trajectory_alpha'],
                   linewidth=self.config['trajectory_width'])
        
        # Plot jumps as dashed lines
        if show_jumps and trajectory.jump_times:
            for state_before, state_after in trajectory.jump_states:
                if len(state_before) >= 3 and len(state_after) >= 3:
                    ax.plot([state_before[0], state_after[0]], 
                           [state_before[1], state_after[1]], 
                           [state_before[2], state_after[2]],
                           color=self.config['trajectory_color'],  # Use trajectory color
                           alpha=self.config['jump_arrow_alpha'],
                           linewidth=self.config['jump_arrow_width'],
                           linestyle='--')
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('x₃')
        
        return ax
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   filename: str, 
                   **kwargs):
        """Save figure with default settings.
        
        Args:
            fig: Figure to save
            filename: Output filename
            **kwargs: Additional arguments for savefig
        """
        save_kwargs = {
            'dpi': self.config['dpi'],
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }
        save_kwargs.update(kwargs)
        
        fig.savefig(filename, **save_kwargs)
    
    def update_config(self, **kwargs):
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
    
    def plot_trajectory_segment_cubification(self,
                                           trajectory: HybridTrajectory,
                                           time_range: Tuple[float, float],
                                           cubifier: DatasetCubifier,
                                           ax: Optional[plt.Axes] = None,
                                           show_full_trajectory: bool = True,
                                           highlight_segment: bool = True) -> plt.Axes:
        """Plot trajectory with cubification of a specific time segment.
        
        Args:
            trajectory: Full HybridTrajectory
            time_range: (t_start, t_end) for segment to cubify
            cubifier: DatasetCubifier instance
            ax: Matplotlib axes to plot on
            show_full_trajectory: Whether to show the complete trajectory
            highlight_segment: Whether to highlight the cubified segment
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config['figure_size'], 
                                 dpi=self.config['dpi'])
        
        # Extract trajectory segment
        segment_trajectory = self._extract_trajectory_segment(trajectory, time_range)
        
        # Plot full trajectory if requested
        if show_full_trajectory:
            self.plot_hybrid_trajectory(trajectory, ax=ax, 
                                       color='lightgray', 
                                       show_jumps=False,
                                       label='Full trajectory')
        
        # Highlight segment if requested
        if highlight_segment:
            self.plot_hybrid_trajectory(segment_trajectory, ax=ax,
                                       color='red',
                                       show_jumps=True,
                                       label=f'Segment [{time_range[0]:.1f}, {time_range[1]:.1f}]')
        
        # Compute and plot cubification
        boxes = cubifier.cubify_trajectory(segment_trajectory, make_square=True)
        
        # Plot boxes
        for box in boxes:
            if hasattr(box, 'plot_2d'):
                box.plot_2d(ax=ax,
                           facecolor=self.config['box_face_color'],
                           edgecolor=self.config['box_edge_color'],
                           alpha=self.config['box_face_alpha'],
                           linewidth=self.config['box_edge_width'])
        
        ax.legend(fontsize=self.config['font_size'])
        self._setup_axes(ax)
        ax.set_title(f'Trajectory Segment Cubification\nTime range: [{time_range[0]:.1f}, {time_range[1]:.1f}]',
                    fontsize=self.config['title_size'])
        
        return ax
    
    def _extract_trajectory_segment(self, 
                                   trajectory: HybridTrajectory, 
                                   time_range: Tuple[float, float]) -> HybridTrajectory:
        """Extract a time segment from a hybrid trajectory.
        
        Args:
            trajectory: Source HybridTrajectory
            time_range: (t_start, t_end) to extract
            
        Returns:
            New HybridTrajectory containing only the specified time range
        """
        from ..src.hybrid_trajectory import HybridTrajectory, TrajectorySegment
        
        t_start, t_end = time_range
        segment_trajectory = HybridTrajectory()
        
        for segment in trajectory.segments:
            # Find overlap between segment time and requested range
            seg_start = segment.time_values[0] if len(segment.time_values) > 0 else t_start
            seg_end = segment.time_values[-1] if len(segment.time_values) > 0 else t_start
            
            # Check for overlap
            if seg_end < t_start or seg_start > t_end:
                continue  # No overlap
            
            # Find indices within time range
            time_mask = (segment.time_values >= t_start) & (segment.time_values <= t_end)
            
            if np.any(time_mask):
                # Extract subset
                subset_times = segment.time_values[time_mask]
                subset_states = segment.state_values[time_mask]
                
                if len(subset_times) > 0:
                    new_segment = TrajectorySegment(
                        time_values=subset_times,
                        state_values=subset_states,
                        jump_index=segment.jump_index
                    )
                    segment_trajectory.add_segment(new_segment)
        
        # Extract jumps within time range
        if trajectory.jump_times:
            for i, jump_time in enumerate(trajectory.jump_times):
                if t_start <= jump_time <= t_end:
                    state_before, state_after = trajectory.jump_states[i]
                    segment_trajectory.add_jump(jump_time, state_before, state_after)
        
        return segment_trajectory
    
    def create_segment_cubification_figure(self,
                                         trajectory: HybridTrajectory,
                                         time_range: Tuple[float, float],
                                         cubifier: DatasetCubifier,
                                         title: str = "Trajectory Segment Analysis") -> plt.Figure:
        """Create comprehensive figure showing trajectory segment and cubification.
        
        Args:
            trajectory: HybridTrajectory to analyze
            time_range: (t_start, t_end) for segment
            cubifier: DatasetCubifier instance
            title: Figure title
            
        Returns:
            Complete matplotlib Figure
        """
        fig = plt.figure(figsize=(15, 5), dpi=self.config['dpi'])
        
        # Phase portrait with full trajectory
        ax1 = fig.add_subplot(131)
        self.plot_hybrid_trajectory(trajectory, ax=ax1, show_jumps=True)
        ax1.set_title("Full Trajectory", fontsize=self.config['title_size'])
        
        # Time series highlighting segment
        ax2 = fig.add_subplot(132)
        self.plot_time_series(trajectory, ax=ax2, show_jumps=True)
        
        # Add vertical lines for segment boundaries
        t_start, t_end = time_range
        ax2.axvline(x=t_start, color='red', linestyle='--', alpha=0.7, label=f't = {t_start}')
        ax2.axvline(x=t_end, color='red', linestyle='--', alpha=0.7, label=f't = {t_end}')
        ax2.legend(fontsize=self.config['font_size'])
        ax2.set_title("Time Series with Segment", fontsize=self.config['title_size'])
        
        # Segment cubification
        ax3 = fig.add_subplot(133)
        self.plot_trajectory_segment_cubification(
            trajectory, time_range, cubifier, ax=ax3,
            show_full_trajectory=True, highlight_segment=True
        )
        
        fig.suptitle(title, fontsize=self.config['title_size'] + 2)
        plt.tight_layout()
        
        return fig
    
    def plot_event_region(self,
                          system,
                          t_eval: float = 0.0,
                          plot_dims: Tuple[int, int] = (0, 1),
                          plot_bounds: Optional[List[Tuple[float, float]]] = None,
                          grid_resolution: int = 100,
                          ax: Optional[plt.Axes] = None) -> None:
        """Plot the event region (0-level set of the event function) for a hybrid system.
        
        Args:
            system: The HybridSystem model
            t_eval: Time at which to evaluate the event function
            plot_dims: Indices of the two state variables for x and y axes
            plot_bounds: [[xmin, xmax], [ymin, ymax]] for the plot
            grid_resolution: Number of points along each axis for contour grid
            ax: Matplotlib axes to plot on
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        event_region_styles = {'colors': 'red', 'linewidths': 2.0, 'alpha': 0.7}
        
        x_dim, y_dim = plot_dims
        
        if plot_bounds is None:
            if hasattr(system, 'domain_bounds'):
                plot_bounds = [
                    (system.domain_bounds[x_dim][0], system.domain_bounds[x_dim][1]),
                    (system.domain_bounds[y_dim][0], system.domain_bounds[y_dim][1])
                ]
            else:
                raise ValueError("plot_bounds must be provided if system has no domain_bounds")

        x_min, x_max = plot_bounds[0]
        y_min, y_max = plot_bounds[1]
        
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Evaluate event function on grid
        Z = np.zeros_like(X)
        state_dim = len(system.domain_bounds) if system.domain_bounds else 2
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                state = np.zeros(state_dim)
                state[x_dim] = X[i, j]
                state[y_dim] = Y[i, j]
                Z[i, j] = system.event_function(t_eval, state)
        
        # Plot event region (0-level set)
        ax.contour(X, Y, Z, levels=[0], **event_region_styles)
    
    def create_phase_portrait_with_trajectories(self,
                                              system: HybridSystem,
                                              initial_conditions: List[List[float]],
                                              time_span: Tuple[float, float],
                                              output_path: str,
                                              max_jumps: int = 50,
                                              max_step: Optional[float] = None,
                                              title: str = "Phase Portrait",
                                              xlabel: str = "x₁",
                                              ylabel: str = "x₂",
                                              domain_bounds: Optional[List[Tuple[float, float]]] = None,
                                              colors: Optional[List[str]] = None,
                                              figsize: Tuple[float, float] = (10, 8),
                                              dpi: int = 150,
                                              show_legend: bool = False) -> None:
        """
        Create a phase portrait by simulating trajectories from given initial conditions.
        
        This method combines trajectory simulation with phase portrait creation,
        providing a convenient interface for the common pattern used in demo scripts.
        
        Args:
            system: HybridSystem to simulate
            initial_conditions: List of initial conditions for trajectories
            time_span: (start_time, end_time) for simulation
            output_path: Path to save the phase portrait
            max_jumps: Maximum number of jumps per trajectory
            max_step: Maximum step size for simulation
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            domain_bounds: Domain bounds for axis limits [(x_min, x_max), (y_min, y_max)]
            colors: Custom colors for trajectories
            figsize: Figure size
            dpi: Figure DPI
        """
        # Simulate trajectories with error handling
        trajectories = []
        for ic in initial_conditions:
            try:
                simulate_kwargs = {
                    'time_span': time_span,
                    'max_jumps': max_jumps,
                    'dense_output': True
                }
                if max_step is not None:
                    simulate_kwargs['max_step'] = max_step
                
                traj = system.simulate(ic, **simulate_kwargs)
                if traj.segments:  # Only add if trajectory has segments
                    trajectories.append(traj)
            except Exception:
                # Silently skip failed simulations
                pass
        
        if not trajectories:
            # Create empty plot if no trajectories succeeded
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if domain_bounds:
                ax.set_xlim(domain_bounds[0])
                ax.set_ylim(domain_bounds[1])
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            return
        
        # Create phase portrait
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Use provided colors or default qualitative colormap
        if colors is None:
            # Use tab10 qualitative colormap for distinct, easily separable colors
            colormap = plt.cm.get_cmap('tab10')
            colors = [colormap(i) for i in range(len(initial_conditions))]
        
        # Plot trajectories using the existing plot_phase_portrait method
        self.plot_phase_portrait(
            system=system,
            trajectories=trajectories,
            ax=ax,
            show_vector_field=False,  # Disable vector field by default for performance
            colors=colors,
            show_legend=show_legend
        )
        
        # Add star markers at initial conditions
        for i, (ic, traj) in enumerate(zip(initial_conditions, trajectories)):
            if traj.segments and len(ic) >= 2:
                color = colors[i % len(colors)]
                ax.scatter(ic[0], ic[1], marker='x', s=100, color=color, 
                          linewidth=0.5, zorder=10)
        
        # Customize the plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if show_legend:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set domain bounds if provided
        if domain_bounds:
            ax.set_xlim(domain_bounds[0])
            ax.set_ylim(domain_bounds[1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


"""
Utilities for plotting and visualization - legacy functions.
"""
import pickle
from pathlib import Path
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np

from .config import config


def visualize_flow_map(
    input_path: str,
    output_path: str,
    margin: Optional[float] = None,
) -> None:
    """
    Visualizes the flow map from saved data.

    Args:
        input_path: Path to the .pkl file with flow map data.
        output_path: Path to save the output figure.
        margin: Margin to add around the domain bounds for plotting.
               If None, uses default from config.
    """
    # Use default margin if not provided
    if margin is None:
        margin = 0.1  # TODO: Add to config system

    print(f"Loading flow map data from {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    grid = data["grid"]
    flow_map_data = data["flow_map_data"]

    initial_points = []
    final_points = []

    for box_index in grid.box_indices:
        points_in_box = grid.get_sample_points(box_index, "corners")
        results_for_box = flow_map_data.get(box_index)

        if results_for_box is not None:
            for i, result_vector in enumerate(results_for_box):
                if not np.isnan(result_vector).any():
                    initial_points.append(points_in_box[i])
                    final_points.append(result_vector)

    initial_points = np.array(initial_points)
    final_points = np.array(final_points)

    print(f"Visualizing {len(initial_points)} valid point mappings.")

    fig, ax = plt.subplots(**config.get_figure_config())

    # Plot initial points (x) as blue stars
    if initial_points.size > 0:
        ax.scatter(initial_points[:, 0], initial_points[:, 1], c="blue", marker="*", label="Initial states (x)")

    # Plot final points (f(x)) as red stars
    if final_points.size > 0:
        ax.scatter(final_points[:, 0], final_points[:, 1], c="red", marker="*", label="Final states (f(x))")

    # Set plot bounds
    domain_bounds = grid.bounds
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("State dim 1")
    ax.set_ylabel("State dim 2")
    ax.set_title("Flow Map Visualization: f(x) = φ(τ, x)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Save the figure
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")
    plt.close(fig)

def visualize_box_map(
    grid: "Grid",
    box_map: "HybridBoxMap",
    output_path: str,
    margin: Optional[float] = None,
) -> None:
    """
    Visualizes a HybridBoxMap.

    Args:
        grid: The grid on which the map is defined.
        box_map: The HybridBoxMap to visualize.
        output_path: Path to save the output figure.
        margin: Margin to add around the domain for plotting.
               If None, uses default from config.
    """
    # Use default margin if not provided
    if margin is None:
        margin = 0.1  # TODO: Add to config system

    print("Visualizing hybrid box map...")
    fig, ax = plt.subplots(**config.get_figure_config())

    # Get all unique source and destination boxes
    source_indices = set(box_map.keys())
    destination_indices = set(idx for dest_set in box_map.values() for idx in dest_set)

    # Plot all boxes with default styling
    for i in grid.box_indices:
        box = grid.get_box(i)
        box.plot_2d(ax, facecolor="lightgray", edgecolor="white", alpha=0.5)

    # Highlight source and destination boxes
    for idx in source_indices:
        grid.get_box(idx).plot_2d(ax, facecolor="lightblue", edgecolor="blue", alpha=0.8, label="Source Box")

    for idx in destination_indices:
        # If a box is both a source and a destination, color it purple
        color = "mediumpurple" if idx in source_indices else "lightcoral"
        edge_color = "purple" if idx in source_indices else "red"
        grid.get_box(idx).plot_2d(ax, facecolor=color, edgecolor=edge_color, alpha=0.8, label="Destination Box")

    # Set plot bounds
    domain_bounds = grid.bounds
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("State dim 1")
    ax.set_ylabel("State dim 2")
    ax.set_title("Hybrid Box Map Visualization")

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.grid(True)
    fig.tight_layout()

    # Save the figure
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=150)
    print(f"Box map visualization saved to {output_path}")
    plt.close(fig)

def visualize_box_map_entry(
    grid: "Grid",
    source_box_index: int,
    destination_indices: Set[int],
    initial_points: np.ndarray,
    final_points: np.ndarray,
    output_path: str,
) -> None:
    """
    Visualizes a single entry of a HybridBoxMap (one source to its destinations).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the full grid in light grey
    for i in range(len(grid)):
        grid.get_box(i).plot_2d(ax, facecolor="#E0E0E0", edgecolor="white", alpha=0.5, zorder=1)

    # Plot the source box
    source_box = grid.get_box(source_box_index)
    source_box.plot_2d(ax, facecolor="#3399FF", edgecolor="blue", alpha=0.4, hatch="//", zorder=2, label="Source Box")

    # Plot the destination boxes
    if destination_indices:
        is_first_dest = True
        for dest_idx in destination_indices:
            dest_box = grid.get_box(dest_idx)
            label = "Destination Boxes" if is_first_dest else None
            dest_box.plot_2d(ax, facecolor="#FF3333", edgecolor="red", alpha=0.4, hatch="\\\\", zorder=3, label=label)
            is_first_dest = False

    # Plot the initial and final points (corners)
    ax.scatter(initial_points[:, 0], initial_points[:, 1], c="blue", marker="*", s=150, zorder=4, label="Source Box Corners (x)",
               edgecolors="black")

    # Filter out any NaN values from final points before plotting
    if final_points.size > 0:
        valid_final_points = final_points[~np.isnan(final_points).any(axis=1)]
        if valid_final_points.size > 0:
            ax.scatter(valid_final_points[:, 0], valid_final_points[:, 1], c="red", marker="*", s=150, zorder=5, label="Destination Points (f(x))",
                       edgecolors="black")

    ax.set_title(f"Box Map for Source Box {source_box_index}")
    ax.set_xlabel("State x1")
    ax.set_ylabel("State x2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Set plot bounds to the grid domain
    domain_bounds = grid.bounds
    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])

    fig.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def plot_morse_sets_on_grid(
    grid: "Grid",
    sccs: List[Set[int]],
    output_path: str,
    margin: Optional[float] = None,
    plot_dims: Tuple[int, int] = (0, 1),
    xlabel: str = "State dim 1",
    ylabel: str = "State dim 2",
    exclude_boxes: Optional[Set[int]] = None,
) -> None:
    """
    Visualizes the boxes of Strongly Connected Components on the grid.

    Args:
        grid: The grid on which the map is defined.
        sccs: A list of SCCs, where each SCC is a set of box indices.
        output_path: Path to save the output figure.
        margin: Margin to add around the domain for plotting.
               If None, uses default from config.
        plot_dims: Tuple of (x_dim, y_dim) indices for projection dimensions.
                  Defaults to (0, 1) for first two dimensions.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        exclude_boxes: Optional set of box indices to exclude from background plotting.
    """
    # Use default margin if not provided
    if margin is None:
        margin = 0.1  # TODO: Add to config system
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define a color cycle for the components using a more vibrant map
    colors = plt.cm.get_cmap("turbo", len(sccs)) if sccs else []

    # Plot all grid boxes faintly (except those in exclude_boxes)
    for i in range(len(grid)):
        if exclude_boxes and i in exclude_boxes:
            continue  # Skip excluded boxes
        box = grid.get_box(i)
        if box.dimension == 2:
            box.plot_2d(ax, facecolor="#F0F0F0", edgecolor="white", alpha=0.7, zorder=1)
        else:
            box.plot_2d_projection(ax, dims=plot_dims, facecolor="#F0F0F0", edgecolor="white", alpha=0.7, zorder=1)

    # Highlight the boxes for each SCC with a unique color
    for i, component in enumerate(sccs):
        color = colors(i)
        is_first_box = True
        for box_idx in component:
            # Add a label for each SCC using M(i) notation
            label = f"M({i})" if is_first_box else None
            box = grid.get_box(box_idx)
            if box.dimension == 2:
                box.plot_2d(ax, facecolor=color, edgecolor="none", alpha=0.4, zorder=2, label=label)
            else:
                box.plot_2d_projection(ax, dims=plot_dims, facecolor=color, edgecolor="none", alpha=0.4, zorder=2, label=label)
            is_first_box = False

    # Set plot bounds using the specified projection dimensions
    domain_bounds = grid.bounds
    x_dim, y_dim = plot_dims
    x_min, x_max = domain_bounds[x_dim]
    y_min, y_max = domain_bounds[y_dim]
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Morse sets")
    if len(sccs) <= 10:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=150)
    # Remove individual save message - will be handled by caller
    plt.close(fig)

def plot_morse_graph_viz(
    morse_graph: "nx.DiGraph",
    sccs: List[Set[int]],
    output_path: str,
) -> None:
    """
    Visualizes the morse graph using PyGraphviz.

    Args:
        morse_graph: The morse graph (Hasse diagram).
        sccs: The list of non-trivial SCCs (recurrent components) corresponding
              to the nodes in the morse_graph.
        output_path: Path to save the output figure (e.g., 'morse_graph.png').
    """
    try:
        import pygraphviz as pgv
    except ImportError:
        print("PyGraphviz is not installed. Cannot create morse graph visualization.")
        print("Please install it: pip install pygraphviz")
        return

    A = pgv.AGraph(directed=True, strict=True, rankdir="TB")
    
    if morse_graph.number_of_nodes() == 0:
        A.add_node("empty", label="No Morse Sets", shape="box", style="filled", fillcolor="lightgray")
    else:
        # Define a color map for the nodes (same as morse sets)
        colors = plt.cm.get_cmap("turbo", len(sccs))
        
        for i, node_id in enumerate(morse_graph.nodes()):
            scc = sccs[node_id]
            node_label = f"M({node_id})\\n({len(scc)} boxes)"
            
            color_rgba = colors(node_id)
            # Apply alpha=0.4 blending with white background (same as morse_sets)
            alpha = 0.4
            blended_r = color_rgba[0] * alpha + (1 - alpha)
            blended_g = color_rgba[1] * alpha + (1 - alpha)
            blended_b = color_rgba[2] * alpha + (1 - alpha)
            fillcolor_hex = f"#{int(blended_r*255):02x}{int(blended_g*255):02x}{int(blended_b*255):02x}"
            
            A.add_node(node_id, label=node_label, shape="ellipse", style="filled", fillcolor=fillcolor_hex)
            
        for u, v in morse_graph.edges():
            A.add_edge(u, v)

    A.layout(prog="dot")
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    A.draw(output_path)
    # Remove individual save message - will be handled by caller
