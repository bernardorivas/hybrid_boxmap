"""
Configuration management for hybrid dynamics library.

This module provides centralized configuration for default parameters,
paths, and settings used throughout the hybrid dynamics library.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, Union


_PACKAGE_LOGGER_NAME = "hybrid_dynamics"
_MANAGED_HANDLER_ATTRIBUTE = "_hybrid_dynamics_managed_handler"


def _resolve_log_level(level: Union[int, str]) -> int:
    """Resolve a logging level name or value, rejecting unknown names."""
    if isinstance(level, int):
        return level

    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown logging level: {level}")
    return resolved


@dataclass
class GridConfig:
    """Configuration for grid-based analysis."""

    default_subdivisions: Tuple[int, int] = (30, 30)
    default_bounds: List[Tuple[float, float]] = None  # Will be set per system
    bloat_factor: float = 0.1
    out_of_bounds_tolerance: float = (
        1e-6  # Tolerance for discarding out-of-bounds points
    )
    sampling_modes: List[str] = None

    def __post_init__(self):
        if self.sampling_modes is None:
            self.sampling_modes = ["corners", "center", "random"]


@dataclass
class SimulationConfig:
    """Configuration for hybrid system simulation."""

    default_time_horizon: float = 5.0
    default_max_jumps: int = 20
    integration_rtol: float = 1e-6
    integration_atol: float = 1e-9
    event_tolerance: float = 1e-8
    jump_time_penalty_epsilon: float = 0.1  # Time deducted for each jump when penalty mode is enabled


@dataclass
class VisualizationConfig:
    """Configuration for plotting and visualization."""

    # Figure settings
    default_figsize: Tuple[float, float] = (10, 8)
    default_dpi: int = 150
    default_bbox_inches: str = "tight"

    # Trajectory settings
    trajectory_color: str = "blue"
    trajectory_alpha: float = 0.8
    trajectory_linewidth: float = 1.5

    # Jump/event settings
    jump_marker: str = "o"
    jump_color: str = "red"
    jump_size: float = 50
    jump_alpha: float = 0.8

    # Grid/box settings
    box_alpha: float = 0.6
    box_edgecolor: str = "none"
    grid_alpha: float = 0.3
    grid_color: str = "gray"
    grid_linewidth: float = 0.5

    # Event region settings
    event_color: str = "red"
    event_linewidth: float = 2.0
    event_alpha: float = 0.7

    # Plotting margins
    plot_margin: float = 0.1


@dataclass
class LoggingConfig:
    """Configuration for logging throughout the library."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    enable_file_logging: bool = False
    log_file: Optional[str] = None
    verbose: bool = False  # For print statement compatibility


class HybridConfig:
    """
    Centralized configuration manager for the hybrid dynamics library.

    This class provides default values and configuration management for
    grid analysis, simulation parameters, and visualization settings.
    """

    def __init__(self):
        self.grid = GridConfig()
        self.simulation = SimulationConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self._output_base_dir = None

    @property
    def output_dir(self) -> Path:
        """Get the base output directory for results and figures."""
        if self._output_base_dir is None:
            # Check environment variable first
            env_dir = os.getenv("HYBRID_OUTPUT_DIR")
            if env_dir:
                self._output_base_dir = Path(env_dir)
            else:
                # Default to ./output relative to current working directory
                self._output_base_dir = Path.cwd() / "output"
        return self._output_base_dir

    def set_output_dir(self, path: Union[str, Path]) -> None:
        """Set the base output directory."""
        self._output_base_dir = Path(path)

    def get_output_subdir(self, subdir: str) -> Path:
        """Get a subdirectory within the output directory."""
        full_path = self.output_dir / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    def get_figure_config(self) -> Dict[str, Any]:
        """Get standard figure configuration for matplotlib."""
        return {
            "figsize": self.visualization.default_figsize,
            "dpi": self.visualization.default_dpi,
        }

    def get_trajectory_style(self) -> Dict[str, Any]:
        """Get standard trajectory plotting style."""
        return {
            "color": self.visualization.trajectory_color,
            "alpha": self.visualization.trajectory_alpha,
            "linewidth": self.visualization.trajectory_linewidth,
        }

    def get_jump_style(self) -> Dict[str, Any]:
        """Get standard jump marker style."""
        return {
            "marker": self.visualization.jump_marker,
            "color": self.visualization.jump_color,
            "s": self.visualization.jump_size,
            "alpha": self.visualization.jump_alpha,
            "zorder": 10,  # Ensure jumps are on top
        }

    def get_box_style(self, facecolor: str = "lightblue") -> Dict[str, Any]:
        """Get standard box plotting style."""
        return {
            "facecolor": facecolor,
            "alpha": self.visualization.box_alpha,
            "edgecolor": self.visualization.box_edgecolor,
        }

    def get_event_style(self) -> Dict[str, Any]:
        """Get standard event region style."""
        return {
            "colors": self.visualization.event_color,
            "linewidths": self.visualization.event_linewidth,
            "alpha": self.visualization.event_alpha,
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        raise ValueError(f"Unknown config option: {section}.{key}")
            else:
                raise ValueError(f"Unknown config section: {section}")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to a dictionary."""
        return {
            "grid": {
                "default_subdivisions": self.grid.default_subdivisions,
                "bloat_factor": self.grid.bloat_factor,
                "sampling_modes": self.grid.sampling_modes,
            },
            "simulation": {
                "default_time_horizon": self.simulation.default_time_horizon,
                "default_max_jumps": self.simulation.default_max_jumps,
                "integration_rtol": self.simulation.integration_rtol,
                "integration_atol": self.simulation.integration_atol,
                "event_tolerance": self.simulation.event_tolerance,
                "jump_time_penalty_epsilon": self.simulation.jump_time_penalty_epsilon,
            },
            "visualization": {
                "default_figsize": self.visualization.default_figsize,
                "default_dpi": self.visualization.default_dpi,
                "trajectory_color": self.visualization.trajectory_color,
                "trajectory_alpha": self.visualization.trajectory_alpha,
                "trajectory_linewidth": self.visualization.trajectory_linewidth,
                "jump_color": self.visualization.jump_color,
                "jump_size": self.visualization.jump_size,
                "event_color": self.visualization.event_color,
                "event_linewidth": self.visualization.event_linewidth,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "date_format": self.logging.date_format,
                "enable_file_logging": self.logging.enable_file_logging,
                "log_file": self.logging.log_file,
                "verbose": self.logging.verbose,
            },
        }

    def get_logger(self, name: str) -> logging.Logger:
        """Return a logger without installing handlers.

        This preserves the historical ``config.get_logger(name)`` API while
        leaving handler selection to the application.  Call
        :meth:`configure_logging` when the library should install its own
        console and optional file handlers.
        """
        return logging.getLogger(name)

    def configure_logging(
        self,
        *,
        level: Optional[Union[int, str]] = None,
        stream: Optional[IO[str]] = None,
        enable_file_logging: Optional[bool] = None,
        log_file: Optional[Union[str, Path]] = None,
    ) -> logging.Logger:
        """Explicitly configure logging for the ``hybrid_dynamics`` package.

        Repeated calls replace only handlers previously installed by this
        method.  User-installed handlers and the package ``NullHandler`` are
        left untouched.
        """
        resolved_level = _resolve_log_level(
            self.logging.level if level is None else level,
        )
        file_logging_enabled = (
            self.logging.enable_file_logging
            if enable_file_logging is None
            else enable_file_logging
        )
        resolved_log_file = self.logging.log_file if log_file is None else log_file
        if log_file is not None and enable_file_logging is None:
            file_logging_enabled = True

        formatter = logging.Formatter(
            self.logging.format,
            self.logging.date_format,
        )
        new_handlers: List[logging.Handler] = []

        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
        new_handlers.append(console_handler)

        try:
            if file_logging_enabled:
                if not resolved_log_file:
                    raise ValueError(
                        "log_file must be set when file logging is enabled",
                    )
                file_handler = logging.FileHandler(resolved_log_file)
                file_handler.setLevel(resolved_level)
                file_handler.setFormatter(formatter)
                setattr(file_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
                new_handlers.append(file_handler)
        except Exception:
            for handler in new_handlers:
                handler.close()
            raise

        package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
        for handler in list(package_logger.handlers):
            if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
                package_logger.removeHandler(handler)
                handler.close()

        package_logger.setLevel(resolved_level)
        package_logger.propagate = False
        for handler in new_handlers:
            package_logger.addHandler(handler)

        return package_logger


# Global configuration instance
# Users can import and modify this directly:
# from hybrid_dynamics.config import config
# config.simulation.default_max_jumps = 50
config = HybridConfig()


def get_logger(name: str) -> logging.Logger:
    """Return a library logger without installing handlers."""
    return config.get_logger(name)


def configure_logging(
    *,
    level: Optional[Union[int, str]] = None,
    stream: Optional[IO[str]] = None,
    enable_file_logging: Optional[bool] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Explicitly configure package console and optional file logging."""
    return config.configure_logging(
        level=level,
        stream=stream,
        enable_file_logging=enable_file_logging,
        log_file=log_file,
    )


# Convenience functions for common access patterns
def get_default_grid_subdivisions() -> Tuple[int, int]:
    """Get default grid subdivisions."""
    return config.grid.default_subdivisions


def get_default_bloat_factor() -> float:
    """Get default bloat factor for box map computation."""
    return config.grid.bloat_factor


def get_default_max_jumps() -> int:
    """Get default maximum number of jumps for simulation."""
    return config.simulation.default_max_jumps


def get_default_time_horizon() -> float:
    """Get default time horizon for simulation."""
    return config.simulation.default_time_horizon


def get_output_dir() -> Path:
    """Get the current output directory."""
    return config.output_dir


def set_output_dir(path: Union[str, Path]) -> None:
    """Set the output directory."""
    config.set_output_dir(path)


def get_jump_time_penalty_epsilon() -> float:
    """Get the time penalty epsilon value for each jump in penalty mode."""
    return config.simulation.jump_time_penalty_epsilon
