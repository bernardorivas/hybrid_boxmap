"""
Utilities for parallel processing with hybrid systems.

This module provides workarounds for the pickling issues with closures
in hybrid systems by using global functions and serialization helpers.
"""

import pickle
from typing import Callable, Dict, Optional, Tuple

import dill
import numpy as np
import numpy.typing as npt

from .config import config
from .hybrid_system import HybridSystem

# Global storage for system instances during parallel processing
_global_system_cache: Dict[int, HybridSystem] = {}
_global_tau_cache: Dict[int, float] = {}
_global_debug_cache: Dict[int, dict] = {}

logger = config.get_logger(__name__)


def register_system_for_parallel(
    system: HybridSystem, tau: float, worker_id: int = 0,
) -> int:
    """
    Register a hybrid system for parallel processing.

    Args:
        system: The hybrid system to register
        tau: The time horizon
        worker_id: Unique identifier for this registration

    Returns:
        The worker_id used for registration
    """
    _global_system_cache[worker_id] = system
    _global_tau_cache[worker_id] = tau
    _global_debug_cache[worker_id] = {}
    return worker_id


def clear_system_cache(worker_id: Optional[int] = None):
    """Clear the global system cache."""
    if worker_id is not None:
        _global_system_cache.pop(worker_id, None)
        _global_tau_cache.pop(worker_id, None)
        _global_debug_cache.pop(worker_id, None)
    else:
        _global_system_cache.clear()
        _global_tau_cache.clear()
        _global_debug_cache.clear()


def _global_evaluate_flow(
    args: Tuple[npt.NDArray[np.float64], int],
) -> Tuple[npt.NDArray[np.float64], int]:
    """
    Global function for evaluating flow that can be pickled.

    Args:
        args: Tuple of (point, worker_id)

    Returns:
        Tuple of (final_state, num_jumps)
    """
    point, worker_id = args

    if worker_id not in _global_system_cache:
        logger.error("System not found for worker_id %s", worker_id)
        return (np.full(point.shape[0], np.nan), -1)

    system = _global_system_cache[worker_id]
    tau = _global_tau_cache[worker_id]
    debug_info = _global_debug_cache[worker_id]

    try:
        traj = system.simulate(point, (0, tau), debug_info=debug_info)

        # Check if the simulation reached tau
        if traj.total_duration >= tau:
            final_state = traj.interpolate(tau)
            num_jumps = traj.num_jumps
            return (final_state, num_jumps)

        # If tau is not reached, use the last available point
        if traj.segments and traj.segments[-1].state_values.size > 0:
            final_state = traj.segments[-1].state_values[-1]
            num_jumps = traj.num_jumps
            return (final_state, num_jumps)

        return (np.full(point.shape[0], np.nan), -1)

    except Exception as e:
        logger.debug("Failed to evaluate flow at point %s: %s", point, e)
        return (np.full(point.shape[0], np.nan), -1)


def create_parallel_evaluator(system: HybridSystem, tau: float) -> Callable:
    """
    Create a picklable evaluator function for a hybrid system.

    This uses dill to serialize the system if possible, falling back
    to the global cache approach if needed.

    Args:
        system: The hybrid system
        tau: Time horizon

    Returns:
        A picklable evaluation function
    """
    # Try to serialize with dill first
    try:
        serialized_system = dill.dumps(system)
        serialized_tau = dill.dumps(tau)

        def dill_evaluator(
            point: npt.NDArray[np.float64],
        ) -> Tuple[npt.NDArray[np.float64], int]:
            """Evaluator using dill serialization."""
            local_system = dill.loads(serialized_system)
            local_tau = dill.loads(serialized_tau)
            debug_info = {}

            try:
                traj = local_system.simulate(
                    point, (0, local_tau), debug_info=debug_info,
                )

                if traj.total_duration >= local_tau:
                    final_state = traj.interpolate(local_tau)
                    return (final_state, traj.num_jumps)

                if traj.segments and traj.segments[-1].state_values.size > 0:
                    final_state = traj.segments[-1].state_values[-1]
                    return (final_state, traj.num_jumps)

                return (np.full(point.shape[0], np.nan), -1)

            except Exception as e:
                logger.debug("Failed to evaluate flow: %s", e)
                return (np.full(point.shape[0], np.nan), -1)

        logger.debug("Successfully created dill-based parallel evaluator")
        return dill_evaluator

    except Exception as e:
        logger.warning(
            "Failed to serialize system with dill: %s. Falling back to global cache.", e,
        )

        # Fall back to global cache approach
        worker_id = id(system)  # Use object id as unique identifier
        register_system_for_parallel(system, tau, worker_id)

        def global_evaluator(
            point: npt.NDArray[np.float64],
        ) -> Tuple[npt.NDArray[np.float64], int]:
            """Evaluator using global cache."""
            return _global_evaluate_flow((point, worker_id))

        return global_evaluator


def test_parallel_capability(system: HybridSystem) -> bool:
    """
    Test if a hybrid system can be used in parallel processing.

    Args:
        system: The hybrid system to test

    Returns:
        True if the system can be parallelized
    """
    try:
        # Try standard pickle first
        pickle.dumps(system)
        return True
    except Exception:
        try:
            # Try dill
            dill.dumps(system)
            return True
        except Exception:
            return False

