# Interval Arithmetic Implementation Plan for HybridBoxMap

## Overview

This document outlines a plan for adding a `rigorous=True` option to `HybridBoxMap.compute()` that uses interval arithmetic to provide mathematically rigorous bounds on the forward image of boxes. Since hybrid systems have discontinuous dynamics (due to reset maps at jumps), the image f(box) will generally be a finite union of boxes rather than a single connected region.

## Key Challenges

1. **Discontinuous dynamics**: A single box may have trajectories that experience different numbers of jumps, leading to multiple disjoint destination regions
2. **Event detection with intervals**: Need rigorous bounds on when/where events occur within an interval
3. **Reset map evaluation**: Reset maps applied to intervals produce intervals
4. **Flow integration**: Need interval-based ODE integration between jumps

## Implementation Strategy

### Phase 1: Infrastructure Setup

1. **Add interval arithmetic library**:
   - Add `pyinterval` or `mpmath` as optional dependency in `pyproject.toml`
   - Create `hybrid_dynamics/src/interval_utils.py` for interval operations
   - Define `IntervalBox` class that represents boxes using intervals

2. **Extend Box class**:
   - Add `to_interval_box()` method to convert Box to IntervalBox
   - Add `from_interval_box()` static method
   - Implement interval-based intersection/union operations

### Phase 2: Core Implementation

1. **Create `RigorousBoxMap` class** in `hybrid_dynamics/src/rigorous_boxmap.py`:
   - Inherits from `HybridBoxMap`
   - Override `compute()` method to use interval arithmetic
   - Track unions of boxes for each source box

2. **Interval-based flow evaluation**:
   - Implement interval ODE integration (e.g., using validated numerics)
   - For each source box:
     - Convert to interval representation
     - Detect all possible event occurrences within time horizon
     - For each event scenario, compute rigorous bounds on destination
     - Apply reset map using interval arithmetic
     - Continue integration after reset

3. **Event detection with intervals**:
   - Evaluate event function on interval boxes
   - Use interval Newton method or bisection to isolate events
   - Track which portions of box experience events vs. pure flow

### Phase 3: Integration

1. **Modify `HybridBoxMap.compute()`**:
   - Add `rigorous: bool = False` parameter
   - If `rigorous=True`, delegate to `RigorousBoxMap`
   - Return structure that can represent unions of boxes

2. **Update data structures**:
   - Modify box map to store `Dict[int, Union[Set[int], List[Box]]]`
   - When rigorous=True, store actual interval boxes, not just indices
   - Add metadata about computation mode

### Phase 4: Testing and Validation

1. **Create `tests/test_interval.py`**:
   - Test interval arithmetic operations
   - Test RigorousBoxMap on rimless wheel example
   - Compare rigorous results with sampling-based approach
   - Verify that rigorous bounds contain all sampled trajectories

2. **Benchmarking**:
   - Compare computation time vs accuracy
   - Test on systems with varying numbers of jumps
   - Validate conservation of probability mass

## File Structure

```
hybrid_dynamics/src/
├── interval_utils.py          # New: Interval arithmetic utilities
├── interval_box.py            # New: IntervalBox class
├── rigorous_boxmap.py         # New: Rigorous computation implementation
├── hybrid_boxmap.py           # Modified: Add rigorous parameter
└── box.py                     # Modified: Add interval conversion methods

tests/
└── test_interval.py           # New: Comprehensive interval arithmetic tests
```

## Example Usage

```python
# Standard computation
box_map = HybridBoxMap.compute(grid, system, tau, rigorous=False)

# Rigorous computation with interval arithmetic
rigorous_map = HybridBoxMap.compute(grid, system, tau, rigorous=True)
# Returns unions of boxes for discontinuous mappings
```

## Implementation Details

### IntervalBox Class Design

```python
class IntervalBox:
    """Box represented using interval arithmetic."""
    
    def __init__(self, intervals: List[Interval]):
        """
        Args:
            intervals: List of intervals, one per dimension
        """
        self.intervals = intervals
        self.dimension = len(intervals)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is definitely inside box."""
        pass
    
    def intersects(self, other: 'IntervalBox') -> bool:
        """Check if boxes might intersect."""
        pass
    
    def hull(self, other: 'IntervalBox') -> 'IntervalBox':
        """Compute interval hull (smallest box containing both)."""
        pass
    
    def split(self, dimension: int) -> Tuple['IntervalBox', 'IntervalBox']:
        """Split box along given dimension."""
        pass
```

### Rigorous Flow Computation Algorithm

```python
def compute_rigorous_flow(box: IntervalBox, system: HybridSystem, tau: float) -> List[IntervalBox]:
    """
    Compute rigorous enclosure of flow from interval box.
    
    Returns:
        List of interval boxes (union represents image)
    """
    result_boxes = []
    
    # Check if event can occur within time horizon
    event_times = detect_events_interval(box, system, tau)
    
    if not event_times:
        # Pure continuous flow
        final_box = integrate_interval_ode(box, system.ode, tau)
        result_boxes.append(final_box)
    else:
        # Split computation at events
        for event_scenario in event_times:
            # Integrate to event
            pre_event_box = integrate_interval_ode(box, system.ode, event_scenario.time)
            
            # Apply reset map
            post_event_box = apply_interval_reset(pre_event_box, system.reset_map)
            
            # Continue integration
            remaining_time = tau - event_scenario.time
            final_box = integrate_interval_ode(post_event_box, system.ode, remaining_time)
            
            result_boxes.append(final_box)
    
    return result_boxes
```

### Event Detection with Intervals

```python
def detect_events_interval(box: IntervalBox, system: HybridSystem, tau: float) -> List[EventScenario]:
    """
    Rigorously detect all possible events within time horizon.
    
    Returns:
        List of possible event scenarios with time intervals
    """
    # Evaluate event function on interval box
    event_range = evaluate_interval_function(system.event_function, box)
    
    if 0 not in event_range:
        # No events possible
        return []
    
    # Use interval Newton or bisection to isolate events
    event_times = []
    
    # Subdivide box if necessary to isolate events
    if event_range.contains_zero():
        # Recursively subdivide to isolate zero crossings
        subboxes = box.subdivide()
        for subbox in subboxes:
            event_times.extend(detect_events_interval(subbox, system, tau))
    
    return event_times
```

## Benefits

1. **Mathematical rigor**: Guaranteed enclosures of all possible trajectories
2. **Uncertainty quantification**: Natural handling of parameter uncertainties  
3. **Verification**: Can prove properties about reachable sets
4. **Robustness**: No sampling artifacts or missed behaviors

## Comparison with CMGDB

CMGDB (Computational Morse Graph Database) also performs rigorous computations on dynamical systems. Key differences:

- **CMGDB**: Focuses on Morse graph construction with interval arithmetic for continuous systems
- **Our approach**: Extends to hybrid systems with discontinuous dynamics
- **Integration**: Could potentially use CMGDB's interval arithmetic infrastructure

## Future Extensions

1. **Parameter uncertainty**: Compute box maps for uncertain parameters
2. **Backward reachability**: Compute preimages using interval arithmetic
3. **Invariant set computation**: Rigorously verify invariant sets
4. **Optimization**: GPU acceleration for interval computations

## References

1. Tucker, W. (2011). Validated numerics: a short introduction to rigorous computations
2. Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009). Introduction to interval analysis
3. Nedialkov, N. S. (2011). Implementing a rigorous ODE solver through literate programming