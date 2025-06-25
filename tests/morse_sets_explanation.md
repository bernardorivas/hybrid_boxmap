# Explanation: Vertical Structures in Morse Sets

## Summary

You expected the Morse sets to consist mostly of horizontal strips, but there are vertical structures on the right boundary. Here's why this happens:

## Key Findings

### 1. **Three Main Morse Sets**
- **Morse Set 0**: 119 boxes including 20 on right boundary (y-indices 0-19, bottom strip)
- **Morse Set 1**: 119 boxes including 20 on right boundary (y-indices 80-99, top strip)  
- **Morse Set 2**: 200 boxes including 2 on right boundary (y-indices 49-50, middle)

### 2. **Self-Loops at Right Boundary**
- 40 boxes on the right boundary have self-loops
- These are primarily at the extreme y-values (top and bottom)

## Why Vertical Structures Exist

### The Mathematical Reason

1. **Reset Map Behavior at Extreme Y**:
   - At y = ±0.99: Reset maps to y = ±1.73 (outside domain!)
   - The system truncates these to stay within [-1, 1]
   - This creates "sticky" behavior at the boundaries

2. **Time Horizon τ = 1.42**:
   - From x = 0.99, it takes 0.01 time to reach x = 1
   - After reset to x = 0, it takes 1.0 time to return to x = 1
   - Remaining time: 0.42 (enough to cross boundary again)
   - This creates cycles that stay near the boundary

3. **Bloat Factor Effect**:
   - Default bloat factor is 0.1 (10% of box size)
   - For boxes at extreme y-values, after reset and flow, the bloated destination includes the source box
   - This creates the self-loops that form vertical structures

### Physical Interpretation

The vertical strips on the right boundary represent **"trapped" regions** where:

1. **Top/Bottom Strips**: Trajectories get amplified beyond the domain bounds by the 1.75 factor and are clipped back, creating recurrent behavior along vertical strips

2. **Horizontal Flow**: While the primary flow is horizontal (as you expected), the combination of:
   - Boundary effects
   - Reset map amplification
   - Domain truncation
   
   creates these unexpected vertical recurrent structures

## Why This Differs from Expectations

You expected horizontal strips because:
- The ODE has dx/dt = 1, dy/dt = 0 (pure horizontal flow)
- This suggests horizontal invariant sets

However, the **hybrid nature** (jumps at x = 1) combined with:
- Y-amplification factor of 1.75
- Domain bounds at y = ±1
- Sufficient time horizon for multiple jumps

creates more complex recurrent behavior than pure horizontal flow would suggest.

## Conclusion

The vertical structures are **real recurrent sets**, not artifacts. They represent regions where the hybrid dynamics create self-reinforcing cycles due to the interplay between:
- Horizontal flow
- Boundary jumps  
- Y-amplification
- Domain truncation

This is a great example of how hybrid systems can have counterintuitive behavior compared to their continuous-only counterparts!